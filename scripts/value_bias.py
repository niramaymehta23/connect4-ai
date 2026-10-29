"""Measure a value network's side-to-move bias against exactly solved positions.

The value convention is +1 = the player to move wins. A sound net's error should
not depend on whose turn it is. For each model this prints, on a fixed seeded set
of positions solved exactly by a small negamax solver:
  error_red_to_move / error_yellow_to_move  mean (prediction - truth) per side
  mover_bias   mean of the two: > 0 means "whoever is to move is winning"
  colour_bias  half their difference: > 0 means Red is favoured regardless of turn
  mse          against the exact result
  turn_plane_effect  mean |v - v with channel 2 flipped|: how much the colour
                     plane alone moves the value (0 when the net ignores it)
  opening      Red-perspective value before and after Red's centre opening (both
               should be positive: Connect Four is a first-player win)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Bitboard solver: 7 bits per column (6 cells + a sentinel), bit 0 = bottom row.
_H1 = 7
_FULL = sum(((1 << 6) - 1) << (c * _H1) for c in range(7))
_ORDER = [3, 2, 4, 1, 5, 0, 6]


class _Budget(Exception):
    pass


def _won(p):
    for s in (1, _H1, 6, 8):
        m = p & (p >> s)
        if m & (m >> (2 * s)):
            return True
    return False


def solve(board, mover, node_limit=200_000):
    """Exact +1/0/-1 for `mover`, or None if the search exceeds `node_limit`."""
    cur = mask = 0
    for c in range(7):
        for r in range(6):
            cell = board[5 - r][c]
            if cell:
                mask |= 1 << (c * _H1 + r)
                if cell == mover:
                    cur |= 1 << (c * _H1 + r)
    table, nodes = {}, [0]

    def negamax(cur, mask, alpha, beta):
        nodes[0] += 1
        if nodes[0] > node_limit:
            raise _Budget
        if mask == _FULL:
            return 0
        children = []
        for c in _ORDER:
            if not mask & (1 << (5 + c * _H1)):
                child = mask | (mask + (1 << (c * _H1)))
                if _won(cur | (child ^ mask)):
                    return 1
                children.append(child)
        key = cur + mask
        low, high = table.get(key, (-1, 1))
        alpha, beta = max(alpha, low), min(beta, high)
        if alpha >= beta:
            return alpha
        start, best = alpha, -1
        for child in children:
            best = max(best, -negamax(cur ^ mask, child, -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if best <= start:
            high = best
        elif best >= beta:
            low = best
        else:
            low = high = best
        table[key] = (low, high)
        return best

    try:
        return negamax(cur, mask, -1, 1)
    except _Budget:
        return None


def _tactical_move(board, mover, rng):
    """Win if possible, else block an immediate loss, else play at random."""
    from src.game.board import clone, place
    from src.game.rules import valid_cols, check_win
    cols = valid_cols(board)
    for token in (mover, 3 - mover):
        for col in cols:
            child = clone(board)
            place(child, col, token)
            if check_win(child, token):
                return col
    return rng.choice(cols)


def solved_positions(n, seed, min_pieces=12):
    """`n` exactly solved (board, mover, value) triples from seeded tactical-random games.

    Purely random games leave immediate wins on the board, so the side to move
    would usually win and a mover-favouring net would look accurate.
    """
    from src.game.board import create_board, clone, place
    from src.game.constants import AI, HUMAN
    from src.game.rules import check_win, is_full
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board, mover, history = create_board(), AI, []
        while True:
            history.append((clone(board), mover))
            place(board, _tactical_move(board, mover, rng), mover)
            if check_win(board, mover) or is_full(board):
                break
            mover = HUMAN if mover == AI else AI
        candidates = [item for item in history if sum(map(bool, sum(item[0], []))) >= min_pieces]
        if candidates:
            board, mover = rng.choice(candidates)
            value = solve(board, mover)
            if value is not None:
                out.append((board, mover, value))
    return out


def measure(net, items):
    import torch
    from src.game.board import create_board, place
    from src.game.constants import AI
    from src.nn.encode import encode_board

    x = torch.stack([encode_board(b, m) for b, m, _ in items])
    truth = torch.tensor([float(v) for _, _, v in items])
    red = torch.tensor([m == AI for _, m, _ in items])
    flipped = x.clone()
    flipped[:, 2] = 1.0 - flipped[:, 2]
    with torch.no_grad():
        pred = net(x).squeeze(1)
        pred_flipped = net(flipped).squeeze(1)
    error = pred - truth
    err_red, err_yellow = error[red].mean().item(), error[~red].mean().item()

    board = create_board()
    with torch.no_grad():
        before = net(encode_board(board, AI).unsqueeze(0)).item()
        place(board, 3, AI)
        after = -net(encode_board(board, 3 - AI).unsqueeze(0)).item()
    return {
        'positions': len(items),
        'error_red_to_move': err_red,
        'error_yellow_to_move': err_yellow,
        'mover_bias': (err_red + err_yellow) / 2,
        'colour_bias': (err_red - err_yellow) / 2,
        'mse': (error ** 2).mean().item(),
        'turn_plane_effect': (pred - pred_flipped).abs().mean().item(),
        'opening_red_value': [before, after],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', type=Path)
    p.add_argument('--positions', type=int, default=400)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()

    import torch
    from src.nn.network import Connect4Net
    torch.set_num_threads(1)
    items = solved_positions(args.positions, args.seed)
    for path in args.models:
        net = Connect4Net.load(path, torch.device('cpu'))
        result = measure(net, items)
        rounded = {k: [round(x, 3) for x in v] if isinstance(v, list) else round(v, 3) if isinstance(v, float) else v
                   for k, v in result.items()}
        print(json.dumps({'model': str(path), **rounded}), flush=True)


if __name__ == '__main__':
    main()
