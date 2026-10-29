"""Position and game analysis from Red's perspective, independent of the evaluator.

A value_fn(board, player_to_move) -> float returns +1 when the player to move is
winning (the src/nn convention). It may expose `many(items)` taking a list of
(board, player_to_move) pairs to evaluate a batch at once.
"""
import math

from ..bots.minimax import DEFAULT_WEIGHTS, _evaluate
from ..game.board import clone, create_board, place
from ..game.constants import AI, COLS, EMPTY, HUMAN, ROWS
from ..game.rules import check_win, is_full, valid_cols
from ..nn.scoring import score_successors


def _other(token):
    return HUMAN if token == AI else AI


def _terminal_red_value(board):
    if check_win(board, AI):
        return 1.0
    if check_win(board, HUMAN):
        return -1.0
    if is_full(board):
        return 0.0
    return None


def _red_values(items, value_fn):
    """Red-perspective values for (board, player_to_move) pairs; terminals are exact."""
    values = [_terminal_red_value(board) for board, _ in items]
    pending = [index for index, value in enumerate(values) if value is None]
    if pending:
        batch = [items[index] for index in pending]
        many = getattr(value_fn, "many", None)
        raw = many(batch) if many else [value_fn(board, player) for board, player in batch]
        for index, value in zip(pending, raw):
            value = float(value)
            values[index] = value if items[index][1] == AI else -value
    return values


def red_value(board, to_move, value_fn):
    return _red_values([(board, to_move)], value_fn)[0]


def cell_heatmap(board, to_move, value_fn):
    """6x7 grid of how much each cell moves Red's value; None where nothing applies.

    Occupied cell: value now minus value with that piece removed (same side to move).
    Each column's playable cell: value after `to_move` plays there minus value now.
    Other empty cells, and playable cells once the game is over, are None.
    """
    probes = [(clone(board), to_move)]
    targets = []
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] != EMPTY:
                probe = clone(board)
                probe[row][col] = EMPTY
                probes.append((probe, to_move))
                targets.append((row, col, "occupied"))
    if _terminal_red_value(board) is None:
        for col in valid_cols(board):
            child = clone(board)
            row = place(child, col, to_move)
            probes.append((child, _other(to_move)))
            targets.append((row, col, "playable"))

    values = _red_values(probes, value_fn)
    base = values[0]
    cells = [[None] * COLS for _ in range(ROWS)]
    for (row, col, kind), value in zip(targets, values[1:]):
        cells[row][col] = base - value if kind == "occupied" else value - base
    return cells


def analyze_game(moves, value_fn, heatmaps=False):
    """Evaluate every position of a game and find the pivot ply.

    Returns {"plies": [{ply, col, token, value, delta, swing}], "values": Red value per
    position (len(moves) + 1), "pivot": index into plies, "winner": AI/HUMAN/None,
    "heatmaps": per-position grids or None}.

    `delta` compares consecutive positions. `swing` compares the position after this
    ply with the one two plies earlier, which has the same side to move: a value
    network that favours the side to move (ours does, by up to ~1.0) biases every
    delta, but that bias cancels in a swing. The pivot is the largest |swing|
    (largest |delta| for a one-move game).
    Raises ValueError for an illegal column or a move after the game ended.
    """
    board, token = create_board(), AI
    positions = [(clone(board), token)]
    plies = []
    winner = None
    for index, col in enumerate(moves):
        if winner is not None or is_full(board):
            raise ValueError(f"move {index + 1} comes after the game ended")
        if not isinstance(col, int) or isinstance(col, bool) or col not in valid_cols(board):
            raise ValueError(f"illegal column {col!r} at move {index + 1}")
        place(board, col, token)
        if check_win(board, token):
            winner = token
        plies.append({"ply": index + 1, "col": col, "token": token})
        token = _other(token)
        positions.append((clone(board), token))

    values = _red_values(positions, value_fn)
    for index, ply in enumerate(plies):
        ply["value"] = values[index + 1]
        ply["delta"] = values[index + 1] - values[index]
        ply["swing"] = values[index + 1] - values[index - 1] if index >= 1 else None
    if len(plies) > 1:
        pivot = max(range(1, len(plies)), key=lambda i: abs(plies[i]["swing"]))
    else:
        pivot = 0 if plies else None
    grids = [cell_heatmap(position, player, value_fn) for position, player in positions] if heatmaps else None
    return {"plies": plies, "values": values, "pivot": pivot, "winner": winner, "heatmaps": grids}


def self_play_replay(value_fn):
    """Greedy one-move-lookahead self-play from the empty board, with Red values."""
    board, token = create_board(), AI
    moves = []
    values = [red_value(board, token, value_fn)]
    while _terminal_red_value(board) is None:
        scores = score_successors(board, token, value_fn)
        col = max((c for c, score in enumerate(scores) if score is not None), key=lambda c: scores[c])
        place(board, col, token)
        moves.append(col)
        token = _other(token)
        values.append(red_value(board, token, value_fn))
    winner = AI if check_win(board, AI) else HUMAN if check_win(board, HUMAN) else None
    return {"moves": moves, "values": values, "winner": winner}


def heuristic_value_fn(weights=None, scale=200.0):
    """Minimax evaluation squashed into (-1, 1), for when no neural model is available."""
    weights = weights or DEFAULT_WEIGHTS
    return lambda board, to_move: math.tanh(_evaluate(board, to_move, weights) / scale)
