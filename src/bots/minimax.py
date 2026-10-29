# Minimax bot with alpha-beta pruning. Accepts a weights dict.
import math
from ..game.rules import valid_cols, check_win, is_full
from ..game.board import place, unplace
from ..game.constants import AI, HUMAN, ROWS, COLS, WIN, EMPTY

DEFAULT_WEIGHTS = {
    'centreBonus':   3,
    'three':         5,
    'two':           2,
    'win':           150,    # seed raised from 100; observed convergence zone is 80-135
    'terminalWin':   1_000_000,
    'depthBonus':    0,
    'threatPenalty': 0,
}


def minimax_bot(board, token, depth=6, weights=None):
    """
    Return the best column for `token` using minimax + alpha-beta.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    cols = _ordered_cols(valid_cols(board))
    opponent = HUMAN if token == AI else AI

    best_score = -math.inf
    best_col = cols[0]

    for col in cols:
        place(board, col, token)
        score = _alphabeta(board, token, opponent, depth - 1,
                           -math.inf, math.inf, False, weights)
        unplace(board, col)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def make_minimax_bot(depth=6, weights=None):
    """
    Return a bot function bound to the given depth and weights.
    The returned function matches the standard bot interface: (board, token) -> col.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    return lambda board, token: minimax_bot(board, token, depth, weights)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _alphabeta(board, token, opponent, depth, alpha, beta, is_maximising, weights):
    """
    Alpha-beta minimax.
    `token` is always the maximising player (the bot we're solving for).
    `is_maximising` tracks whose turn it is at this node.
    """
    # Terminal: check if the player who just moved has won.
    just_moved = opponent if is_maximising else token
    if check_win(board, just_moved):
        score = weights['terminalWin'] if just_moved == token else -weights['terminalWin']
        return score + (depth if just_moved == token else -depth) * weights['depthBonus']
    if is_full(board):
        return 0
    if depth == 0:
        return _evaluate(board, token, weights)

    cols = _ordered_cols(valid_cols(board))

    if is_maximising:
        best = -math.inf
        for col in cols:
            place(board, col, token)
            best = max(best, _alphabeta(board, token, opponent, depth - 1,
                                        alpha, beta, False, weights))
            unplace(board, col)
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # β cut-off
        return best
    else:
        best = math.inf
        for col in cols:
            place(board, col, opponent)
            best = min(best, _alphabeta(board, token, opponent, depth - 1,
                                        alpha, beta, True, weights))
            unplace(board, col)
            beta = min(beta, best)
            if alpha >= beta:
                break  # α cut-off
        return best


def minimax_trace(board, token, depth=4, weights=None, max_events=4000):
    """
    Run the same root loop and alpha-beta search as `minimax_bot`, recording it.

    Returns {"best_col", "depth", "root", "events", "truncated"}. `root` lists each
    root column in search order with its score, nodes searched and moves pruned.
    `events` (capped at `max_events`) are {"path", "kind", "score"} in visit order,
    where kind is "leaf", "prune" (score = moves skipped) or "done" (root column finished).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    board = [row[:] for row in board]
    opponent = HUMAN if token == AI else AI
    events = []
    truncated = False

    def emit(path, kind, score):
        nonlocal truncated
        if len(events) < max_events:
            events.append({"path": list(path), "kind": kind, "score": score})
        else:
            truncated = True

    def search(depth, alpha, beta, is_maximising, path, stats):
        stats["nodes"] += 1
        just_moved = opponent if is_maximising else token
        if check_win(board, just_moved):
            score = weights['terminalWin'] if just_moved == token else -weights['terminalWin']
            score += (depth if just_moved == token else -depth) * weights['depthBonus']
            emit(path, "leaf", score)
            return score
        if is_full(board):
            emit(path, "leaf", 0)
            return 0
        if depth == 0:
            score = _evaluate(board, token, weights)
            emit(path, "leaf", score)
            return score

        cols = _ordered_cols(valid_cols(board))
        best = -math.inf if is_maximising else math.inf
        for index, col in enumerate(cols):
            place(board, col, token if is_maximising else opponent)
            path.append(col)
            score = search(depth - 1, alpha, beta, not is_maximising, path, stats)
            path.pop()
            unplace(board, col)
            if is_maximising:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if alpha >= beta:
                skipped = len(cols) - index - 1
                if skipped:
                    stats["pruned"] += skipped
                    emit(path, "prune", skipped)
                break
        return best

    root = []
    best_score = -math.inf
    best_col = None
    for col in _ordered_cols(valid_cols(board)):
        stats = {"nodes": 0, "pruned": 0}
        place(board, col, token)
        score = search(depth - 1, -math.inf, math.inf, False, [col], stats)
        unplace(board, col)
        emit([col], "done", score)
        root.append({"col": col, "score": score, **stats})
        if best_col is None or score > best_score:
            best_score = score
            best_col = col

    return {"best_col": best_col, "depth": depth, "root": root, "events": events, "truncated": truncated}


def _evaluate(board, token, weights):
    """
    Heuristic evaluation of a non-terminal position from `token`'s perspective.
    Scores all WIN-length windows and applies the centre-column bonus.
    """
    opponent = HUMAN if token == AI else AI
    score = 0

    # Centre column bonus
    mid = COLS // 2
    for r in range(ROWS):
        if board[r][mid] == token:
            score += weights['centreBonus']
        elif board[r][mid] == opponent:
            score -= weights['centreBonus']

    # Horizontal windows
    for r in range(ROWS):
        for c in range(COLS - WIN + 1):
            score += _score_window(
                [board[r][c], board[r][c + 1], board[r][c + 2], board[r][c + 3]],
                token, opponent, weights,
            )

    # Vertical windows
    for c in range(COLS):
        for r in range(ROWS - WIN + 1):
            score += _score_window(
                [board[r][c], board[r + 1][c], board[r + 2][c], board[r + 3][c]],
                token, opponent, weights,
            )

    # Diagonal down-right (\)
    for r in range(ROWS - WIN + 1):
        for c in range(COLS - WIN + 1):
            score += _score_window(
                [board[r][c], board[r + 1][c + 1], board[r + 2][c + 2], board[r + 3][c + 3]],
                token, opponent, weights,
            )

    # Diagonal up-right (/)
    for r in range(WIN - 1, ROWS):
        for c in range(COLS - WIN + 1):
            score += _score_window(
                [board[r][c], board[r - 1][c + 1], board[r - 2][c + 2], board[r - 3][c + 3]],
                token, opponent, weights,
            )

    return score


def _score_window(window, token, opponent, weights):
    """Score a single 4-cell window from `token`'s perspective."""
    token_count = window.count(token)
    opp_count   = window.count(opponent)
    empty_count = window.count(EMPTY)

    # Mixed window — blocked in both directions, no value
    if token_count > 0 and opp_count > 0:
        return 0

    # Token's threats
    if token_count == 4:
        return weights['win']
    if token_count == 3 and empty_count == 1:
        return weights['three']
    if token_count == 2 and empty_count == 2:
        return weights['two']

    # Opponent's threats (negative)
    if opp_count == 3 and empty_count == 1:
        return -(weights['three'] * (1 + weights['threatPenalty']))
    if opp_count == 2 and empty_count == 2:
        return -weights['two']

    return 0


def _ordered_cols(cols):
    """
    Sort valid columns so centre columns are tried first —
    dramatically improves alpha-beta pruning efficiency.
    """
    mid = COLS // 2
    return sorted(cols, key=lambda c: abs(c - mid))


# ---------------------------------------------------------------------------
# Public utility used by MCTS weighted rollouts
# ---------------------------------------------------------------------------

def score_move(board, col, token, weights):
    """
    Score a single candidate move without recursion.

    Places token in col, evaluates all WIN-length windows that intersect
    the cell where the token lands, unplaces, and returns the raw heuristic
    score.  Scoring only the intersecting windows (rather than the full board)
    keeps this fast enough to call on every rollout step.

    Returns -inf if the column is full.
    """
    row = place(board, col, token)
    if row == -1:
        return -math.inf  # column already full

    opponent = HUMAN if token == AI else AI
    score = 0

    # Centre-column bonus: apply if the landing cell is in the centre column.
    mid = COLS // 2
    if col == mid:
        score += weights['centreBonus']

    # Collect every WIN-length window that passes through (row, col).

    # Horizontal — row fixed, c_start varies.
    for c_start in range(max(0, col - WIN + 1), min(COLS - WIN, col) + 1):
        window = [board[row][c_start + i] for i in range(WIN)]
        score += _score_window(window, token, opponent, weights)

    # Vertical — col fixed, r_start varies.
    for r_start in range(max(0, row - WIN + 1), min(ROWS - WIN, row) + 1):
        window = [board[r_start + i][col] for i in range(WIN)]
        score += _score_window(window, token, opponent, weights)

    # Diagonal down-right (\): window starts at (row-k, col-k).
    for k in range(WIN):
        rs, cs = row - k, col - k
        if 0 <= rs <= ROWS - WIN and 0 <= cs <= COLS - WIN:
            window = [board[rs + i][cs + i] for i in range(WIN)]
            score += _score_window(window, token, opponent, weights)

    # Diagonal up-right (/): window starts at (row+k, col-k).
    for k in range(WIN):
        rs, cs = row + k, col - k
        if WIN - 1 <= rs < ROWS and 0 <= cs <= COLS - WIN:
            window = [board[rs - i][cs + i] for i in range(WIN)]
            score += _score_window(window, token, opponent, weights)

    unplace(board, col)
    return score
