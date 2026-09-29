"""Use a value model trained on the player-to-move perspective."""

from ..game.board import clone, place
from ..game.constants import AI, HUMAN
from ..game.rules import check_win, is_full, valid_cols


def score_successors(board, token, value_fn):
    """Exact terminal values; otherwise negate the next player's estimated value.

    value_fn(board, player_to_move) -> float. The caller's board is never changed.
    """
    opponent = HUMAN if token == AI else AI
    scores = [None] * 7
    for col in valid_cols(board):
        child = clone(board)
        place(child, col, token)
        if check_win(child, token):
            scores[col] = 1.0
        elif is_full(child):
            scores[col] = 0.0
        else:
            scores[col] = -float(value_fn(child, opponent))
    return scores
