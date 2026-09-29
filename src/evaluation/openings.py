"""Generate varied, reproducible, non-terminal Connect Four openings."""

import random

from ..game.board import create_board, place
from ..game.constants import AI, HUMAN
from ..game.rules import check_win, is_full, valid_cols


def _opponent(token):
    return HUMAN if token == AI else AI


def generate_openings(count, plies=4, seed=0):
    """
    Return `count` unique opening sequences.

    Openings are random but reproducible for a given seed. Any sequence that ends
    the game is discarded, because the agents must still have a position to play.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if plies < 0:
        raise ValueError("plies cannot be negative")
    if plies == 0:
        if count > 1:
            raise ValueError("only one unique zero-ply opening exists")
        return [()]

    rng = random.Random(seed)
    openings = set()
    attempts = 0
    max_attempts = max(1_000, count * 100)

    while len(openings) < count and attempts < max_attempts:
        attempts += 1
        board = create_board()
        token = AI
        moves = []
        valid = True

        for _ in range(plies):
            cols = valid_cols(board)
            if not cols:
                valid = False
                break
            col = rng.choice(cols)
            place(board, col, token)
            moves.append(col)
            if check_win(board, token) or is_full(board):
                valid = False
                break
            token = _opponent(token)

        if valid:
            openings.add(tuple(moves))

    if len(openings) < count:
        raise ValueError(
            f"could only generate {len(openings)} unique {plies}-ply openings"
        )
    return sorted(openings)
