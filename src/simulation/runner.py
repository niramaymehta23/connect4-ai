# Play one or more complete games between two bots and return results.
import time
from ..game.board import create_board, place
from ..game.rules import check_win, is_full, valid_cols
from ..game.constants import AI, HUMAN


def _opponent(token):
    return HUMAN if token == AI else AI


def prepare_opening(opening_moves=None):
    """Apply a legal, non-terminal opening and return board, next token, history."""
    board = create_board()
    token = AI
    history = []

    for col in opening_moves or ():
        if col not in valid_cols(board):
            raise ValueError(f"illegal opening move: column {col}")
        place(board, col, token)
        history.append(col)
        if check_win(board, token) or is_full(board):
            raise ValueError("opening moves must not contain a finished game")
        token = _opponent(token)

    return board, token, history


def run_game(bot_a, bot_b, bot_a_goes_first=True, opening_moves=None):
    """
    Run a single game between two bot functions.

    Token assignment: the first player always uses AI (2), second uses HUMAN (1).
    `bot_a_goes_first` controls which bot is the first player.

    `opening_moves` can provide a legal, non-terminal sequence of columns. This is
    useful for evaluating deterministic agents across more than one trajectory.

    Returns dict with winner, move count, duration, and the complete column history.
    """
    board, current_token, move_history = prepare_opening(opening_moves)
    bot_for_token = {
        AI: bot_a if bot_a_goes_first else bot_b,
        HUMAN: bot_b if bot_a_goes_first else bot_a,
    }
    owner_for_token = {
        AI: 'A' if bot_a_goes_first else 'B',
        HUMAN: 'B' if bot_a_goes_first else 'A',
    }

    start = time.perf_counter()

    while True:
        token = current_token
        col = bot_for_token[token](board, token)
        if col not in valid_cols(board):
            raise ValueError(f"bot selected illegal column: {col}")

        place(board, col, token)
        move_history.append(col)

        if check_win(board, token):
            duration = (time.perf_counter() - start) * 1000
            return {
                'winner': owner_for_token[token],
                'moves': len(move_history),
                'duration': duration,
                'move_history': move_history,
            }

        if is_full(board):
            duration = (time.perf_counter() - start) * 1000
            return {
                'winner': None,
                'moves': len(move_history),
                'duration': duration,
                'move_history': move_history,
            }

        current_token = _opponent(token)


def run_games(bot_a, bot_b, n=100):
    """
    Run N games between bot_a and bot_b, alternating who goes first each game.
    Returns win/loss/draw counts from bot_a's perspective.

    Returns dict: { 'wins': int, 'losses': int, 'draws': int }
    """
    wins = losses = draws = 0

    for i in range(n):
        result = run_game(bot_a, bot_b, bot_a_goes_first=(i % 2 == 0))
        if result['winner'] == 'A':
            wins += 1
        elif result['winner'] == 'B':
            losses += 1
        else:
            draws += 1

    return {'wins': wins, 'losses': losses, 'draws': draws}
