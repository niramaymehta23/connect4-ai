import unittest

from src.bots.minimax import DEFAULT_WEIGHTS, minimax_bot
from src.game.board import create_board, place
from src.game.constants import AI, HUMAN
from src.game.rules import valid_cols
from src.simulation.runner import run_game


def board_from(moves):
    board, token = create_board(), AI
    for col in moves:
        place(board, col, token)
        token = HUMAN if token == AI else AI
    return board, token


class MinimaxTraceTests(unittest.TestCase):
    def test_trace_best_col_matches_bot(self):
        from src.bots.minimax import minimax_trace

        any_pruned = False
        for moves in ([], [3, 3, 4, 2, 2, 4, 5], [3, 0, 3, 6, 3]):
            with self.subTest(moves=moves):
                board, token = board_from(moves)
                trace = minimax_trace(board, token, depth=4, weights=DEFAULT_WEIGHTS)
                self.assertEqual(board, board_from(moves)[0])
                self.assertEqual(trace["best_col"], minimax_bot(board, token, 4, DEFAULT_WEIGHTS))
                self.assertEqual(sorted(entry["col"] for entry in trace["root"]), valid_cols(board))
                for entry in trace["root"]:
                    self.assertGreaterEqual(entry["nodes"], 1)
                    self.assertIn("score", entry)
                    any_pruned = any_pruned or entry["pruned"] > 0
        self.assertTrue(any_pruned)

    def test_trace_root_scores_match_alphabeta_with_depth_bonus(self):
        import math
        from src.bots.minimax import _alphabeta, minimax_trace

        weights = {**DEFAULT_WEIGHTS, "depthBonus": 10}
        # Red can make an open-ended three on the bottom row: a forced win within depth 4.
        for moves in ([2, 6, 3, 6], [3, 0, 3, 6, 3], [3, 3, 4, 2, 2, 4, 5]):
            with self.subTest(moves=moves):
                board, token = board_from(moves)
                opponent = HUMAN if token == AI else AI
                trace = minimax_trace(board, token, depth=4, weights=weights)
                self.assertEqual(trace["best_col"], minimax_bot(board, token, 4, weights))
                for entry in trace["root"]:
                    place(board, entry["col"], token)
                    expected = _alphabeta(board, token, opponent, 3, -math.inf, math.inf, False, weights)
                    board, _ = board_from(moves)
                    self.assertEqual(entry["score"], expected)

    def test_trace_events_respect_cap(self):
        from src.bots.minimax import minimax_trace

        trace = minimax_trace(create_board(), AI, depth=4, weights=DEFAULT_WEIGHTS, max_events=50)
        self.assertEqual(len(trace["events"]), 50)
        self.assertTrue(trace["truncated"])
        self.assertEqual(trace["depth"], 4)


def leftmost(board, _token):
    return valid_cols(board)[0]


def rightmost(board, _token):
    return valid_cols(board)[-1]


class RunGameTimingTests(unittest.TestCase):
    def test_move_times_align_with_history(self):
        result = run_game(leftmost, rightmost, opening_moves=[3, 3])
        self.assertEqual(set(result) >= {"winner", "moves", "duration", "move_history"}, True)
        times = result["move_times_ms"]
        self.assertEqual(len(times), len(result["move_history"]))
        self.assertEqual(times[:2], [None, None])
        self.assertTrue(all(isinstance(t, float) and t >= 0 for t in times[2:]))
