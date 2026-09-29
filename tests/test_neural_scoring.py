import unittest
from src.game.board import create_board, place
from src.game.constants import AI, HUMAN
from src.nn.scoring import score_successors


class NeuralScoringTests(unittest.TestCase):
    def test_uses_and_negates_opponent_perspective_without_mutation(self):
        board = create_board()
        seen = []
        def value(child, token):
            seen.append(token)
            return 0.8
        self.assertEqual(score_successors(board, AI, value), [-0.8] * 7)
        self.assertEqual(seen, [HUMAN] * 7)
        self.assertEqual(board, create_board())

    def test_immediate_win_is_exact_and_does_not_query_model(self):
        board = create_board()
        for col in [0, 1, 2]:
            place(board, col, AI)
        def value(child, token):
            self.assertEqual(child[5][3], 0)
            return 0.4
        self.assertEqual(score_successors(board, AI, value)[3], 1.0)
