import random
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional for core tests')
class SelfPlayTests(unittest.TestCase):
    def test_exploration_is_limited_to_opening_and_outcomes_have_correct_sign(self):
        from src.nn.self_play import generate_games
        calls = []
        # First player wins in seven plies.
        moves = iter([0, 1, 0, 1, 0, 1, 0])
        def choose(*args, **kwargs):
            calls.append(kwargs['temperature'])
            return next(moves)
        with patch('src.nn.self_play._run_nn_mcts', side_effect=choose):
            samples = generate_games(None, 'cpu', 1, 10, exploration_plies=3)
        self.assertEqual(calls, [1.0] * 3 + [0.0] * 4)
        self.assertEqual([outcome for _, outcome in samples], [1, -1, 1, -1, 1, -1, 1])

    def test_seeded_neural_mcts_returns_legal_move_without_mutation(self):
        from src.nn.self_play import _run_nn_mcts
        from src.game.board import create_board
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1))
        board = create_board()
        state = random.getstate()
        try:
            random.seed(1)
            first = _run_nn_mcts(board, 2, Constant(), 'cpu', 15)
            random.seed(1)
            second = _run_nn_mcts(board, 2, Constant(), 'cpu', 15)
        finally:
            random.setstate(state)
        self.assertEqual(first, second)
        self.assertIn(first, range(7))
        self.assertEqual(board, create_board())
