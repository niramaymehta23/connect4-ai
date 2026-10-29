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

    def _scripted_samples(self, moves):
        from src.nn.self_play import generate_games
        moves = iter(moves)
        with patch('src.nn.self_play._run_nn_mcts', side_effect=lambda *a, **k: next(moves)):
            return generate_games(None, 'cpu', 1, 10)

    def test_second_player_win_is_credited_to_yellow_positions(self):
        # Yellow (second player) completes the bottom row of columns 1-4.
        samples = self._scripted_samples([0, 1, 0, 2, 6, 3, 6, 4])
        turn = [state[2, 0, 0].item() for state, _ in samples]
        self.assertEqual(turn, [1.0, 0.0] * 4)
        self.assertEqual([outcome for _, outcome in samples], [-1, 1] * 4)

    def test_states_are_encoded_from_the_mover_before_their_move(self):
        samples = self._scripted_samples([0, 1, 0, 1, 0, 1, 0])
        for ply, (state, outcome) in enumerate(samples):
            own, opponent = state[0].sum().item(), state[1].sum().item()
            # The mover has made ply // 2 moves; the opponent has made the rest.
            self.assertEqual((own, opponent), (ply // 2, ply - ply // 2))
            self.assertEqual(state[0, 5, 0].item(), float(ply >= 1 and ply % 2 == 0))
        # Each position's target is the negation of the next one's: one game,
        # two perspectives, no side gets a bonus.
        outcomes = [outcome for _, outcome in samples]
        self.assertEqual(outcomes[1:], [-o for o in outcomes[:-1]])

    def test_drawn_game_gives_every_position_zero(self):
        draw = [3, 6, 1, 3, 1, 3, 6, 4, 5, 5, 3, 4, 3, 1, 1, 0, 4, 3, 0, 4, 6,
                6, 2, 6, 1, 0, 5, 5, 2, 4, 5, 1, 5, 6, 0, 0, 2, 2, 4, 2, 2, 0]
        samples = self._scripted_samples(draw)
        self.assertEqual(len(samples), 42)
        self.assertEqual({outcome for _, outcome in samples}, {0.0})

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
