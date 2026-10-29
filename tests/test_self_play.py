import hashlib
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

    def test_single_process_games_match_the_original_implementation(self):
        from src.nn.self_play import generate_games
        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1))
        state = random.getstate()
        try:
            random.seed(5)
            samples = generate_games(Zero(), 'cpu', 2, 10, exploration_plies=4)
        finally:
            random.setstate(state)
        # Recorded from generate_games before worker pools existed.
        flat = repr([(s.to(torch.int8).flatten().tolist(), o) for s, o in samples])
        self.assertEqual(len(samples), 49)
        self.assertEqual(hashlib.sha256(flat.encode()).hexdigest()[:16], '886de9703b9e0de7')


@unittest.skipIf(torch is None, 'PyTorch is optional for core tests')
class WorkerPoolTests(unittest.TestCase):
    """Spawned workers start slowly, so each test shares what it can."""

    def setUp(self):
        from src.nn.network import Connect4Net
        torch.manual_seed(0)
        self.net = Connect4Net().eval()

    def test_parallel_games_are_seeded_per_game_and_returned_in_order(self):
        from src.nn.self_play import generate_games, worker_pool, _play_game
        with worker_pool(2) as pool:
            first = generate_games(self.net, 'cpu', 3, 8, pool=pool, seed='23:1')
            second = generate_games(self.net, 'cpu', 3, 8, pool=pool, seed='23:1')
            other = generate_games(self.net, 'cpu', 3, 8, pool=pool, seed='23:2')
        with worker_pool(3) as pool:
            three_workers = generate_games(self.net, 'cpu', 3, 8, pool=pool, seed='23:1')
        state = random.getstate()
        try:
            expected = []
            for game in range(3):
                random.seed(f'23:1:{game}')
                expected += _play_game(self.net, torch.device('cpu'), 8, 8)
        finally:
            random.setstate(state)
        as_lists = lambda samples: [(s.tolist(), o) for s, o in samples]
        self.assertEqual(as_lists(first), as_lists(expected))
        self.assertEqual(as_lists(second), as_lists(expected))
        self.assertEqual(as_lists(three_workers), as_lists(expected))
        self.assertNotEqual(as_lists(other), as_lists(expected))

    def test_parallel_evaluation_plays_the_same_games(self):
        from src.nn.self_play import make_nn_mcts_bot, worker_pool
        from src.evaluation.compare import compare_on_openings
        bot = make_nn_mcts_bot(self.net, torch.device('cpu'), iterations=8)
        openings = [(3,), (2, 4)]
        serial = compare_on_openings(bot, bot, openings, seed=7)
        with worker_pool(2) as pool:
            parallel = compare_on_openings(bot, bot, openings, seed=7, pool=pool)
        games = lambda result: [(g['move_history'], g['winner'], g['seed']) for g in result['game_records']]
        self.assertEqual(games(parallel), games(serial))
        self.assertEqual(parallel['score_rate'], serial['score_rate'])
