import unittest
import random

from src.evaluation.compare import compare_on_openings
from src.evaluation.openings import generate_openings
from src.game.rules import valid_cols
from src.simulation.runner import run_game
from src.bots.random import random_bot


def leftmost(board, _token):
    return valid_cols(board)[0]


def rightmost(board, _token):
    return valid_cols(board)[-1]


class EvaluationTests(unittest.TestCase):
    def test_deterministic_agents_repeat_from_identical_start(self):
        first = run_game(leftmost, rightmost)
        second = run_game(leftmost, rightmost)
        self.assertEqual(first['move_history'], second['move_history'])

    def test_openings_are_unique_and_reproducible(self):
        first = generate_openings(count=8, plies=4, seed=23)
        second = generate_openings(count=8, plies=4, seed=23)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 8)

    def test_balanced_openings_create_independent_trajectories(self):
        openings = generate_openings(count=8, plies=4, seed=23)
        result = compare_on_openings(leftmost, rightmost, openings)
        self.assertEqual(result['games'], 16)
        self.assertGreater(result['unique_trajectories'], 2)
        self.assertGreater(result['trajectory_diversity'], 0.5)

    def test_illegal_opening_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "illegal opening move"):
            run_game(leftmost, rightmost, opening_moves=[0] * 7)

    def test_random_agents_are_reproducible_and_rng_is_restored(self):
        openings = generate_openings(4, seed=4)
        before = random.getstate()
        first = compare_on_openings(random_bot, random_bot, openings, seed=10)
        self.assertEqual(random.getstate(), before)
        second = compare_on_openings(random_bot, random_bot, openings, seed=10)
        self.assertEqual([g['move_history'] for g in first['game_records']],
                         [g['move_history'] for g in second['game_records']])

    def test_colours_swap_even_after_odd_length_opening(self):
        result = compare_on_openings(leftmost, rightmost, [(3,)])
        self.assertEqual(result['game_records'][0]['move_history'][1], 6)
        self.assertEqual(result['game_records'][1]['move_history'][1], 0)

    def test_terminal_opening_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finished game'):
            run_game(leftmost, rightmost, opening_moves=[0, 1, 0, 1, 0, 1, 0])

    def test_illegal_bot_move_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'illegal column'):
            run_game(lambda b, t: -1, rightmost)


if __name__ == '__main__':
    unittest.main()
