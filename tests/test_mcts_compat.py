import random
import unittest

from src.bots.mcts import make_mcts_bot
from src.bots.minimax import DEFAULT_WEIGHTS
from src.game.board import create_board, place
from src.game.constants import AI, HUMAN

BOARDS = {
    "empty": [],
    "midgame": [3, 3, 4, 2, 2, 4, 5],
    "win_available": [0, 6, 1, 6, 2],
    "must_block": [3, 0, 3, 6, 3],
}
CONFIGS = {
    "plain": dict(iterations=60),
    "weighted": dict(iterations=60, rollout_weights=DEFAULT_WEIGHTS, temperature=3.0),
    "sampled": dict(iterations=60, sample_moves=True),
}
# (config, board, seed, chosen column, next 32 random bits) recorded from the pre-refactor bot.
EXPECTED = [
    ('plain', 'empty', 0, 3, 925220108),
    ('plain', 'empty', 1, 3, 1514474787),
    ('plain', 'empty', 2, 2, 1560845295),
    ('plain', 'midgame', 0, 6, 3435741175),
    ('plain', 'midgame', 1, 6, 1930811753),
    ('plain', 'midgame', 2, 6, 4255138899),
    ('plain', 'win_available', 0, 3, 3132966725),
    ('plain', 'win_available', 1, 3, 3882726895),
    ('plain', 'win_available', 2, 3, 2047127512),
    ('plain', 'must_block', 0, 3, 3364750262),
    ('plain', 'must_block', 1, 3, 3969871807),
    ('plain', 'must_block', 2, 4, 814094739),
    ('weighted', 'empty', 0, 2, 3761864295),
    ('weighted', 'empty', 1, 3, 2789084210),
    ('weighted', 'empty', 2, 4, 1674655481),
    ('weighted', 'midgame', 0, 6, 3012989015),
    ('weighted', 'midgame', 1, 6, 3260379189),
    ('weighted', 'midgame', 2, 6, 3429834246),
    ('weighted', 'win_available', 0, 3, 2893842435),
    ('weighted', 'win_available', 1, 3, 363698845),
    ('weighted', 'win_available', 2, 3, 2496840614),
    ('weighted', 'must_block', 0, 3, 2098601769),
    ('weighted', 'must_block', 1, 1, 855625789),
    ('weighted', 'must_block', 2, 3, 2351829274),
    ('sampled', 'empty', 0, 4, 1141911726),
    ('sampled', 'empty', 1, 3, 1397104307),
    ('sampled', 'empty', 2, 4, 4069408226),
    ('sampled', 'midgame', 0, 0, 2218470013),
    ('sampled', 'midgame', 1, 3, 1309102551),
    ('sampled', 'midgame', 2, 0, 927301052),
    ('sampled', 'win_available', 0, 5, 1434624957),
    ('sampled', 'win_available', 1, 0, 103726226),
    ('sampled', 'win_available', 2, 1, 3837832731),
    ('sampled', 'must_block', 0, 1, 3339169529),
    ('sampled', 'must_block', 1, 5, 4171304560),
    ('sampled', 'must_block', 2, 5, 2914009266),
]


def board_from(moves):
    board, token = create_board(), AI
    for col in moves:
        place(board, col, token)
        token = HUMAN if token == AI else AI
    return board, token


class SeededBotMoves(unittest.TestCase):
    def test_seeded_columns_and_random_consumption_unchanged(self):
        previous = random.getstate()
        try:
            for config, board_name, seed, col, tail in EXPECTED:
                with self.subTest(config=config, board=board_name, seed=seed):
                    board, token = board_from(BOARDS[board_name])
                    random.seed(seed)
                    self.assertEqual(make_mcts_bot(**CONFIGS[config])(board, token), col)
                    self.assertEqual(random.getrandbits(32), tail)
        finally:
            random.setstate(previous)


class SearchSnapshots(unittest.TestCase):
    def test_snapshots_grow_to_final_visits(self):
        from src.bots.mcts import mcts_search

        board, token = board_from(BOARDS["midgame"])
        result = mcts_search(board, token, iterations=100, snapshot_every=25)
        self.assertEqual(len(result["visits"]), 7)
        self.assertEqual(len(result["snapshots"]), 4)
        for earlier, later in zip(result["snapshots"], result["snapshots"][1:]):
            self.assertTrue(all(a <= b for a, b in zip(earlier, later)))
        self.assertEqual(result["snapshots"][-1], result["visits"])
        self.assertEqual(result["visits"][result["col"]], max(result["visits"]))
        self.assertEqual(board, board_from(BOARDS["midgame"])[0])
