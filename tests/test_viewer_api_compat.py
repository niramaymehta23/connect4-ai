import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import flask  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Flask is optional for core tests")

from src.viewer import server
from src.game.rules import valid_cols

EMPTY_BOARD = [[0] * 7 for _ in range(6)]
RUN_RECORD = {
    "wins": 1, "losses": 1, "draws": 0, "games": 2, "score_rate": 0.5,
    "game_records": [
        {"winner": "A", "moves": 7, "duration": 3.0, "move_history": [0, 6, 1, 6, 2, 6, 3]},
        {"winner": "B", "moves": 8, "duration": 5.0, "move_history": [6, 0, 6, 1, 6, 2, 5, 3]},
    ],
    "config": {"a": "minimax", "b": "random"},
    "timestamp": "2026-01-01T00:00:00+00:00",
}


class ExistingRoutes(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_pages_and_static_files(self):
        for path in ("/", "/dashboard", "/training"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.content_type)
        self.assertEqual(self.client.get("/game.js").status_code, 200)

    def test_bots_and_status(self):
        bots = self.client.get("/api/bots").get_json()
        self.assertTrue({"random", "minimax", "mcts_fast", "mcts_strong", "neural"} <= {b["id"] for b in bots})
        for bot in bots:
            self.assertEqual(set(bot), {"id", "label", "available", "reason"})
        status = self.client.get("/api/status").get_json()
        self.assertTrue({"neural_available", "best_weights_generation", "best_weights_fitness"} <= set(status))

    def test_move_contract(self):
        expected_scores = {"random": None, "minimax": 7, "mcts_fast": 7}
        expected_kind = {"random": "flat", "minimax": "minimax", "mcts_fast": "mcts"}
        for bot_id, scores in expected_scores.items():
            with self.subTest(bot=bot_id):
                body = self.client.post("/api/move", json={"board": EMPTY_BOARD, "token": 2, "bot_id": bot_id}).get_json()
                self.assertIn(body["col"], valid_cols(EMPTY_BOARD))
                self.assertEqual(body["kind"], expected_kind[bot_id])
                if scores is None:
                    self.assertIsNone(body["scores"])
                else:
                    self.assertEqual(len(body["scores"]), scores)

    def test_unknown_bot_is_400(self):
        response = self.client.post("/api/move", json={"board": EMPTY_BOARD, "token": 2, "bot_id": "nope"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_training_is_a_list(self):
        self.assertIsInstance(self.client.get("/api/training").get_json(), list)

    def test_dashboard_reads_top_level_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "match.json").write_text(json.dumps(RUN_RECORD))
            (runs / "learning-01").mkdir()
            (runs / "learning-01" / "history.json").write_text("[]")
            with mock.patch.object(server, "RUNS_DIR", runs):
                items = self.client.get("/api/dashboard").get_json()
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertTrue({"file", "a", "b", "wins", "losses", "draws", "games", "score_rate", "avg_moves",
                         "avg_duration", "move_lengths", "durations", "timestamp"} <= set(item))
        self.assertEqual((item["a"], item["b"], item["move_lengths"]), ("minimax", "random", [7, 8]))
