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
from src.viewer.analysis import heuristic_value_fn
from src.game.rules import valid_cols

EMPTY_BOARD = [[0] * 7 for _ in range(6)]


def learning_entry(iteration, diversity):
    return {
        "iteration": iteration, "games_played": iteration * 10, "samples": 120, "replay_size": 120 * iteration,
        "mean_training_loss": 0.5 - iteration / 100, "champion_promoted": iteration == 2, "seconds": 1.0,
        "evaluation": {"wins": 3, "losses": 1, "draws": 0, "games": 4, "score_rate": 0.75,
                       "unique_trajectories": 3, "trajectory_diversity": diversity, "game_records": []},
    }


class NewEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def post(self, path, payload):
        return self.client.post(path, json=payload)

    def test_heatmap_endpoint(self):
        body = self.post("/api/heatmap", {"board": EMPTY_BOARD, "token": 2}).get_json()
        self.assertEqual((len(body["cells"]), len(body["cells"][0])), (6, 7))
        self.assertIn(body["evaluator"], {"neural", "heuristic"})
        self.assertEqual(self.post("/api/heatmap", {"board": [[0] * 7], "token": 2}).status_code, 400)

    def test_minimax_trace_endpoint(self):
        body = self.post("/api/minimax-trace", {"board": EMPTY_BOARD, "token": 2}).get_json()
        self.assertIn(body["best_col"], valid_cols(EMPTY_BOARD))
        self.assertEqual(body["depth"], 4)
        self.assertEqual(len(body["root"]), 7)
        self.assertTrue(body["events"])

    def test_analyze_endpoint(self):
        body = self.post("/api/analyze", {"moves": [3, 4, 3]}).get_json()
        self.assertEqual(len(body["plies"]), 3)
        self.assertEqual(len(body["values"]), 4)
        self.assertEqual(len(body["heatmaps"]), 4)
        self.assertIsInstance(body["pivot"], int)
        self.assertIn(body["evaluator"], {"neural", "heuristic"})
        self.assertEqual(self.post("/api/analyze", {"moves": [9]}).status_code, 400)
        self.assertEqual(self.post("/api/analyze", {"moves": "33"}).status_code, 400)

    def test_move_reports_elapsed_ms_and_mcts_snapshots(self):
        for bot_id in ("random", "mcts_fast"):
            with self.subTest(bot=bot_id):
                body = self.post("/api/move", {"board": EMPTY_BOARD, "token": 2, "bot_id": bot_id}).get_json()
                self.assertGreaterEqual(body["elapsed_ms"], 0)
        body = self.post("/api/move", {"board": EMPTY_BOARD, "token": 2, "bot_id": "mcts_fast"}).get_json()
        self.assertEqual(len(body["snapshots"]), 20)
        self.assertEqual([float(v) for v in body["snapshots"][-1]], body["scores"])

    def test_compare_position(self):
        body = self.post("/api/compare-position",
                         {"board": EMPTY_BOARD, "token": 2, "bots": ["random", "minimax", "bogus"]}).get_json()
        self.assertEqual([row["bot_id"] for row in body], ["random", "minimax"])
        for row in body:
            self.assertTrue({"bot_id", "label", "col", "elapsed_ms", "kind", "scores"} <= set(row))
            self.assertIn(row["col"], valid_cols(EMPTY_BOARD))


class InputValidation(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_finished_positions_are_rejected(self):
        board = [[0] * 7 for _ in range(6)]
        board[5][:4] = [2, 2, 2, 2]
        for path in ("/api/minimax-trace", "/api/compare-position"):
            with self.subTest(path=path):
                self.assertEqual(self.client.post(path, json={"board": board, "token": 1}).status_code, 400)

    def test_non_finite_and_malformed_input_is_400(self):
        response = self.client.post("/api/heatmap", data='{"board": [[0,0,0,0,0,0,0]], "token": Infinity}',
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/heatmap", data='{"board": %s, "token": Infinity}' % json.dumps(EMPTY_BOARD),
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/heatmap", json={"board": EMPTY_BOARD, "token": 3}).status_code, 400)

    def test_compare_position_ignores_non_strings_and_duplicates(self):
        body = self.client.post("/api/compare-position",
                                json={"board": EMPTY_BOARD, "token": 2, "bots": [{}, "random", "random", 7]}).get_json()
        self.assertEqual([row["bot_id"] for row in body], ["random"])

    def test_compare_position_omits_snapshots(self):
        body = self.client.post("/api/compare-position",
                                json={"board": EMPTY_BOARD, "token": 2, "bots": ["mcts_fast"]}).get_json()
        self.assertNotIn("snapshots", body[0])
        self.assertEqual(body[0]["kind"], "mcts")

    def test_heatmap_marks_only_occupied_and_playable_cells(self):
        cells = self.client.post("/api/heatmap", json={"board": EMPTY_BOARD, "token": 2}).get_json()["cells"]
        self.assertTrue(all(value is None for row in cells[:5] for value in row))
        self.assertTrue(all(isinstance(value, float) for value in cells[5]))


class DataEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_dashboard_budget_and_agent_times(self):
        timed = {
            "wins": 1, "losses": 0, "draws": 0, "games": 1, "score_rate": 1.0,
            "config": {"a": "mcts", "b": "random", "simulations": 50, "depth": 3},
            "game_records": [{"winner": "A", "moves": 4, "duration": 10.0, "a_first": False,
                              "move_history": [0, 1, 2, 3], "move_times_ms": [None, 2.0, 8.0, 4.0]}],
        }
        old = {"wins": 0, "losses": 1, "draws": 0, "games": 1, "score_rate": 0.0, "config": {"a": "random", "b": "minimax"},
               "game_records": [{"winner": "B", "moves": 7, "duration": 3.0, "move_history": [0] * 7}]}
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "timed.json").write_text(json.dumps(timed))
            (runs / "old.json").write_text(json.dumps(old))
            with mock.patch.object(server, "RUNS_DIR", runs):
                items = {item["file"]: item for item in self.client.get("/api/dashboard").get_json()}
        self.assertEqual((items["timed.json"]["simulations"], items["timed.json"]["depth"]), (50, None))
        # b moves first (a_first False): plies 1 and 3 belong to a (2 and 4 ms), ply 2 to b; ply 0 is an opening.
        self.assertEqual(items["timed.json"]["agent_ms"], {"a": 3.0, "b": 8.0})
        self.assertIsNone(items["old.json"]["agent_ms"])
        # compare.py records a default budget even for agents that never search; it must not be plotted.
        self.assertIsNone(items["old.json"]["simulations"])

    def test_training_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "learning-01").mkdir()
            (runs / "learning-01" / "history.json").write_text(json.dumps([learning_entry(1, 1.0), learning_entry(2, 0.5)]))
            (runs / "stray.json").write_text("{}")
            with mock.patch.object(server, "RUNS_DIR", runs):
                body = self.client.get("/api/training/runs").get_json()
            with mock.patch.object(server, "RUNS_DIR", runs / "missing"):
                self.assertEqual(self.client.get("/api/training/runs").get_json(), [])
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["run"], "learning-01")
        second = body[0]["iterations"][1]
        self.assertEqual((second["iteration"], second["trajectory_diversity"], second["promoted"]), (2, 0.5, True))
        self.assertTrue({"loss", "score_rate", "unique_trajectories", "games", "replay_size"} <= set(second))

    def test_training_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            with mock.patch.object(server, "CHECKPOINT_DIR", models):
                self.assertEqual(self.client.get("/api/training/replays").get_json(), [])
            for iteration in (3, 10, 50, 120, 240, 365):
                (models / f"checkpoint_{iteration:03d}.pt").write_bytes(b"")
            with mock.patch.object(server, "CHECKPOINT_DIR", models), \
                    mock.patch.object(server, "_load_checkpoint_value_fn", lambda _path: heuristic_value_fn()):
                body = self.client.get("/api/training/replays").get_json()
        self.assertEqual([item["iteration"] for item in body], [3, 50, 120, 365])
        for item in body:
            self.assertEqual(len(item["values"]), len(item["moves"]) + 1)
            self.assertTrue(item["checkpoint"].startswith("checkpoint_"))

    def test_replay_cache_refreshes_when_checkpoint_changes(self):
        import os

        loads = []

        def loader(path):
            loads.append(path)
            return heuristic_value_fn()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_001.pt"
            checkpoint.write_bytes(b"")
            with mock.patch.object(server, "CHECKPOINT_DIR", Path(tmp)), \
                    mock.patch.object(server, "_load_checkpoint_value_fn", loader):
                self.client.get("/api/training/replays")
                self.client.get("/api/training/replays")
                self.assertEqual(len(loads), 1)
                stat = checkpoint.stat()
                os.utime(checkpoint, (stat.st_atime, stat.st_mtime + 10))
                self.client.get("/api/training/replays")
        self.assertEqual(len(loads), 2)

    def test_replays_work_without_a_promoted_best_model(self):
        from src.viewer import bot_registry

        if bot_registry.load_checkpoint_value_fn is None:
            self.skipTest("torch is not installed")
        from src.nn.network import Connect4Net

        with tempfile.TemporaryDirectory() as tmp:
            Connect4Net().save(Path(tmp) / "checkpoint_007.pt")
            with mock.patch.object(server, "CHECKPOINT_DIR", Path(tmp)), \
                    mock.patch.object(bot_registry, "NEURAL_VALUE_FN", None):
                body = self.client.get("/api/training/replays").get_json()
        self.assertEqual([item["iteration"] for item in body], [7])
