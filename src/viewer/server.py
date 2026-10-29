import json
import re
import time
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.bots.minimax import minimax_trace, score_move
from src.bots.mcts import mcts_search
from src.game.constants import AI, HUMAN
from src.game.rules import check_win, valid_cols
from src.io.persistence import load_best
from src.viewer import bot_registry
from src.viewer.analysis import analyze_game, cell_heatmap, heuristic_value_fn, self_play_replay
from src.viewer.bot_registry import BEST_WEIGHTS, BOT_FNS, BOT_LIST, BOT_SPECS, MINIMAX_DEPTH

STATIC_DIR = Path(__file__).parents[2] / "static"
RUNS_DIR = Path(__file__).parents[2] / "runs"
NN_HISTORY = Path(__file__).parents[2] / "data" / "metrics" / "nn_history.json"
CHECKPOINT_DIR = bot_registry.CHECKPOINT_DIR
TRACE_DEPTH = 4
# compare.py records --simulations and --depth for every run; only these agents use them.
SIMULATION_AGENTS = {"mcts", "weighted-mcts", "neural-mcts"}
DEPTH_AGENTS = {"minimax", "evolved"}
MAX_REPLAYS = 4

app = Flask(__name__, static_folder=str(STATIC_DIR))


class BadRequest(ValueError):
    pass


@app.errorhandler(BadRequest)
def bad_request(error):
    return jsonify({"error": str(error)}), 400


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/bots")
def bots():
    return jsonify(BOT_LIST)


@app.route("/api/status")
def status():
    best = load_best(BEST_WEIGHTS)
    neural_meta = next((bot for bot in BOT_LIST if bot["id"] == "neural"), None)
    return jsonify(
        {
            "neural_available": bool(neural_meta and neural_meta["available"]),
            "best_weights_generation": best["generation"],
            "best_weights_fitness": best["fitness"],
        }
    )


def _normalise_board(value):
    board = [[int(cell) for cell in row] for row in value]
    if len(board) != 6 or any(len(row) != 7 for row in board):
        raise ValueError("board must be 6x7")
    return board


def _position_payload():
    """Validated (board, token) from a JSON body; BadRequest on malformed input."""
    payload = request.get_json(force=True, silent=True) or {}
    try:
        board = _normalise_board(payload["board"])
        token = int(payload["token"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BadRequest(f"expected a 6x7 board and a token: {exc}") from exc
    if token not in (AI, HUMAN) or any(cell not in (0, AI, HUMAN) for row in board for cell in row):
        raise BadRequest("cells and token must be 0, 1 or 2")
    return board, token


def _value_fn():
    if bot_registry.NEURAL_VALUE_FN is not None:
        return bot_registry.NEURAL_VALUE_FN, "neural"
    return heuristic_value_fn(BEST_WEIGHTS), "heuristic"


def _minimax_scores(board, token):
    scores = []
    for col in range(7):
        raw_score = score_move(board, col, token, BEST_WEIGHTS)
        scores.append(None if raw_score == float("-inf") else float(raw_score))
    return scores


def _neural_scores(board, token):
    score_fn = BOT_SPECS.get("neural_scores")
    if score_fn is None:
        return [None] * 7
    return score_fn(board, token)


def _decide(bot_id, board, token):
    """Ask one registered bot for a move; elapsed_ms times the decision only."""
    kind = BOT_SPECS.get(bot_id, {}).get("kind", "flat")
    started = time.perf_counter()
    if kind == "mcts":
        iterations = BOT_SPECS[bot_id]["iterations"]
        search = mcts_search(board, token, iterations=iterations, snapshot_every=max(1, iterations // 20))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "col": int(search["col"]),
            "scores": [float(v) if v > 0 else None for v in search["visits"]],
            "snapshots": search["snapshots"],
            "kind": kind,
            "elapsed_ms": elapsed_ms,
        }
    col = int(BOT_FNS[bot_id](board, token))
    elapsed_ms = (time.perf_counter() - started) * 1000
    if kind == "minimax":
        scores = _minimax_scores(board, token)
    elif kind == "neural":
        scores = _neural_scores(board, token)
    else:
        scores = None
    return {"col": col, "scores": scores, "kind": kind, "elapsed_ms": elapsed_ms}


@app.route("/api/move", methods=["POST"])
def move():
    payload = request.get_json(force=True)
    board = _normalise_board(payload["board"])
    token = int(payload["token"])
    bot_id = payload["bot_id"]

    if BOT_FNS.get(bot_id) is None:
        return jsonify({"error": f"bot unavailable: {bot_id}"}), 400
    return jsonify(_decide(bot_id, board, token))


@app.route("/api/compare-position", methods=["POST"])
def compare_position():
    board, token = _position_payload()
    if not valid_cols(board) or check_win(board, AI) or check_win(board, HUMAN):
        raise BadRequest("the game is already over in this position")
    requested = (request.get_json(force=True, silent=True) or {}).get("bots")
    bot_ids = [b["id"] for b in BOT_LIST if b["id"] in BOT_FNS]
    if isinstance(requested, list):
        names = dict.fromkeys(name for name in requested if isinstance(name, str))
        bot_ids = [bot_id for bot_id in names if bot_id in BOT_FNS]
    labels = {bot["id"]: bot["label"] for bot in BOT_LIST}
    rows = []
    for bot_id in bot_ids:
        result = _decide(bot_id, board, token)
        result.pop("snapshots", None)
        rows.append({"bot_id": bot_id, "label": labels[bot_id], **result})
    return jsonify(rows)


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    board, token = _position_payload()
    value_fn, evaluator = _value_fn()
    return jsonify({"cells": cell_heatmap(board, token, value_fn), "evaluator": evaluator})


@app.route("/api/minimax-trace", methods=["POST"])
def minimax_trace_route():
    board, token = _position_payload()
    if not valid_cols(board) or check_win(board, AI) or check_win(board, HUMAN):
        raise BadRequest("the game is already over in this position")
    trace = minimax_trace(board, token, depth=TRACE_DEPTH, weights=BEST_WEIGHTS)
    return jsonify({**trace, "play_depth": MINIMAX_DEPTH})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    moves = (request.get_json(force=True, silent=True) or {}).get("moves")
    if not isinstance(moves, list):
        raise BadRequest("moves must be a list of column indexes")
    value_fn, evaluator = _value_fn()
    try:
        result = analyze_game(moves, value_fn, heatmaps=True)
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    return jsonify({**result, "evaluator": evaluator})


@app.route("/dashboard")
def dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/training")
def training():
    return send_from_directory(app.static_folder, "training.html")


def _agent_ms(records):
    """Average real decision time per agent, or None when a run has no per-move timing."""
    totals = {"a": [0.0, 0], "b": [0.0, 0]}
    for record in records:
        times = record.get("move_times_ms")
        if times is None:
            return None
        first = "a" if record.get("a_first", True) else "b"
        second = "b" if first == "a" else "a"
        for ply, elapsed in enumerate(times):
            if elapsed is not None:
                owner = first if ply % 2 == 0 else second
                totals[owner][0] += elapsed
                totals[owner][1] += 1
    if not records:
        return None
    return {agent: (total / count if count else None) for agent, (total, count) in totals.items()}


@app.route("/api/dashboard")
def api_dashboard():
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            cfg = data.get("config", {})
            records = data.get("game_records", [])
            agents = {cfg.get("a"), cfg.get("b")}
            runs.append({
                "file": path.name,
                "a": cfg.get("a", "?"),
                "b": cfg.get("b", "?"),
                "simulations": cfg.get("simulations") if agents & SIMULATION_AGENTS else None,
                "depth": cfg.get("depth") if agents & DEPTH_AGENTS else None,
                "wins": data.get("wins", 0),
                "losses": data.get("losses", 0),
                "draws": data.get("draws", 0),
                "games": data.get("games", 0),
                "score_rate": data.get("score_rate", 0),
                "avg_moves": sum(r["moves"] for r in records) / len(records) if records else 0,
                "avg_duration": sum(r["duration"] for r in records) / len(records) if records else 0,
                "move_lengths": [r["moves"] for r in records],
                "durations": [r["duration"] for r in records],
                "agent_ms": _agent_ms(records),
                "timestamp": data.get("timestamp", ""),
            })
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # Not a compare.py result; other JSON in runs/ is ignored.
    return jsonify(runs)


@app.route("/api/training")
def api_training():
    if not NN_HISTORY.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(NN_HISTORY.read_text()))
    except (OSError, ValueError):
        return jsonify([])


@app.route("/api/training/runs")
def api_training_runs():
    runs = []
    if not RUNS_DIR.is_dir():
        return jsonify(runs)
    for history_path in sorted(RUNS_DIR.glob("*/history.json")):
        try:
            history = json.loads(history_path.read_text())
            iterations = [{
                "iteration": entry["iteration"],
                "loss": entry["mean_training_loss"],
                "score_rate": entry["evaluation"]["score_rate"],
                "trajectory_diversity": entry["evaluation"]["trajectory_diversity"],
                "unique_trajectories": entry["evaluation"]["unique_trajectories"],
                "games": entry["evaluation"]["games"],
                "promoted": entry["champion_promoted"],
                "replay_size": entry["replay_size"],
            } for entry in history]
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Not a learn.py run directory.
        runs.append({"run": history_path.parent.name, "iterations": iterations})
    return jsonify(runs)


def _load_checkpoint_value_fn(path):
    loader = bot_registry.load_checkpoint_value_fn
    return loader(path) if loader else None


@lru_cache(maxsize=16)
def _checkpoint_replay(path, _mtime):
    value_fn = _load_checkpoint_value_fn(path)
    return self_play_replay(value_fn) if value_fn else None


def _selected_checkpoints():
    found = []
    for path in CHECKPOINT_DIR.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    if len(found) <= MAX_REPLAYS:
        return found
    last = len(found) - 1
    return [found[round(i * last / (MAX_REPLAYS - 1))] for i in range(MAX_REPLAYS)]


@app.route("/api/training/replays")
def api_training_replays():
    replays = []
    for iteration, path in _selected_checkpoints():
        try:
            replay = _checkpoint_replay(str(path), path.stat().st_mtime)
        except Exception as exc:  # A corrupt or incompatible checkpoint must not hide the others.
            app.logger.warning("skipping %s: %s", path.name, exc)
            continue
        if replay:
            replays.append({"checkpoint": path.name, "iteration": iteration, **replay})
    return jsonify(replays)


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)
