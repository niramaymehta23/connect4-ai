from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.bots.minimax import score_move
from src.io.persistence import load_best
from src.viewer.bot_registry import BEST_WEIGHTS, BOT_FNS, BOT_LIST, BOT_SPECS

STATIC_DIR = Path(__file__).parents[2] / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


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


def _scores_for(bot_id, board, token):
    kind = BOT_SPECS.get(bot_id, {}).get("kind")
    if kind == "minimax":
        return _minimax_scores(board, token)
    if kind == "neural":
        return _neural_scores(board, token)
    return None  # Search visit statistics are not exposed by these agents yet.


@app.route("/api/move", methods=["POST"])
def move():
    payload = request.get_json(force=True)
    board = _normalise_board(payload["board"])
    token = int(payload["token"])
    bot_id = payload["bot_id"]

    bot_fn = BOT_FNS.get(bot_id)
    if bot_fn is None:
        return jsonify({"error": f"bot unavailable: {bot_id}"}), 400

    col = int(bot_fn(board, token))
    scores = _scores_for(bot_id, board, token)
    return jsonify({"col": col, "scores": scores})


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)
