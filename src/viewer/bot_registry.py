from pathlib import Path

from src.bots.mcts import make_mcts_bot
from src.bots.minimax import DEFAULT_WEIGHTS, make_minimax_bot
from src.bots.random import random_bot
from src.io.persistence import load_best

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "best_model.pt"

BEST_RECORD = load_best(DEFAULT_WEIGHTS)
BEST_WEIGHTS = BEST_RECORD["weights"]

BOT_FNS = {}
BOT_LIST = []
BOT_SPECS = {}


def _register(bot_id, label, fn=None, reason=None, kind=None):
    available = fn is not None
    if available:
        BOT_FNS[bot_id] = fn
    BOT_LIST.append(
        {
            "id": bot_id,
            "label": label,
            "available": available,
            "reason": reason,
        }
    )
    BOT_SPECS[bot_id] = {"kind": kind or bot_id, "label": label}


_register("random", "CHAOS (Random)", random_bot, kind="flat")
_register(
    "minimax",
    "DARWIN (Minimax)",
    make_minimax_bot(depth=6, weights=BEST_WEIGHTS),
    kind="minimax",
)
_register("mcts_fast", "SCOUT (MCTS Fast)", make_mcts_bot(iterations=200), kind="flat")
_register(
    "mcts_strong",
    "ORACLE (MCTS Strong)",
    make_mcts_bot(iterations=1000),
    kind="flat",
)

if not MODEL_PATH.exists():
    _register("neural", "NEXUS (Neural Network)", reason="requires trained model", kind="neural")
else:
    try:
        import torch

        from src.nn.encode import encode_board
        from src.nn.network import Connect4Net
        from src.nn.scoring import score_successors
        from src.nn.self_play import make_nn_mcts_bot

        _device = torch.device("cpu")
        _neural_model = Connect4Net.load(MODEL_PATH, _device)

        def _neural_scores(board, token):
            def value(position, player):
                with torch.no_grad():
                    return _neural_model(encode_board(position, player).unsqueeze(0).to(_device)).item()
            return score_successors(board, token, value)

        def _neural_bot(board, token):
            scores = _neural_scores(board, token)
            valid = [(idx, score) for idx, score in enumerate(scores) if score is not None]
            return max(valid, key=lambda item: item[1])[0]

        BOT_SPECS["neural_scores"] = _neural_scores
        _register("neural", "Neural value (one move ahead)", _neural_bot, kind="neural")
        _register("neural_mcts", "Neural + MCTS (200 simulations)",
                  make_nn_mcts_bot(_neural_model, _device, iterations=200), kind="flat")
    except ImportError:
        _register("neural", "NEXUS (Neural Network)", reason="requires torch", kind="neural")
    except Exception as exc:
        _register("neural", "NEXUS (Neural Network)", reason=f"failed to load model: {exc}", kind="neural")
