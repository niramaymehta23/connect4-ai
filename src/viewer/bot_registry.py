from pathlib import Path

from src.bots.mcts import make_mcts_bot
from src.bots.minimax import DEFAULT_WEIGHTS, make_minimax_bot
from src.bots.random import random_bot
from src.io.persistence import load_best

ROOT = Path(__file__).parents[2]
CHECKPOINT_DIR = ROOT / "models"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
MINIMAX_DEPTH = 6

# value_fn(board, player_to_move) -> float with a batched `.many`; None without a model.
NEURAL_VALUE_FN = None
load_checkpoint_value_fn = None

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
    make_minimax_bot(depth=MINIMAX_DEPTH, weights=BEST_WEIGHTS),
    kind="minimax",
)
MCTS_ITERATIONS = {"mcts_fast": 200, "mcts_strong": 1000}
_register("mcts_fast", "SCOUT (MCTS Fast)", make_mcts_bot(iterations=MCTS_ITERATIONS["mcts_fast"]), kind="mcts")
_register("mcts_strong", "ORACLE (MCTS Strong)", make_mcts_bot(iterations=MCTS_ITERATIONS["mcts_strong"]), kind="mcts")
for _bot_id, _iterations in MCTS_ITERATIONS.items():
    BOT_SPECS[_bot_id]["iterations"] = _iterations

try:
    import torch

    from src.nn.encode import encode_board
    from src.nn.network import Connect4Net
    from src.nn.scoring import score_successors
    from src.nn.self_play import make_nn_mcts_bot
except ImportError:
    torch = None

if torch is not None:
    _device = torch.device("cpu")

    def _make_value_fn(model):
        def value(position, player):
            with torch.no_grad():
                return model(encode_board(position, player).unsqueeze(0).to(_device)).item()

        def many(items):
            with torch.no_grad():
                batch = torch.stack([encode_board(position, player) for position, player in items])
                return model(batch.to(_device)).squeeze(1).tolist()

        value.many = many
        return value

    def load_checkpoint_value_fn(path):
        return _make_value_fn(Connect4Net.load(path, _device))

if not MODEL_PATH.exists():
    _register("neural", "NEXUS (Neural Network)", reason="requires trained model", kind="neural")
elif torch is None:
    _register("neural", "NEXUS (Neural Network)", reason="requires torch", kind="neural")
else:
    try:
        _neural_model = Connect4Net.load(MODEL_PATH, _device)
    except Exception as exc:
        _register("neural", "NEXUS (Neural Network)", reason=f"failed to load model: {exc}", kind="neural")
    else:
        NEURAL_VALUE_FN = _make_value_fn(_neural_model)

        def _neural_scores(board, token):
            return score_successors(board, token, NEURAL_VALUE_FN)

        def _neural_bot(board, token):
            scores = _neural_scores(board, token)
            valid = [(idx, score) for idx, score in enumerate(scores) if score is not None]
            return max(valid, key=lambda item: item[1])[0]

        BOT_SPECS["neural_scores"] = _neural_scores
        _register("neural", "Neural value (one move ahead)", _neural_bot, kind="neural")
        _register("neural_mcts", "Neural + MCTS (200 simulations)",
                  make_nn_mcts_bot(_neural_model, _device, iterations=200), kind="flat")
