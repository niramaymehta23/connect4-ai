"""
self_play.py — MCTS self-play loop driven by the neural network value function.

Key difference from Phase C rollout MCTS:
  Instead of simulating a random game to the end (rollout), we query the
  network for a value estimate of the leaf position.  The network outputs
  a float in (-1, 1) from the leaf node's player's perspective.

  We map that to [0, 1] so it is compatible with the existing _backpropagate
  logic (which flips with 1-result):
      value_for_node_player ∈ (-1, 1)
      mapped = (value_for_node_player + 1) / 2  ∈ (0, 1)
  Then, if the node's player differs from the "chooser" (the parent's player
  who decided to visit this node), we flip: result = 1 - mapped.

All network calls are wrapped in torch.no_grad() as required.
"""
import random
import math
import torch

from ..game.board import create_board, place, clone
from ..game.rules  import valid_cols, check_win, is_full
from ..game.constants import AI, HUMAN

# Re-use MCTSNode and _backpropagate from Phase C — no duplication.
from ..bots.mcts import MCTSNode, _backpropagate

from .encode import encode_board


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opp(token):
    return HUMAN if token == AI else AI


def _run_nn_mcts(board, token, network, device, iterations, C=1.414, temperature=0.0):
    """
    Run MCTS from position `board` with `token` to move, using `network` as
    the leaf evaluator instead of random rollouts.

    Parameters
    ----------
    board      : current board (not mutated)
    token      : AI or HUMAN
    network    : Connect4Net in eval() mode
    device     : torch.device
    iterations : number of MCTS iterations
    C          : UCB1 exploration constant

    Returns
    -------
    int — chosen column index
    """
    if iterations < 1 or temperature < 0:
        raise ValueError("iterations must be positive and temperature nonnegative")
    root = MCTSNode(clone(board), None, None, token)

    for _ in range(iterations):
        # ── 1. Selection ───────────────────────────────────────────────────
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(C)

        # ── 2. Expansion ───────────────────────────────────────────────────
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # ── 3. Evaluation ──────────────────────────────────────────────────
        # chooser = the player whose win count this node's result contributes to
        chooser = node.parent.token if node.parent is not None else token

        if node.is_terminal():
            # Use exact outcome: find who moved last
            last_player = node.parent.token if node.parent is not None else token
            if check_win(node.board, last_player):
                result = 1.0 if last_player == chooser else 0.0
            else:
                result = 0.5  # draw
        else:
            # Query the network for a value estimate.
            # network output is in (-1, 1) from node.token's perspective.
            with torch.no_grad():
                t = encode_board(node.board, node.token).unsqueeze(0).to(device)
                raw_value = network(t).item()

            # Map (-1, 1) → (0, 1) for node.token's perspective, then
            # orient to chooser's perspective.
            val_for_node = (raw_value + 1.0) / 2.0  # ∈ (0, 1)
            result = val_for_node if node.token == chooser else 1.0 - val_for_node

        # ── 4. Backpropagation ─────────────────────────────────────────────
        _backpropagate(node, result)

    if not root.children:
        return random.choice(valid_cols(board))
    if temperature == 0:
        return max(root.children, key=lambda n: n.visits).move
    # Exploration during training: sample proportional to visits^(1 / T).
    # Dividing by max visits keeps powers bounded without changing probabilities.
    peak = max(child.visits for child in root.children)
    weights = [(child.visits / peak) ** (1.0 / temperature) for child in root.children]
    return random.choices(root.children, weights=weights, k=1)[0].move


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_games(network, device, n_games, mcts_iterations, exploration_plies=8):
    """
    Play `n_games` self-play games using network-backed MCTS for every move.

    Parameters
    ----------
    network         : Connect4Net — must be in eval() mode before calling
    device          : torch.device
    n_games         : int
    mcts_iterations : int — MCTS iterations per move

    Returns
    -------
    list of (state_tensor, outcome_float) pairs.
    state_tensor is shape (3, ROWS, COLS), float32.
    outcome_float is +1.0 (win), -1.0 (loss), 0.0 (draw)
    from the perspective of the player who was to move at that state.
    """
    all_samples = []

    for _ in range(n_games):
        board   = create_board()
        current = AI          # AI always goes first in runner convention
        history = []          # (state_tensor, move_token) accumulated during the game

        while True:
            # Record state *before* the move so we capture the position the
            # player evaluated when deciding.
            history.append((encode_board(board, current), current))

            col = _run_nn_mcts(
                board, current, network, device, mcts_iterations,
                temperature=1.0 if len(history) <= exploration_plies else 0.0,
            )
            place(board, col, current)

            if check_win(board, current):
                winner = current
                break
            if is_full(board):
                winner = None
                break

            current = _opp(current)

        # Assign outcomes retroactively from each state's player's perspective.
        for tensor, move_token in history:
            if winner is None:
                outcome = 0.0
            elif winner == move_token:
                outcome = 1.0
            else:
                outcome = -1.0
            all_samples.append((tensor, outcome))

    return all_samples


def make_nn_mcts_bot(network, device, iterations=200, C=1.414):
    """
    Return a bot function with the standard (board, token) -> col interface.
    Uses network-backed MCTS internally.  Suitable for use with runner.run_games.

    Network should be in eval() mode before the returned bot is called.
    """
    def bot(board, token):
        return _run_nn_mcts(board, token, network, device, iterations, C)
    return bot
