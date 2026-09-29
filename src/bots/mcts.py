import math
import random
from ..game.rules import valid_cols, check_win, is_full
from ..game.board import clone, place
from ..game.constants import AI, HUMAN
from .minimax import score_move


class MCTSNode:
    __slots__ = ('board', 'move', 'parent', 'children', 'wins', 'visits',
                 'unvisited_moves', 'token')

    def __init__(self, board, move, parent, token):
        self.board = board
        self.move = move          # column that led to this node (None for root)
        self.parent = parent
        self.children = []
        self.wins = 0.0           # wins for the player who chose this node (parent.token)
        self.visits = 0
        self.token = token        # the player whose turn it is AT this node
        self.unvisited_moves = valid_cols(board)

    def is_fully_expanded(self):
        return len(self.unvisited_moves) == 0

    def is_terminal(self):
        # The player who just moved (opponent of self.token) may have won.
        opponent = HUMAN if self.token == AI else AI
        return (self.move is not None and check_win(self.board, opponent)) or is_full(self.board)

    def ucb1(self, C):
        if self.visits == 0:
            return math.inf
        return self.wins / self.visits + C * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, C):
        return max(self.children, key=lambda n: n.ucb1(C))

    def expand(self):
        col = self.unvisited_moves.pop(random.randrange(len(self.unvisited_moves)))
        new_board = clone(self.board)
        place(new_board, col, self.token)
        child_token = HUMAN if self.token == AI else AI
        child = MCTSNode(new_board, col, self, child_token)
        self.children.append(child)
        return child


def _opponent(token):
    return HUMAN if token == AI else AI


def _rollout(board, token):
    """
    Play random moves to game end from `board` with `token` to move.
    Returns the winning token, or None for a draw.
    """
    board = clone(board)
    current = token
    while True:
        cols = valid_cols(board)
        if not cols:
            return None
        col = random.choice(cols)
        place(board, col, current)
        if check_win(board, current):
            return current
        if is_full(board):
            return None
        current = _opponent(current)


def _weighted_rollout(board, token, weights, temperature=1.0):
    """
    Same contract as _rollout — returns winning token or None for draw.

    Instead of picking uniformly at random, scores each valid column with
    score_move and converts those scores to selection probabilities via a
    temperature-scaled softmax:

      1. Shift scores so the minimum is 0  (eliminates scale bias from the
         minimax weight magnitudes; all inputs to softmax are now ≥ 0)
      2. If all shifted scores are 0, fall back to uniform random
      3. Divide by temperature: higher T → flatter distribution (more random);
         lower T → more peaked (more greedy)
      4. Standard softmax with max-subtraction for numerical stability

    temperature=1.0 gives the raw shifted-softmax.
    temperature→∞ approaches uniform random.
    temperature→0 approaches pure greedy.
    """
    board = clone(board)
    current = token
    while True:
        cols = valid_cols(board)
        if not cols:
            return None

        # 1. Score each legal move from `current`'s perspective.
        raw_scores = [score_move(board, col, current, weights) for col in cols]

        # 2. Shift so minimum is 0 — removes the scale problem where one large
        #    score swamped everything else in the C2 max-subtraction approach.
        min_s   = min(raw_scores)
        shifted = [s - min_s for s in raw_scores]

        if max(shifted) == 0:
            # No signal at all (e.g. empty board, no threats yet) — uniform.
            col = random.choice(cols)
        else:
            # 3. Temperature scaling, then softmax with max-subtraction.
            scaled  = [s / temperature for s in shifted]
            max_sc  = max(scaled)
            exps    = [math.exp(s - max_sc) for s in scaled]
            total   = sum(exps)
            probs   = [e / total for e in exps]
            col     = random.choices(cols, weights=probs, k=1)[0]

        place(board, col, current)
        if check_win(board, current):
            return current
        if is_full(board):
            return None
        current = _opponent(current)


def _backpropagate(node, result):
    """
    Walk from `node` up to root updating wins/visits.

    `result` is 1.0 / 0.5 / 0.0 from the perspective of the player who
    chose to go to `node` (i.e. node.parent.token).  At each level we
    flip the result because the grandparent chose the opposite player.
    """
    while node is not None:
        node.visits += 1
        node.wins += result
        result = 1.0 - result   # flip perspective for the node above
        node = node.parent


def make_mcts_bot(iterations=500, C=1.414, rollout_weights=None, temperature=1.0, sample_moves=False):
    """
    Return a bot function with signature (board, token) -> col.
    Compatible with runner.py and tournament.py without any changes.

    Parameters
    ----------
    iterations      : number of MCTS iterations per move (default 500)
    C               : UCB1 exploration constant (default 1.414 = √2)
    rollout_weights : when not None, use _weighted_rollout with these Phase A
                      weights instead of the plain uniform _rollout.
    temperature     : softmax temperature for _weighted_rollout (default 1.0).
                      Higher → more random; lower → more greedy.
                      Only used when rollout_weights is not None.
    """
    if iterations < 1 or temperature <= 0:
        raise ValueError('iterations and rollout temperature must be positive')
    def bot(board, token):
        root = MCTSNode(clone(board), None, None, token)

        for _ in range(iterations):
            # 1. Selection — walk down picking best child until a non-fully-expanded
            #    or terminal node is reached.
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(C)

            # 2. Expansion — add one new child for an unvisited move.
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # 3. Simulation — determine the outcome from this position.
            if node.is_terminal():
                # The player who just moved to reach this node won (or it's a draw).
                # The player who moved = node.parent.token (or root.token if node==root).
                last_player = node.parent.token if node.parent is not None else token
                if check_win(node.board, last_player):
                    winner = last_player
                else:
                    winner = None  # draw
            elif rollout_weights is not None:
                winner = _weighted_rollout(node.board, node.token,
                                           rollout_weights, temperature)
            else:
                winner = _rollout(node.board, node.token)

            # Convert winner → result from the perspective of the player who
            # chose to go to `node` (= node.parent.token, or `token` for root).
            chooser = node.parent.token if node.parent is not None else token
            if winner is None:
                result = 0.5
            elif winner == chooser:
                result = 1.0
            else:
                result = 0.0

            # 4. Backpropagation — walk up, flipping perspective at each level.
            _backpropagate(node, result)

        # Pick move with the most visits (robust to outliers).
        if not root.children:
            return random.choice(valid_cols(board))

        if not sample_moves:
            return max(root.children, key=lambda child: child.visits).move
        # Sample proportionally from visit counts instead of always taking argmax.
        # Breaks determinism while still strongly preferring the most-visited move.
        children = root.children
        visits = [c.visits for c in children]
        total = sum(visits)
        probs = [v / total for v in visits]
        best = random.choices(children, weights=probs, k=1)[0]
        return best.move

    return bot
