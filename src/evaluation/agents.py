"""One catalogue for command-line comparisons; neural imports are optional."""
from pathlib import Path
from ..bots.random import random_bot
from ..bots.minimax import DEFAULT_WEIGHTS, make_minimax_bot
from ..bots.mcts import make_mcts_bot
from ..io.persistence import load_best

NAMES = ('random', 'minimax', 'evolved', 'mcts', 'weighted-mcts', 'neural', 'neural-mcts')


def build_agent(name, depth=3, simulations=100, checkpoint=None):
    if depth < 1 or simulations < 1:
        raise ValueError('depth and simulations must be positive')
    if name == 'random':
        return random_bot
    if name == 'minimax':
        return make_minimax_bot(depth, DEFAULT_WEIGHTS)
    weights = load_best(DEFAULT_WEIGHTS)
    if name in ('evolved', 'weighted-mcts') and weights['generation'] == 0:
        raise ValueError('evolved agent requires saved evolved weights')
    if name == 'evolved':
        return make_minimax_bot(depth, weights['weights'])
    if name in ('mcts', 'weighted-mcts'):
        return make_mcts_bot(simulations, rollout_weights=weights['weights'] if name == 'weighted-mcts' else None,
                             temperature=3.0, sample_moves=False)
    if name in ('neural', 'neural-mcts'):
        if checkpoint is None or not Path(checkpoint).is_file():
            raise ValueError('neural agents require an explicit existing --checkpoint')
        import torch
        from ..nn.network import Connect4Net
        from ..nn.encode import encode_board
        from ..nn.scoring import score_successors
        from ..nn.self_play import make_nn_mcts_bot
        device = torch.device('cpu')
        net = Connect4Net.load(checkpoint, device)
        if name == 'neural-mcts':
            return make_nn_mcts_bot(net, device, iterations=simulations)
        def bot(board, token):
            def value(position, player):
                with torch.no_grad():
                    return net(encode_board(position, player).unsqueeze(0)).item()
            scores = score_successors(board, token, value)
            return max((c for c, v in enumerate(scores) if v is not None), key=lambda c: scores[c])
        return bot
    raise ValueError(f'unknown agent: {name}')
