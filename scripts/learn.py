"""A small, explicit neural self-play experiment. Every invocation creates a new run."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='new directory, e.g. runs/lesson-01')
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--games', type=int, default=4)
    p.add_argument('--simulations', type=int, default=30)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-pairs', type=int, default=4)
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--exploration-plies', type=int, default=8)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--checkpoint', type=Path, help='optional warm start; optimizer and replay start fresh')
    args = p.parse_args()
    if args.run.exists():
        p.error('run directory exists: use a new name to preserve the previous experiment')
    if min(args.iterations, args.games, args.simulations, args.steps, args.batch_size, args.eval_pairs) < 1:
        p.error('iteration, game, search, batch, step and evaluation counts must be positive')
    if args.exploration_plies < 0:
        p.error('exploration plies cannot be negative')

    import torch
    from src.nn.network import Connect4Net
    from src.nn.replay import ReplayBuffer
    from src.nn.self_play import generate_games, make_nn_mcts_bot
    from src.evaluation.openings import generate_openings
    from src.evaluation.compare import compare_on_openings

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    champion = Connect4Net.load(args.checkpoint, device) if args.checkpoint else Connect4Net().to(device).eval()
    candidate = copy.deepcopy(champion)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    replay = ReplayBuffer()
    openings = generate_openings(args.eval_pairs, plies=4, seed=args.seed + 100_000)
    args.run.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.run / 'config.json').write_text(json.dumps(config, indent=2))
    champion.save(args.run / 'champion.pt')
    history = []
    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        candidate.eval()
        samples = generate_games(candidate, device, args.games, args.simulations, args.exploration_plies)
        for state, outcome in samples:
            replay.push(state, outcome)
        candidate.train()
        losses = []
        for _ in range(args.steps):
            states, outcomes = replay.sample(args.batch_size)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(candidate(states.to(device)), outcomes.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        candidate.eval()
        evaluation = compare_on_openings(
            make_nn_mcts_bot(candidate, device, args.simulations),
            make_nn_mcts_bot(champion, device, args.simulations),
            openings, seed=args.seed + 200_000,
        )
        # A learning heuristic, not a statistical guarantee of improvement.
        promoted = evaluation['score_rate'] > 0.55
        if promoted:
            champion = copy.deepcopy(candidate).eval()
            champion.save(args.run / 'champion.pt')
        candidate.save(args.run / f'candidate-{iteration:04d}.pt')
        entry = {
            'iteration': iteration, 'games_played': iteration * args.games,
            'samples': len(samples), 'replay_size': len(replay),
            'mean_training_loss': sum(losses) / len(losses),
            'evaluation': evaluation, 'champion_promoted': promoted,
            'seconds': time.perf_counter() - started,
        }
        history.append(entry)
        (args.run / 'history.json').write_text(json.dumps(history, indent=2))
        print(f"Iteration {iteration}: {len(samples)} positions; loss {entry['mean_training_loss']:.4f}; "
              f"candidate score {evaluation['score_rate']:.1%}; promoted={promoted}", flush=True)
    print(f'Saved {args.run}. Use a separate seed/opening suite for final evaluation.')


if __name__ == '__main__':
    main()
