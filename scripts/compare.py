"""Compare two agents with paired openings and a reproducible random seed."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.agents import NAMES, build_agent
from src.evaluation.compare import compare_on_openings
from src.evaluation.openings import generate_openings
from src.io.persistence import load_best
from src.bots.minimax import DEFAULT_WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', choices=NAMES, default='minimax')
    parser.add_argument('--b', choices=NAMES, default='random')
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--opening-plies', type=int, default=4)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, help='new JSON output file; never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('output exists; choose a new experiment file')
    try:
        openings = generate_openings(args.pairs, args.opening_plies, args.seed)
        a = build_agent(args.a, args.depth, args.simulations, args.checkpoint)
        b = build_agent(args.b, args.depth, args.simulations, args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'{args.a} vs {args.b}: {args.pairs} openings, both colours, seed {args.seed}', flush=True)
    result = compare_on_openings(a, b, openings, seed=args.seed)
    result['schema_version'] = 1
    result['config'] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result['timestamp'] = datetime.now(timezone.utc).isoformat()
    result['python'] = platform.python_version()
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True)
    result['git_commit'] = git.stdout.strip() if git.returncode == 0 else None
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=root, text=True, capture_output=True)
    result['working_tree_dirty'] = bool(status.stdout.strip())
    result['weights'] = load_best(DEFAULT_WEIGHTS) if {args.a, args.b} & {'evolved', 'weighted-mcts'} else None
    result['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    print(f"A: {result['wins']} wins / {result['losses']} losses / {result['draws']} draws")
    print(f"Score: {result['score_rate']:.1%}; unique full trajectories: {result['unique_trajectories']}/{result['games']}")
    print('This measures this opening suite and compute budget; diversity is not strength or statistical independence.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
