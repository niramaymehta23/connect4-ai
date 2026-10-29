"""Run compare.py at several search budgets so /dashboard can plot score rate vs. budget."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.agents import NAMES

BUDGETED = {'mcts', 'weighted-mcts', 'neural-mcts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=NAMES, default='mcts', help='agent whose budget varies (side A)')
    parser.add_argument('--opponent', choices=NAMES, default='random')
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 25, 50, 100, 200],
                        help='simulation counts to try')
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--checkpoint', type=Path, help='required for neural agents')
    args = parser.parse_args()
    if args.agent not in BUDGETED:
        parser.error(f'--agent must use a search budget: {sorted(BUDGETED)}')
    if args.opponent in BUDGETED:
        parser.error('compare.py gives both sides the same --simulations; pick an opponent without a budget')

    for budget in args.budgets:
        output = ROOT / 'runs' / f'budget-{args.agent}-vs-{args.opponent}-s{budget}-p{args.pairs}-seed{args.seed}.json'
        if output.exists():
            print(f'skip {output.name}: already exists')
            continue
        command = [sys.executable, str(ROOT / 'scripts' / 'compare.py'), '--a', args.agent, '--b', args.opponent,
                   '--simulations', str(budget), '--pairs', str(args.pairs), '--seed', str(args.seed),
                   '--output', str(output)]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
        print(f'budget {budget}:', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
