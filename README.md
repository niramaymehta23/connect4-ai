# Connect Four Learning Lab

[![Tests](https://github.com/niramaymehta23/connect4-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/niramaymehta23/connect4-ai/actions/workflows/ci.yml)

A hands-on experiment in how game-playing AI works: compare search algorithms,
evolve a small scoring function, and train a neural network through self-play.
The goal is to understand and measure improvement before claiming a strong agent.

**Start here:** [the learning guide](docs/GUIDE.md). It explains what each agent
does, why games repeat, and how to interpret an experiment. Old phase documents
are preserved in `docs/archive/` as historical notes, not current instructions.

## Set up

Python 3.12+ for the pinned dependencies; neural training needs a compatible PyTorch installation.

```sh
git clone https://github.com/niramaymehta23/connect4-ai.git
cd connect4-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The non-neural comparison command and basic tests use only Python's standard library.

## Compare two agents

```sh
python scripts/compare.py --a minimax --b random --depth 2 --pairs 6 --seed 23
python scripts/compare.py --a mcts --b minimax --depth 3 --simulations 100 \
  --pairs 20 --seed 41 --output runs/mcts-vs-minimax.json
```

Each opening is played twice with the agents' colours swapped. The report includes
win/loss/draw counts, score (win = 1, draw = 0.5), complete move histories, seed,
search settings, and the number of distinct trajectories. Existing output files
are never overwritten. Different trajectories still share positions; their count
is a diversity diagnostic, not a claim of statistical independence.

Agent names: `random`, `minimax`, `evolved`, `mcts`, `weighted-mcts`, `neural`,
`neural-mcts`. Neural agents require `--checkpoint path/to/model.pt`. Depth and
simulation counts are different kinds of compute budgets: also examine latency
before claiming that an algorithm is better.

## Learn through self-play

```sh
python scripts/learn.py --run runs/lesson-01 --iterations 3 --games 4 \
  --simulations 30 --eval-pairs 4 --seed 23
python scripts/compare.py --a neural-mcts --b random --simulations 30 \
  --checkpoint runs/lesson-01/champion.pt --pairs 10 --seed 999
```

This deliberately small experiment teaches the loop; it does not train a strong
model. It starts fresh unless `--checkpoint` supplies initial weights. Each run
has its own configuration, history, candidate checkpoints, and champion.

`--device cpu` is the default; `mps` and `cuda` are explicit alternatives. An existing
run directory is rejected to prevent accidentally combining unrelated experiments.
Warm starts load model weights only; optimizer state and replay memory start fresh.

## Play or watch

```sh
python scripts/viewer.py
```

Open http://localhost:5000. The viewer offers local two-player, human-vs-bot,
and bot-vs-bot modes. The neural agents load `models/best_model.pt` when present:
**Neural value** looks one move ahead; **Neural + MCTS** searches using that network.
The viewer performs inference only. Watching a match does not train a model.

## Where things live

| Location | Purpose |
| --- | --- |
| `src/game/` | Board and rules |
| `src/bots/` | Random, minimax and rollout MCTS |
| `src/nn/` | Network, self-play, replay memory and neural scoring |
| `src/evaluation/` | Shared agent catalogue and paired-opening comparisons |
| `src/simulation/` | Game execution and legacy tournaments |
| `src/viewer/`, `static/` | Playable browser viewer |
| `scripts/compare.py`, `scripts/learn.py` | Current experiment entry points |
| `tests/` | Rules around comparison and neural integration |
| `runs/` | Ignored local experiment outputs |
| `data/`, `models/` | Preserved historical weights and training artifacts |

## Current limits

- The network is value-only; this is not a full AlphaZero implementation.
- Candidate promotion above 55% score is an experimental heuristic. Small matches
  cannot establish reliable improvement, and repeatedly testing one suite can overfit it.
- Historical metrics combine runs and settings; old percentages are not validated
  performance claims. Preserve those records, but use fresh experiments for graphs.
- Legacy `train.py`, `train_nn.py`, `simulate.py`, and `benchmark.py` remain available
  for archaeology. Their resume/fitness/evaluation semantics differ; the guide lists
  the problems. Prefer the commands above for new neural experiments and comparisons.
- The strongest agent has not been established. A held-out suite, multiple seeds,
  compute measurements, and stronger opponents are still needed.

## License and safety

MIT licensed; see [LICENSE](LICENSE). Read [SECURITY.md](SECURITY.md) before
hosting the viewer or loading third-party checkpoints. No API credentials are needed.
