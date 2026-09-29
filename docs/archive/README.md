# Connect4 AI

> Historical snapshot. Some results, commands and design claims are outdated.
> Start with the repository's current README instead.

A multi-phase system for building and training the strongest possible Connect 4 bot — combining evolutionary weight tuning, Monte Carlo Tree Search, and neural network self-play.

Play against trained bots in a browser-based viewer, or run training from the terminal.

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/connect4-ai.git
cd connect4-ai

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Launch the viewer

```bash
python3 scripts/viewer.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 3. Pick a mode

| Mode | Description |
|------|-------------|
| Player vs Player | Two humans, same machine |
| Player vs Bot | Choose your colour and opponent |
| Bot vs Bot | Watch two bots play — step through moves or simulate with speed control |

### Available bots

| Bot | Description |
|-----|-------------|
| Random | Picks a random valid column. Useful as a baseline. |
| Minimax (evolved) | Depth-6 alpha-beta search with weights tuned by evolution |
| MCTS Fast | Monte Carlo Tree Search, 200 iterations per move |
| MCTS Strong | Monte Carlo Tree Search, 1000 iterations per move |
| Neural Network | CNN trained via self-play *(requires a trained model — see below)* |

### Run a simulation from the terminal

```bash
# 100 games, minimax vs random (default)
python3 scripts/simulate.py

# MCTS vs minimax, 20 games, 2000 iterations
python3 scripts/simulate.py --mode mcts_vs_minimax --games 20 --iterations 2000

# Weighted rollouts with temperature tuning
python3 scripts/simulate.py --mode mcts_vs_minimax --games 30 --rollout weighted --temperature 3.0
```

### Benchmark the trained bot

```bash
python3 scripts/benchmark.py
# Tests evolved minimax vs random and vs default weights, 200 games each
```

---

## Training

### Phase A — Evolutionary weight tuning

Evolve the minimax heuristic weights across generations using a round-robin tournament.

```bash
python3 scripts/train.py
python3 scripts/train.py --gens 50 --pop 20 --games 20 --elite 4 --workers 5
```

Best weights are saved to `data/weights/best.json` after every generation.
Training resumes automatically from the last saved generation on restart.

| Flag | Default | Description |
|------|---------|-------------|
| `--gens` | 100 | Number of generations |
| `--pop` | 20 | Population size |
| `--games` | 20 | Games per matchup |
| `--elite` | 4 | Survivors per generation |
| `--workers` | 5 | Parallel worker processes |

### Phase B — Neural network self-play

Train a CNN via self-play. Requires PyTorch (MPS-accelerated on Apple Silicon).

```bash
python3 scripts/train_nn.py
python3 scripts/train_nn.py --iterations 50 --games 20 --mcts-iters 100
```

Trained models are saved to `models/best_model.pt`.

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 100 | Training iterations |
| `--games` | 50 | Self-play games per iteration |
| `--mcts-iters` | 200 | MCTS iterations per move during self-play |
| `--batch-size` | 256 | Training batch size |
| `--eval-every` | 10 | Evaluate vs champion every N iterations |

Training prints loss, buffer size, and evaluation results to the terminal.
A 4-panel fitness chart is saved to `data/metrics/nn_fitness_chart.png`.

---

## Project Structure

```
connect4-ai/
├── scripts/               # Entry points — run these directly
│   ├── viewer.py          # Launch the browser game viewer
│   ├── train.py           # Phase A evolutionary training
│   ├── train_nn.py        # Phase B neural network training
│   ├── simulate.py        # Run bot matchups from the terminal
│   └── benchmark.py       # Benchmark trained bot vs baselines
│
├── src/
│   ├── game/              # Pure game logic — no AI, no UI
│   │   ├── board.py       # Board state: create, place, unplace, clone
│   │   ├── rules.py       # Win detection, draw detection, valid moves
│   │   └── constants.py   # ROWS, COLS, WIN, HUMAN, AI, EMPTY
│   │
│   ├── bots/              # Bot implementations
│   │   ├── minimax.py     # Minimax + alpha-beta, accepts weights dict
│   │   ├── mcts.py        # MCTS with UCB1, weighted rollouts, temperature
│   │   └── random.py      # Random bot baseline
│   │
│   ├── training/          # Phase A training loop
│   │   ├── evolution.py   # Evolutionary loop: seed, tournament, select, mutate
│   │   ├── mutate.py      # Gaussian weight mutation with caps
│   │   └── fitness.py     # Fitness formula and standalone evaluation
│   │
│   ├── nn/                # Phase B neural network
│   │   ├── network.py     # CNN (3×Conv+BN → Flatten → 2×Linear → Tanh)
│   │   ├── encode.py      # Board → (3, 6, 7) float32 tensor
│   │   ├── self_play.py   # Generate training games using MCTS + network
│   │   ├── replay.py      # Replay buffer (capacity 50,000)
│   │   ├── train_nn.py    # Training loop, evaluation, champion promotion
│   │   └── chart.py       # Matplotlib training charts
│   │
│   ├── simulation/        # Run games between bots
│   │   ├── runner.py      # Single game and N-game runner
│   │   ├── tournament.py  # Round-robin tournament with parallel workers
│   │   └── worker.py      # Multiprocessing worker for tournament matchups
│   │
│   ├── viewer/            # Browser viewer backend
│   │   ├── server.py      # Flask routes: /, /api/bots, /api/move, /api/status
│   │   └── bot_registry.py # Maps bot names to callable functions
│   │
│   ├── io/
│   │   └── persistence.py # Save/load weights and metrics history
│   │
│   └── viz/               # Terminal visualisation
│       ├── board.py       # ANSI colour board printer
│       ├── metrics.py     # Per-generation stats formatter
│       └── chart.py       # Matplotlib fitness chart
│
├── data/
│   ├── weights/           # Saved bot weights (JSON)
│   │   ├── best.json      # Best weights across all training runs
│   │   └── gen_NNN.json   # Per-generation snapshots
│   └── metrics/           # Training history
│       ├── history.json
│       └── fitness_chart.png
│
├── models/                # Neural network checkpoints (.pt files, not in repo)
├── static/                # Browser viewer frontend (HTML, CSS, JS)
└── docs/                  # Design documents
    ├── ARCHITECTURE.md
    ├── PHASE_A.md
    ├── PHASE_B.md
    ├── PHASE_C.md
    ├── ROADMAP.md
    └── METRICS.md
```

---

## How It Works

### Phase A — Evolutionary weight tuning

The minimax bot scores board positions using a heuristic with tunable weights:

| Weight | Meaning |
|--------|---------|
| `centreBonus` | Points per piece in the centre column |
| `two` | Score for 2-in-a-row with 2 open gaps |
| `three` | Score for 3-in-a-row with 1 open gap |
| `win` | Score for 4-in-a-row (non-terminal heuristic) |
| `depthBonus` | Extra reward for winning sooner |

A population of bots plays a round-robin tournament each generation. The top performers survive and are mutated to form the next generation. After 26 generations the champion reached **fitness 0.9425** and wins **100% of games against a random bot**.

Key design decisions:
- **3 random-bot anchors** injected into every tournament — breaks the structural mean=0.5 trap that occurs in a pure minimax vs minimax pool
- **Fitness formula:** `wins / (wins + losses + draws × 2)` — penalises draws heavily to discourage defensive convergence
- **Mutation:** 50% probability per weight, sigma = `|w| × 0.30`, with per-weight caps to prevent runaway drift
- **Convergence gate:** 30-generation window peak + elite variance check — both must pass to stop early
- **Parallelisation:** matchups distributed across worker processes via `multiprocessing.Pool` (~4.4× speedup on M3 Pro)

### Phase C — Monte Carlo Tree Search

MCTS replaces the heuristic entirely. Instead of scoring positions with a formula, it simulates games to the end and uses win rate as the score.

Each move runs: **Selection** (UCB1) → **Expansion** → **Rollout** → **Backpropagation**.

Two rollout policies:
- **Random rollout** — standard MCTS, pure random simulation
- **Weighted rollout** — Phase A weights bias move selection during simulation via softmax with temperature. At T=3.0, 200-iteration weighted MCTS matches the quality of 500-iteration random MCTS at 2.2× less compute.

### Phase B — Neural network

A CNN learns purely from self-play, replacing the rollout step in MCTS with a learned value function.

**Architecture:**
```
Input: (3, 6, 7) tensor
  Channel 0: current player's pieces
  Channel 1: opponent's pieces
  Channel 2: whose turn (all 1s or all 0s)

Conv2d(3, 64, 3×3) → BatchNorm2d → ReLU   ×3
Flatten
Linear(64×6×7, 256) → ReLU
Linear(256, 1) → Tanh

Output: value in [-1, +1]
  +1 = current player is winning
  -1 = opponent is winning
```

Training uses Apple Metal (MPS) for GPU acceleration on M3. The network replaces random rollouts in MCTS — during self-play, every leaf node is evaluated by the network instead of simulated to the end.

### Layer dependency rules

Strict one-way imports prevent circular dependencies:

```
game/        ← no imports from this project
bots/        ← imports from game/ only
simulation/  ← imports from bots/, game/
training/    ← imports from simulation/, bots/, game/
nn/          ← imports from bots/, game/, simulation/
viewer/      ← imports from bots/, game/, io/
scripts/     ← imports from everything
```

---

## Requirements

- Python 3.10+
- `matplotlib` — fitness charts during training
- `flask` — browser viewer backend
- `torch`, `numpy`, `tqdm` — neural network training *(optional — all other features work without PyTorch)*

```bash
pip install -r requirements.txt
```

For neural network training on Apple Silicon, verify MPS is available:
```bash
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

---

## Included trained weights

The repo includes evolved weights from Phase A (generation 26, fitness 0.9425):

```json
{
  "centreBonus":   2.63,
  "two":           1.19,
  "three":         4.53,
  "win":           200,
  "depthBonus":    0.30,
  "threatPenalty": 0
}
```

These are loaded automatically by the Minimax (evolved) bot in the viewer and benchmark script. No training required to use them.

Neural network weights (`models/best_model.pt`) are not included in the repo. Run `scripts/train_nn.py` to generate them, or download a pretrained checkpoint from the releases page if available.
