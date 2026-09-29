# Roadmap

> Archived roadmap; see ../GUIDE.md for current limitations and next experiments.

## Phase A — Evolutionary weight tuning (start here)

### A1 — Core foundation
- [ ] src/game/constants.js       -- ROWS, COLS, WIN, HUMAN, AI
- [ ] src/game/board.js           -- createBoard, place, unplace, clone
- [ ] src/game/rules.js           -- checkWin, isFull, validCols, lowestRow
- [ ] src/bots/random.js          -- picks a random valid column (baseline)
- [ ] src/bots/minimax.js         -- minimax + alpha-beta, accepts weights object
- [ ] src/simulation/runner.js    -- plays one game between two bots, returns result
- [ ] scripts/simulate.js         -- CLI: play N games, print summary

Milestone: two bots can play each other on the terminal, results printed.

### A2 — Training loop
- [ ] src/training/mutate.js      -- mutate a weights object
- [ ] src/training/fitness.js     -- score a bot by win rate vs a pool
- [ ] src/simulation/tournament.js -- round-robin between a population
- [ ] src/training/evolution.js   -- full evolutionary loop
- [ ] scripts/train.js            -- CLI entry point for training

Milestone: training loop runs, weights improve over generations.

### A3 — Persistence
- [ ] save/load weights to data/weights/best.json
- [ ] resume training from last saved generation
- [ ] per-generation snapshots to data/weights/gen_NNN.json

Milestone: training survives restarts, accumulates across sessions.

### A4 — Metrics and visualisation
- [ ] src/viz/board.js            -- print board to terminal
- [ ] src/viz/metrics.js          -- print generation stats table
- [ ] src/viz/chart.js            -- ASCII fitness chart
- [ ] append metrics to data/metrics/history.json
- [ ] scripts/benchmark.js        -- test saved bot vs random and baseline

Milestone: full training run is visible in terminal with charts.

### A5 — Parallelisation
- [ ] wrap simulation runner in worker_threads
- [ ] run 5 games in parallel per worker (M3 Pro: 5 perf cores)
- [ ] tournament distributes matchups across workers

Milestone: training runs ~5x faster.

## Phase C — MCTS

### C1 — MCTS bot
- [ ] src/bots/mcts.js            -- MCTS with UCB1, configurable iterations
- [ ] integrate with simulation runner (same interface as minimax bot)
- [ ] MCTS vs Phase A champion matchup

### C2 — Weighted rollouts
- [ ] rollout policy that uses Phase A weights to bias random play
- [ ] benchmark: random rollout vs weighted rollout

### C3 — MCTS tuning
- [ ] tune exploration constant C
- [ ] tune iteration budget vs time limit

## Phase B — Neural network

### B1 — Python project setup
- [ ] create connect4_nn/ Python project
- [ ] rewrite game logic in Python (board.py, rules.py)
- [ ] verify against JS version with identical test cases

### B2 — Network and training
- [ ] define network architecture in PyTorch
- [ ] self-play loop with MCTS-guided move selection
- [ ] replay buffer and training step
- [ ] Metal (MPS) device setup for M3 GPU

### B3 — Evaluation and export
- [ ] evaluation loop: new network vs champion
- [ ] save best model as models/best_model.pt
- [ ] (optional) export to ONNX for use in Node.js

## Bot Viewer UI

A browser-based viewer where you can watch two trained bots play each other in real time.

Features:
- Load any two bot weight files from data/weights/ directory
- Watch them play move by move in the existing Connect 4 UI
- Heatmap showing each bot's column scores on their turn
- Speed control: slow / normal / fast
- Game stats: winner, move count, which weights were used
- Head-to-head record if multiple games are played

Implementation plan:
- Add a bot loader to the browser game that reads a .json weight file
- Add Bot vs Bot mode to game.js alongside the existing human vs AI mode
- The weight files are plain JSON so the browser can load them directly
- Build on top of the existing game UI — no new framework needed

This milestone sits after Phase C so there are interesting (MCTS vs evolved minimax) matchups to watch.
