# Architecture

> Historical plan, including obsolete JavaScript references. See ../GUIDE.md.

## Core principle

Game logic, bot logic, training logic, and visualisation are completely separate.
The game layer knows nothing about AI. The AI layer knows nothing about training.
This makes it easy to swap in a new bot type (MCTS, neural net) without touching
anything else.

## Data flow

```
Training loop
    |
    |-- creates population of bots (weight sets)
    |
    |-- Tournament runner
    |       |
    |       |-- picks pairs from population
    |       |
    |       +-- Simulation runner
    |               |
    |               |-- Game engine (pure, no AI)
    |               |       board.js + rules.js
    |               |
    |               |-- Bot A (minimax + weights A)
    |               +-- Bot B (minimax + weights B)
    |                       |
    |                       +-- returns { winner, moves, duration }
    |
    |-- Fitness evaluator
    |       assigns win rate score to each bot
    |
    |-- Selection + mutation
    |       keeps top N, mutates copies to form next generation
    |
    |-- Saves best weights  -->  data/weights/
    |-- Appends metrics     -->  data/metrics/history.json
    +-- Prints progress     -->  terminal (table + ASCII chart)
```

## Key design decisions

**Why not store the tree between moves?**
Minimax builds the tree implicitly on the call stack and discards it.
Storing it would use significant memory for no benefit since the opponent's
actual move invalidates most branches anyway.

**Why JSON for weights, not a database?**
Weights are small (< 1KB per bot). JSON files are human-readable, easy to
inspect, diff, and copy. A database would add complexity with no benefit at
this scale.

**Why Node.js worker_threads for simulation?**
The M3 Pro has 5 performance cores. Running 5 games in parallel across workers
cuts wall-clock training time by ~5x. Each worker is stateless -- it receives
two weight sets, plays N games, returns results. No shared state.

**Why separate terminal viz from training logic?**
So training can run headlessly (no output) for maximum speed, or with full
visualisation for monitoring. A --silent flag will suppress all terminal output.

## Module dependency rules

```
viz        -->  can import from: simulation, game
scripts    -->  can import from: training, simulation, viz, bots, game
training   -->  can import from: simulation, bots, game
simulation -->  can import from: bots, game
bots       -->  can import from: game
game       -->  no imports from this project
```

No circular dependencies. Lower layers never import from higher layers.
