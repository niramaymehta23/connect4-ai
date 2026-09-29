# Phase A — Evolutionary Weight Tuning

> Historical design notes; see ../GUIDE.md for the current learning path.

## What we are optimising

The minimax heuristic scorer has several tunable parameters:

| Weight | Default | Meaning |
|--------|---------|---------|
| centreBonus | 3 | points per AI piece in the centre column |
| two | 2 | score for 2-in-a-row with 2 gaps |
| three | 5 | score for 3-in-a-row with 1 gap |
| win | 100 | score for 4-in-a-row (non-terminal, heuristic) |
| terminalWin | 1000000 | score for confirmed win (terminal node) |
| depthBonus | 0 | extra reward for winning sooner vs later |
| threatPenalty | 0 | extra penalty for opponent near-wins |

These weights determine how the bot evaluates non-terminal board positions.
The search algorithm (minimax + alpha-beta) stays fixed. We are tuning
what the bot thinks is a "good" position.

## Algorithm — (1+lambda) evolutionary strategy

```
1. Load existing best weights from data/weights/best.json (or use defaults)
2. Create a population of N bots by mutating the best weights
3. Run a round-robin tournament: every bot plays every other bot
   - each matchup: M games (alternating who goes first)
   - record win/loss/draw for each bot
4. Rank bots by win rate
5. Keep top K bots (elites), discard the rest
6. Generate next population by mutating the elites
7. Save best bot to data/weights/best.json
8. Append generation stats to data/metrics/history.json
9. Repeat from step 3
```

## Mutation

Each weight is mutated independently with:
- mutation probability: 0.3 (each weight has 30% chance of changing)
- mutation scale: gaussian noise, sigma = weight * 0.15
- bounds: all weights must stay positive; terminalWin is never mutated

Small mutations preserve what's working. Sigma scales with the weight
value so large weights and small weights mutate proportionally.

## Population parameters (defaults, configurable)

| Parameter | Default | Notes |
|-----------|---------|-------|
| populationSize | 20 | bots per generation |
| eliteCount | 4 | survivors per generation |
| gamesPerMatchup | 10 | 5 as first player, 5 as second |
| searchDepth | 4 | fixed during training for speed |
| generations | 100 | total generations per run |
| workerCount | 5 | parallel workers (M3 Pro performance cores) |

Search depth is fixed at 4 during training (not 6) because training
runs thousands of games. Depth 4 is fast and discriminates between
good and bad weights well enough. The final saved weights are then
used with depth 6 when playing interactively.

## Fitness function

```
fitness = wins / (wins + losses + draws * 0.5)
```

Draws count as half a win. This rewards decisive wins over drawing.

## Convergence criteria

Training stops early if:
- the best bot's fitness has not improved by more than 0.001 over
  the last 20 generations (plateau detection)
- or the fitness reaches 0.95+ (dominates the pool)

## What "best" means here

The best bot from Phase A will have found weight combinations that
the default values miss. It won't be qualitatively different from
minimax -- it's still the same algorithm, just better calibrated.
The ceiling is perfect minimax play at sufficient depth.

## Output files

- data/weights/best.json          -- best bot overall
- data/weights/gen_NNN.json       -- best bot snapshot each generation
- data/metrics/history.json       -- full metrics history
- data/logs/phase_a_TIMESTAMP.log -- full run log
