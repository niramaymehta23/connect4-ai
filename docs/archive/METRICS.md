# Metrics — What We Measure and Why

> Historical metrics specification; current evaluation is documented in ../GUIDE.md.

## Per-game metrics (simulation runner)

| Metric | Why it matters |
|--------|---------------|
| winner (1/2/draw) | primary outcome |
| moveCount | short games = decisive play, long = defensive |
| duration_ms | track performance, catch slowdowns |
| firstPlayer | track whether going first is an advantage |
| botAId, botBId | link result to specific weight sets |
| generation | when in training this game was played |

## Per-generation metrics (training loop)

| Metric | Why it matters |
|--------|---------------|
| generation | x-axis for all charts |
| bestFitness | primary measure of improvement |
| meanFitness | health of the whole population |
| worstFitness | are weak bots getting culled properly |
| bestWinRate | win rate of champion vs pool |
| drawRate | too many draws = bots playing too defensively |
| avgMoveCount | tracks game length trend |
| mutationSigma | how much weights are changing per generation |
| wallClockTime_ms | how long each generation takes |
| gamesPlayed | total games run so far |

## Convergence indicators

| Indicator | What it signals |
|-----------|----------------|
| fitness plateau (< 0.001 improvement over 20 gens) | training has converged |
| draw rate rising | bots learning to defend rather than win |
| fitness variance collapsing | population becoming too homogeneous |
| bestFitness > 0.95 | champion dominates pool |

## Terminal output format (per generation)

```
Gen 042 | best: 0.847 | mean: 0.731 | draws: 12% | moves/game: 28.4 | 4.2s
  weights: centre=3.8  two=1.9  three=6.2  win=98.4
  top 3: bot_019(0.847)  bot_007(0.821)  bot_014(0.803)
```

## ASCII win-rate chart (every 10 generations)

```
  1.0 |                                        *
  0.9 |                              * * * * *
  0.8 |                    * * * * *
  0.7 |          * * * * *
  0.6 | * * * * *
  0.5 |
      +-------------------------------------------> generation
      0         10        20        30        40
```

## Saved metrics format (data/metrics/history.json)

```json
[
  {
    "generation": 1,
    "timestamp": "2024-01-15T10:23:44Z",
    "bestFitness": 0.612,
    "meanFitness": 0.501,
    "worstFitness": 0.388,
    "drawRate": 0.18,
    "avgMoveCount": 31.2,
    "wallClockTime_ms": 4200,
    "gamesPlayed": 200,
    "bestWeights": {
      "centreBonus": 3.1,
      "two": 2.2,
      "three": 5.4,
      "win": 101.3
    }
  }
]
```

## What good progress looks like

- Generations 1-10: rapid improvement, best fitness jumps from ~0.5 to ~0.75
- Generations 10-40: steady improvement, fitness climbs to ~0.85-0.90
- Generations 40+: slow refinement, marginal gains, possible plateau
- Draw rate should stay below 20% -- rising draws mean defensive convergence
- Move count trending down means bots are finding wins faster
