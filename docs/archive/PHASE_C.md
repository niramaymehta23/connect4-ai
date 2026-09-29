# Phase C — Monte Carlo Tree Search (MCTS)

> Historical design notes; see ../GUIDE.md for the current learning path.

## Why MCTS after Phase A

Phase A optimises weights for a fixed heuristic. MCTS eliminates the
heuristic entirely -- instead of scoring a position with a formula,
it plays random games to the end and uses the win rate as the score.
This means it can discover strategies that the heuristic would never
find because they don't look good on the scoring formula.

## How MCTS works

Each move, the bot runs a loop (as many iterations as time/budget allows):

```
1. Selection
   Walk the tree from root, always choosing the child with the
   highest UCB1 score:
   UCB1 = wins/visits + C * sqrt(ln(parent_visits) / visits)
   C is the exploration constant (default 1.414 = sqrt(2))
   This balances exploiting known good moves vs exploring unknown ones.

2. Expansion
   When we reach a node that hasn't been fully expanded,
   add one new child (an unexplored move).

3. Simulation (rollout)
   From the new node, play random moves until the game ends.
   This is the "Monte Carlo" part -- no intelligence, just random play.

4. Backpropagation
   Walk back up to the root, updating win/visit counts at each node.

After the budget is exhausted, pick the move with the most visits
(not highest win rate -- visits is more robust).
```

## Key difference from minimax

Minimax looks ahead a fixed depth and scores with a heuristic.
MCTS looks ahead to the actual end of the game but plays randomly
after the explored frontier. It gets better with more iterations,
not more depth. On your M3 Pro with worker_threads, you can run
multiple rollouts in parallel.

## MCTS vs Phase A bot

These two bots can play each other during Phase C development.
MCTS with enough iterations will beat the Phase A minimax bot
because it has no heuristic blind spots.

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| iterations | 1000 | rollouts per move |
| C (exploration) | 1.414 | sqrt(2), standard value |
| rolloutPolicy | random | can be upgraded to use Phase A weights |

## Upgrade path: MCTS + heuristic rollout

Instead of random rollouts, use the Phase A weights to bias the
random rollout toward better-looking moves. This is called a
"heavy rollout" and significantly improves MCTS strength without
a neural network.

## Output

MCTS bots are defined by their parameters, not weights.
A saved MCTS bot is just:

```json
{
  "type": "mcts",
  "iterations": 1000,
  "C": 1.414,
  "rolloutPolicy": "weighted",
  "rolloutWeights": "<path to Phase A best.json>"
}
```
