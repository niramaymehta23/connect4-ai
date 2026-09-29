# Understanding the experiment

## Your original idea: winner, copy, repeat

You imagined two versions of the same type of model playing, keeping the winner,
copying it, and repeating. This combines **self-play** (agents generating experience
by playing each other) with **selection** (choosing which version to keep).

Copying alone changes nothing. You also need an improvement mechanism:

- **Evolution:** slightly change the copy's numerical weights, test those mutations,
  and keep the variants that perform well over many games.
- **Neural learning:** use completed games as training examples and adjust the copy's
  weights with gradient descent. Then compare the trained candidate with a frozen champion.

Architecture means the arrangement of layers. Weights are the numbers learned inside
that arrangement. Two networks can have identical architecture and very different weights.
Winning one game does not prove superiority: who starts, which opening is used, and
chance in search all matter. Keep checkpoints rather than deleting the loser.

## The agents, in increasing conceptual complexity

| Agent | How it chooses | What learns? |
| --- | --- | --- |
| Random | Choose a legal column at random | Nothing |
| Minimax | Look ahead assuming the opponent responds well; score leaf boards | Nothing during play |
| Evolved minimax | Same search, with tuned scoring numbers | Small set of heuristic weights, by mutation and selection |
| MCTS | Explore a tree using simulations and accumulated results | Search statistics for this move; no lasting model by itself |
| Neural value | Try each move, estimate its resulting board value | CNN weights, during a separate training run |
| Neural + MCTS | Search a tree; use the CNN to evaluate unfinished leaf positions | CNN weights during training; tree statistics during search |

The small heuristic weights are ordinary numbers, not 'binary weights'. For example,
you might value a centre piece at 3 and an open three-in-a-row at 5. Evolution can
test 2.8 and 6.1 instead. Neural weights are also numbers, but there are many more,
and their useful patterns are learned rather than written as explicit board rules.

## Minimax in one example

Imagine trying column 3. You then examine the opponent's possible replies. If one
reply wins immediately, you must account for it even if their other replies are bad.
Repeat this reasoning several moves ahead, then choose the move with the best result
assuming sensible opposition. At the depth limit, a scoring formula estimates how
good unfinished boards are. Increasing depth costs more computation.

## MCTS in one example

Imagine you can explore only 100 possible continuations. MCTS balances looking at
promising moves with investigating moves it has barely tried. Each simulation:

1. Follows the search tree using its current statistics.
2. Adds an unexplored legal move.
3. Estimates the outcome: play a rollout to the end, or ask a neural value model.
4. Updates the visited nodes, flipping the perspective for alternating players.

After the search budget is spent, evaluation chooses the root move visited most often.
The tree is currently rebuilt for every move. MCTS alone does not save a smarter brain
after a match. Its random choices can vary, but a clearly preferred move can still repeat.

## What the neural network actually learns

The CNN sees a 3 × 6 × 7 tensor: current player's pieces, opponent pieces, and a plane
identifying the token to move. It outputs one number, approximately from -1 to +1.
That is a learned estimate of outcome from the player-to-move's perspective, not a
calibrated probability or a guarantee.

The training loop is:

```text
candidate plays itself using MCTS + exploration
  → record positions and final outcomes (+1 / 0 / -1)
  → sample old and new positions from replay memory
  → predict outcomes, compute error, adjust weights
  → compare candidate with frozen champion over paired openings
  → promote if the chosen experimental threshold is passed
```

Losing positions are useful examples too. Training uses them to learn which boards
lead to losses. A lower training loss means better fitting of sampled outcomes;
it does not by itself mean stronger play. A moving champion is also a moving target,
so its score curve is not an absolute strength curve.

This is a small value-based self-play system. AlphaGo Zero/AlphaZero also learn a
move policy and use additional search/exploration machinery. See the
[original DeepMind explanation](https://deepmind.google/blog/alphago-zero-starting-from-scratch/).

## Why you kept seeing the same game

- The same fixed policy, start board and tie-breaking rules produce the same moves.
- Copying weights produces the same policy. Inference does not update weights.
- The old minimax tournament restarted from the empty board every time. Alternating
  colours could simply replay two trajectories many times.
- Neural self-play always chose the most-visited move. The revised implementation
  samples from visits during the first eight plies, then chooses the most visited.
- MCTS contains random expansion/rollouts already; repetition does not establish a
  random-number bug. We have not reproduced every historical MCTS session.
- The old viewer's neural opponent used a different procedure from neural training.
  It also queried a post-move board from the wrong player's perspective. Scoring now
  negates the next player's value and uses exact values for immediate wins/draws.

A strong player may consistently open in the same column. Variety is useful for
training coverage and evaluation; it is not the definition of intelligence.

## Experiments to do in order

1. **Determinism:** run the tests and examine how fixed agents replay a trajectory.
2. **Search:** compare random, minimax at depths 1–3, and MCTS at several budgets.
   Record strength alongside elapsed time. Keep openings fixed for a comparison.
3. **Weights:** compare default vs historical evolved weights at the same search depth.
   Do not assume the historical champion is better until it wins a held-out comparison.
4. **Learning:** run `scripts/learn.py` with a small budget. Inspect the saved config,
   samples, losses, candidate scores, and champion decisions.
5. **Generalisation:** compare checkpoints on a separate seed/opening suite against
   fixed opponents. Repeat with multiple training seeds and report uncertainty.
6. **Improve:** only then consider a policy head, richer exploration, symmetry
   augmentation, search-tree reuse, batched inference, or a stronger reference solver.

The opening suite contains sampled legal positions, not necessarily equally strong
positions. Swapping colours controls assignment advantage within each pair; it does
not prove the suite represents every kind of position. Shared positions and correlated
games also mean you should not treat every recorded move as an independent data point.

## Evidence and remaining legacy problems

Old `data/metrics/history.json` contains repeated generation numbers. Fitness is
defined differently in `tournament.py` and `fitness.py`. The old evolutionary loop
compares 'best ever' scores against changing opponents, which is not a stable test.
Its resume generation comes from the best model rather than a complete training state.
The old neural loop reloads the champion but resets optimizer and replay memory,
and infers total games from the current configuration. Historical totals may mix budgets.

The current `learn.py` avoids those resume ambiguities with separate run directories.
It still uses a small heuristic promotion gate, value-only targets, and in-memory replay.
Full-state resume, reliable champion selection, controlled evolutionary training, and
held-out strength curves remain explicit next experiments. Existing models/data are
retained so past work is not lost or silently rewritten.

For a future shareable graph, plot checkpoint performance against **fixed opponents**
at a documented compute budget, with multiple seeds and uncertainty. Keep raw results
with the figure. Do not relabel historical within-population fitness as overall win rate.
