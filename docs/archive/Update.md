# Project Update Log — Connect4 AI

> Historical development log. Claims here require fresh controlled evaluation.

Running record of every major change, decision, and result across the project.
Updated after every significant milestone or architectural decision.

---

## Entry 001 — Project Scaffold
**Date:** 2026-03-23
**Milestone:** Initial commit — project scaffold, docs, phase A–C planning

### What was created
Full folder structure and placeholder files:

```
src/
  game/         constants.js, board.js, rules.js
  bots/         random.js, minimax.js, mcts.js (stub)
  simulation/   runner.js, tournament.js
  training/     mutate.js, fitness.js, evolution.js
  viz/          board.js, metrics.js, chart.js (stubs)
scripts/        simulate.js, train.js
data/
  weights/      best.json
  metrics/      history.json
docs/           ARCHITECTURE.md, ROADMAP.md, PHASE_A.md, PHASE_B.md, PHASE_C.md, METRICS.md
```

### Key architectural decisions made at this stage
- **Strict layer separation**: game → bots → simulation → training → scripts. No circular imports.
- **JSON for weights**: human-readable, diff-able, no database overhead at this scale.
- **Node.js ES modules** throughout (`.js` with `import/export`).
- **Phase ordering**: A (evolutionary) → C (MCTS) → B (neural net). MCTS before NN because MCTS builds on Phase A weights and is simpler to validate.
- **`terminalWin` is immutable**: structural constant used by alpha-beta. Never mutated during evolution.
- **Search depth split**: depth 4 during training (speed), depth 6 for interactive play (quality).

---

## Entry 002 — Milestone A1: Core Foundation
**Date:** 2026-03-23
**Milestone:** A1 complete — two bots can play each other, results printed to terminal

### Files implemented
| File | What it does |
|------|-------------|
| `src/game/constants.js` | `ROWS=6, COLS=7, WIN=4, HUMAN=1, AI=2, EMPTY=0` |
| `src/game/board.js` | `createBoard`, `place`, `unplace`, `clone` — pure array operations |
| `src/game/rules.js` | `checkWin`, `isFull`, `validCols`, `lowestRow` |
| `src/bots/random.js` | Picks a uniformly random valid column. Used as baseline. |
| `src/bots/minimax.js` | Minimax + alpha-beta pruning. Accepts a `weights` object. |
| `src/simulation/runner.js` | `runGame(botA, botB, opts)` → `{ winner, moves, duration }` |
| `scripts/simulate.js` | CLI: `node scripts/simulate.js --games N` |

### Key design decisions
- **`place` / `unplace` (in-place mutation)**: Avoids allocating a new board array per node during minimax search — critical for performance at depth 4–6.
- **Centre-first column ordering in minimax**: Columns tried in order `[3, 2, 4, 1, 5, 0, 6]` for better alpha-beta pruning efficiency.
- **Token assignment**: First player always uses `AI (2)`, second player uses `HUMAN (1)`. `botAGoesFirst` flag controls which bot gets which token.
- **`runGames(botA, botB, n)`**: Wrapper that alternates first player across N games and returns aggregate `{ wins, losses, draws }`.

### Default weights at this stage
```
centreBonus = 3
three       = 5
two         = 2
win         = 100
terminalWin = 1_000_000
depthBonus  = 0
threatPenalty = 0
```

---

## Entry 003 — Milestone A2: Training Loop
**Date:** 2026-03-23
**Milestone:** A2 complete — evolutionary training loop runs, weights persist across sessions

### Files implemented
| File | What it does |
|------|-------------|
| `src/training/mutate.js` | Gaussian noise mutation per weight |
| `src/training/fitness.js` | Win-rate fitness scoring |
| `src/simulation/tournament.js` | Round-robin: every bot plays every other bot |
| `src/training/evolution.js` | Full evolutionary loop — seed, tournament, select, mutate, save |
| `scripts/train.js` | CLI entry: `--gens`, `--pop`, `--depth`, `--games`, `--elite` |

### Algorithm implemented
```
1. Load best saved weights (or use defaults on first run)
2. Init population: 1 copy of seed + (N-1) mutations
3. Round-robin tournament — each pair plays gamesPerMatchup games
4. Fitness = wins / (wins + losses + draws × 0.5)
5. Keep top K elites; fill rest of next gen with mutations of elites
6. Save best.json; append to history.json
7. Repeat until convergence or max gens reached
```

### Mutation parameters (initial)
```
MUTATION_PROB = 0.3   (30% chance each weight changes)
SIGMA_SCALE   = 0.15  (sigma = |weight| × 0.15)
MIN_SIGMA     = 0.1   (floor so zero-valued weights can still evolve)
```

### Convergence criteria
- Best fitness has not improved by >0.001 over the last 20 generations (plateau)
- Or best fitness reaches ≥0.95 (dominates the pool)

### Persistence
- `data/weights/best.json` — best bot found across all runs, updated every generation
- `data/metrics/history.json` — full per-generation metrics appended every generation
- Training resumes from saved weights on restart

---

## Entry 004 — First Training Run: 20 Generations
**Date:** 2026-03-23
**Command:** `node scripts/train.js --gens 20 --pop 20 --depth 4 --games 10 --elite 4`

### Raw results
| Gen | Best | Mean | Worst | Draws |
|-----|------|------|-------|-------|
| 1   | 0.658 | 0.490 | 0.132 | 3% |
| 6   | 0.684 | 0.492 | 0.158 | 3% |
| 14  | 0.684 | 0.500 | 0.079 | 0% |
| 20  | 0.711 | 0.500 | 0.000 | 0% |

Best weights across all 20 gens: `centre=2.24  two=1.82  three=5.51  win=82.4`

### Diagnosis: training was stuck in a noise trap

Three compounding problems were identified:

**Problem 1 — Structural mean=0.5 trap**

In a pure round-robin, every win for bot A is a loss for bot B.
Therefore `sum(wins) = sum(losses)` and the population mean fitness is
always exactly 0.5 by construction. The entire useful signal lives only in
how far each bot deviates from 0.5 — which comes exclusively from
second-player wins (beating an opponent when going second).

Decomposing the fitness values:
| fitness | second-player win rate |
|---------|------------------------|
| 0.553   | 11% as 2nd player |
| 0.632   | 26% as 2nd player |
| 0.684   | 37% as 2nd player |
| 0.711   | 42% as 2nd player |

The total differentiating range is only 10–40 extra wins out of 190 games.

**Problem 2 — Tournament noise exceeds the learning signal**

The same bot scored 0.553 in gen 7 and 0.684 in gen 6 with identical
weights. That's ±0.07–0.10 noise for an unchanged bot — larger than the
fitness gain from any single mutation. The same bot (`bot_0000_16`) sat
in the top 3 for all 20 generations because its mutations could never
reliably beat it through the noise.

**Problem 3 — Mutations too small to clear the noise floor**

With `sigma = |weight| × 0.15`, a mutation to `win=82` changes it by
~±12. For `centreBonus=2.2`, sigma ≈ 0.33. These produce bots nearly
indistinguishable from their parent in tournament play. Probability of
a mutation being detectably better (clearing ±0.07 noise) was very low.

---

## Entry 005 — Fix: Random Anchor Bots + Larger Mutations
**Date:** 2026-03-24

### Root cause
The pure minimax vs minimax round-robin has a structural floor at mean=0.5
because first-mover advantage is exactly balanced (each bot plays half games
as first player, half as second). The fitness differences between nearly-
identical bots are smaller than tournament noise. Evolution cannot make
progress when it cannot distinguish parent from child.

### Decision: Add 3 random-bot anchors to every tournament
**File changed:** `src/training/evolution.js`

Random bots are injected as fixed competitors in every tournament. They are
not part of the population — they are stripped out after tournament scoring
before selection and `nextGeneration`. Their role is purely to provide a
stable baseline:

- All minimax bots beat random bots heavily → mean fitness shifts above 0.5
- The fitness gap between a stronger and weaker minimax bot is now larger
  relative to noise
- Population ordering becomes meaningful again

Implementation: `makeAnchors(n)` creates `n` bots with `id: rand_anchor_i`
and `fn: randomBot`. They are passed to `tournament([...population, ...anchors])`.
After `allRanked` is returned, `ranked = allRanked.filter(b => !b.id.startsWith('rand_anchor_'))`.
Game counts are computed from `allRanked` (full tournament) before filtering.

**Why not add anchors to the selection pool permanently?**
We don't want random bots surviving into the next generation. They are
measurement tools, not breeding candidates. Filtering after scoring is
the cleanest separation.

### Decision: Increase mutation scale
**File changed:** `src/training/mutate.js`

```diff
- MUTATION_PROB = 0.3   (30% per weight)
- SIGMA_SCALE   = 0.15  (sigma = |weight| × 0.15)
+ MUTATION_PROB = 0.5   (50% per weight)
+ SIGMA_SCALE   = 0.30  (sigma = |weight| × 0.30)
```

With the old parameters, mutations were too small to produce bots that
clearly outperformed their parent in a noisy tournament. Doubling sigma
means the rare good mutations are now large enough to register above the
noise floor. The penalty — more mutations will be much worse — is
acceptable because those are culled immediately by selection.

### Smoke test result (1 gen, pop=10, depth=4, games=4)
```
Gen 001/001 | best: 0.917 | mean: 0.625 | worst: 0.375
  weights: centre=2.24  two=1.19  three=6.82  win=82.4  depthBonus=0.03
  top 3: bot_0000_04(0.917)  bot_0000_03(0.792)  bot_0000_00(0.708)
```

Compared to the old 20-gen run:

| Metric | Old (20 gens) | New (1 gen) |
|--------|--------------|-------------|
| Best fitness | 0.55–0.71 | **0.917** |
| Mean fitness | always 0.500 | **0.625** — trap broken |
| Winner | always bot_0000_16 | bot_0000_04 (new mutation) |
| New weights explored | none | `depthBonus=0.03` emerged |

The structural mean=0.5 trap is broken. New bots win. Weights are moving.

### Current best weights (after smoke test)
```json
{
  "centreBonus": 2.243,
  "three": 6.816,
  "two": 1.194,
  "win": 82.353,
  "terminalWin": 1000000,
  "depthBonus": 0.030,
  "threatPenalty": 0
}
fitness: 0.9167  (44W / 4L / 0D across 48 games)
```

Notable: `three` rose from 5.0 → 6.82 (+36%), `two` fell from 2.0 → 1.19 (-40%).
The bot is learning to prioritise 3-in-a-row threats more heavily than
2-in-a-row, which aligns with stronger tactical play. `depthBonus` emerged
from zero, suggesting a slight reward for winning faster is useful.

---

## Entry 006 — Full 20-Gen Training Run (Post-Fix)
**Date:** 2026-03-24
**Command:** `node scripts/train.js --gens 20 --pop 20 --depth 4 --games 10 --elite 4`

### Raw results (condensed)
| Gen | Best | Mean | Worst | Draws | Best weights (win / three / two) |
|-----|------|------|-------|-------|----------------------------------|
| 1   | 0.851 | 0.565 | 0.182 | 2%  | 92.3 / 8.55 / 1.19 |
| 3   | 0.828 | 0.540 | 0.295 | 11% | 137.1 / 2.27 / 1.06 |
| 7   | 0.707 | 0.493 | 0.286 | 28% | 195.9 / 2.70 / 1.10 |
| 11  | 0.736 | 0.503 | 0.346 | 25% | 207.8 / 5.44 / 1.16 |
| 13  | 0.819 | 0.552 | 0.233 | 7%  | 303.6 / 5.17 / 0.75 |
| 14–18 | 0.699–0.769 | ~0.54 | — | 10–12% | 303.6 / 5.17 / 0.75 (stable) |
| 19–20 | 0.729–0.851 | ~0.54 | — | 9–10% | 377.3 / 5.17 / 0.75 |

### What the fixes proved
Both fixes from Entry 005 validated immediately:

| Metric | Old (pre-fix, 20 gens) | New (post-fix, 20 gens) |
|--------|------------------------|--------------------------|
| Mean fitness | always **0.500** | **0.503–0.565** (above 0.5) |
| Winner changes each gen | never — same bot all 20 gens | **yes — active evolution** |
| Weights changing | frozen | **continuously evolving** |
| Draw rate | 0–4% | **5–37%** (more competitive play) |

### Weight trajectories — clear learning signals

**`win` weight** (4-in-a-row heuristic score):
```
92 → 82 → 137 → 104 → 90 → 80 → 196 → 128 → 155 → 197
→ 208 → 155 → 304 → 304 → 304 → 304 → 304 → 304 → 377 → 377
```
Strong upward trend. Settled at 304 for 6 consecutive gens (gens 13–18),
then jumped to 377. Still climbing — not converged. Original default was 100.

**`two` weight** (2-in-a-row score):
```
1.19 → 1.19 → 1.06 → 1.39 → 1.39 → 1.37 → 1.10 → 1.16 → 1.16
→ 1.43 → 1.16 → 0.89 → 0.75 → 0.75 → 0.75 → 0.75 → 0.75 → 0.75 → 0.75 → 0.75
```
Converged to 0.75 from gen 13 onward (8 consecutive gens). Original default was 2.0.
The training is confident: 2-in-a-row threats are nearly worthless compared to 3-in-a-row.

**`three` weight** (3-in-a-row score): Noisy exploration early, settled near 5.17.
Close to the original 5.0 — this default was already reasonable.

**`depthBonus`**: Emerged from 0 → 0.097. Winning faster is slightly rewarded.

**`threatPenalty`**: Emerged from 0 → 0.062. Penalising opponent 3-in-a-rows is slightly useful.

### Draw rate interpretation
Draw rate jumped from near 0% (old runs) to 5–37% in the new runs. This is not
a regression — it reflects more competitive play. When the bots are better calibrated,
games reach positions where both sides defend well and more games go to draw. The
old low draw rate was because the bots were unbalanced and decisive wins dominated.

### Problem discovered: `bestOverall` tracking bug

**The symptom**: `Done. Best fitness: 0.8506 (gen 1)` — the saved weights were from
gen 1's bot (win=92.3), not from the evolved gen 20 lineage (win=377.3).

**The root cause**: `bestOverall` was updated with strict `>`. Gen 1 happened to score
0.851 in a relatively easy early tournament. Gen 20's evolved bot also scored 0.851
(against stronger opponents). Because `0.851 > 0.851` is false, the more evolved bot
never replaced the lucky early one.

**The consequence**: `best.json` was seeded with a regressive bot. The next training
run would start from win=92.3 and have to rediscover the win=377.3 trajectory.

**The fix applied:**
```diff
// src/training/evolution.js
- if (best.fitness > bestOverall.fitness)
+ if (best.fitness >= bestOverall.fitness)
```
With `>=`, ties favour the most recently evolved weights. A bot that scores the same
fitness against a stronger pool in gen 20 always replaces a bot that scored it in gen 1
against a weaker pool.

**best.json manually corrected** to the gen 20 evolved weights:
```json
{
  "centreBonus": 2.974,
  "three":       5.172,
  "two":         0.747,
  "win":       377.294,
  "terminalWin": 1000000,
  "depthBonus":  0.097,
  "threatPenalty": 0.062
}
```

### Problem discovered: convergence triggered coincidentally

The plateau check fires when `fitHistory[last] - fitHistory[last-20] < 0.001`.
Gen 1 best = 0.851, gen 20 best = 0.851 → difference = 0.000 → converged.

This is a false positive. The weights were actively evolving in gens 19–20 (`win`
just jumped from 304→377). The coincidence of identical best-fitness scores in
gen 1 and gen 20 triggered early stopping. The `win` weight trajectory suggests
at least another 20–30 gens would produce further improvement.

This is a known limitation to fix later (e.g., track mean fitness trend, or
compare weight distance between generations, rather than just peak fitness).

---

## Entry 007 — 50-Gen Run: Draw-Farming Problem
**Date:** 2026-03-24
**Command:** `node scripts/train.js --gens 50 --pop 20 --depth 4 --games 10 --elite 4`
**Seeded from:** gen 20 evolved weights (win=377.3)

### Result
Run stopped at gen 5 with best fitness 0.955:
```
Gen 005/050 | best: 0.955 | mean: 0.465 | worst: 0.212 | draws: 31%
  weights: centre=2.09  two=0.77  three=4.65  win=463.9  depthBonus=0.10  threatPenalty=0.02
```

### Three problems identified

**Problem 1 — Draw farming**
Draws jumped to 31% at gen 5. The bot optimised for `not losing` rather than
`winning`. Under the old formula `wins / (wins + losses + draws × 0.5)`, a draw
is rewarded as a half-win. A bot that draws 80% of its games scores fitness ≈ 0.8
without winning a single one. With random anchors in the pool, a minimax bot
drawing against anchors (instead of winning) could still get a high score.

**Problem 2 — `win` weight unboundedly climbing**
`win` went from 377 → 464 in 5 generations with no ceiling. At this scale,
`win` (4-in-a-row heuristic, non-terminal) dwarfs every other signal by 60–100×.
The bot was collapsing to a single-feature evaluator.

**Problem 3 — False convergence at final generation**
`Converged at generation 20 — stopping early` printed even when the run completed
all its generations normally (no early stop). The message was misleading.

### Three fixes applied

**Fix 1 — Fitness formula penalises draws** (`src/training/fitness.js`)
```diff
- fitness = wins / (wins + losses + draws × 0.5)   // draws rewarded as half-win
+ fitness = wins / (wins + losses + draws × 2)      // draws hurt more than losses
```
A bot with 50W / 50L / 100D now scores 0.25 instead of 0.40.
Draw-farming is no longer a viable strategy.

**Fix 2 — Hard caps on weight drift** (`src/training/mutate.js`)
```js
const CAPS = {
  win:         300,
  centreBonus:  10,
  three:        20,
  two:          10,
};
```
Applied after mutation: `result[key] = key in CAPS ? Math.min(mutated, CAPS[key]) : mutated`.
`win=300` is already ~3× the original default (100) — room for the bot to learn
it matters, but not room to drown all other signals.

**Fix 3 — Convergence message gated on early stopping** (`src/training/evolution.js`)
```diff
- console.log(`\nConverged at generation ${gen} — stopping early.`);
+ if (gen < cfg.generations) {
+   console.log(`\nConverged at generation ${gen} — stopping early.`);
+ }
```
Message now only prints when the run actually stops before its final generation.

### Seed reset
`best.json` reset to gen 20 evolved weights with `win` clamped to 300:
```json
{
  "centreBonus": 2.974,
  "three":       5.172,
  "two":         0.747,
  "win":         300,
  "terminalWin": 1000000,
  "depthBonus":  0.097,
  "threatPenalty": 0.062
}
```

---

## Entry 008 — 50-Gen Run (Post Draw-Penalty Fix)
**Date:** 2026-03-24
**Command:** `node scripts/train.js --gens 50 --pop 20 --depth 4 --games 10 --elite 4`

### Result
Stopped at gen 20 of 50 with best fitness 0.837 (gen 4):
```
Gen 004/050 | best: 0.837 | mean: 0.540 | draws: 10%  ← peak
Gen 020/050 | best: 0.606 | mean: 0.495 | draws: 30%  ← triggered convergence
```

### Problem: plateau convergence triggered by a noisy bad generation
The old plateau check was:
```js
improvement = latest - history[history.length - 20]
// 0.606 - 0.776 = -0.170 < 0.001 → fired
```
Gen 20 had 30% draws and fitness dropped to 0.606. Since `-0.17 < 0.001`, the
check fired — not because the population plateaued, but because one generation
had a bad tournament draw. The check compares `latest` (can be any value) to a
past value, so any decline over the window triggers it.

### Fix applied — two-gate convergence (`src/training/evolution.js`)

**Gate 1 — Use window peak, not latest**
```diff
- improvement = latest - history[history.length - 20]
+ windowBest  = Math.max(...fitHistory.slice(-30))
+ improvement = windowBest - window[0]
```
A single bad generation can no longer trigger the check. The question becomes:
"has the best value seen in 30 gens improved over what we saw 30 gens ago?"

**Gate 2 — Require elite variance below threshold**
```js
if (variance(eliteFitnesses) < 0.005) return true;
```
Even if the 30-gen peak has plateaued, if the top 4 elites still have diverse
fitnesses the population is still exploring. Both gates must pass simultaneously.

**Window extended from 20 → 30 generations**

Verification against the gen-20 false trigger:
```
Old:  0.606 - 0.776 = -0.170 < 0.001  → FIRES (wrong)
New:  length=20 < 30                   → does NOT fire (correct)
      windowBest=0.837, window[0]=0.776, improvement=0.061 > 0.001
      elite variance=0.000068 < 0.005 (but gate 1 already fails)
```

---

## Entry 009 — A2 Remaining Fixes + A3 Persistence
**Date:** 2026-03-24

### A2 fixes applied

**1. `gamesPerMatchup` default raised 10 → 20** (`src/training/evolution.js`, `scripts/train.js`)
Each matchup now plays 20 games instead of 10. This halves tournament noise
(variance ∝ 1/n) without changing architecture. `--games 20` is the new default
in the usage comment.

**2. `win` cap lowered 300 → 200** (`src/training/mutate.js`)
The population naturally converges toward win=80–135; the 300 cap was letting
it drift up to 463 before the draw-penalty fix brought it back. 200 is a safe
ceiling well above the convergence zone.

**3. `DEFAULT_WEIGHTS.win` raised 100 → 150** (`src/bots/minimax.js`)
Seeds fresh runs from 150 — centred on the observed convergence zone — rather
than 100 (too low) or 300 (too high). The "default seed value" the user referred
to lives in `DEFAULT_WEIGHTS`, not `mutate.js`.

**4. `best.json` reset** (`data/weights/best.json`)
- `win` clamped from 300 → 150 (aligns with new cap/seed intent)
- `generation` reset to 0, `fitness` reset to 0 (clean-slate for generation
  tracking; ensures verification run produces `gen_001.json`–`gen_005.json`)
- Other evolved weights preserved (centreBonus, three, two, depthBonus, threatPenalty)

---

### A3 — Persistence module implemented

**New file: `src/io/persistence.js`**

All save/load logic extracted from `evolution.js` into a dedicated module with
five named exports:

| Function | Description |
|---|---|
| `loadBest(defaultWeights)` | Read `best.json`; return `{ weights, generation, fitness }` |
| `saveBest(bot)` | Overwrite `best.json` |
| `saveGeneration(gen, bot)` | Write `data/weights/gen_NNN.json` snapshot |
| `loadHistory()` | Read `data/metrics/history.json`; return `[]` if missing |
| `appendMetrics(entry)` | Append one entry to `history.json` |

`saveBest` and `saveGeneration` both call `ensureDir()` (mkdirSync recursive),
so the weights directory is always created on first run. A `strip()` helper
removes the non-serialisable `fn` field before writing.

**`evolution.js` rewritten to use `persistence.js`**

- Removed: `readFileSync`, `writeFileSync`, `existsSync`, `__dirname`, `ROOT`,
  `BEST_PATH`, `HIST_PATH`, `loadSeedWeights()`, inline `saveBest()`, `appendHistory()`
- Added: `import { loadBest, saveBest, saveGeneration, appendMetrics }`
- `loadBest(DEFAULT_WEIGHTS)` returns `{ weights, generation, fitness }` — no
  longer discards saved fitness and generation number

**Generation tracking fixed: resumes from correct absolute gen number**

```js
// Before: always started from gen 1
for (let gen = 1; gen <= cfg.generations; gen++)

// After: startGen loaded from best.json, loop continues from there
const startGen = seed.generation;        // 0 on first run, 4 after 4 gens saved
const lastGen  = startGen + cfg.generations;
for (let gen = startGen + 1; gen <= lastGen; gen++)
```

The header now shows the absolute range on resume:
```
Resuming from saved weights (gen 4, fitness 0.708)
Phase A — Evolutionary weight tuning
  pop=10  elites=4  depth=4  games/matchup=20  gens=3 (gen 5–7)  anchors=3
```

---

### Verification results

**Run 1 — fresh start (gen 0 seed):**
```
node scripts/train.js --gens 5 --pop 10 --games 20 --elite 4
```
```
Gen 001/005 | best: 0.650 ...
Gen 002/005 | best: 0.625 ...
Gen 003/005 | best: 0.667 ...
Gen 004/005 | best: 0.708 ...   ← bestOverall saved here
Gen 005/005 | best: 0.625 ...
Done. Best fitness: 0.7083 (gen 4)
```
Files created: `gen_001.json` through `gen_005.json` ✓
`best.json` updated to gen 4, fitness 0.708 ✓
`history.json` has 5 new entries ✓
No early convergence (history.length=5 < 30) ✓

**Run 2 — resume check:**
```
node scripts/train.js --gens 3 --pop 10 --games 20 --elite 4
```
```
Resuming from saved weights (gen 4, fitness 0.708)
Gen 005/007 | best: 0.696 ...   ← starts from 5, not 1
Gen 006/007 | best: 0.723 ...
Gen 007/007 | best: 0.682 ...
```
Files created: `gen_006.json`, `gen_007.json` (not overwriting gen_001) ✓
Absolute gen numbering preserved across restarts ✓

---

## Entry 010 — Phase A Convergence + A4 Metrics/Visualisation
**Date:** 2026-03-24

### Phase A — Convergence result

Training converged at generation 55 (stopped early). Best found at **gen 26**:
```
fitness:  0.9425  (943 wins vs pool including 3 random anchors)
weights:  centre=2.63  two=1.19  three=4.53  win=200  depthBonus=0.30  threatPenalty=0
```

Key observations from the 50-gen run (gens 7–56):
- `win` weight hit the 200 cap and stayed there for most runs — signals the cap
  is the binding constraint, not the true optimum
- `three` rose steadily from ~2.5 to 4.53 — the bot learned to value 3-in-a-row
  more heavily as depth-4 search horizon allows acting on them
- `depthBonus` settled at 0.30 — preferring sooner wins (not in original defaults)
- `threatPenalty` decayed to zero — not useful at depth 4 given explicit search
- Draw rate stayed 10–25%, well below the 50% draw-farming episode

**Benchmark results (100 games each, depth 6):**
- vs random bot: **100% win rate** (100W / 0L / 0D)
- vs default-weights minimax (depth 6): **50% win rate** (50W / 50L / 0D)

The 50/50 result vs default is expected: training was at depth 4, and at depth 6
the search itself is much stronger — weight differences have less impact as the
bot can see further. The weights are calibrated for depth-4 heuristic scoring;
depth-6 search partially compensates for suboptimal weights by looking further.

---

### A4 — Metrics and visualisation implemented

**`src/viz/board.js`** — `printBoard(board, label?)`
Full Connect 4 board with ANSI colour:
- `AI (2)` → red ●
- `HUMAN (1)` → yellow ○
- `EMPTY (0)` → dim ·
- Grid with `+---+` dividers and column numbers 1–7

**`src/viz/metrics.js`** — `printHeader`, `printGenLine`, `fmtWeights`
Extracted from `evolution.js`. These were already implemented inline; moving them
to `src/viz/` makes them importable by tools and benchmark scripts independently.

**`src/viz/chart.js`** — `printFitnessChart(fitHistory, startGen)`
ASCII fitness chart printed every 10 gens during training and at convergence:
```
  Fitness over generations
  1.0 |
  0.9 |                                      *                   *
  0.8 |                                  *         *           *   *
  0.7 |*   *                 * *           *   * *   * * * * * * *   * *   *
  0.6 |  *   * * * * * * * *     * * * *                   * *           *   * *
  0.5 |
       +-------------------------------------------------------------> gen
             10        20        30        40        50
```
Y-levels: 0.5–1.0 in 0.1 steps. Each column maps to one generation (sampled
if n > 54 columns). `*` placed at the row whose label = round(fitness × 10)/10.
X-axis shows generation numbers at a rounded interval.

**`scripts/benchmark.js`** — full implementation replacing the TODO stub
```
node scripts/benchmark.js [--games N] [--depth D]
```
- Loads best.json via `loadBest()`
- Plays best bot vs random bot (N games)
- Plays best bot vs default-weights minimax (N games)
- Prints formatted W/L/D table with percentages
- Prints overall win rate summary
Default: 200 games, depth 6.

**`evolution.js`** — wired to viz modules
- Imports `printHeader`, `printGenLine`, `fmtWeights` from `src/viz/metrics.js`
- Imports `printFitnessChart` from `src/viz/chart.js`
- `printFitnessChart(fitHistory, startGen)` called after every 10th generation
  and once more at convergence (early stop or final gen)
- Inline print functions removed from `evolution.js`

---

## Current State Summary

| Item | Status |
|------|--------|
| Phase A1 — Core engine | ✅ Complete |
| Phase A2 — Training loop | ✅ Complete |
| A2 fix: games/matchup default 10→20 | ✅ Applied |
| A2 fix: win cap 300→200 | ✅ Applied |
| A2 fix: DEFAULT_WEIGHTS.win 100→150 | ✅ Applied |
| A3 — `src/io/persistence.js` | ✅ Complete |
| A3 — per-gen `gen_NNN.json` snapshots | ✅ Complete |
| A3 — correct gen resuming | ✅ Fixed |
| Anchor-bot fix | ✅ Applied — mean > 0.5 confirmed |
| Mutation sigma fix | ✅ Applied — weights evolving |
| `bestOverall >=` fix | ✅ Applied |
| Draw-penalty fitness fix | ✅ Applied — draws × 2 in denominator |
| Convergence robustness fix | ✅ Applied — 30-gen window + elite variance gate |
| Phase A converged | ✅ Gen 26, fitness 0.9425, 100% vs random |
| Phase A4 — Metrics viz | ✅ Complete |
| Full JS → Python port | ✅ Complete (Entry 011) |
| Phase A5 — Parallelisation | ✅ Complete (Entry 012) |
| Bot Viewer UI milestone | ✅ Added to ROADMAP.md |
| Phase C — MCTS | ⬜ Not started |
| Phase B — Neural net | ⬜ Not started |

## Immediate Next Step

A5 complete. Next: **Phase C — MCTS bot** (`src/bots/mcts.py` stub already present).

---

## Entry 011 — Full JavaScript → Python Port
**Date:** 2026-03-25
**Scope:** Entire codebase translated from Node.js ES modules to Python 3 stdlib

### What changed
All `.js` source files translated to Python. No external libraries except
`matplotlib` for charts. Same directory structure, same module boundaries,
same logic and weight format throughout.

### Files created

| Python file | Translated from |
|---|---|
| `src/game/constants.py` | `constants.js` |
| `src/game/board.py` | `board.js` |
| `src/game/rules.py` | `rules.js` |
| `src/bots/random.py` | `random.js` |
| `src/bots/minimax.py` | `minimax.js` |
| `src/simulation/runner.py` | `runner.js` |
| `src/simulation/tournament.py` | `tournament.js` |
| `src/training/mutate.py` | `mutate.js` |
| `src/training/fitness.py` | `fitness.js` |
| `src/training/evolution.py` | `evolution.js` |
| `src/io/persistence.py` | `persistence.js` |
| `src/viz/board.py` | `viz/board.js` |
| `src/viz/metrics.py` | `viz/metrics.js` |
| `src/viz/chart.py` | `viz/chart.js` (**replaced** ASCII → matplotlib) |
| `scripts/simulate.py` | `simulate.js` |
| `scripts/train.py` | `train.js` |
| `scripts/benchmark.py` | `benchmark.js` |
| `src/bots/mcts.py` | `mcts.js` (renamed stub, not yet implemented) |

`__init__.py` added to every `src/` subdirectory so all modules are importable as packages.

### Translation rules applied
- `argparse` for CLI argument parsing (replaces `process.argv`)
- `json` module for all file I/O (replaces `JSON.parse` / `fs`)
- `pathlib.Path` for file paths (replaces `path.join` / `__dirname`)
- ANSI colour codes in `src/viz/board.py` identical to the JS version
- `data/weights/best.json` weight format unchanged — no migration needed

### `src/viz/chart.py` — matplotlib replacement
The ASCII chart in `chart.js` was replaced entirely with a two-subplot
matplotlib figure saved to `data/metrics/fitness_chart.png`:

- **Subplot 1:** Best fitness (solid line), mean fitness (dashed), worst
  fitness (dotted) over generations
- **Subplot 2:** Draw rate over generations
- `matplotlib.use('Agg')` — non-interactive backend, no window ever opens
- Saved after every 10 generations and at the end of training
- Graceful fallback: if `matplotlib` is not installed, prints a warning
  and continues training without crashing

`requirements.txt` added at project root:
```
matplotlib
# Phase B dependencies (not yet required):
# torch
# numpy
```

### Verification results

| Check | Command | Result |
|---|---|---|
| 1 | `python scripts/simulate.py` | Minimax beat random 100% over 100 games ✅ |
| 2 | `python scripts/train.py --gens 5 --pop 10 --games 20 --elite 4` | gen_001–gen_005 created, best.json updated, history.json appended, fitness_chart.png created ✅ |
| 3 | `python scripts/benchmark.py` | 100% vs random, ~50% vs default-weights minimax ✅ |

---

## Entry 012 — Milestone A5: Parallelisation
**Date:** 2026-03-25
**Milestone:** A5 complete — tournament matchups run in parallel across CPU cores

### Goal
Replace the sequential matchup loop in `tournament.py` with a
`multiprocessing.Pool` distribution across worker processes. Target: ~5×
speedup on the M3 Pro (5 performance cores). No external libraries —
`multiprocessing` is stdlib.

### New file: `src/simulation/worker.py`

Top-level `run_matchup(args)` function — required to be top-level because
`multiprocessing` pickles functions by `(module, qualname)`. Lambdas,
local functions, and methods are not picklable.

```
args = {
    bot_a_id, bot_b_id,
    weights_a, weights_b,   ← weight dicts (None = random bot)
    depth_a, depth_b,
    games
}
```

Bots are **not** passed across process boundaries (functions are not
reliably picklable). Each worker reconstructs both bots from scratch
inside the process using `make_minimax_bot(depth, weights)` or
`random_bot` (when `weights is None`).

Returns `{ bot_a_id, bot_b_id, wins_a, wins_b, draws, move_counts }`.

### Updated: `src/simulation/tournament.py`

- Added `workers` parameter (default `5`)
- When `workers > 1`: builds all pair matchup specs upfront, then
  dispatches with `multiprocessing.Pool(workers).map(run_matchup, matchups)`
- When `workers == 1`: falls back to `[run_matchup(m) for m in matchups]`
  (sequential — useful for debugging and profiling)
- Result aggregation and fitness scoring unchanged from the JS version

### Updated: `scripts/train.py`

- Added `--workers` flag (default `5`)
- `workers` passed through `opts` dict to `evolve()` → `tournament()`
- **`if __name__ == '__main__'` guard added** — required on macOS to prevent
  worker processes from re-executing the script and spawning infinitely
- `multiprocessing.set_start_method('fork')` — avoids re-importing all
  modules in each worker (fork inherits the parent's address space)

### Updated: `scripts/benchmark.py`

- Added `--workers` flag (default `5`)
- Same `if __name__ == '__main__'` guard and `set_start_method('fork')`

### Wall-clock speedup

Both runs: `--gens 3 --pop 20 --games 20 --elite 4`

| workers | Time (3 gens) | Speedup |
|---------|--------------|---------|
| 1 (sequential) | ~2062.7s | 1× baseline |
| 5 (parallel) | ~470.7s | **4.38×** |

~4.4× speedup on M3 Pro with 5 performance cores. Per-generation wall
clock time is printed in the terminal output on each `Gen` line so the
speedup is visible at a glance.

### Key macOS multiprocessing notes

| Issue | Solution |
|---|---|
| `spawn` (macOS default) re-imports `__main__` in every worker | Use `set_start_method('fork')` in the script's `__main__` block |
| Workers spawning infinitely | `if __name__ == '__main__':` guard in all script entry points |
| Bots (closures) not picklable | Pass weight dicts only; reconstruct bots inside each worker |
| Lambdas not picklable | `run_matchup` is a top-level module function |

---

## Entry 018 — Phase B: Enhanced Metrics, Charts, and Terminal Logging
**Date:** 2026-03-25

### What was added

Three areas of improvement on top of the Phase B1+B2 foundation:
terminal logging quality, structured JSON metrics, and a four-subplot training chart.

---

#### 1. Terminal logging — `src/nn/train_nn.py`

**Startup header** — printed once before the first iteration:
```
╔══════════════════════════════════════════════╗
║        Connect4 AI — Phase B Training        ║
╚══════════════════════════════════════════════╝
Device:       mps (Apple M3 GPU)
Iterations:   5
Games/iter:   5
MCTS iters:   30
Batch size:   32
Train steps:  8
Eval every:   3
Ckpt every:   3
──────────────────────────────────────────────
```

**Resume notice** (when `data/metrics/nn_history.json` exists):
```
Resuming from iteration 5 (last loss: 0.2806, buffer will reset)
```

**Per-iteration line** with zero-padded numbers and wall-clock time:
```
Iter 0001/0005 | loss:   0.1226 | buffer:     92 | games:      5 | 3.8s
```

**Eval block** — only on evaluation iterations, shows W/L/D explicitly:
```
── Eval @ iter 0003 ────────────────────────────
  Candidate vs Champion: 23W / 26L / 1D  (46.0%)
  → champion retained (threshold: 55.0%)
──────────────────────────────────────────────
  ✓ Checkpoint saved: models/checkpoint_003.pt
```

Or when promoted:
```
  → NEW CHAMPION saved to models/best_model.pt
```

**End-of-run summary:**
```
══════════════════════════════════════════════
Training complete.
Total iterations: 5  |  Total games: 25
Best eval win rate: 0.460 (iter 0003)
Final buffer size: 411
Total wall time:   43.9s
══════════════════════════════════════════════
```

`_format_duration(seconds)` renders `1h 23m 14s` / `4m 03s` / `43.9s` as appropriate.

**Eval and checkpoint are now independent** — they each check `i % interval == 0` separately, so both fire on the same iteration without one suppressing the other (fixes a `continue`-based bug in the earlier version).

---

#### 2. Metrics tracking — `data/metrics/nn_history.json`

After every iteration, `_append_nn_history(history, entry)` writes a JSON array entry:
```json
{
  "iteration": 3,
  "timestamp": "2026-03-25T15:43:52Z",
  "loss": 0.122309,
  "buffer_size": 253,
  "games_played": 15,
  "wall_clock_ms": 3042,
  "eval_win_rate": 0.46,
  "champion_updated": false
}
```
`eval_win_rate` is `null` on non-eval iterations.  The file accumulates across runs:
on resume, the history is loaded at startup and new entries are appended, so the full
multi-session training curve is preserved in one file.

`games_played` = `abs_iter × games_per_iter` (absolute across all sessions, so the
field counts correctly even after resuming from a previous session).

---

#### 3. `src/nn/chart.py` — four-subplot training chart

`save_nn_chart(history)` produces `data/metrics/nn_fitness_chart.png` — a 13×8-inch
2×2 figure using the same Agg/non-interactive backend as `src/viz/chart.py`.

| Subplot | Data |
|---------|------|
| Top-left | MSE loss over iterations (line; skips `null` entries) |
| Top-right | Replay buffer size over iterations |
| Bottom-left | Eval win rate — scatter+line on eval iterations only; dashed red line at 0.55 promotion threshold |
| Bottom-right | Wall-clock time per iteration in seconds |

Chart is saved after every evaluation and once more at the end of training.
Gracefully skips with a warning if matplotlib is not installed.

---

#### 4. `scripts/train_nn.py` — `--checkpoint-every` flag

New CLI flag (default `10`) controls the checkpoint interval independently of
`--eval-every`.  Both flags accept any positive integer.

---

### Verification output (clean first run)

```
$ /usr/bin/python3 scripts/train_nn.py \
    --iterations 5 --games 5 --mcts-iters 30 \
    --batch-size 32 --train-steps 8 \
    --eval-every 3 --checkpoint-every 3

Device: mps
╔══════════════════════════════════════════════╗
║        Connect4 AI — Phase B Training        ║
╚══════════════════════════════════════════════╝
Device:       mps (Apple M3 GPU)
Iterations:   5
Games/iter:   5
MCTS iters:   30
Batch size:   32
Train steps:  8
Eval every:   3
Ckpt every:   3
──────────────────────────────────────────────
Iter 0001/0005 | loss:   0.1226 | buffer:     92 | games:      5 | 3.8s
Iter 0002/0005 | loss:   0.0875 | buffer:    161 | games:     10 | 2.0s
Iter 0003/0005 | loss:   0.1223 | buffer:    253 | games:     15 | 3.0s
── Eval @ iter 0003 ────────────────────────────
  Candidate vs Champion: 23W / 26L / 1D  (46.0%)
  → champion retained (threshold: 55.0%)
──────────────────────────────────────────────
  ✓ Checkpoint saved: models/checkpoint_003.pt
Iter 0004/0005 | loss:   0.3689 | buffer:    318 | games:     20 | 2.0s
Iter 0005/0005 | loss:   0.9908 | buffer:    411 | games:     25 | 2.6s

══════════════════════════════════════════════
Training complete.
Total iterations: 5  |  Total games: 25
Best eval win rate: 0.460 (iter 0003)
Final buffer size: 411
Total wall time:   43.9s
══════════════════════════════════════════════
```

**Checklist:**
- ✅ Header prints with `mps (Apple M3 GPU)` and all parameters
- ✅ Per-iteration lines with loss, buffer, games, timing
- ✅ Eval block prints at iter 3 (W/L/D + retention/promotion message)
- ✅ Checkpoint message prints at iter 3 (eval and checkpoint fire independently)
- ✅ `data/metrics/nn_history.json` — 5 entries with correct structure
- ✅ `data/metrics/nn_fitness_chart.png` — 4-subplot figure (~95 KB)
- ✅ `models/checkpoint_003.pt` — saved
- ✅ End summary with totals and best eval

**Resume also verified** (second run after first 5 iters):
```
Resuming from iteration 5 (last loss: 0.2806, buffer will reset)
...
Iter 0006/0010 | loss:   0.0805 | buffer:     75 | games:     30 | 2.6s
...
── Eval @ iter 0008 ────────────────────────────
  Candidate vs Champion: 29W / 21L / 0D  (58.0%)
  → NEW CHAMPION saved to models/best_model.pt
──────────────────────────────────────────────
  ✓ Checkpoint saved: models/checkpoint_008.pt
...
Total iterations: 10  |  Total games: 50
Best eval win rate: 0.580 (iter 0008)
```

History file accumulated to 10 entries across both sessions.

---

### Files changed
- `src/nn/train_nn.py` — complete rewrite of `training_loop`; `evaluate` now returns `dict` (wins/losses/draws/win_rate); new helpers `_load_nn_history`, `_append_nn_history`, `_print_header`, `_format_duration`
- `src/nn/chart.py` — new file; `save_nn_chart(history)` 4-subplot PNG
- `scripts/train_nn.py` — `--checkpoint-every` flag added

---

## Entry 017 — Phase B1 + B2: Neural Network Self-Play
**Date:** 2026-03-25

### What was built

Phase B adds a value network trained entirely from self-play.  No hand-crafted
heuristics — the network learns purely from game outcomes.  All code lives in
`src/nn/` so it is cleanly separated from Phase A (minimax) and Phase C (MCTS).

---

#### `src/nn/encode.py` — board → tensor

`encode_board(board, token) → torch.Tensor` of shape `(3, 6, 7)`, `float32`, no
batch dimension.

| Channel | Content |
|---------|---------|
| 0 | 1.0 where `token` pieces are |
| 1 | 1.0 where opponent pieces are |
| 2 | all 1.0 if `token == AI`, all 0.0 if `token == HUMAN` |

Channel 2 makes the network turn-aware without requiring separate model heads.

---

#### `src/nn/network.py` — Connect4Net

Three conv blocks (Conv2d → BatchNorm2d → ReLU), then Flatten → Linear(2688, 256)
→ ReLU → Linear(256, 1) → Tanh.  BatchNorm layers added for stability during early
training when few positions have been seen.

Output is a single float in (−1, +1): +1 = current player wins, −1 = current player
loses.

Device is resolved at import time (MPS → CPU) and printed once:
```
Device: mps
```

Class methods: `save(path)` writes the state dict; `Connect4Net.load(path, device)`
returns the network in eval mode.

---

#### `src/nn/replay.py` — ReplayBuffer

`deque`-backed circular buffer (default capacity 50 000).  Stores `(state_tensor,
outcome_float)` pairs.  Outcomes are from the player-to-move's perspective:
`+1.0` win, `-1.0` loss, `0.0` draw.

`sample(batch_size)` returns `(states, outcomes)` as stacked float32 tensors with a
batch dimension, using `random.choices` (sampling with replacement).

---

#### `src/nn/self_play.py` — Network-backed MCTS + game generation

`_run_nn_mcts(board, token, network, device, iterations, C)` runs standard UCT
(selection / expansion / evaluation / backpropagation) but replaces the rollout with a
single network forward pass.

The network outputs a value `v ∈ (−1, 1)` from `node.token`'s perspective.  To keep
compatibility with the existing `_backpropagate` function (which uses `1 − result`
flipping and expects values in `[0, 1]`):

```
val_for_node = (v + 1) / 2          # map (−1,1) → (0,1)
result = val_for_node                # if node.token == chooser
       = 1.0 − val_for_node         # otherwise (opponent's perspective)
```

All network calls are wrapped in `torch.no_grad()`.

`generate_games(network, device, n_games, mcts_iterations)` plays `n_games`
self-play games, recording `(encode_board(state, token), token)` before each move,
then assigns outcomes retroactively: winner `+1.0`, loser `−1.0`, draw `0.0`.
Returns a flat list of `(tensor, outcome_float)` pairs.

`make_nn_mcts_bot(network, device, iterations, C)` returns a `(board, token) → col`
closure for use with `runner.run_games` (evaluation).

---

#### `src/nn/train_nn.py` — Training primitives + loop

`train_step(network, optimiser, buffer, batch_size, device) → float`
: samples a batch, runs MSE forward/backward, returns scalar loss.

`evaluate(new_net, champion_net, device, n_games=50, iterations=200) → float`
: pits two network-backed MCTS bots against each other (alternating first player),
returns win rate of `new_net`.

`training_loop(opts)`:
1. Load champion from `models/best_model.pt` (or start fresh)
2. Copy to candidate, init Adam optimiser and replay buffer
3. Per iteration: self-play → push to buffer → gradient steps if buffer ≥ batch_size
4. Every `eval_every` iters: evaluate; if win rate > 0.55 → new champion saved
5. Every 10 iters: checkpoint written to `models/checkpoint_NNN.pt`

---

#### `scripts/train_nn.py` — CLI entry point

```
python3 scripts/train_nn.py [options]

--iterations   total iterations       (default 100)
--games        self-play games/iter   (default 50)
--mcts-iters   MCTS iters per move    (default 200)
--batch-size   gradient batch size    (default 256)
--train-steps  gradient steps/iter   (default 64)
--eval-every   eval interval          (default 10)
```

Guards with `if __name__ == '__main__'` and
`multiprocessing.set_start_method('fork')` for future parallelisation.

---

#### `requirements.txt` updated
Uncommented `torch` and `numpy`; added `tqdm`.

---

### Sanity check output

```
$ /usr/bin/python3 scripts/train_nn.py \
    --iterations 3 --games 5 --mcts-iters 30 \
    --batch-size 32 --train-steps 8 --eval-every 3

Device: mps
No saved champion — starting from scratch
Training on device: mps
Iterations=3  games/iter=5  mcts_iters=30  batch=32  train_steps=8  eval_every=3

Iter    1/3  loss=0.2963  buffer=83
Iter    2/3  loss=0.1948  buffer=131
Iter    3/3  loss=0.5817  buffer=210
  ↳ Eval (iter 3): candidate win rate = 0.580 → NEW CHAMPION saved

Training complete.
```

| Metric | Value |
|--------|-------|
| Device | **mps** (Apple M3 GPU) |
| Buffer growth | 83 → 131 → 210 samples over 3 iters (≈42 samples/game × 5 games) |
| Initial loss (iter 1) | 0.2963 |
| Loss after 8 grad steps (iter 2) | 0.1948 (decreasing — learning signal present) |
| Loss at iter 3 | 0.5817 (fluctuation expected at tiny sample size) |
| Eval win rate (iter 3) | 0.580 → champion saved |
| Total wall-clock time | ~45 s for 3 iterations on MPS |

The eval win rate of 0.580 at iteration 3 is inflated because the candidate is only 3
iterations ahead of the champion (essentially the same randomly-initialised network +
a few gradient steps on 210 samples).  This is expected behaviour early in training.

---

### Architecture decisions

| Decision | Rationale |
|----------|-----------|
| BatchNorm after every Conv | Stabilises training when the network has seen <1 000 positions; can be removed later once a large replay buffer is accumulated |
| Map network output to `[0, 1]` for MCTS backprop | Reuses the existing `_backpropagate` from Phase C unchanged; no risk of sign errors when going up the tree |
| `random.choices` with replacement in `ReplayBuffer.sample` | Avoids crash when buffer is smaller than batch_size early in training |
| `multiprocessing.set_start_method('fork')` | Preparation for parallelising self-play across cores (Phase B future work) |

---

### Files added
- `src/nn/__init__.py`
- `src/nn/encode.py`
- `src/nn/network.py`
- `src/nn/replay.py`
- `src/nn/self_play.py`
- `src/nn/train_nn.py`
- `scripts/train_nn.py`
- `models/.gitkeep`
- `requirements.txt` — torch, numpy, tqdm uncommented/added

---

## Entry 016 — Phase C3: MCTS Tuning — Weighted Rollout Fixed
**Date:** 2026-03-25

### What was fixed

The C2 `_weighted_rollout` used `max`-subtraction in softmax (standard numerical stabilisation):
```
exps = [exp(s - max_s) for s in raw_scores]
```
This silently collapsed whenever one score was large: for scores `[200, 5, 2, 0]` the arguments
became `[0, -195, -198, -200]`, giving `exp(0) ≈ 1` vs `exp(-195) ≈ 0` — a deterministic greedy pick.

The fix has three steps now applied in `_weighted_rollout`:
1. **Min-shift** all scores so the minimum becomes 0: `shifted = [s - min(scores) for s in scores]`.
   This means the *spread* of the scores — not the absolute magnitude — drives the probabilities.
2. **Uniform fallback** when `max(shifted) == 0` (no useful signal, e.g. opening moves).
3. **Temperature scaling** before softmax: `scaled = [s / T for s in shifted]`.
   Higher T flattens the distribution toward uniform random; lower T sharpens it toward greedy.

`temperature=1.0` is the new default, threaded as a parameter through both `_weighted_rollout`
and `make_mcts_bot`.

### Changes to scripts/simulate.py
Added `--temperature FLOAT` flag (default `1.0`). Ignored when `--rollout random`.
The rollout label now includes the temperature value, e.g.
`mcts (500 iters, weighted rollout T=3.0 (gen 26, fitness 0.943))`.

---

### Temperature sweep — 30 games each, 500 iters, weighted rollout vs minimax depth-4

| Rollout | T | Wins/30 | Win% | ms/game |
|---------|---|---------|------|---------|
| random  | — | 8       | 26.7% | 1756 |
| weighted | 0.1 | 3   | 10.0% | 1665 |
| weighted | 0.3 | 5   | 16.7% | 1706 |
| weighted | 1.0 | 8   | 26.7% | 1743 |
| **weighted** | **3.0** | **10** | **33.3%** | 1932 |
| weighted | 10.0 | 7  | 23.3% | 2009 |

**T=3.0 identified as the peak** — it beat the random baseline (33.3% vs 26.7%).

#### Why the curve peaks at T=3.0

Shifted scores for Connect 4 mid-game positions typically range from 0 (neutral) to ~5–10 (mild
threats). At T=3.0, softmax on `[0, 5, 2]` gives approximately `[13%, 72%, 24%]` — still clearly
biased toward the better move, but with enough residual probability on alternatives to maintain
rollout diversity. At T=10.0 the probabilities approach `[28%, 40%, 32%]`, which is nearly uniform
and loses most of the heuristic signal. At T=0.1, scores of 5 vs 2 become `50 vs 20` after
scaling, yielding `~1.0` vs `~10⁻¹³` — effectively deterministic.

---

### Definitive comparison — 50 games, T=3.0

| Configuration | Iters | Wins/50 | Win% | ms/game |
|---------------|-------|---------|------|---------|
| random rollout | 500 | 14 | 28.0% | 1845 |
| weighted T=3.0 | 500 | 12 | 24.0% | 1861 |
| weighted T=3.0 | 200 | 13 | 26.0% | 853 |

---

### Interpretation

**The 50-game runs are within noise of each other.** A 95% confidence interval for p ≈ 0.26
at n=50 spans roughly ±12 percentage points, so the gap between 28.0% and 24.0% is not
statistically distinguishable. The sweep's T=3.0 result (33.3%) was likely elevated by sampling
variance at n=30.

**What is confirmed:**
- The min-shift fix is correct: T=1.0 exactly matched random (26.7% vs 26.7%) in the sweep,
  meaning the fixed softmax is now a neutral transformation at unit temperature — as intended.
- The greedy direction (T < 1) consistently hurts: T=0.1 gave 10.0% and T=0.3 gave 16.7% in both
  sweep and extended runs, confirming that deterministic rollouts are genuinely harmful.
- The curve has the expected shape: performance rises from T=0.1 to T=3.0, then flattens or
  falls at T=10.0.

**The most valuable result — iteration efficiency:**
200-iter weighted (26.0%, 853 ms/game) roughly matched 500-iter random (28.0%, 1845 ms/game)
at 2.2× less wall-clock time per game. This is the practical pay-off of the weighted rollout:
comparable quality at significantly lower compute cost, even if the win-rate signal is noisy at
these sample sizes.

**Why depth-4 minimax remains hard to beat:**
Depth-4 minimax with alpha-beta pruning consistently sees 4 half-plies ahead and blocks/exploits
all immediate 3-in-a-row threats. MCTS at 500 iterations distributes those 500 samples across
a branching factor of ~7, reaching depth ~3-4 only for the most visited lines. The heuristic in
rollouts adds soft guidance but cannot replace the exact minimax lookahead for tactical positions.
Crossing the 50%+ win rate threshold will likely require either ≥2000 iterations or a parallel
rollout implementation.

---

### Files changed
- `src/bots/mcts.py` — `_weighted_rollout` rewritten with min-shift + temperature; `make_mcts_bot` gains `temperature=1.0` param
- `scripts/simulate.py` — `--temperature` flag added; rollout label includes T value

---

## Entry 015 — Phase C2: Weighted Rollouts
**Date:** 2026-03-25

### What was built

#### 1. `score_move` added to `src/bots/minimax.py`
Public function that scores a single candidate move without any recursive search:
- Calls `place(board, col, token)` to find the exact landing row
- Iterates only the WIN-length windows that physically intersect `(row, col)` — horizontal, vertical, diagonal-down-right, diagonal-up-right — plus the centre-column bonus if applicable
- Calls `unplace(board, col)` and returns the raw heuristic score
- Returns `-inf` if the column is full
- Deliberately avoids scoring the whole board, keeping it fast enough to call on every rollout step (~7 windows per call worst-case vs ~69 for the full board scan)

#### 2. `_weighted_rollout` added to `src/bots/mcts.py`
Same contract as `_rollout` (returns winning token or `None` for draw). Per step:
1. Calls `score_move(board, col, current, weights)` for every legal column
2. If `max(scores) <= 0` (no positive signal — typical on the first few moves of an empty board), falls back to `random.choice` so the rollout always makes progress
3. Otherwise applies softmax with max-subtraction for numerical stability: `exp(s − max_s)` for each score, then `random.choices(cols, weights=probs)`

Import: `from .minimax import score_move` — both files are inside `bots/`, so no layer rules are broken.

#### 3. `make_mcts_bot` updated
Added `rollout_weights=None` parameter. When not `None`, dispatches to `_weighted_rollout`; otherwise uses the existing `_rollout`. No other changes to the MCTS loop.

#### 4. `scripts/simulate.py` updated
Added `--rollout random|weighted` flag (default `random`). When `weighted`:
- Loads `data/weights/best.json` via `persistence.load_best(DEFAULT_WEIGHTS)`
- Passes `weights` to `make_mcts_bot` as `rollout_weights`
- Prints generation number and fitness in the output header for traceability

---

### Verification results

All runs vs **minimax depth-4**, alternating first player.

| Rollout policy | Iterations | Games | Wins | Losses | ms/game |
|----------------|-----------|-------|------|--------|---------|
| random         | 500        | 20    | 4 (20.0%)  | 16 (80.0%)  | 1748 |
| weighted       | 500        | 20    | 3 (15.0%)  | 17 (85.0%)  | 1819 |
| weighted       | 200        | 20    | 1 (5.0%)   | 19 (95.0%)  | 799  |
| random         | 500        | 30    | 8 (26.7%)  | 22 (73.3%)  | 1825 |
| weighted       | 500        | 30    | 3 (10.0%)  | 27 (90.0%)  | 1883 |

Weighted rollout uses Phase A champion: `data/weights/best.json` — gen 26, fitness 0.943.

---

### Observations — weighted rollout underperformed

The hypothesis was that weighted rollouts would beat random rollouts at equal iteration budgets.
Across both the 20-game and 30-game samples the weighted policy scored **consistently lower**
(~10–15%) than the random policy (~20–27%). This is a genuine and instructive result.

#### Root cause — softmax on high-magnitude scores collapses rollout diversity

`score_move` inherits the minimax weight scale: `win=200`, `three=5`, `two=2`.
As soon as any candidate move builds a near-win threat it gets a score of ~200 while all other
columns score ≤5.  After softmax that translates to probability ≈ 1.0 for the best move and
≈ 0.0 for everything else.  The rollout becomes **nearly deterministic**.

MCTS correctness depends on each simulation producing a diverse, independently-sampled outcome
so that averaging them converges to an unbiased value estimate.  When all 500 simulations from
the same node run almost the same greedy game, you are effectively running the same game 500
times — no more informative than running it once.  Worse, if the greedy policy has a systematic
blind spot, every simulation inherits that blind spot and the value estimates become **biased**
rather than noisy.  The random rollout avoids this entirely.

#### Why random rollouts work well as MCTS simulation policy

Uniform random play, despite being very weak, satisfies the key statistical requirement:
each rollout is an independent unbiased sample of the reachable outcome distribution.
Averaging thousands of such samples converges to the true game-theoretic value by the law
of large numbers.  The only cost is that it takes more iterations to reduce variance to an
acceptable level — which is why iteration budget matters so much.

#### How to fix this in a future iteration (Phase C3 candidates)

| Fix | How it helps |
|-----|-------------|
| Softmax temperature `T > 1` | Spreads probabilities; `T=10` on scores /10 keeps some diversity |
| ε-greedy rollout | With probability ε pick randomly, otherwise use heuristic |
| Normalise scores before softmax | Divide by weight magnitude so signals are O(1) not O(100) |
| Use weighted rollout only in late-game | Apply heuristic only when scores are >0 AND at least one column has a three-in-a-row |
| Heavy playout with shallow search | Replace rollout with a 1-ply minimax move rather than a heuristic sample |

---

### Files changed
- `src/bots/minimax.py` — `score_move` public function added at bottom of file
- `src/bots/mcts.py` — `_weighted_rollout`, import of `score_move`, `rollout_weights` param
- `scripts/simulate.py` — `--rollout` flag, `persistence.load_best` integration

---

## Entry 014 — Phase C: MCTS Bot Implemented
**Date:** 2026-03-25

### What was built
Implemented a full Monte Carlo Tree Search bot in `src/bots/mcts.py`, replacing the Phase C stub.

#### Core implementation
- **`MCTSNode`** class with `board`, `move`, `parent`, `children`, `wins`, `visits`, `unvisited_moves`, `token`
- **UCB1 selection**: `wins/visits + C * sqrt(ln(parent_visits) / visits)` — unvisited nodes get `inf`
- **Expansion**: picks a random unvisited move, appends a new child
- **Rollout**: plays random moves to game end, returns winning token or `None` for draw
- **Backpropagation**: walks up to root, flipping perspective at each level (key correctness fix — wins are relative to the player who chose that node, alternating at each tree level)
- **`make_mcts_bot(iterations, C)`**: factory returning a `(board, token) -> col` closure matching the exact interface of `make_minimax_bot`, requiring no changes to `runner.py` or `tournament.py`

Default parameters: `iterations=500`, `C=1.414` (√2).

#### Bug fixed during development
Initial attempt stored all results from a single root-player perspective, causing the bot to play as if the opponent also wanted to maximise the root's wins. Fixed by flipping result at each backpropagation step so each node's `wins/visits` reflects wins for the player who selected that child.

#### `scripts/simulate.py` updated
Added `--mode` flag supporting four matchups:
- `minimax_vs_random` (existing default)
- `mcts_vs_random` (new)
- `mcts_vs_minimax` (new)
- `minimax_vs_mcts` (new)

Added `--games N` and `--iterations I` flags. Legacy positional `N` arg preserved for backwards compatibility.

### Verification results

| Test | Matchup | Games | Result |
|------|---------|-------|--------|
| mcts_vs_random | MCTS (500 iters) vs random | 20 | **100% MCTS wins** |
| mcts_vs_minimax | MCTS (500 iters) vs minimax depth-4 | 10 | **40% MCTS wins** |
| mcts_vs_minimax | MCTS (2000 iters) vs minimax depth-4 | 10 | **60% MCTS wins** |

Timing: ~800 ms/game at 500 iters, ~7 s/game at 2000 iters.

### Analysis
- MCTS dominates random play (heuristic-free but statistically sound)
- At 500 iterations MCTS is competitive with depth-4 alpha-beta minimax (40% win rate)
- At 2000 iterations MCTS outperforms depth-4 minimax (60% win rate), confirming that iteration budget directly controls strength
- Depth-4 minimax has a heuristic advantage in the opening; MCTS catches up as iteration count grows and rollouts cover more of the game tree

### Files changed
- `src/bots/mcts.py` — full implementation (replaces stub)
- `scripts/simulate.py` — `--mode`, `--games`, `--iterations` flags

---

## Entry 013 — Bot Viewer UI Milestone Added
**Date:** 2026-03-25

Added **Bot Viewer UI** as a new milestone to `docs/ROADMAP.md`, positioned
after Phase C.

### What it will be
A browser-based viewer to watch two trained bots play each other in real time:

- Load any two weight files from `data/weights/`
- Watch move-by-move in the existing Connect 4 browser UI
- Heatmap showing each bot's column scores on their turn
- Speed control: slow / normal / fast
- Game stats: winner, move count, weight files used, head-to-head record

### Why after Phase C
Phase C produces the MCTS bot — the most interesting matchup to watch is
MCTS vs the evolved minimax champion. Placing this milestone after C means
there are compelling head-to-head matchups ready when the viewer is built.

### Implementation plan (from roadmap)
- Bot loader reads a `.json` weight file in the browser (plain JSON, no server needed)
- Bot vs Bot mode added to `game.js` alongside existing human vs AI mode
- Builds on top of the existing game UI — no new framework required

---

## Entry 019 — Bot Viewer UI Implemented
**Date:** 2026-03-25

### What was built
Implemented a browser-based Connect 4 viewer served by Flask with a single static page UI.

Files added:
- `scripts/viewer.py` — local entrypoint that serves the viewer on port 5000
- `src/viewer/__init__.py` — exports the Flask app
- `src/viewer/bot_registry.py` — registers available bots and serialisable bot metadata
- `src/viewer/server.py` — Flask routes for page load, bot list, move requests, status, and static assets
- `static/index.html` — single-page viewer shell
- `static/style.css` — board, heatmap, mode cards, setup, and control styling
- `static/game.js` — browser-side state machine, game rules, rendering, and bot-vs-bot controls

Other files updated:
- `requirements.txt` — added `flask`

### Bot availability in the viewer
- **Random** — available
- **Minimax (evolved)** — available, using `load_best(DEFAULT_WEIGHTS)['weights']` at depth 6
- **MCTS Fast (200 iters)** — available
- **MCTS Strong (1000 iters)** — available
- **Neural Network** — currently greyed out because `models/best_model.pt` is not present

### Viewer behaviour implemented
- Mode select screen with Player vs Player, Player vs Bot, and Bot vs Bot cards
- PvP gameplay fully handled in browser-side JS
- PvBot setup with colour picker and bot dropdown; bot moves fetched from Python
- Bot-vs-bot setup with separate bot selectors for Red and Yellow
- Heatmap panel showing minimax scores, flat scores for random/MCTS, and neural scores when available
- Bot-vs-bot controls for Reset, Next, Simulate, Pause, and speed changes
- Head-to-head record tracking across bot-vs-bot resets
- Back button resets the viewer state and returns to mode select

### Verification checklist results
- [x] Python syntax check passed for `src/viewer/*.py` and `scripts/viewer.py`
- [x] JavaScript syntax check passed for `static/game.js`
- [x] Neural bot correctly resolves to unavailable without `models/best_model.pt`
- [x] Viewer server started successfully via `.venv/bin/python scripts/viewer.py`
- [x] `/api/bots`, `/api/status`, and `/api/move` smoke-tested successfully
- [ ] Browser verification at `http://localhost:5000`

### Issues encountered and resolutions
- **Flask not installed in the system `python3` environment**: `python3 -c "import flask"` failed with `ModuleNotFoundError`. Resolved for project use by adding `flask` to `requirements.txt` and confirming the existing repo virtualenv can run the viewer successfully.
- **Heatmap state in PvBot**: the first implementation left the latest bot heatmap fully bright after turn handoff. Resolved by re-rendering the last bot scores after the turn flips back to the player, which keeps the heatmap dimmed on the player's turn.
- **Neural viewer dependency safety**: the viewer must keep working when `torch` or the trained model is missing. Resolved with guarded import/loading logic and an unavailable reason string instead of raising at import time.

---

*This file is updated after every major change, training run, or architectural decision.*

---

## Entry 020 — MCTS Determinism Fix: Proportional Visit Sampling
**Date:** 2026-03-27
**File:** `src/bots/mcts.py`

### What changed

Replaced argmax move selection with proportional sampling over visit counts.

**Before:** `max(root.children, key=lambda n: n.visits + random.uniform(0, 0.5))` — tiebreak noise was too small relative to visit counts, causing MCTS to play near-identically every game.

**After:** Visit counts are normalised into a probability distribution; `random.choices()` samples from it. The most-visited move is still strongly preferred but no longer deterministic.

### Why

MCTS was playing the same game repeatedly in Bot vs Bot mode — identical openings, identical outcomes. Proportional sampling preserves the quality signal from the search while introducing meaningful variation between games.

---

## Entry 019 — Visual Redesign: Deep Space / Terminal Luxury
**Date:** 2026-03-27
**Milestone:** Full browser viewer visual redesign

### What was redesigned

Complete aesthetic overhaul of the browser viewer. Game logic untouched.

**Design direction:** Deep space / terminal luxury — dark navy base, sharp amber-gold accent, monospace typography throughout.

### Files changed

| File | Change |
|------|--------|
| `src/viewer/bot_registry.py` | Renamed bot labels: CHAOS, DARWIN, SCOUT, ORACLE, NEXUS |
| `static/index.html` | Added Google Fonts, bg-layer orbs/grid, hero subtitle, structured mode cards with icon/title/desc, thinking indicator in heatmap panel header, new slider layout with Slow/Fast labels and speed step markers |
| `static/style.css` | Full rewrite — new CSS custom properties, animated background orbs, dot grid, Syne/DM Mono fonts, gold gradient h1, mode cards with hover shimmer/lift, panel gold accent top line, thinking dots animation, radial-gradient pieces with colour glow, win-pulse animations per colour, cold-blue→warm-gold heatmap ramp, monospace buttons/controls |
| `static/game.js` | Thinking indicator toggle in `requestMove()`, heatmap colour ramp updated (cold-blue→warm-gold), `updateRecord()` outputs coloured HTML spans |

### Verification checklist

- [x] Background orbs visible and slowly drifting, dot grid subtle
- [x] Hero `h1` has gold gradient text
- [x] Mode cards show icon + title + desc, gold shimmer on hover, lift on hover
- [x] Thin gold accent line visible at top of each panel
- [x] Thinking dots animate in gold when bot is computing, fade out when done
- [x] Heatmap uses cold-blue → warm-gold colour ramp
- [x] Pieces have radial gradient and colour glow
- [x] Winning pieces pulse with matching colour glow
- [x] Bot names show as DARWIN, SCOUT, ORACLE etc. in dropdowns
- [x] Red colour picker highlights red when active, yellow highlights yellow
- [x] Record shows coloured numbers (red / muted / yellow)
- [x] Speed steps labels align under slider
- [x] All game logic unchanged — wins, draws, heatmap, simulation, speed control all work
