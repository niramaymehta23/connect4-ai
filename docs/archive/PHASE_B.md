# Phase B — Neural Network Reinforcement Learning

> Historical design notes; see ../GUIDE.md for the current learning path.

## Overview

Replace the heuristic scorer with a neural network that learns purely
from self-play. No hand-crafted weights. The network takes a board
state as input and outputs a value (who is winning) and optionally a
policy (which moves look promising).

## Why Python, not Node.js

Node.js has no mature neural net library with hardware acceleration.
Python with PyTorch supports Apple Metal (M3 GPU) via the MPS backend.
For a small network this means 10-50x faster training than CPU-only.

Phase B will be a separate Python project that imports the Connect 4
game logic (rewritten in Python, keeping the same structure).

## Network architecture (starting point)

Input: 6x7x3 tensor
  - channel 0: cells occupied by AI (1 or 0)
  - channel 1: cells occupied by human (1 or 0)
  - channel 2: whose turn it is (all 1s or all 0s)

Architecture:
  Conv2d(3, 64, kernel=3, padding=1)  -->  ReLU
  Conv2d(64, 64, kernel=3, padding=1) -->  ReLU
  Conv2d(64, 64, kernel=3, padding=1) -->  ReLU
  Flatten
  Linear(64*6*7, 256)  -->  ReLU
  Linear(256, 1)       -->  Tanh  (output: -1 = human wins, +1 = AI wins)

This is a value network only. A policy head can be added in a later
iteration (as in AlphaZero) to guide the tree search directly.

## Training loop

```
1. Self-play: current network plays against itself for N games
   - moves chosen by MCTS guided by the network's value estimates
   - every (state, outcome) pair is stored in a replay buffer

2. Training: sample batches from the replay buffer
   - loss = MSE(network(state), actual_outcome)
   - backpropagate, update weights via Adam optimiser

3. Evaluation: new network plays 100 games vs previous best
   - if win rate > 55%, new network becomes the champion

4. Repeat
```

## Metal (M3 GPU) usage

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

The M3 Pro's 14-core GPU will handle forward/backward passes.
Rollouts (game simulation) still run on CPU.

## Relationship to earlier phases

Phase B does not replace Phase A or C. The Phase A weights can be used
to bootstrap the neural net training with better-than-random initial
rollouts. The Phase C MCTS can use the Phase B network as its value
function, combining the best of both approaches (this is AlphaZero's
actual architecture).

## Output

- models/best_model.pt      -- PyTorch model weights
- models/checkpoint_NNN.pt  -- checkpoint every N generations
- metrics/nn_history.json   -- loss, win rate over training
