"""
train_nn.py — Training loop for the Phase B neural network.

Loop structure (per iteration):
  1. Self-play   : candidate network generates games → push to replay buffer
  2. Train step  : sample buffer, gradient descent on value cross-entropy (MSE logged)
  3. Evaluate    : every eval_every iters, candidate vs champion (50 games)
                   if win_rate > 0.55 → candidate becomes new champion
  4. Checkpoint  : every checkpoint_every iters → models/checkpoint_NNN.pt
  5. Metrics     : append to data/metrics/nn_history.json after every iteration
  6. Chart       : save 4-subplot PNG after every eval and at end of training
"""
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..nn.network   import Connect4Net
from ..nn.replay    import ReplayBuffer
from ..nn.self_play import generate_games, make_nn_mcts_bot
from ..nn.chart     import save_nn_chart
from ..simulation.runner import run_games

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT            = Path(__file__).parents[2]
_MODELS_DIR      = _ROOT / 'models'
_METRICS_DIR     = _ROOT / 'data' / 'metrics'
_BEST_PATH       = _MODELS_DIR / 'best_model.pt'
_NN_HISTORY_PATH = _METRICS_DIR / 'nn_history.json'

_PROMOTION_THRESHOLD = 0.55
_BAR_WIDTH = 46   # width of the decorative separator lines


# ---------------------------------------------------------------------------
# History persistence helpers
# ---------------------------------------------------------------------------

def _load_nn_history():
    """Load existing nn_history.json, return [] if missing or unreadable."""
    if _NN_HISTORY_PATH.exists():
        try:
            return json.loads(_NN_HISTORY_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return []


def _append_nn_history(history, entry):
    """Append one entry to history list and persist to disk."""
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history.append(entry)
    _NN_HISTORY_PATH.write_text(
        json.dumps(history, indent=2), encoding='utf-8'
    )


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

def _sep(char='─'):
    return char * _BAR_WIDTH


def _format_duration(total_seconds):
    """Return a human-readable duration string: 1h 23m 14s / 4m 03s / 12.4s"""
    secs = int(total_seconds)
    h, rem = divmod(secs, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {s:02d}s'
    if m:
        return f'{m}m {s:02d}s'
    return f'{total_seconds:.1f}s'


def _device_label(device):
    """Return a friendly device string."""
    d = str(device)
    if d == 'mps':
        return 'mps (Apple M3 GPU)'
    if d.startswith('cuda'):
        return f'{d} (NVIDIA GPU)'
    return d


def _print_header(opts, device, resuming_from=None):
    """Print the startup header block."""
    width = _BAR_WIDTH
    title = 'Connect4 AI — Phase B Training'
    pad   = (width - len(title)) // 2

    print(f'╔{"═" * width}╗')
    print(f'║{" " * pad}{title}{" " * (width - pad - len(title))}║')
    print(f'╚{"═" * width}╝')
    print(f'Device:       {_device_label(device)}')
    print(f'Iterations:   {opts.iterations}')
    print(f'Games/iter:   {opts.games}')
    print(f'MCTS iters:   {opts.mcts_iters}')
    print(f'Batch size:   {opts.batch_size}')
    print(f'Train steps:  {opts.train_steps}')
    print(f'Eval every:   {opts.eval_every}')
    print(f'Ckpt every:   {opts.checkpoint_every}')
    if resuming_from is not None:
        last      = resuming_from
        loss_str  = f'{last["loss"]:.4f}' if last["loss"] is not None else 'n/a'
        print(f'\nResuming from iteration {last["iteration"]} '
              f'(last loss: {loss_str}, buffer will reset)')
    print(_sep())


# ---------------------------------------------------------------------------
# Core training primitives
# ---------------------------------------------------------------------------

def train_step(network, optimiser, buffer, batch_size, device):
    """
    One gradient-descent step.

    Parameters
    ----------
    network    : Connect4Net in train() mode
    optimiser  : torch.optim.Optimizer
    buffer     : ReplayBuffer
    batch_size : int
    device     : torch.device

    Returns
    -------
    float — scalar MSE for this step (the optimised loss is cross-entropy)
    """
    states, outcomes = buffer.sample(batch_size)
    states   = states.to(device)
    outcomes = outcomes.to(device)

    optimiser.zero_grad()
    loss, mse = network.training_losses(states, outcomes)
    loss.backward()
    optimiser.step()

    return mse.item()


def evaluate(new_net, champion_net, device, n_games=50, iterations=200):
    """
    Play `n_games` games between `new_net` and `champion_net` using
    network-backed MCTS bots (alternating first player).

    Returns
    -------
    dict with keys: win_rate (float), wins (int), losses (int), draws (int)
    """
    new_net.eval()
    champion_net.eval()

    bot_new      = make_nn_mcts_bot(new_net,      device, iterations=iterations)
    bot_champion = make_nn_mcts_bot(champion_net, device, iterations=iterations)

    result   = run_games(bot_new, bot_champion, n_games)
    win_rate = result['wins'] / n_games
    return {
        'win_rate': win_rate,
        'wins':     result['wins'],
        'losses':   result['losses'],
        'draws':    result['draws'],
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def training_loop(opts):
    """
    Run the full self-play → train → evaluate loop.

    opts attributes used:
        iterations, games, mcts_iters, batch_size,
        train_steps, eval_every, checkpoint_every
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load existing history (resume support) ────────────────────────────────
    history = _load_nn_history()
    resuming_from = history[-1] if history else None
    start_iter    = history[-1]['iteration'] if history else 0

    # ── Initialise champion ───────────────────────────────────────────────────
    if _BEST_PATH.exists():
        champion = Connect4Net.load(_BEST_PATH, device)
    else:
        champion = Connect4Net().to(device)
        champion.eval()

    # ── Initialise candidate as a deep copy of champion ───────────────────────
    candidate = copy.deepcopy(champion).to(device)
    optimiser = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    buffer    = ReplayBuffer()

    _print_header(opts, device, resuming_from)

    # ── Tracking variables ────────────────────────────────────────────────────
    run_start         = time.perf_counter()
    best_eval_rate    = max((h['eval_win_rate'] for h in history
                             if h['eval_win_rate'] is not None), default=None)
    best_eval_iter    = None
    if best_eval_rate is not None:
        # Find which iteration that came from
        for h in history:
            if h['eval_win_rate'] == best_eval_rate:
                best_eval_iter = h['iteration']

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(1, opts.iterations + 1):
        abs_iter  = start_iter + i
        iter_start = time.perf_counter()

        # ── a. Self-play ──────────────────────────────────────────────────────
        champion.eval()
        candidate.eval()
        samples = generate_games(candidate, device, opts.games, opts.mcts_iters)
        for tensor, outcome in samples:
            buffer.push(tensor, outcome)

        # ── b. Train ──────────────────────────────────────────────────────────
        loss_val = None
        if len(buffer) >= opts.batch_size:
            candidate.train()
            for _ in range(opts.train_steps):
                loss_val = train_step(candidate, optimiser, buffer,
                                      opts.batch_size, device)
            candidate.eval()

        iter_elapsed = time.perf_counter() - iter_start
        total_games  = abs_iter * opts.games

        loss_str = f'{loss_val:.4f}' if loss_val is not None else 'n/a'
        print(f'Iter {abs_iter:04d}/{start_iter + opts.iterations:04d} | '
              f'loss: {loss_str:>8s} | '
              f'buffer: {len(buffer):>6d} | '
              f'games: {total_games:>6d} | '
              f'{iter_elapsed:.1f}s')

        # ── c. Metrics dict for this iteration ────────────────────────────────
        entry = {
            'iteration':       abs_iter,
            'timestamp':       datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'loss':            round(loss_val, 6) if loss_val is not None else None,
            'buffer_size':     len(buffer),
            'games_played':    total_games,
            'wall_clock_ms':   int(iter_elapsed * 1000),
            'eval_win_rate':   None,
            'champion_updated': False,
        }

        # ── d. Evaluate ───────────────────────────────────────────────────────
        ran_eval = False
        if i % opts.eval_every == 0:
            ran_eval    = True
            eval_result = evaluate(candidate, champion, device,
                                   n_games=50, iterations=opts.mcts_iters)
            win_rate = eval_result['win_rate']
            promoted = win_rate > _PROMOTION_THRESHOLD

            entry['eval_win_rate']    = round(win_rate, 4)
            entry['champion_updated'] = promoted

            # Track best eval
            if best_eval_rate is None or win_rate > best_eval_rate:
                best_eval_rate = win_rate
                best_eval_iter = abs_iter

            # Pretty eval block
            w, l, d = eval_result['wins'], eval_result['losses'], eval_result['draws']
            print(f'── Eval @ iter {abs_iter:04d} {"─" * (_BAR_WIDTH - 18)}')
            print(f'  Candidate vs Champion: {w}W / {l}L / {d}D  ({win_rate:.1%})')
            if promoted:
                champion = copy.deepcopy(candidate).to(device)
                champion.eval()
                champion.save(_BEST_PATH)
                print(f'  → NEW CHAMPION saved to {_BEST_PATH.relative_to(_ROOT)}')
            else:
                print(f'  → champion retained (threshold: {_PROMOTION_THRESHOLD:.1%})')
            print(_sep())

        # ── e. Checkpoint (independent of eval) ───────────────────────────────
        if i % opts.checkpoint_every == 0:
            ckpt = _MODELS_DIR / f'checkpoint_{abs_iter:03d}.pt'
            candidate.save(ckpt)
            print(f'  ✓ Checkpoint saved: {ckpt.relative_to(_ROOT)}')

        # ── f. Persist metrics + chart ─────────────────────────────────────────
        _append_nn_history(history, entry)
        if ran_eval:
            save_nn_chart(history)

    # ── End-of-run summary ────────────────────────────────────────────────────
    total_wall = time.perf_counter() - run_start
    total_iters = opts.iterations
    total_games = (start_iter + opts.iterations) * opts.games

    save_nn_chart(history)

    print()
    print('═' * _BAR_WIDTH)
    print('Training complete.')
    print(f'Total iterations: {start_iter + total_iters}  |  Total games: {total_games}')
    if best_eval_rate is not None:
        print(f'Best eval win rate: {best_eval_rate:.3f} (iter {best_eval_iter:04d})')
    else:
        print('Best eval win rate: n/a (no eval ran)')
    print(f'Final buffer size: {len(buffer)}')
    print(f'Total wall time:   {_format_duration(total_wall)}')
    print('═' * _BAR_WIDTH)
