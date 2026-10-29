"""Balanced comparisons that expose trajectory diversity as well as win rate."""

import pickle
import random
from ..simulation.runner import run_game


def _play_seeded_game(bot_a, bot_b, opening, bot_a_goes_first, game_seed):
    previous_rng = random.getstate()
    try:
        random.seed(game_seed)
        result = run_game(bot_a, bot_b, bot_a_goes_first=bot_a_goes_first, opening_moves=opening)
    finally:
        random.setstate(previous_rng)
    result.update(opening=list(opening), a_first=bot_a_goes_first, seed=game_seed)
    return result


def _play_pickled_game(task):
    """Worker entry point: bots arrive as one plain pickle shared by every game."""
    bots, *game = task
    return _play_seeded_game(*pickle.loads(bots), *game)


def compare_on_openings(bot_a, bot_b, openings, seed=0, pool=None):
    """
    Play every opening twice, swapping which bot owns the first-player token.

    This balances colour assignment within each opening and exposes duplicate
    trajectories. Varied openings are not a guarantee of statistical independence.

    Every game is seeded on its own, so passing a multiprocessing `pool` (for
    example `nn.self_play.worker_pool`) gives the same games, provided both
    bots pickle (top-level functions or objects, not closures).
    """
    games = [
        (opening, bot_a_goes_first, seed + opening_index * 2 + int(not bot_a_goes_first))
        for opening_index, opening in enumerate(openings)
        for bot_a_goes_first in (True, False)
    ]
    if pool is None:
        games = [_play_seeded_game(bot_a, bot_b, *game) for game in games]
    else:
        bots = pickle.dumps((bot_a, bot_b))
        games = pool.map(_play_pickled_game, [(bots, *game) for game in games], chunksize=1)

    wins = sum(game['winner'] == 'A' for game in games)
    losses = sum(game['winner'] == 'B' for game in games)
    draws = len(games) - wins - losses
    trajectories = {tuple(game['move_history']) for game in games}
    total = len(games)
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'games': total,
        'score_rate': (wins + 0.5 * draws) / total if total else 0.0,
        'unique_trajectories': len(trajectories),
        'trajectory_diversity': len(trajectories) / total if total else 0.0,
        'game_records': games,
    }
