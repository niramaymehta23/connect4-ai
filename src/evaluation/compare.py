"""Balanced comparisons that expose trajectory diversity as well as win rate."""

import random
from ..simulation.runner import run_game


def compare_on_openings(bot_a, bot_b, openings, seed=0):
    """
    Play every opening twice, swapping which bot owns the first-player token.

    This balances colour assignment within each opening and exposes duplicate
    trajectories. Varied openings are not a guarantee of statistical independence.
    """
    wins = losses = draws = 0
    games = []

    for opening_index, opening in enumerate(openings):
        for bot_a_goes_first in (True, False):
            previous_rng = random.getstate()
            game_seed = seed + opening_index * 2 + int(not bot_a_goes_first)
            try:
                random.seed(game_seed)
                result = run_game(
                    bot_a, bot_b, bot_a_goes_first=bot_a_goes_first,
                    opening_moves=opening,
                )
            finally:
                random.setstate(previous_rng)
            result.update(opening=list(opening), a_first=bot_a_goes_first, seed=game_seed)
            games.append(result)
            if result['winner'] == 'A':
                wins += 1
            elif result['winner'] == 'B':
                losses += 1
            else:
                draws += 1

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
