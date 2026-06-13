import numpy as np
from scipy.stats import poisson

from . import config
from .utils import as_float


def poisson_probability(lam, k):
    """Return P(X = k) for a Poisson random variable with mean lam."""
    if k < 0:
        return 0.0

    lam = max(float(lam), config.MIN_EXPECTED_GOALS)
    return float(poisson.pmf(k, lam))


def build_score_matrix(home_lambda, away_lambda, max_goals=config.MAX_GOALS):
    """Build a normalized matrix of exact scoreline probabilities."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")

    home_lambda = max(float(home_lambda), config.MIN_EXPECTED_GOALS)
    away_lambda = max(float(away_lambda), config.MIN_EXPECTED_GOALS)

    goal_range = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goal_range, home_lambda)
    away_probs = poisson.pmf(goal_range, away_lambda)

    score_matrix = np.outer(home_probs, away_probs)
    total_probability = score_matrix.sum()
    if total_probability <= 0:
        raise ValueError("Score matrix has zero probability mass.")

    return score_matrix / total_probability


def derive_probabilities(score_matrix):
    """Derive match probabilities from a normalized scoreline matrix."""
    score_matrix = np.asarray(score_matrix, dtype=float)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be a square 2D matrix.")

    total_mass = score_matrix.sum()
    if total_mass <= 0:
        raise ValueError("score_matrix must contain positive probability mass.")

    score_matrix = score_matrix / total_mass
    max_goal_value = score_matrix.shape[0] - 1
    home_goals = np.arange(max_goal_value + 1)[:, None]
    away_goals = np.arange(max_goal_value + 1)[None, :]
    total_goals = home_goals + away_goals

    home_win_prob = as_float(score_matrix[home_goals > away_goals].sum())
    draw_prob = as_float(score_matrix[home_goals == away_goals].sum())
    away_win_prob = as_float(score_matrix[home_goals < away_goals].sum())

    total_goals_distribution = {}
    for goals in range(6):
        total_goals_distribution[goals] = as_float(
            score_matrix[total_goals == goals].sum()
        )
    total_goals_distribution["6+"] = as_float(score_matrix[total_goals >= 6].sum())

    over_under = {}
    for threshold_goals in [0, 1, 2, 3]:
        over_key = f"over_{threshold_goals}_5"
        under_key = f"under_{threshold_goals}_5"
        over_under[over_key] = as_float(score_matrix[total_goals >= threshold_goals + 1].sum())
        over_under[under_key] = as_float(score_matrix[total_goals <= threshold_goals].sum())

    winner_goals = np.maximum(home_goals, away_goals)
    has_winner = home_goals != away_goals
    winner_goals_distribution = {
        "no_winner": draw_prob,
        "winner_1_goal": as_float(
            score_matrix[has_winner & (winner_goals == 1)].sum()
        ),
        "winner_2_goals": as_float(
            score_matrix[has_winner & (winner_goals == 2)].sum()
        ),
        "winner_3_goals": as_float(
            score_matrix[has_winner & (winner_goals == 3)].sum()
        ),
        "winner_4_plus_goals": as_float(
            score_matrix[has_winner & (winner_goals >= 4)].sum()
        ),
    }

    flat_indexes = np.argsort(score_matrix.ravel())[::-1][:5]
    most_likely_scorelines = []
    for flat_index in flat_indexes:
        home_score, away_score = np.unravel_index(flat_index, score_matrix.shape)
        most_likely_scorelines.append(
            {
                "scoreline": f"{home_score}-{away_score}",
                "home_goals": int(home_score),
                "away_goals": int(away_score),
                "probability": as_float(score_matrix[home_score, away_score]),
            }
        )

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "total_goals_distribution": total_goals_distribution,
        "over_under": over_under,
        "winner_goals_distribution": winner_goals_distribution,
        "most_likely_scorelines": most_likely_scorelines,
    }
