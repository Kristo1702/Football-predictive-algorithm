import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from . import config
from .poisson import build_score_matrix, derive_probabilities
from .predict import load_prediction_assets
from .utils import ensure_columns_exist


def _normalize_probability_rows(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def evaluate_models(
    df,
    home_model=None,
    away_model=None,
    feature_columns=None,
    label="Test",
):
    """Evaluate goal and probability quality on a dated holdout split."""
    if home_model is None or away_model is None or feature_columns is None:
        home_model, away_model, feature_columns = load_prediction_assets()

    feature_columns = list(feature_columns)
    required_columns = feature_columns + [
        "home_goals",
        "away_goals",
        "total_goals",
        "result",
    ]
    ensure_columns_exist(df, required_columns, context=f"{label.lower()} data")

    X = df[feature_columns]
    actual_home_goals = df["home_goals"].to_numpy()
    actual_away_goals = df["away_goals"].to_numpy()
    actual_total_goals = df["total_goals"].to_numpy()
    actual_results = df["result"].astype(int).to_numpy()

    home_lambdas = np.maximum(
        home_model.predict(X),
        config.MIN_EXPECTED_GOALS,
    )
    away_lambdas = np.maximum(
        away_model.predict(X),
        config.MIN_EXPECTED_GOALS,
    )
    predicted_total_goals = home_lambdas + away_lambdas

    result_probabilities = []
    over_2_5_probabilities = []

    for home_lambda, away_lambda in zip(home_lambdas, away_lambdas):
        score_matrix = build_score_matrix(
            home_lambda,
            away_lambda,
            max_goals=config.MAX_GOALS,
        )
        derived = derive_probabilities(score_matrix)
        result_probabilities.append(
            [
                derived["away_win_prob"],
                derived["draw_prob"],
                derived["home_win_prob"],
            ]
        )
        over_2_5_probabilities.append(derived["over_under"]["over_2_5"])

    result_probabilities = _normalize_probability_rows(result_probabilities)
    over_2_5_probabilities = np.asarray(over_2_5_probabilities, dtype=float)

    one_hot_results = np.eye(3)[actual_results]
    predicted_results = result_probabilities.argmax(axis=1)
    actual_over_2_5 = (actual_total_goals > 2.5).astype(int)

    metrics = {
        "home_goals_mae": mean_absolute_error(actual_home_goals, home_lambdas),
        "away_goals_mae": mean_absolute_error(actual_away_goals, away_lambdas),
        "total_goals_mae": mean_absolute_error(
            actual_total_goals,
            predicted_total_goals,
        ),
        "result_log_loss": log_loss(
            actual_results,
            result_probabilities,
            labels=[0, 1, 2],
        ),
        "result_brier_score": np.mean(
            np.sum((result_probabilities - one_hot_results) ** 2, axis=1)
        ),
        "over_2_5_brier_score": brier_score_loss(
            actual_over_2_5,
            over_2_5_probabilities,
        ),
        "accuracy": np.mean(predicted_results == actual_results),
    }

    print(f"\n{label} evaluation")
    print(f"  Home goals MAE: {metrics['home_goals_mae']:.4f}")
    print(f"  Away goals MAE: {metrics['away_goals_mae']:.4f}")
    print(f"  Total goals MAE: {metrics['total_goals_mae']:.4f}")
    print(f"  1X2 log loss: {metrics['result_log_loss']:.4f}")
    print(f"  1X2 Brier score: {metrics['result_brier_score']:.4f}")
    print(f"  Over 2.5 Brier score: {metrics['over_2_5_brier_score']:.4f}")
    print(f"  Simple accuracy: {metrics['accuracy']:.4f}")

    return metrics
