import joblib
import pandas as pd

from . import config
from .poisson import build_score_matrix, derive_probabilities
from .utils import ensure_columns_exist, format_probability


def load_prediction_assets(
    home_model_path=config.HOME_MODEL_PATH,
    away_model_path=config.AWAY_MODEL_PATH,
    feature_columns_path=config.FEATURE_COLUMNS_PATH,
):
    """Load the saved models and feature order used during training."""
    if not home_model_path.exists():
        raise FileNotFoundError(f"Missing home model: {home_model_path}")
    if not away_model_path.exists():
        raise FileNotFoundError(f"Missing away model: {away_model_path}")
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Missing feature columns: {feature_columns_path}")

    home_model = joblib.load(home_model_path)
    away_model = joblib.load(away_model_path)
    feature_columns = joblib.load(feature_columns_path)
    return home_model, away_model, feature_columns


def prepare_feature_row(feature_row, feature_columns):
    """Convert one match feature dictionary into a model-ready dataframe."""
    if isinstance(feature_row, pd.DataFrame):
        row_df = feature_row.copy()
    else:
        row_df = pd.DataFrame([feature_row])

    ensure_columns_exist(row_df, feature_columns, context="prediction input")
    row_df = row_df[feature_columns].copy()

    for column in feature_columns:
        row_df[column] = pd.to_numeric(row_df[column], errors="coerce")

    if row_df.isna().any().any():
        missing_columns = row_df.columns[row_df.isna().any()].tolist()
        raise ValueError(
            "Prediction input has missing or non-numeric values for: "
            + ", ".join(missing_columns)
        )

    return row_df


def predict_match(feature_row):
    """Predict expected goals and all derived match probabilities."""
    home_model, away_model, feature_columns = load_prediction_assets()
    X = prepare_feature_row(feature_row, feature_columns)

    home_lambda = max(float(home_model.predict(X)[0]), config.MIN_EXPECTED_GOALS)
    away_lambda = max(float(away_model.predict(X)[0]), config.MIN_EXPECTED_GOALS)

    score_matrix = build_score_matrix(
        home_lambda,
        away_lambda,
        max_goals=config.MAX_GOALS,
    )
    derived = derive_probabilities(score_matrix)
    scorelines = derived.pop("most_likely_scorelines")

    return {
        "home_expected_goals": home_lambda,
        "away_expected_goals": away_lambda,
        "probabilities": derived,
        "scorelines": scorelines,
    }


def print_prediction(prediction, home_team=None, away_team=None):
    """Print a readable match prediction report."""
    home_label = home_team or "Home"
    away_label = away_team or "Away"
    probabilities = prediction["probabilities"]

    print(f"\n{home_label} vs {away_label}")
    print("Expected goals")
    print(f"  {home_label}: {prediction['home_expected_goals']:.2f}")
    print(f"  {away_label}: {prediction['away_expected_goals']:.2f}")

    print("\nHome win / draw / away win")
    print(f"  {home_label} win: {format_probability(probabilities['home_win_prob'])}")
    print(f"  Draw: {format_probability(probabilities['draw_prob'])}")
    print(f"  {away_label} win: {format_probability(probabilities['away_win_prob'])}")

    print("\nOver/under probabilities")
    over_under = probabilities["over_under"]
    for key in [
        "over_0_5",
        "under_0_5",
        "over_1_5",
        "under_1_5",
        "over_2_5",
        "under_2_5",
        "over_3_5",
        "under_3_5",
    ]:
        print(f"  {key}: {format_probability(over_under[key])}")

    print("\nTotal goals distribution")
    for key, value in probabilities["total_goals_distribution"].items():
        print(f"  {key}: {format_probability(value)}")

    print("\nWinner goals distribution")
    for key, value in probabilities["winner_goals_distribution"].items():
        print(f"  {key}: {format_probability(value)}")

    print("\nMost likely scorelines")
    for scoreline in prediction["scorelines"]:
        print(
            f"  {scoreline['scoreline']}: "
            f"{format_probability(scoreline['probability'])}"
        )
