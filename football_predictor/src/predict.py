import joblib
import pandas as pd

from . import config
from .feature_builder import create_upcoming_match_features
from .load_data import load_dataset
from .poisson import build_score_matrix, derive_probabilities
from .preprocessing import create_targets
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


def predict_feature_row_with_models(feature_row):
    """Predict only the two expected-goals values for one feature row."""
    home_model, away_model, feature_columns = load_prediction_assets()
    X = prepare_feature_row(feature_row, feature_columns)

    home_lambda = max(float(home_model.predict(X)[0]), config.MIN_EXPECTED_GOALS)
    away_lambda = max(float(away_model.predict(X)[0]), config.MIN_EXPECTED_GOALS)
    return home_lambda, away_lambda


def _prediction_from_lambdas(home_lambda, away_lambda):
    """Build the standard prediction dictionary from expected goals."""
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


def predict_match(feature_row):
    """Predict expected goals and all derived match probabilities."""
    home_lambda, away_lambda = predict_feature_row_with_models(feature_row)
    return _prediction_from_lambdas(home_lambda, away_lambda)


def predict_neutral_match_symmetric(
    home_team,
    away_team,
    match_date,
    tournament="Friendly",
    is_neutral=1,
):
    """Predict a neutral match by averaging both team-order orientations."""
    df = load_dataset(config.DATASET_PATH)
    df = create_targets(df)

    feature_row_ab = create_upcoming_match_features(
        df=df,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        tournament=tournament,
        is_neutral=is_neutral,
    )
    feature_row_ba = create_upcoming_match_features(
        df=df,
        home_team=away_team,
        away_team=home_team,
        match_date=match_date,
        tournament=tournament,
        is_neutral=is_neutral,
    )

    ab_home_lambda, ab_away_lambda = predict_feature_row_with_models(feature_row_ab)
    ba_home_lambda, ba_away_lambda = predict_feature_row_with_models(feature_row_ba)

    # In the reversed orientation, ba_away_lambda belongs to the original
    # home_team and ba_home_lambda belongs to the original away_team.
    home_team_lambda = (ab_home_lambda + ba_away_lambda) / 2.0
    away_team_lambda = (ab_away_lambda + ba_home_lambda) / 2.0

    prediction = _prediction_from_lambdas(home_team_lambda, away_team_lambda)
    prediction["feature_row"] = feature_row_ab
    prediction["feature_row_ab"] = feature_row_ab
    prediction["feature_row_ba"] = feature_row_ba
    prediction["raw_lambdas"] = {
        "ab_home_lambda": ab_home_lambda,
        "ab_away_lambda": ab_away_lambda,
        "ba_home_lambda": ba_home_lambda,
        "ba_away_lambda": ba_away_lambda,
        "home_team_lambda": home_team_lambda,
        "away_team_lambda": away_team_lambda,
    }
    prediction["symmetric_neutral_used"] = True
    return prediction


def predict_upcoming_match(
    home_team,
    away_team,
    match_date,
    tournament="Friendly",
    is_neutral=1,
):
    """Build features from team names, then predict an upcoming match."""
    if int(is_neutral) == 1:
        return predict_neutral_match_symmetric(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            tournament=tournament,
            is_neutral=is_neutral,
        )

    df = load_dataset(config.DATASET_PATH)
    df = create_targets(df)

    feature_row = create_upcoming_match_features(
        df=df,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        tournament=tournament,
        is_neutral=is_neutral,
    )

    prediction = predict_match(feature_row)
    prediction["feature_row"] = feature_row
    prediction["feature_row_ab"] = feature_row
    prediction["feature_row_ba"] = None
    prediction["raw_lambdas"] = {
        "home_lambda": prediction["home_expected_goals"],
        "away_lambda": prediction["away_expected_goals"],
    }
    prediction["symmetric_neutral_used"] = False
    return prediction


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
