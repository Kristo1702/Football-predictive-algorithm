import joblib
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .preprocessing import create_time_weights
from .utils import ensure_columns_exist, ensure_directory


def build_goal_model(alpha=config.POISSON_ALPHA, max_iter=config.POISSON_MAX_ITER):
    """Create a stable Poisson regression pipeline for goal prediction."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "poisson",
                PoissonRegressor(alpha=alpha, max_iter=max_iter),
            ),
        ]
    )


def train_models(
    train_df,
    val_df=None,
    test_df=None,
    feature_columns=config.FEATURE_COLUMNS,
    alpha=config.POISSON_ALPHA,
    max_iter=config.POISSON_MAX_ITER,
):
    """Train home-goals and away-goals expected goal models."""
    feature_columns = list(feature_columns)
    required_columns = feature_columns + ["home_goals", "away_goals"]
    ensure_columns_exist(train_df, required_columns, context="training data")
    if val_df is not None:
        ensure_columns_exist(val_df, required_columns, context="validation data")
    if test_df is not None:
        ensure_columns_exist(test_df, required_columns, context="test data")

    X_train = train_df[feature_columns]
    y_home = train_df["home_goals"]
    y_away = train_df["away_goals"]
    sample_weights = create_time_weights(train_df)

    home_model = build_goal_model(alpha=alpha, max_iter=max_iter)
    away_model = build_goal_model(alpha=alpha, max_iter=max_iter)

    home_model.fit(X_train, y_home, poisson__sample_weight=sample_weights)
    away_model.fit(X_train, y_away, poisson__sample_weight=sample_weights)

    print(f"Training rows: {len(train_df)}")
    if val_df is not None:
        print(f"Validation rows: {len(val_df)}")
    if test_df is not None:
        print(f"Test rows: {len(test_df)}")
    print("Selected feature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    if val_df is not None:
        X_val = val_df[feature_columns]
        val_home_pred = home_model.predict(X_val)
        val_away_pred = away_model.predict(X_val)
        print(f"Average predicted home goals on validation: {val_home_pred.mean():.3f}")
        print(f"Average predicted away goals on validation: {val_away_pred.mean():.3f}")

    return home_model, away_model, feature_columns


def save_models(
    home_model,
    away_model,
    feature_columns,
    home_model_path=config.HOME_MODEL_PATH,
    away_model_path=config.AWAY_MODEL_PATH,
    feature_columns_path=config.FEATURE_COLUMNS_PATH,
):
    """Save trained models and the exact feature order used at training time."""
    ensure_directory(config.MODEL_DIR)
    joblib.dump(home_model, home_model_path)
    joblib.dump(away_model, away_model_path)
    joblib.dump(list(feature_columns), feature_columns_path)


def train_and_save_models(train_df, val_df, test_df):
    """Train both goal models and persist the artifacts."""
    home_model, away_model, feature_columns = train_models(train_df, val_df, test_df)
    save_models(home_model, away_model, feature_columns)
    return home_model, away_model, feature_columns
