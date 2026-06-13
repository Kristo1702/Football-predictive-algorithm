import numpy as np
import pandas as pd

from . import config
from .utils import ensure_columns_exist


def create_targets(df):
    """Create target columns from final match scores."""
    df = df.copy()
    ensure_columns_exist(df, ["home_goals", "away_goals"], context="raw data")

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result"] = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ],
        [2, 1, 0],
        default=np.nan,
    )
    df["winner_goals"] = np.where(
        df["home_goals"] == df["away_goals"],
        np.nan,
        df[["home_goals", "away_goals"]].max(axis=1),
    )
    return df


def clean_data(df, feature_columns):
    """Keep only rows with complete targets and selected pre-match features."""
    df = df.copy()
    required_columns = list(feature_columns) + [
        "_date",
        "home_goals",
        "away_goals",
        "total_goals",
        "result",
    ]
    ensure_columns_exist(df, required_columns, context="prepared data")

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in config.BINARY_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["total_goals"] = pd.to_numeric(df["total_goals"], errors="coerce")
    df["result"] = pd.to_numeric(df["result"], errors="coerce")

    before_rows = len(df)
    df = df.dropna(subset=required_columns).copy()
    dropped_rows = before_rows - len(df)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing targets or features.")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["total_goals"] = df["total_goals"].astype(int)
    df["result"] = df["result"].astype(int)

    return df.reset_index(drop=True)


def filter_recent_matches(
    df,
    lookback_years=config.LOOKBACK_YEARS,
    reference_date=None,
):
    """Keep only matches inside the recent training lookback window."""
    if lookback_years <= 0:
        raise ValueError("lookback_years must be positive.")

    ensure_columns_exist(df, ["_date"], context="clean data")
    df = df.copy()
    df["_date"] = pd.to_datetime(df["_date"], errors="coerce")
    if df["_date"].isna().any():
        raise ValueError("Cannot filter recent matches with missing _date values.")

    if reference_date is None:
        reference_date = df["_date"].max()
    else:
        reference_date = pd.Timestamp(reference_date)

    cutoff_date = reference_date - pd.DateOffset(years=lookback_years)
    before_rows = len(df)
    df = df[df["_date"] >= cutoff_date].copy()
    removed_rows = before_rows - len(df)

    print(
        f"Filtered to matches from {cutoff_date.date()} onward "
        f"using reference date {reference_date.date()}."
    )
    print(f"Removed {removed_rows} matches older than {lookback_years} years.")

    return df.sort_values("_date").reset_index(drop=True)


def time_based_split(
    df,
    train_end_date=config.TRAIN_END_DATE,
    validation_end_date=config.VALIDATION_END_DATE,
):
    """Split chronologically to avoid training on future matches."""
    ensure_columns_exist(df, ["_date"], context="clean data")
    df = df.copy()
    df["_date"] = pd.to_datetime(df["_date"], errors="coerce")

    train_end = pd.Timestamp(train_end_date)
    validation_end = pd.Timestamp(validation_end_date)
    if train_end >= validation_end:
        raise ValueError("TRAIN_END_DATE must be earlier than VALIDATION_END_DATE.")

    train_df = df[df["_date"] <= train_end].copy()
    val_df = df[(df["_date"] > train_end) & (df["_date"] <= validation_end)].copy()
    test_df = df[df["_date"] > validation_end].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        date_min = df["_date"].min().date()
        date_max = df["_date"].max().date()
        raise ValueError(
            "Time split produced an empty split. "
            f"Dataset date range is {date_min} to {date_max}; "
            f"train_end={train_end.date()}, validation_end={validation_end.date()}."
        )

    return train_df, val_df, test_df


def create_time_weights(df, half_life_days=config.HALF_LIFE_DAYS):
    """Return exponential time-decay sample weights for a dated dataframe."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")

    ensure_columns_exist(df, ["_date"], context="training data")
    dates = pd.to_datetime(df["_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Cannot create time weights with missing _date values.")

    latest_date = dates.max()
    days_old = (latest_date - dates).dt.days.clip(lower=0)
    weights = 0.5 ** (days_old / float(half_life_days))
    weights = weights.to_numpy(dtype=float)

    average_weight = weights.mean()
    if average_weight <= 0:
        raise ValueError("Time weights have non-positive average.")

    # Keep the relative recency emphasis, but make model regularization easier
    # to reason about by keeping the average training weight close to 1.0.
    return weights / average_weight
