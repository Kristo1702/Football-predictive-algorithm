from pathlib import Path

import pandas as pd

from .utils import ensure_columns_exist


def load_dataset(path):
    """Load the match dataset, parse dates, and sort chronologically."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    ensure_columns_exist(df, ["_date"], context=str(path))

    df["_date"] = pd.to_datetime(df["_date"], errors="coerce")
    if df["_date"].isna().any():
        missing_dates = int(df["_date"].isna().sum())
        raise ValueError(f"Found {missing_dates} rows with invalid _date values.")

    df = df.sort_values("_date").reset_index(drop=True)
    return df
