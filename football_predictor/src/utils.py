from pathlib import Path

import numpy as np


def ensure_columns_exist(df, columns, context="dataframe"):
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns in {context}: {missing_text}")


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def as_float(value):
    """Convert numpy numeric values to plain Python floats for clean output."""
    return float(np.asarray(value))


def format_probability(value):
    """Format a probability as a percentage string."""
    return f"{100.0 * float(value):.1f}%"
