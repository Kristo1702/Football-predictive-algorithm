from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import config
from src.evaluate import evaluate_models
from src.load_data import load_dataset
from src.preprocessing import clean_data, create_targets, time_based_split
from src.train import train_and_save_models


def main():
    df = load_dataset(config.DATASET_PATH)
    df = create_targets(df)
    df = clean_data(df, config.FEATURE_COLUMNS)

    train_df, val_df, test_df = time_based_split(df)
    home_model, away_model, feature_columns = train_and_save_models(
        train_df,
        val_df,
        test_df,
    )

    evaluate_models(
        val_df,
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        label="Validation",
    )
    evaluate_models(
        test_df,
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        label="Test",
    )

    print("\nSaved model files")
    print(f"  {config.HOME_MODEL_PATH}")
    print(f"  {config.AWAY_MODEL_PATH}")
    print(f"  {config.FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
