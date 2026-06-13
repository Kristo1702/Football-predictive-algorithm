from pathlib import Path
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import config
from src.load_data import load_dataset
from src.predict import predict_upcoming_match
from src.utils import format_probability


def print_feature_row(feature_row):
    print("Generated feature row")
    for key, value in feature_row.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def print_distribution(title, distribution):
    print(f"\n{title}")
    for key, value in distribution.items():
        print(f"  {key}: {format_probability(value)}")


def main():
    home_team = "Denmark"
    away_team = "Norway"
    tournament = "Friendly"
    is_neutral = 1

    df = load_dataset(config.DATASET_PATH)
    latest_match_date = df["_date"].max()
    match_date = latest_match_date + pd.Timedelta(days=1)

    prediction = predict_upcoming_match(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        tournament=tournament,
        is_neutral=is_neutral,
    )

    probabilities = prediction["probabilities"]

    print(f"{home_team} vs {away_team}")
    print(f"Match date: {match_date.date()}")
    print(f"Tournament: {tournament}")
    print(f"Neutral venue: {is_neutral}")
    print()

    print_feature_row(prediction["feature_row"])

    print("\nExpected goals")
    print(f"  {home_team}: {prediction['home_expected_goals']:.2f}")
    print(f"  {away_team}: {prediction['away_expected_goals']:.2f}")

    print("\nHome / draw / away probabilities")
    print(f"  {home_team} win: {format_probability(probabilities['home_win_prob'])}")
    print(f"  Draw: {format_probability(probabilities['draw_prob'])}")
    print(f"  {away_team} win: {format_probability(probabilities['away_win_prob'])}")

    print_distribution(
        "Total goals distribution",
        probabilities["total_goals_distribution"],
    )
    print_distribution(
        "Winner goals distribution",
        probabilities["winner_goals_distribution"],
    )

    print("\nTop 5 scorelines")
    for scoreline in prediction["scorelines"]:
        print(
            f"  {scoreline['scoreline']}: "
            f"{format_probability(scoreline['probability'])}"
        )


if __name__ == "__main__":
    main()
