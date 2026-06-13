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


def print_feature_row(title, feature_row):
    print(title)
    if feature_row is None:
        print("  Not used")
        return

    for key, value in feature_row.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def print_distribution(title, distribution):
    print(f"\n{title}")
    for key, value in distribution.items():
        print(f"  {key}: {format_probability(value)}")


def print_raw_lambdas(raw_lambdas):
    print("\nRaw lambdas from both orientations")
    for key, value in raw_lambdas.items():
        print(f"  {key}: {value:.4f}")


def print_prediction_report(home_team, away_team, match_date, tournament, is_neutral, prediction):
    probabilities = prediction["probabilities"]

    print(f"{home_team} vs {away_team}")
    print(f"Match date: {match_date.date()}")
    print(f"Tournament: {tournament}")
    print(f"Neutral venue: {is_neutral}")
    print(f"Symmetric neutral prediction used: {prediction['symmetric_neutral_used']}")
    print()

    print_feature_row(
        "Generated feature row, original order",
        prediction["feature_row_ab"],
    )
    print()
    print_feature_row(
        "Generated feature row, reversed order",
        prediction["feature_row_ba"],
    )
    print_raw_lambdas(prediction["raw_lambdas"])

    print("\nFinal averaged expected goals")
    print(f"  {home_team}: {prediction['home_expected_goals']:.2f}")
    print(f"  {away_team}: {prediction['away_expected_goals']:.2f}")

    print("\nFinal home / draw / away probabilities")
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


def run_neutral_symmetry_sanity_check(match_date, tournament):
    """Check that reversing neutral team order mirrors win probabilities."""
    denmark_vs_norway = predict_upcoming_match(
        home_team="Denmark",
        away_team="Norway",
        match_date=match_date,
        tournament=tournament,
        is_neutral=1,
    )
    norway_vs_denmark = predict_upcoming_match(
        home_team="Norway",
        away_team="Denmark",
        match_date=match_date,
        tournament=tournament,
        is_neutral=1,
    )

    dn_probs = denmark_vs_norway["probabilities"]
    nd_probs = norway_vs_denmark["probabilities"]

    denmark_win_delta = abs(dn_probs["home_win_prob"] - nd_probs["away_win_prob"])
    norway_win_delta = abs(dn_probs["away_win_prob"] - nd_probs["home_win_prob"])
    draw_delta = abs(dn_probs["draw_prob"] - nd_probs["draw_prob"])

    print("\nNeutral symmetry sanity check")
    print("  Denmark vs Norway:")
    print(f"    Denmark win: {format_probability(dn_probs['home_win_prob'])}")
    print(f"    Draw:        {format_probability(dn_probs['draw_prob'])}")
    print(f"    Norway win:  {format_probability(dn_probs['away_win_prob'])}")
    print("  Norway vs Denmark:")
    print(f"    Norway win:  {format_probability(nd_probs['home_win_prob'])}")
    print(f"    Draw:        {format_probability(nd_probs['draw_prob'])}")
    print(f"    Denmark win: {format_probability(nd_probs['away_win_prob'])}")
    print("  Mirroring differences:")
    print(f"    Denmark win delta: {denmark_win_delta:.8f}")
    print(f"    Norway win delta:  {norway_win_delta:.8f}")
    print(f"    Draw delta:        {draw_delta:.8f}")

    tolerance = 1e-10
    if (
        denmark_win_delta <= tolerance
        and norway_win_delta <= tolerance
        and draw_delta <= tolerance
    ):
        print("  Result: PASS")
    else:
        print("  Result: CHECK REQUIRED")


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

    print_prediction_report(
        home_team,
        away_team,
        match_date,
        tournament,
        is_neutral,
        prediction,
    )
    run_neutral_symmetry_sanity_check(match_date, tournament)


if __name__ == "__main__":
    main()
