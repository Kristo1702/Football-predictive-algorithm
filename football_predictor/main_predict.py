from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict import predict_match, print_prediction


example_match = {
    "home_elo": 1900,
    "away_elo": 1750,
    "elo_diff": 150,
    "home_form_scored": 2.0,
    "home_form_conceded": 0.8,
    "home_form_win_rate": 0.7,
    "away_form_scored": 1.3,
    "away_form_conceded": 1.2,
    "away_form_win_rate": 0.45,
    "is_neutral": 1,
    "is_world_cup": 1,
    "is_continental": 0,
}


def main():
    prediction = predict_match(example_match)
    print_prediction(
        prediction,
        home_team="Example Home",
        away_team="Example Away",
    )


if __name__ == "__main__":
    main()
