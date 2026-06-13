from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "teams_match_features.csv"

HOME_MODEL_PATH = MODEL_DIR / "home_goals_model.pkl"
AWAY_MODEL_PATH = MODEL_DIR / "away_goals_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

# Version 1 feature set. These columns should all be known before kickoff.
FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_scored",
    "home_form_conceded",
    "home_form_win_rate",
    "away_form_scored",
    "away_form_conceded",
    "away_form_win_rate",
    "is_neutral",
    "is_world_cup",
    "is_continental",
]

# Candidate additions for a later model version.
EXTRA_FEATURE_COLUMNS = [
    "overall_diff",
    "attack_diff",
    "defense_diff",
    "home_avg_attack",
    "away_avg_attack",
    "home_avg_defense",
    "away_avg_defense",
]

TARGET_COLUMNS = ["home_goals", "away_goals"]
METADATA_COLUMNS = ["_home_team", "_away_team", "_date", "_tournament"]
BINARY_COLUMNS = ["is_neutral", "is_world_cup", "is_continental"]

MAX_GOALS = 10

LOOKBACK_YEARS = 10
FORM_MATCH_WINDOW = 5

# Time-based split:
# train:      _date <= TRAIN_END_DATE
# validation: TRAIN_END_DATE < _date <= VALIDATION_END_DATE
# test:       _date > VALIDATION_END_DATE
TRAIN_END_DATE = "2021-12-31"
VALIDATION_END_DATE = "2023-12-31"

HALF_LIFE_DAYS = 1095  # about 3 years
TRAIN_FINAL_MODEL_ON_ALL_RECENT_DATA = True
POISSON_ALPHA = 0.01
POISSON_MAX_ITER = 1000
MIN_EXPECTED_GOALS = 1e-6
