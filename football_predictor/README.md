# Football Match Predictor

This project predicts football matches by estimating expected goals first, then deriving all match probabilities from a Poisson scoreline matrix.

It predicts:

- Home win / draw / away win probabilities
- Total goals probability distribution
- Over/under goal probabilities
- How many goals the winner scores
- Most likely exact scorelines

The core algorithm is:

```text
historical match data
-> train home and away expected-goals models
-> predict home_expected_goals and away_expected_goals
-> build a Poisson scoreline probability matrix
-> derive all other probabilities from that matrix
```

The project does not train a black-box winner classifier first.

## Project Structure

```text
football_predictor/
|-- data/
|   `-- teams_match_features.csv
|-- models/
|   |-- home_goals_model.pkl
|   |-- away_goals_model.pkl
|   `-- feature_columns.pkl
|-- src/
|   |-- config.py
|   |-- load_data.py
|   |-- preprocessing.py
|   |-- feature_builder.py
|   |-- train.py
|   |-- poisson.py
|   |-- predict.py
|   |-- evaluate.py
|   `-- utils.py
|-- main_train.py
|-- main_predict.py
`-- README.md
```

## Data

The dataset is read from:

```python
DATASET_PATH = DATA_DIR / "teams_match_features.csv"
```

`home_goals` and `away_goals` are targets. They are never input features because they are only known after the match.

`_home_team`, `_away_team`, `_date`, and `_tournament` are metadata. `_date` is used for sorting, filtering, time splitting, and feature building.

## Features

Version 1 uses:

```python
[
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
```

Extra candidate features are listed in `src/config.py`, but are not enabled by default.

## Training Window

Training does not use very old matches.

Before splitting, the dataset is filtered to the latest `LOOKBACK_YEARS` years:

```python
LOOKBACK_YEARS = 10
```

The reference date is the latest `_date` found in the dataset, not the current calendar year. For example, if the newest match in the file is in 2025, training only uses matches from roughly 2015 onward.

## Time Splitting

After the 10-year filter, the data is split chronologically:

- Train: old part of the recent data
- Validation: middle part
- Test: newest part

There is no random train/test split.

This avoids training on future matches and testing on earlier matches.

## Time Weighting

Newer matches receive higher sample weight than older matches:

```text
weight = 0.5 ** (days_old / HALF_LIFE_DAYS)
```

The default half-life is:

```python
HALF_LIFE_DAYS = 1095
```

That is about 3 years, so a match about 3 years older than the newest training match counts about half as much before normalization. Weights are normalized so their average is about 1.0.

## Final Saved Model

Training first evaluates honestly on validation and test splits.

If this flag is enabled:

```python
TRAIN_FINAL_MODEL_ON_ALL_RECENT_DATA = True
```

the final saved production models are then trained again on all recent matches from the 10-year window. This lets evaluation stay honest while the saved model still uses the newest available information.

## Automatic Prediction Features

`src/feature_builder.py` can create model features from team names:

```python
create_upcoming_match_features(
    df,
    home_team="Denmark",
    away_team="Norway",
    match_date="2025-12-12",
    tournament="Friendly",
    is_neutral=1,
)
```

The feature builder uses only matches before `match_date`.

It computes:

- Latest known home-team Elo before the match date
- Latest known away-team Elo before the match date
- Elo difference
- Each team's form from its latest `FORM_MATCH_WINDOW` matches before the match date
- Tournament flags

Warning: Elo is taken from the latest available pre-match Elo before the prediction date. It is an approximation unless the dataset provides fully updated current Elo values.

## Training

From the `football_predictor` directory:

```bash
python main_train.py
```

From the repository root:

```bash
python football_predictor/main_train.py
```

Training will:

1. Load the dataset
2. Create target columns
3. Clean missing targets and features
4. Filter to the latest 10 years
5. Split the recent data chronologically
6. Train expected-goals models
7. Evaluate validation and test splits
8. Optionally train final saved models on all recent data

## Prediction

Train first so model files exist. Then run:

```bash
python main_predict.py
```

or from the repository root:

```bash
python football_predictor/main_predict.py
```

`main_predict.py` predicts an automatic Denmark vs Norway example. It sets the match date to one day after the latest match in the dataset, builds features from team names, then prints the generated feature row and prediction output.

Programmatic use:

```python
from src.predict import predict_upcoming_match

prediction = predict_upcoming_match(
    home_team="Denmark",
    away_team="Norway",
    match_date="2025-12-12",
    tournament="Friendly",
    is_neutral=1,
)
```

The returned dictionary includes:

- `feature_row`
- `home_expected_goals`
- `away_expected_goals`
- derived probability dictionaries
- top scorelines

## Output Interpretation

Expected goals are scoring-rate estimates, not exact score predictions.

The 1X2 probabilities are calculated by summing scoreline matrix cells:

- Home win: home goals greater than away goals
- Draw: home goals equal away goals
- Away win: home goals less than away goals

The most likely scorelines are the five highest-probability cells in the matrix.

## Evaluation

`src/evaluate.py` reports:

- MAE for home goals
- MAE for away goals
- MAE for total goals
- 1X2 log loss
- 1X2 Brier score
- Over 2.5 goals Brier score
- Simple 1X2 accuracy

Accuracy is secondary. Log loss and Brier score are more important because this is a probability model.

## Leakage Warnings

Do not include post-match or future information in features.

Never use these as inputs:

- `home_goals`
- `away_goals`
- `total_goals`
- `result`
- `winner_goals`

For upcoming predictions, the feature builder uses only matches before the prediction date. If the underlying dataset itself contains leaked feature engineering, the model can still be misleading.

## Limitations

- Independent Poisson score assumptions are simple and may miss score correlation.
- Lineups, injuries, red cards, travel, and weather are not modeled unless added as clean pre-match features.
- Scoreline tails above `MAX_GOALS` are truncated and normalized.
- Predictions are probabilities, not guarantees.
