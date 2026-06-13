# Football Match Predictor

This project predicts football matches by estimating expected goals first, then deriving every market-style probability from a Poisson scoreline matrix.

It predicts:

- Home win / draw / away win probabilities
- Total goals probability distribution
- Over/under goal probabilities
- How many goals the winner scores
- Most likely exact scorelines

The model is not a direct black-box winner classifier. It follows this pipeline:

```text
historical match data
-> train home and away expected-goals models
-> predict home_expected_goals and away_expected_goals
-> build a Poisson scoreline probability matrix
-> sum scoreline probabilities into match probabilities
```

## Project Structure

```text
football_predictor/
├── data/
│   └── teams_match_features.csv
├── models/
│   ├── home_goals_model.pkl
│   ├── away_goals_model.pkl
│   └── feature_columns.pkl
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── poisson.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── main_train.py
├── main_predict.py
└── README.md
```

The `.pkl` model files are created when training runs.

## Data

The CSV should contain pre-match features plus final score targets. The default file path is configured in `src/config.py`:

```python
DATASET_PATH = DATA_DIR / "teams_match_features.csv"
```

`home_goals` and `away_goals` are targets. They must never be used as input features because they are only known after the match.

`_home_team`, `_away_team`, `_date`, and `_tournament` are metadata. `_date` is used for sorting and time-based splitting, but not as a model input.

## Features

Version 1 uses only:

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

Extra feature candidates are listed in `src/config.py` for later model versions, but they are not enabled by default.

## Algorithm

Two separate `PoissonRegressor` models are trained:

- One model predicts `home_goals`
- One model predicts `away_goals`

Those predictions are expected goals:

```text
home_expected_goals = home_lambda
away_expected_goals = away_lambda
```

The project then builds a scoreline matrix:

```text
matrix[i][j] = P(home scores i goals and away scores j goals)
```

For example:

- `matrix[2][1]` is the probability of a 2-1 home win
- `matrix[0][0]` is the probability of a 0-0 draw
- `matrix[1][3]` is the probability of a 1-3 away win

The matrix is normalized so its probabilities sum to 1. All downstream outputs come from this matrix.

## Time Splitting

The dataset is sorted by `_date`. Splits are chronological:

- Training: old matches
- Validation: middle period
- Test: newest matches

The defaults are in `src/config.py`:

```python
TRAIN_END_DATE = "2021-12-31"
VALIDATION_END_DATE = "2023-12-31"
```

This avoids random train/test splitting, which can leak future football information into the model.

## Time Weighting

Training uses exponential time decay:

```text
weight = 0.5 ** (days_old / half_life_days)
```

Newer matches get higher weight. Older matches still contribute, but less strongly.

## Training

From the `football_predictor` directory:

```bash
python main_train.py
```

Or from the repository root:

```bash
python football_predictor/main_train.py
```

Training will:

1. Load `data/teams_match_features.csv`
2. Parse and sort `_date`
3. Create targets
4. Drop rows with missing selected features or missing targets
5. Split by time
6. Train home and away expected-goals models
7. Evaluate validation and test splits
8. Save models into `models/`

## Prediction

Train first so the model files exist. Then run:

```bash
python main_predict.py
```

Or from the repository root:

```bash
python football_predictor/main_predict.py
```

`main_predict.py` contains an example feature row:

```python
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
```

For real predictions, replace those values with information known before kickoff.

## Output Interpretation

Expected goals are model estimates of scoring rates, not exact score predictions.

Example:

```text
Expected goals
  Home: 1.72
  Away: 1.05
```

This means the model estimates the home team scoring rate at 1.72 goals and the away team scoring rate at 1.05 goals.

The 1X2 probabilities are calculated by summing exact scorelines:

- Home win: sum of all scorelines where home goals > away goals
- Draw: sum of all scorelines where home goals = away goals
- Away win: sum of all scorelines where home goals < away goals

The most likely scorelines are the five individual cells in the scoreline matrix with the highest probabilities.

## Evaluation

`src/evaluate.py` reports:

- MAE for home goals
- MAE for away goals
- MAE for total goals
- 1X2 log loss
- 1X2 Brier score
- Over 2.5 goals Brier score
- Simple 1X2 accuracy

Accuracy is secondary. Log loss and Brier score matter more because this is a probability model.

## Data Leakage Warnings

Do not add features that are only known after the match starts or after it ends.

Do not include:

- `home_goals`
- `away_goals`
- `total_goals`
- `result`
- `winner_goals`
- Any future form, future rating, or post-match statistic

Even strong validation numbers are not trustworthy if the feature table leaks future information.

## Limitations

This is a clean baseline, not a finished betting or production system.

Important limitations:

- Independent Poisson assumptions can understate correlation between team scores.
- Red cards, injuries, lineups, weather, travel, and tactical context are not modeled unless supplied as clean pre-match features.
- Scoreline tails above `MAX_GOALS` are truncated and normalized.
- The model quality depends heavily on whether ELO and form features were generated without future leakage.
- Predictions are probabilities, not guarantees.

Use the outputs as calibrated uncertainty estimates, not certainties.
