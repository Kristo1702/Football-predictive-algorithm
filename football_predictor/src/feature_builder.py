import unicodedata

import pandas as pd

from . import config
from .utils import ensure_columns_exist


def _normalize_team_name(name):
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().strip().split())


def _team_mask(df, team_name):
    normalized_name = _normalize_team_name(team_name)
    home_names = df["_home_team"].map(_normalize_team_name)
    away_names = df["_away_team"].map(_normalize_team_name)
    return (home_names == normalized_name) | (away_names == normalized_name)


def _team_side(row, team_name):
    normalized_name = _normalize_team_name(team_name)
    if _normalize_team_name(row["_home_team"]) == normalized_name:
        return "home"
    if _normalize_team_name(row["_away_team"]) == normalized_name:
        return "away"
    raise ValueError(f"Team '{team_name}' does not appear in this match row.")


def get_team_matches_before(df, team_name, cutoff_date):
    """Return all matches for a team before a cutoff date."""
    required_columns = ["_home_team", "_away_team", "_date"]
    ensure_columns_exist(df, required_columns, context="match data")

    cutoff_date = pd.Timestamp(cutoff_date)
    working_df = df.copy()
    working_df["_date"] = pd.to_datetime(working_df["_date"], errors="coerce")

    matches = working_df[(working_df["_date"] < cutoff_date) & _team_mask(working_df, team_name)]
    matches = matches.sort_values("_date").reset_index(drop=True)

    if matches.empty:
        raise ValueError(
            f"No matches found for '{team_name}' before {cutoff_date.date()}."
        )

    return matches


def get_latest_team_elo(df, team_name, cutoff_date):
    """Return the latest known pre-match Elo for a team before cutoff_date."""
    required_columns = ["home_elo", "away_elo"]
    ensure_columns_exist(df, required_columns, context="match data")

    matches = get_team_matches_before(df, team_name, cutoff_date)
    latest_match = matches.iloc[-1]
    side = _team_side(latest_match, team_name)
    elo_column = f"{side}_elo"
    elo = pd.to_numeric(latest_match[elo_column], errors="coerce")

    if pd.isna(elo):
        raise ValueError(
            f"No usable Elo found for '{team_name}' before "
            f"{pd.Timestamp(cutoff_date).date()}."
        )

    return float(elo)


def compute_team_form(
    df,
    team_name,
    cutoff_date,
    window=config.FORM_MATCH_WINDOW,
):
    """Compute goals scored, goals conceded, and win rate before cutoff_date."""
    if window <= 0:
        raise ValueError("window must be positive.")

    required_columns = ["home_goals", "away_goals"]
    ensure_columns_exist(df, required_columns, context="match data")

    matches = get_team_matches_before(df, team_name, cutoff_date)
    matches = matches.tail(window)

    if len(matches) < window:
        print(
            f"Warning: '{team_name}' has only {len(matches)} matches before "
            f"{pd.Timestamp(cutoff_date).date()}; using available matches."
        )

    scored = []
    conceded = []
    wins = []

    for _, row in matches.iterrows():
        side = _team_side(row, team_name)
        home_goals = pd.to_numeric(row["home_goals"], errors="coerce")
        away_goals = pd.to_numeric(row["away_goals"], errors="coerce")
        if pd.isna(home_goals) or pd.isna(away_goals):
            raise ValueError(
                f"Missing historical goals for '{team_name}' before "
                f"{pd.Timestamp(cutoff_date).date()}."
            )
        home_goals = int(home_goals)
        away_goals = int(away_goals)

        if side == "home":
            team_goals = home_goals
            opponent_goals = away_goals
        else:
            team_goals = away_goals
            opponent_goals = home_goals

        scored.append(team_goals)
        conceded.append(opponent_goals)
        wins.append(1 if team_goals > opponent_goals else 0)

    return {
        "scored": float(sum(scored) / len(scored)),
        "conceded": float(sum(conceded) / len(conceded)),
        "win_rate": float(sum(wins) / len(wins)),
    }


def infer_tournament_flags(tournament_name, is_neutral):
    """Infer simple tournament flags from a tournament name."""
    tournament_text = _normalize_team_name(tournament_name)
    is_world_cup = 1 if "world cup" in tournament_text else 0

    continental_keywords = [
        "uefa euro",
        "european championship",
        "copa america",
        "africa cup of nations",
        "african cup of nations",
        "asian cup",
        "gold cup",
        "nations league",
        "uefa nations league",
        "concacaf nations league",
        "afc asian cup",
    ]
    is_continental = int(
        any(keyword in tournament_text for keyword in continental_keywords)
    )

    return {
        "is_neutral": int(is_neutral),
        "is_world_cup": is_world_cup,
        "is_continental": is_continental,
    }


def create_upcoming_match_features(
    df,
    home_team,
    away_team,
    match_date,
    tournament="Friendly",
    is_neutral=1,
):
    """Create model-ready features for a future match from team names."""
    match_date = pd.Timestamp(match_date)

    home_elo = get_latest_team_elo(df, home_team, match_date)
    away_elo = get_latest_team_elo(df, away_team, match_date)
    home_form = compute_team_form(df, home_team, match_date)
    away_form = compute_team_form(df, away_team, match_date)
    flags = infer_tournament_flags(tournament, is_neutral)

    feature_row = {
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "elo_diff": float(home_elo - away_elo),
        "home_form_scored": float(home_form["scored"]),
        "home_form_conceded": float(home_form["conceded"]),
        "home_form_win_rate": float(home_form["win_rate"]),
        "away_form_scored": float(away_form["scored"]),
        "away_form_conceded": float(away_form["conceded"]),
        "away_form_win_rate": float(away_form["win_rate"]),
        "is_neutral": int(flags["is_neutral"]),
        "is_world_cup": int(flags["is_world_cup"]),
        "is_continental": int(flags["is_continental"]),
    }

    return {column: feature_row[column] for column in config.FEATURE_COLUMNS}
