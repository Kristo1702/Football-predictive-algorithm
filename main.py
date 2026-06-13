from difflib import get_close_matches
from pathlib import Path
import sys
import os

import pandas as pd

from football_predictor.src import config
from football_predictor.src.load_data import load_dataset
from football_predictor.src.predict import predict_upcoming_match


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

def clear_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    elif os.name == 'posix':  # Linux og macOS
        os.system('clear')
    else:  # Fallback
        print("\033c", end="")

def color(text, style):
    return f"{style}{text}{RESET}"


def line(width=78, char="-"):
    return char * width


def title(text):
    print(color(line(char="="), CYAN))
    print(color(text.center(78), BOLD + CYAN))
    print(color(line(char="="), CYAN))


def section(text):
    print()
    print(color(text.upper(), BOLD))
    print(color(line(len(text), "-"), DIM))


def pause(message="Press Enter to continue..."):
    input(color(f"\n{message}", DIM))


def probability_bar(probability, width=28):
    probability = max(0.0, min(1.0, float(probability)))
    filled = int(round(probability * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def percent(probability):
    return f"{100.0 * float(probability):5.1f}%"


def over_under_label(key):
    direction, whole_goals, _ = key.split("_")
    return f"{direction.title()} {whole_goals}.5"


def total_goals_label(goals):
    if goals == 1:
        return "1 goal"
    return f"{goals} goals"


def print_metric(label, probability):
    print(f"  {label:<24} {probability_bar(probability)} {percent(probability)}")


def print_key_value(label, value):
    print(f"  {label:<24} {value}")


def check_project_files():
    missing_files = []
    for path in [
        config.DATASET_PATH,
        config.HOME_MODEL_PATH,
        config.AWAY_MODEL_PATH,
        config.FEATURE_COLUMNS_PATH,
    ]:
        if not Path(path).exists():
            missing_files.append(str(path))

    if missing_files:
        clear_terminal()
        title("FOOTBALL MATCH PREDICTOR")
        print(color("Missing required project files:", RED))
        for path in missing_files:
            print(f"  - {path}")
        print("\nTrain the models first with:")
        print(color("  python football_predictor/main_train.py", GREEN))
        return False

    return True


def load_team_data():
    df = load_dataset(config.DATASET_PATH)
    required = [
        "_home_team",
        "_away_team",
        "_date",
        "home_elo",
        "away_elo",
        "home_form_scored",
        "home_form_conceded",
        "home_form_win_rate",
        "away_form_scored",
        "away_form_conceded",
        "away_form_win_rate",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    return df


def get_all_teams(df):
    teams = pd.concat([df["_home_team"], df["_away_team"]]).dropna().astype(str)
    return sorted(teams.unique())


def normalize_name(name):
    return " ".join(str(name).strip().lower().split())


def resolve_team_name(raw_name, teams):
    raw_name = raw_name.strip()
    normalized_input = normalize_name(raw_name)
    lookup = {normalize_name(team): team for team in teams}

    if normalized_input in lookup:
        return lookup[normalized_input]

    partial_matches = [
        team for team in teams if normalized_input and normalized_input in normalize_name(team)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    if partial_matches:
        return choose_from_matches(raw_name, partial_matches[:12])

    close_keys = get_close_matches(normalized_input, lookup.keys(), n=8, cutoff=0.55)
    close_matches = [lookup[key] for key in close_keys]
    if close_matches:
        return choose_from_matches(raw_name, close_matches)

    return None


def choose_from_matches(raw_name, matches):
    print(color(f"\nI found multiple possible teams for '{raw_name}':", YELLOW))
    for index, team in enumerate(matches, start=1):
        print(f"  {index}. {team}")
    print("  0. Type again")

    while True:
        choice = input("\nChoose team number: ").strip()
        if choice == "0":
            return None
        if choice.isdigit():
            number = int(choice)
            if 1 <= number <= len(matches):
                return matches[number - 1]
        print(color("Please choose a valid number.", RED))


def ask_team(label, teams):
    while True:
        raw_name = input(f"{label}: ").strip()
        if not raw_name:
            print(color("Please enter a team name.", RED))
            continue

        resolved = resolve_team_name(raw_name, teams)
        if resolved:
            return resolved

        print(color("No team found. Try a full name or another spelling.", RED))


def ask_yes_no(prompt, default=False):
    default_text = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "j", "ja"}:
            return True
        if answer in {"n", "no", "nej"}:
            return False
        print(color("Please answer yes or no.", RED))


def latest_team_profile(df, team):
    matches = df[(df["_home_team"] == team) | (df["_away_team"] == team)].copy()
    if matches.empty:
        raise ValueError(f"No rows found for team: {team}")

    matches = matches.sort_values("_date")
    row = matches.iloc[-1]
    side = "home" if row["_home_team"] == team else "away"
    opponent = row["_away_team"] if side == "home" else row["_home_team"]

    return {
        "team": team,
        "side_in_source": side,
        "source_date": row["_date"],
        "source_tournament": row.get("_tournament", "Unknown"),
        "source_opponent": opponent,
        "elo": float(row[f"{side}_elo"]),
        "form_scored": float(row[f"{side}_form_scored"]),
        "form_conceded": float(row[f"{side}_form_conceded"]),
        "form_win_rate": float(row[f"{side}_form_win_rate"]),
    }


def build_match_features(home_profile, away_profile, is_neutral, is_world_cup, is_continental):
    return {
        "home_elo": home_profile["elo"],
        "away_elo": away_profile["elo"],
        "elo_diff": home_profile["elo"] - away_profile["elo"],
        "home_form_scored": home_profile["form_scored"],
        "home_form_conceded": home_profile["form_conceded"],
        "home_form_win_rate": home_profile["form_win_rate"],
        "away_form_scored": away_profile["form_scored"],
        "away_form_conceded": away_profile["form_conceded"],
        "away_form_win_rate": away_profile["form_win_rate"],
        "is_neutral": int(is_neutral),
        "is_world_cup": int(is_world_cup),
        "is_continental": int(is_continental),
    }


def print_team_source(profile, label):
    source_date = profile["source_date"].strftime("%Y-%m-%d")
    print_key_value(f"{label} team", profile["team"])
    print_key_value("Latest data date", source_date)
    print_key_value("Latest tournament", profile["source_tournament"])
    print_key_value("Latest opponent", profile["source_opponent"])
    print_key_value("ELO", f"{profile['elo']:.1f}")
    print_key_value(
        "Form",
        (
            f"scored {profile['form_scored']:.2f}, "
            f"conceded {profile['form_conceded']:.2f}, "
            f"win rate {profile['form_win_rate']:.2f}"
        ),
    )


def print_prediction_dashboard(prediction, home_team, away_team, home_profile, away_profile, feature_row):
    probabilities = prediction["probabilities"]

    clear_terminal()
    title("FOOTBALL MATCH PREDICTOR")

    print(color(f"{home_team} vs {away_team}".center(78), BOLD))
    
    section("Expected goals")
    print_key_value(home_team, f"{prediction['home_expected_goals']:.2f} xG")
    print_key_value(away_team, f"{prediction['away_expected_goals']:.2f} xG")

    section("1X2 probabilities")
    print_metric(f"{home_team} win", probabilities["home_win_prob"])
    print_metric("Draw", probabilities["draw_prob"])
    print_metric(f"{away_team} win", probabilities["away_win_prob"])

    section("Over / under")
    over_under = probabilities["over_under"]
    for key in ["over_0_5", "over_1_5", "over_2_5", "over_3_5"]:
        print_metric(over_under_label(key), over_under[key])
    print()
    for key in ["under_0_5", "under_1_5", "under_2_5", "under_3_5"]:
        print_metric(over_under_label(key), over_under[key])

    section("Total goals distribution")
    for goals, probability in probabilities["total_goals_distribution"].items():
        print_metric(total_goals_label(goals), probability)

    section("Winner goals distribution")
    winner_goals = probabilities["winner_goals_distribution"]
    print_metric("No winner / draw", winner_goals["no_winner"])
    print_metric("Winner scores 1", winner_goals["winner_1_goal"])
    print_metric("Winner scores 2", winner_goals["winner_2_goals"])
    print_metric("Winner scores 3", winner_goals["winner_3_goals"])
    print_metric("Winner scores 4+", winner_goals["winner_4_plus_goals"])

    section("Most likely exact scorelines")
    for index, scoreline in enumerate(prediction["scorelines"], start=1):
        label = f"{index}. {scoreline['scoreline']}"
        print_metric(label, scoreline["probability"])

    print("\n")
    print_key_value("Neutral venue", "yes" if feature_row["is_neutral"] else "no")
    print_key_value("World Cup", "yes" if feature_row["is_world_cup"] else "no")
    print_key_value("Continental", "yes" if feature_row["is_continental"] else "no")
    print_key_value(
        "Symmetric neutral",
        "yes" if prediction.get("symmetric_neutral_used") else "no",
    )

    print(color("\nPredictions are probabilities, not guarantees.", YELLOW))


def show_team_examples(teams):
    clear_terminal()
    title("AVAILABLE TEAMS")
    print("Examples from the dataset:\n")
    for team in teams[:80]:
        print(f"  - {team}")
    if len(teams) > 80:
        print(color(f"\nShowing 80 of {len(teams)} teams. Search works with partial names.", DIM))
    pause()


def predict_flow(df, teams):
    clear_terminal()
    title("NEW MATCH PREDICTION")
    print("Enter the home team first and the away team second.")
    print(color("Tip: partial names work, for example 'denmark' or 'brazil'.\n", DIM))

    home_team = ask_team("Home team", teams)
    away_team = ask_team("Away team", teams)

    while away_team == home_team:
        print(color("Home and away team cannot be the same.", RED))
        away_team = ask_team("Away team", teams)

    print()
    is_neutral = ask_yes_no("Neutral venue?", default=False)
    is_world_cup = ask_yes_no("World Cup match?", default=False)
    is_continental = ask_yes_no("Continental tournament match?", default=False)

    home_profile = latest_team_profile(df, home_team)
    away_profile = latest_team_profile(df, away_team)
    tournament = "Friendly"
    if is_world_cup:
        tournament = "World Cup"
    elif is_continental:
        tournament = "UEFA Euro"

    match_date = df["_date"].max() + pd.Timedelta(days=1)
    prediction = predict_upcoming_match(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        tournament=tournament,
        is_neutral=int(is_neutral),
    )
    print_prediction_dashboard(
        prediction,
        home_team,
        away_team,
        home_profile,
        away_profile,
        prediction["feature_row"],
    )
    pause()


def main_menu(df, teams):
    while True:
        clear_terminal()
        title("FOOTBALL MATCH PREDICTOR")
        print("CHOOSE AN OPTION:\n")
        print("  1. Predict a match")
        print("  2. Show team examples")
        print("  3. Exit")
        print()

        choice = input("Choose an option: ").strip()
        if choice == "1":
            predict_flow(df, teams)
        elif choice == "2":
            show_team_examples(teams)
        elif choice == "3":
            clear_terminal()
            print("Goodbye.")
            return
        else:
            print(color("Please choose 1, 2, or 3.", RED))
            pause()


def main():
    try:
        if not check_project_files():
            return
        df = load_team_data()
        teams = get_all_teams(df)
        main_menu(df, teams)
    except KeyboardInterrupt:
        clear_terminal()
        print("Goodbye.")
        sys.exit(0)
    except Exception as error:
        clear_terminal()
        title("FOOTBALL MATCH PREDICTOR")
        print(color("An error occurred:", RED))
        print(f"  {error}")
        print("\nIf the models are missing or outdated, run:")
        print(color("  python football_predictor/main_train.py", GREEN))
        sys.exit(1)


if __name__ == "__main__":
    main()
