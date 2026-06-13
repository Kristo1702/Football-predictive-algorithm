import functions as f



def get_team_data(team, results):
    home_teams = results.get("home_team")
    away_teams = results.get("away_team")
    home_score = results.get("home_score")
    away_score = results.get("away_score")
    dates = results.get("date")

    matches = []
    wins = 0
    losses = 0
    draws = 0
    positive_goals = 0
    negative_goals = 0

    for i in range(len(home_teams)):
        if team == home_teams[i] or team == away_teams[i]:

            match_date = dates[i]
            try:
                home_goals = int(home_score[i])
            except:
                home_goals = 0
                print(f"fail: {i}")
            
            try:
                away_goals = int(away_score[i])
            except:
                away_goals = 0
                print(f"fail: {i}")

            if team == home_teams[i]:
                positive_goals += home_goals
                negative_goals += away_goals
                opposite_team = away_teams[i]
                match_positive_goals = home_goals
                match_negative_goals = away_goals
                home = True

                if home_goals > away_goals:
                    wins += 1
                    outcome = "won"
                elif home_goals == away_goals:
                    draws += 1
                    outcome = "draw"
                elif home_goals < away_goals:
                    losses += 1
                    outcome = "loss"

            elif team == away_teams[i]:
                positive_goals += away_goals
                negative_goals += home_goals
                opposite_team = home_teams[i]
                match_positive_goals = away_goals
                match_negative_goals = home_goals
                home = False

                if home_goals < away_goals:
                    wins += 1
                    outcome = "win"
                elif home_goals == away_goals:
                    draws += 1
                    outcome = "draw"
                elif home_goals > away_goals:
                    losses += 1
                    outcome = "loss"

            this_match = {
                "outcome": outcome,
                "positive_goals": match_positive_goals,
                "negative_goals": match_negative_goals,
                "opposite_team": opposite_team,
                "home": home,
                "date": match_date
            }
            matches.append(this_match)

    return {
        "matches": matches,
        "wins": wins,
        "losses": losses,
        "draws": draws
    }





def input_teams(results):
    while True:
        f.clear_terminal()
        team_home = input("Enter home team: ").lower()
        if team_home not in results.get("home_team") or team_home not in results.get("away_team"):
            f.clear_terminal()
            print(f"Team '{team_home}' not found")

        f.clear_terminal()
        team_away = input("Enter home team: ").lower()
        if team_away not in results.get("home_team") or team_away not in results.get("away_team"):
            f.clear_terminal()
            print(f"Team '{team_away}' not found")
        
        f.clear_terminal()
        return team_home, team_away

def main():
    results = f.load_results()
    goal_scorers = f.load_goalscorers()

    team_home, team_away = input_teams(results)

    home_team_data = get_team_data(team_home, results)
    away_team_data = get_team_data(team_away, results)

    matches = home_team_data.get("matches")
    wins = home_team_data.get("wins")
    losses = home_team_data.get("losses")
    draws = home_team_data.get("draws")

    



if __name__ == "__main__":
    main()