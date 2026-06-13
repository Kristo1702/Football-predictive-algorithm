import csv
import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILES = ("results.csv", "goalscorers.csv")

def clear_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    elif os.name == 'posix':  # Linux og macOS
        os.system('clear')
    else:  # Fallback
        print("\033c", end="")

def header(path):
    with open(path, newline="", encoding="utf-8") as file:
        return next(csv.reader(file), [])


def _load_csv(filename):
    path = DATA_DIR / filename

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data = {field: [] for field in reader.fieldnames or []}

        for row in reader:
            for field in data:
                data[field].append(row[field])

    return data

def load_results():
    return _load_csv("results.csv")

def load_goalscorers():
    return _load_csv("goalscorers.csv")