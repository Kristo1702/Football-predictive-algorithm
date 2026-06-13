import csv
import os
from pathlib import Path

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