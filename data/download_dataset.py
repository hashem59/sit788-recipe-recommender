"""Download Food.com recipes, sample 5,000 rows, write recipes_sample.csv.

Run: python data/download_dataset.py
Prereq: Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)
"""
from __future__ import annotations
import ast
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT_PATH = ROOT / "recipes_sample.csv"
SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def download() -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    csv_path = RAW_DIR / "RAW_recipes.csv"
    if csv_path.exists():
        return csv_path
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "shuyangli94/food-com-recipes-and-user-interactions",
        "-f", "RAW_recipes.csv",
        "-p", str(RAW_DIR), "--unzip",
    ], check=True)
    return csv_path


def parse_list_str(s: str) -> list[str]:
    """Food.com stores list columns as Python-repr strings."""
    try:
        return [str(x).strip() for x in ast.literal_eval(s)]
    except (ValueError, SyntaxError):
        return []


def main() -> None:
    csv_path = download()
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["name", "ingredients", "steps"])
    df["ingredients_list"] = df["ingredients"].apply(parse_list_str)
    df["steps_list"] = df["steps"].apply(parse_list_str)
    df["n_ingredients"] = df["ingredients_list"].str.len()
    df["n_steps"] = df["steps_list"].str.len()
    df = df[(df["n_ingredients"].between(4, 15)) & (df["n_steps"].between(3, 25))]
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    keep = ["id", "name", "minutes", "ingredients_list", "steps_list",
            "n_ingredients", "n_steps", "description"]
    df = df[keep].reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
