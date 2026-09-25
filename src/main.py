"""
main.py -- Entry point for the AI-Powered Interview Question Generator.

Day 1: Project skeleton           [done]
Day 2: Load the dataset           [done]
Day 3: Explore and clean the data [done]
"""

from __future__ import annotations

import sys
import io

# Fix Windows console encoding (cp1252 cannot handle emoji/unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from src.data_loader import load_dataset, summarize
from src.data_cleaner import (
    explore_dataset,
    format_exploration_report,
    clean_dataset,
    save_cleaned_dataset,
)


def main() -> None:
    """Entry point for the Interview Question Generator."""
    print("=" * 60)
    print(">> AI-Powered Interview Question Generator")
    print("=" * 60)
    print()

    # -- Day 2: Load dataset --
    print("[1] Loading raw dataset ...")
    raw_df = load_dataset()
    print(f"    Loaded {len(raw_df):,} raw question(s).\n")

    # -- Day 3: Explore the dataset (EDA) --
    print("[2] Running Exploratory Data Analysis (EDA) ...\n")
    stats = explore_dataset(raw_df)
    print(format_exploration_report(stats))
    print()

    # -- Day 3: Clean & Preprocess --
    print("[3] Cleaning and standardizing dataset ...")
    cleaned_df, audit = clean_dataset(raw_df, enrich_metrics=True)
    print(f"    • Initial records      : {audit['initial_rows']}")
    print(f"    • Empty Qs dropped     : {audit['empty_questions_dropped']}")
    print(f"    • Duplicates removed   : {audit['duplicates_dropped']}")
    print(f"    • Final clean records  : {audit['final_rows']}")
    print()

    # -- Save Cleaned Dataset --
    saved_path = save_cleaned_dataset(cleaned_df)
    print(f"[4] Cleaned dataset saved to: {saved_path}\n")

    # -- Show Sample Cleaned Questions with Word Count Metrics --
    print("-- Sample Cleaned Questions with Metrics " + "-" * 26)
    for i, row in cleaned_df.head(4).iterrows():
        print(
            f"\n  #{i+1} [{row['Difficulty']}] ({row['Category']} | {row['Role']})"
            f" [Words: Q={row['Question_Words']}, A={row['Answer_Words']}]\n"
            f"  Q: {row['Question']}\n"
            f"  A: {row['Ideal_Answer'][:120]}..."
        )
    print("\n" + "-" * 66)
    print(f"\n[OK] Day 3 Data Exploration & Cleaning complete!\n")


if __name__ == "__main__":
    main()
