"""
main.py -- Entry point for the AI-Powered Interview Question Generator.

Day 1: Project skeleton  [done]
Day 2: Load the dataset  [done]
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


def main() -> None:
    """Entry point for the Interview Question Generator."""
    print(
        ">> Welcome to AI-Powered Interview Question Generator!\n"
        ">> Project skeleton is set up correctly.\n"
    )

    # -- Day 2: Load and preview the dataset --
    print("[*] Loading interview-question dataset ...\n")
    df = load_dataset()

    print(summarize(df))
    print()

    # Show a few sample questions
    print("-- Sample Questions " + "-" * 46)
    for i, row in df.head(5).iterrows():
        print(
            f"\n  [{row['Difficulty']}] ({row['Category']} | {row['Role']})\n"
            f"  Q: {row['Question']}"
        )
    print("\n" + "-" * 66)
    print(f"\n[OK] Dataset loaded successfully -- {len(df):,} questions ready.\n")


if __name__ == "__main__":
    main()
