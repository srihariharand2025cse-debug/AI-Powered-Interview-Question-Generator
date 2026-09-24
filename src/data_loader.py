"""
data_loader.py — Day 2: Load and validate the interview‑question dataset.

Supports loading data from:
  1. A local CSV file (default: data/sample_questions.csv)
  2. The Kaggle dataset (requires kaggle API credentials)

Expected CSV columns:
  Question | Ideal_Answer | Category | Role | Difficulty
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "sample_questions.csv"

EXPECTED_COLUMNS = {"Question", "Ideal_Answer", "Category", "Role", "Difficulty"}

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the interview‑question CSV into a DataFrame.

    Parameters
    ----------
    filepath : str, optional
        Path to a CSV file.  When *None*, falls back to
        ``data/sample_questions.csv`` shipped with the repo.

    Returns
    -------
    pd.DataFrame
        The loaded (and lightly validated) dataset.

    Raises
    ------
    FileNotFoundError
        If the resolved file does not exist.
    ValueError
        If required columns are missing from the CSV.
    """
    path = Path(filepath) if filepath else _DEFAULT_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "  • Place your CSV in the data/ folder, or\n"
            "  • Run `python -m src.download_kaggle` to fetch the Kaggle dataset."
        )

    df = pd.read_csv(path)
    _validate(df, path)
    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, source: Path) -> None:
    """Run basic sanity checks on the loaded DataFrame."""
    # 1. Check required columns exist
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {source} is missing required columns: {missing}\n"
            f"Expected columns: {sorted(EXPECTED_COLUMNS)}"
        )

    # 2. Warn about empty rows
    empty_rows = df[df["Question"].isna()].shape[0]
    if empty_rows:
        print(f"[WARN] {empty_rows} row(s) have empty 'Question' field.")

    # 3. Report unexpected difficulty values
    if "Difficulty" in df.columns:
        unique_diff = set(df["Difficulty"].dropna().unique())
        unexpected = unique_diff - VALID_DIFFICULTIES
        if unexpected:
            print(
                f"[WARN] Unexpected difficulty values found: {unexpected}. "
                f"Expected one of {VALID_DIFFICULTIES}."
            )


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> str:
    """Return a human‑readable summary of the dataset."""
    lines = [
        "[Dataset Summary]",
        f"   Total questions : {len(df):,}",
        f"   Columns         : {', '.join(df.columns)}",
    ]

    if "Difficulty" in df.columns:
        dist = df["Difficulty"].value_counts()
        lines.append(f"   Difficulty split :")
        for level in ["Easy", "Medium", "Hard"]:
            count = dist.get(level, 0)
            lines.append(f"      {level:8s} -> {count:,}")

    if "Category" in df.columns:
        lines.append(f"   Categories       : {df['Category'].nunique()} unique")
        for cat, cnt in df["Category"].value_counts().items():
            lines.append(f"      {cat:24s} -> {cnt:,}")

    if "Role" in df.columns:
        lines.append(f"   Roles            : {df['Role'].nunique()} unique")
        for role, cnt in df["Role"].value_counts().items():
            lines.append(f"      {role:24s} -> {cnt:,}")

    return "\n".join(lines)
