"""
data_cleaner.py — Day 3: Explore and clean the interview-question dataset.

Provides text cleaning, normalization, validation, dataset health checks,
and descriptive exploration analytics for interview questions and answers.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CLEAN_CSV = _PROJECT_ROOT / "data" / "cleaned_questions.csv"

DIFFICULTY_MAPPING = {
    "easy": "Easy",
    "beginner": "Easy",
    "basic": "Easy",
    "simple": "Easy",
    "1": "Easy",
    "medium": "Medium",
    "intermediate": "Medium",
    "moderate": "Medium",
    "mid": "Medium",
    "2": "Medium",
    "hard": "Hard",
    "advanced": "Hard",
    "expert": "Hard",
    "complex": "Hard",
    "3": "Hard",
}

DEFAULT_DIFFICULTY = "Medium"
DEFAULT_CATEGORY = "General"
DEFAULT_ROLE = "General"
DEFAULT_ANSWER = "No answer provided."


# ---------------------------------------------------------------------------
# Individual Text Cleaning Helpers
# ---------------------------------------------------------------------------

def clean_text(text: Any, default: str = "") -> str:
    """Clean a single string of text.

    - Handles null/NaN/non-string types
    - Decodes HTML entities (e.g. &amp; -> &)
    - Replaces smart/curly quotes and dashes with standard ASCII
    - Removes HTML tags
    - Normalizes multiple spaces, tabs, and newlines into a single space
    - Strips leading/trailing whitespace
    """
    if pd.isna(text) or text is None:
        return default

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize smart quotes and dashes
    quote_replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " - ",
        "\u2013": " - ",
        "\u2026": "...",
        "\xa0": " ",
    }
    for orig, repl in quote_replacements.items():
        text = text.replace(orig, repl)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else default


def normalize_difficulty(val: Any) -> str:
    """Normalize difficulty value to one of: 'Easy', 'Medium', 'Hard'."""
    if pd.isna(val) or val is None:
        return DEFAULT_DIFFICULTY

    cleaned = str(val).strip().lower()
    return DIFFICULTY_MAPPING.get(cleaned, DEFAULT_DIFFICULTY)


def normalize_label(val: Any, default: str = "General") -> str:
    """Clean and title-case category and role labels."""
    cleaned = clean_text(val, default=default)
    if not cleaned or cleaned.lower() == "nan":
        return default
    return cleaned.title()


# ---------------------------------------------------------------------------
# Exploration & Dataset Health Check
# ---------------------------------------------------------------------------

def explore_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive exploratory data analysis (EDA) metrics.

    Returns
    -------
    dict
        Dictionary containing shapes, missing counts, duplicate stats,
        distributions, and text length statistics.
    """
    total_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    missing_pct = {k: round((v / total_rows) * 100, 2) if total_rows > 0 else 0 for k, v in missing_counts.items()}

    # Duplicate check on question column
    q_col = "Question" if "Question" in df.columns else df.columns[0]
    cleaned_q = df[q_col].dropna().astype(str).str.strip().str.lower()
    duplicate_count = int(cleaned_q.duplicated().sum())

    # Text length analytics
    question_char_lens = df[q_col].dropna().astype(str).str.len()
    question_word_counts = df[q_col].dropna().astype(str).apply(lambda x: len(x.split()))

    answer_col = "Ideal_Answer" if "Ideal_Answer" in df.columns else None
    if answer_col and answer_col in df.columns:
        answer_word_counts = df[answer_col].dropna().astype(str).apply(lambda x: len(x.split()))
    else:
        answer_word_counts = pd.Series([], dtype=int)

    def stats_summary(series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": int(series.min()),
            "max": int(series.max()),
            "mean": round(float(series.mean()), 1),
            "median": round(float(series.median()), 1),
        }

    # Categorical distributions
    category_dist = df["Category"].value_counts().to_dict() if "Category" in df.columns else {}
    role_dist = df["Role"].value_counts().to_dict() if "Role" in df.columns else {}
    difficulty_dist = df["Difficulty"].value_counts().to_dict() if "Difficulty" in df.columns else {}

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "duplicate_questions": duplicate_count,
        "question_word_stats": stats_summary(question_word_counts),
        "question_char_stats": stats_summary(question_char_lens),
        "answer_word_stats": stats_summary(answer_word_counts),
        "category_distribution": category_dist,
        "role_distribution": role_dist,
        "difficulty_distribution": difficulty_dist,
    }


def format_exploration_report(stats: Dict[str, Any]) -> str:
    """Format exploration metrics into a clean CLI readable report."""
    lines = [
        "=" * 60,
        " DATASET EXPLORATION & HEALTH REPORT (Day 3)",
        "=" * 60,
        f"  Total Records  : {stats['total_rows']:,}",
        f"  Total Columns  : {stats['total_columns']} ({', '.join(stats['columns'])})",
        f"  Duplicate Qs   : {stats['duplicate_questions']:,} duplicate question(s) found",
        "",
        "--- Missing Values ---",
    ]

    for col, count in stats["missing_counts"].items():
        pct = stats["missing_percentages"][col]
        status = "[OK]" if count == 0 else f"[WARN: {count} missing ({pct}%)]"
        lines.append(f"  {col:<16}: {status}")

    lines.append("")
    lines.append("--- Question Text Statistics ---")
    q_words = stats["question_word_stats"]
    q_chars = stats["question_char_stats"]
    lines.append(
        f"  Word Count     : Min {q_words['min']} | Max {q_words['max']} | Mean {q_words['mean']} | Median {q_words['median']}"
    )
    lines.append(
        f"  Char Length    : Min {q_chars['min']} | Max {q_chars['max']} | Mean {q_chars['mean']} | Median {q_chars['median']}"
    )

    if stats["answer_word_stats"]["max"] > 0:
        a_words = stats["answer_word_stats"]
        lines.append("")
        lines.append("--- Answer Text Statistics ---")
        lines.append(
            f"  Word Count     : Min {a_words['min']} | Max {a_words['max']} | Mean {a_words['mean']} | Median {a_words['median']}"
        )

    lines.append("")
    lines.append("--- Breakdown Distributions ---")
    if stats["difficulty_distribution"]:
        diff_str = ", ".join([f"{k}: {v}" for k, v in stats["difficulty_distribution"].items()])
        lines.append(f"  Difficulty     : {diff_str}")
    if stats["category_distribution"]:
        lines.append(f"  Categories     : {len(stats['category_distribution'])} unique categories")
    if stats["role_distribution"]:
        lines.append(f"  Roles          : {len(stats['role_distribution'])} unique roles")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Cleaning Pipeline
# ---------------------------------------------------------------------------

def add_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add word count and character length columns for questions and answers."""
    df = df.copy()
    if "Question" in df.columns:
        df["Question_Length"] = df["Question"].astype(str).str.len()
        df["Question_Words"] = df["Question"].astype(str).apply(lambda s: len(s.split()))
    if "Ideal_Answer" in df.columns:
        df["Answer_Length"] = df["Ideal_Answer"].astype(str).str.len()
        df["Answer_Words"] = df["Ideal_Answer"].astype(str).apply(lambda s: len(s.split()))
    return df


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_empty_questions: bool = True,
    enrich_metrics: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, standardize, and preprocess the dataset.

    Transformations applied:
      1. Strips HTML tags, fixes smart quotes, normalizes spaces on all text fields.
      2. Drops rows with missing or blank Questions.
      3. Imputes missing Ideal_Answers with a standard fallback.
      4. Standardizes Difficulty values to 'Easy', 'Medium', or 'Hard'.
      5. Standardizes Category and Role casing.
      6. Removes duplicate questions (case-insensitive deduplication).
      7. Optionally computes text length and word count metrics.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (cleaned_dataframe, audit_summary_dict)
    """
    initial_count = len(df)
    clean_df = df.copy()

    # 1. Clean Question column
    if "Question" in clean_df.columns:
        clean_df["Question"] = clean_df["Question"].apply(lambda q: clean_text(q, default=""))
        if drop_empty_questions:
            clean_df = clean_df[clean_df["Question"].str.len() > 0]
    empty_dropped = initial_count - len(clean_df)

    # 2. Clean Ideal_Answer column
    if "Ideal_Answer" in clean_df.columns:
        clean_df["Ideal_Answer"] = clean_df["Ideal_Answer"].apply(
            lambda a: clean_text(a, default=DEFAULT_ANSWER)
        )
    else:
        clean_df["Ideal_Answer"] = DEFAULT_ANSWER

    # 3. Standardize Difficulty
    if "Difficulty" in clean_df.columns:
        clean_df["Difficulty"] = clean_df["Difficulty"].apply(normalize_difficulty)
    else:
        clean_df["Difficulty"] = DEFAULT_DIFFICULTY

    # 4. Standardize Category & Role
    if "Category" in clean_df.columns:
        clean_df["Category"] = clean_df["Category"].apply(
            lambda c: normalize_label(c, default=DEFAULT_CATEGORY)
        )
    else:
        clean_df["Category"] = DEFAULT_CATEGORY

    if "Role" in clean_df.columns:
        clean_df["Role"] = clean_df["Role"].apply(
            lambda r: normalize_label(r, default=DEFAULT_ROLE)
        )
    else:
        clean_df["Role"] = DEFAULT_ROLE

    # 5. Deduplication
    pre_dedup_count = len(clean_df)
    if drop_duplicates and "Question" in clean_df.columns:
        # Create a normalized lowercased key for deduplication
        clean_df["_dedup_key"] = clean_df["Question"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
        clean_df = clean_df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    duplicates_dropped = pre_dedup_count - len(clean_df)

    # 6. Optional text metrics
    if enrich_metrics:
        clean_df = add_text_metrics(clean_df)

    # Reset index
    clean_df = clean_df.reset_index(drop=True)

    summary = {
        "initial_rows": initial_count,
        "final_rows": len(clean_df),
        "empty_questions_dropped": empty_dropped,
        "duplicates_dropped": duplicates_dropped,
        "total_dropped": initial_count - len(clean_df),
    }

    return clean_df, summary


def save_cleaned_dataset(df: pd.DataFrame, output_path: Optional[str | Path] = None) -> Path:
    """Save cleaned DataFrame to CSV file."""
    path = Path(output_path) if output_path else _DEFAULT_CLEAN_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
