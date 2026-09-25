import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_cleaner import (
    clean_text,
    normalize_difficulty,
    normalize_label,
    explore_dataset,
    clean_dataset,
    add_text_metrics,
)


def test_clean_text():
    # Test HTML entities & tags
    raw = "What is &quot;OOP&quot;? <p>Explain in detail.</p>"
    cleaned = clean_text(raw)
    assert cleaned == 'What is "OOP"? Explain in detail.'

    # Test whitespace and smart quotes
    raw_quotes = "  “Explain” the difference…  "
    assert clean_text(raw_quotes) == '"Explain" the difference...'

    # Test null handling
    assert clean_text(None, default="fallback") == "fallback"


def test_normalize_difficulty():
    assert normalize_difficulty("easy") == "Easy"
    assert normalize_difficulty("BEGINNER") == "Easy"
    assert normalize_difficulty("medium") == "Medium"
    assert normalize_difficulty("hard") == "Hard"
    assert normalize_difficulty("expert") == "Hard"
    assert normalize_difficulty("unknown_level") == "Medium"
    assert normalize_difficulty(None) == "Medium"


def test_normalize_label():
    assert normalize_label("software engineer") == "Software Engineer"
    assert normalize_label(None, default="General") == "General"


def test_clean_dataset_with_dirty_data():
    dirty_data = pd.DataFrame(
        [
            {
                "Question": "<b>What is Python?</b>",
                "Ideal_Answer": "A programming language.",
                "Category": "technical",
                "Role": "software engineer",
                "Difficulty": "easy",
            },
            {
                "Question": "What is Python?",  # Duplicate question
                "Ideal_Answer": "Duplicate entry.",
                "Category": "Technical",
                "Role": "Software Engineer",
                "Difficulty": "Easy",
            },
            {
                "Question": None,  # Empty question row
                "Ideal_Answer": "No question here.",
                "Category": "General",
                "Role": "General",
                "Difficulty": "Medium",
            },
        ]
    )

    cleaned_df, audit = clean_dataset(dirty_data, enrich_metrics=True)

    assert audit["initial_rows"] == 3
    assert audit["empty_questions_dropped"] == 1
    assert audit["duplicates_dropped"] == 1
    assert audit["final_rows"] == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["Question"] == "What is Python?"
    assert cleaned_df.iloc[0]["Difficulty"] == "Easy"
    assert cleaned_df.iloc[0]["Category"] == "Technical"
    assert cleaned_df.iloc[0]["Role"] == "Software Engineer"
    assert "Question_Words" in cleaned_df.columns
    assert cleaned_df.iloc[0]["Question_Words"] == 3


def test_explore_dataset():
    df = pd.DataFrame(
        [
            {
                "Question": "What is Python?",
                "Ideal_Answer": "A programming language.",
                "Category": "Technical",
                "Role": "Software Engineer",
                "Difficulty": "Easy",
            }
        ]
    )
    stats = explore_dataset(df)
    assert stats["total_rows"] == 1
    assert stats["duplicate_questions"] == 0
    assert stats["question_word_stats"]["mean"] == 3.0


if __name__ == "__main__":
    test_clean_text()
    test_normalize_difficulty()
    test_normalize_label()
    test_clean_dataset_with_dirty_data()
    test_explore_dataset()
    print("All cleaner tests passed successfully!")
