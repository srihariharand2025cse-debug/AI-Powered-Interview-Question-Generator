"""
test_filter.py — Day 4: Unit tests for question filtering, skill extraction,
and interview set generation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.question_filter import (
    extract_skills,
    tag_skills,
    filter_questions,
    generate_interview_set,
    format_interview_set,
    QuestionFilter,
)


def _get_mock_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    return pd.DataFrame(
        [
            {
                "Question": "Explain the concept of Object-Oriented Programming.",
                "Ideal_Answer": "OOP is based on objects, encapsulation, inheritance, polymorphism in Python.",
                "Category": "Technical",
                "Role": "Software Engineer",
                "Difficulty": "Easy",
            },
            {
                "Question": "What is the difference between SQL and NoSQL databases?",
                "Ideal_Answer": "SQL databases like PostgreSQL are relational. NoSQL like MongoDB are flexible.",
                "Category": "Technical",
                "Role": "Software Engineer",
                "Difficulty": "Medium",
            },
            {
                "Question": "How would you design a URL shortener?",
                "Ideal_Answer": "Design with hashing, Redis caching, and distributed system principles.",
                "Category": "System Design",
                "Role": "Software Engineer",
                "Difficulty": "Hard",
            },
            {
                "Question": "Describe your experience with Agile methodology.",
                "Ideal_Answer": "I have worked with Scrum, sprints, standups, and retrospectives.",
                "Category": "Process",
                "Role": "Software Engineer",
                "Difficulty": "Easy",
            },
            {
                "Question": "How do you resolve conflicts in a team?",
                "Ideal_Answer": "Address conflict early through open communication and leadership.",
                "Category": "Conflict Resolution",
                "Role": "HR",
                "Difficulty": "Medium",
            },
            {
                "Question": "Tell me about yourself.",
                "Ideal_Answer": "I am a dedicated professional passionate about engineering.",
                "Category": "Introduction",
                "Role": "General",
                "Difficulty": "Easy",
            },
        ]
    )


def test_extract_skills():
    text1 = "How do you implement polymorphism in Python with OOP?"
    skills1 = extract_skills(text1)
    assert "Python" in skills1
    assert "OOP" in skills1

    text2 = "Explain how you handle team conflict and communicate with stakeholders."
    skills2 = extract_skills(text2)
    assert "Soft Skills & HR" in skills2

    # Empty / null input handling
    assert extract_skills("") == []
    assert extract_skills("   ") == []


def test_tag_skills():
    df = _get_mock_dataset()
    tagged_df = tag_skills(df)
    assert "Skills" in tagged_df.columns
    assert "OOP" in tagged_df.iloc[0]["Skills"]
    assert "SQL & Databases" in tagged_df.iloc[1]["Skills"]


def test_filter_by_role():
    df = _get_mock_dataset()

    # Including General questions
    res = filter_questions(df, role="Software Engineer", include_general_role=True)
    roles = res["Role"].unique().tolist()
    assert "Software Engineer" in roles
    assert "HR" not in roles

    # Excluding General questions
    res_strict = filter_questions(df, role="Software Engineer", include_general_role=False)
    assert all(r == "Software Engineer" for r in res_strict["Role"])


def test_filter_by_difficulty():
    df = _get_mock_dataset()
    res_easy = filter_questions(df, difficulty="Easy")
    assert all(d == "Easy" for d in res_easy["Difficulty"])
    assert len(res_easy) == 3

    res_multi = filter_questions(df, difficulty=["Medium", "Hard"])
    assert set(res_multi["Difficulty"].unique()).issubset({"Medium", "Hard"})


def test_filter_by_category():
    df = _get_mock_dataset()
    res_tech = filter_questions(df, category="Technical")
    assert all(c == "Technical" for c in res_tech["Category"])
    assert len(res_tech) == 2


def test_filter_by_skills():
    df = _get_mock_dataset()
    res_sql = filter_questions(df, skills=["SQL", "Databases"])
    assert len(res_sql) >= 1
    assert any("SQL" in q for q in res_sql["Question"])

    # Test match_all_skills (AND logic)
    res_all = filter_questions(df, skills=["Python", "OOP"], match_all_skills=True)
    assert len(res_all) == 1
    assert "Object-Oriented" in res_all.iloc[0]["Question"]


def test_filter_by_keywords_and_exclusions():
    df = _get_mock_dataset()
    res_kw = filter_questions(df, keywords="Redis")
    assert len(res_kw) == 1
    assert "URL shortener" in res_kw.iloc[0]["Question"]

    res_ex = filter_questions(df, keywords="engineering", exclude_keywords="Object-Oriented")
    assert "Explain the concept of Object-Oriented Programming." not in res_ex["Question"].values


def test_generate_interview_set():
    df = _get_mock_dataset()
    mock_set = generate_interview_set(df, role="Software Engineer", num_questions=3, random_state=42)
    assert len(mock_set) == 3

    # Test balanced difficulty
    balanced_set = generate_interview_set(
        df,
        role="Software Engineer",
        num_questions=3,
        balance_difficulty=True,
        random_state=42,
    )
    assert len(balanced_set) <= len(df)
    diffs = balanced_set["Difficulty"].unique().tolist()
    assert len(diffs) >= 2


def test_question_filter_class():
    df = _get_mock_dataset()
    qf = QuestionFilter(df)

    assert "Software Engineer" in qf.get_roles()
    assert "Technical" in qf.get_categories()
    assert "Python" in qf.get_skills_catalog()

    results = qf.filter(skills="Agile")
    assert len(results) == 1
    assert "Agile" in results.iloc[0]["Question"]

    formatted = format_interview_set(results)
    assert "Generated Interview Question Set" in formatted
    assert "Agile" in formatted


if __name__ == "__main__":
    test_extract_skills()
    test_tag_skills()
    test_filter_by_role()
    test_filter_by_difficulty()
    test_filter_by_category()
    test_filter_by_skills()
    test_filter_by_keywords_and_exclusions()
    test_generate_interview_set()
    test_question_filter_class()
    print("All Day 4 question filter tests passed successfully!")
