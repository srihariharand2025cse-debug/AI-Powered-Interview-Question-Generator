"""
test_semantic_generator.py — Day 6: Unit tests for Semantic Match Ranking
and Dynamic Question Generation.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.semantic_generator import (
    JobProfile,
    parse_job_profile,
    generate_evaluation_rubric,
    HybridMatchRanker,
    DynamicQuestionGenerator,
    format_interview_guide,
    export_markdown_interview_guide,
    SENIORITY_LEVELS,
)


def _get_mock_df() -> pd.DataFrame:
    """Create a diverse mock dataset for testing."""
    return pd.DataFrame([
        {
            "Question": "Tell me about yourself.",
            "Ideal_Answer": "Professional background, career highlights, and passion for the industry.",
            "Category": "Introduction",
            "Role": "General",
            "Difficulty": "Easy",
            "Skills": "General",
        },
        {
            "Question": "Explain the concept of Object-Oriented Programming.",
            "Ideal_Answer": "OOP is based on objects, classes, encapsulation, inheritance, and polymorphism.",
            "Category": "Technical",
            "Role": "Software Engineer",
            "Difficulty": "Easy",
            "Skills": "Python, OOP",
        },
        {
            "Question": "What is the difference between SQL and NoSQL databases?",
            "Ideal_Answer": "SQL is relational with ACID transactions, while NoSQL is schema-flexible.",
            "Category": "Technical",
            "Role": "Software Engineer",
            "Difficulty": "Medium",
            "Skills": "SQL & Databases, System Design",
        },
        {
            "Question": "How would you design a URL shortener?",
            "Ideal_Answer": "Use hash functions, caching with Redis, load balancer, and database.",
            "Category": "System Design",
            "Role": "Software Engineer",
            "Difficulty": "Hard",
            "Skills": "System Design, Web & Networking",
        },
        {
            "Question": "What is the difference between a process and a thread?",
            "Ideal_Answer": "A process has isolated memory; threads share memory and require synchronization.",
            "Category": "Technical",
            "Role": "Software Engineer",
            "Difficulty": "Medium",
            "Skills": "Concurrency & OS",
        },
        {
            "Question": "How do you resolve conflicts in a team?",
            "Ideal_Answer": "Through active listening, empathy, open communication, and finding common ground.",
            "Category": "Conflict Resolution",
            "Role": "HR",
            "Difficulty": "Medium",
            "Skills": "Soft Skills & HR",
        },
        {
            "Question": "Describe your experience with Agile methodology.",
            "Ideal_Answer": "Worked with Scrum, sprints, daily standups, and retrospectives.",
            "Category": "Process",
            "Role": "Software Engineer",
            "Difficulty": "Easy",
            "Skills": "Agile & Process",
        },
        {
            "Question": "Explain the CAP theorem.",
            "Ideal_Answer": "Distributed systems can only choose 2 of Consistency, Availability, Partition tolerance.",
            "Category": "Technical",
            "Role": "Software Engineer",
            "Difficulty": "Hard",
            "Skills": "System Design",
        },
    ])


def test_parse_job_profile():
    # String input with role, seniority, and skill keywords
    p1 = parse_job_profile("Senior Backend Software Engineer with Python and Redis caching experience")
    assert p1.role == "Software Engineer"
    assert p1.seniority == "Senior"
    assert "Python" in p1.target_skills
    assert "SQL & Databases" in p1.target_skills or "System Design" in p1.target_skills
    assert p1.difficulty_preferences["Hard"] >= 0.40

    # Dict input
    p2 = parse_job_profile({
        "role": "HR",
        "seniority": "Junior",
        "skills": ["Soft Skills & HR"],
        "query": "conflict resolution and onboarding",
    })
    assert p2.role == "HR"
    assert p2.seniority == "Junior"
    assert "Soft Skills & HR" in p2.target_skills
    assert p2.difficulty_preferences["Easy"] >= 0.50

    # JobProfile instance pass-through
    p3 = parse_job_profile(p1)
    assert p3 is p1


def test_generate_evaluation_rubric():
    # System design question rubric
    sd_row = {
        "Category": "System Design",
        "Skills": "System Design, Web & Networking",
        "Difficulty": "Hard",
        "Question": "How would you design a URL shortener?",
    }
    rubric = generate_evaluation_rubric(sd_row)
    assert len(rubric["follow_up_probes"]) > 0
    assert len(rubric["key_criteria"]) > 0
    assert len(rubric["red_flags"]) > 0
    assert rubric["estimated_minutes"] == 15
    assert any("caching" in p.lower() or "throughput" in p.lower() or "partition" in p.lower() for p in rubric["follow_up_probes"])

    # Conflict / HR question rubric
    hr_row = {
        "Category": "Conflict Resolution",
        "Skills": "Soft Skills & HR",
        "Difficulty": "Medium",
        "Question": "How do you resolve conflicts in a team?",
    }
    hr_rubric = generate_evaluation_rubric(hr_row)
    assert hr_rubric["estimated_minutes"] == 10
    assert any("STAR" in c or "empathy" in c.lower() for c in hr_rubric["key_criteria"])


def test_hybrid_match_ranker_scoring():
    df = _get_mock_df()
    ranker = HybridMatchRanker(df)

    # 1. Target skills scoring
    skill_scores = ranker.compute_skill_scores(["Python", "OOP"])
    assert skill_scores[1] > 0.5  # Question 1 is OOP in Python

    # 2. Role scoring
    role_scores = ranker.compute_role_scores("Software Engineer")
    assert role_scores[1] == 1.0
    assert role_scores[0] == 0.65  # General role gets partial affinity
    assert role_scores[5] == 0.15  # HR role gets lower affinity for SE

    # 3. Difficulty scoring
    senior_diff = SENIORITY_LEVELS["senior"]
    diff_scores = ranker.compute_difficulty_scores(senior_diff)
    assert diff_scores[3] == senior_diff["Hard"]  # URL shortener is Hard
    assert diff_scores[1] == senior_diff["Easy"]  # OOP is Easy


def test_hybrid_match_ranking_and_mmr():
    df = _get_mock_df()
    ranker = HybridMatchRanker(df)

    # Query targeting distributed architecture
    results = ranker.rank(
        profile_or_query="Senior Software Engineer distributed system caching and load balancing",
        top_k=4,
        apply_mmr=True,
        diversity_lambda=0.7,
    )

    assert not results.empty
    assert len(results) <= 4
    assert "Composite_Score" in results.columns
    assert "Semantic_Score" in results.columns
    assert "Skill_Score" in results.columns
    assert "Rank" in results.columns

    # Top result should be system design or database related
    top_question = results.iloc[0]["Question"]
    assert "URL shortener" in top_question or "CAP" in top_question or "SQL" in top_question


def test_dynamic_question_generator_presets():
    df = _get_mock_df()
    generator = DynamicQuestionGenerator(df)

    # Balanced Round (5 questions across stages)
    balanced_set = generator.generate_interview(
        profile_or_query="Mid Software Engineer with Python and SQL",
        num_questions=4,
        round_type="balanced",
        include_rubrics=True,
    )

    assert len(balanced_set) == 4
    assert "Stage" in balanced_set.columns
    assert "Follow_Up_Probes" in balanced_set.columns
    assert "Key_Criteria" in balanced_set.columns
    assert "Estimated_Minutes" in balanced_set.columns
    assert "Question_Number" in balanced_set.columns
    assert list(balanced_set["Question_Number"]) == [1, 2, 3, 4]

    # Technical Deep Dive Round
    tech_set = generator.generate_interview(
        profile_or_query="Senior Software Engineer System Design and Concurrency",
        num_questions=3,
        round_type="technical_deep_dive",
    )
    assert len(tech_set) == 3
    # Check that technical categories dominate
    tech_cats = set(tech_set["Category"])
    assert any(c in tech_cats for c in ["Technical", "System Design", "Process"])


def test_formatting_and_markdown_export():
    df = _get_mock_df()
    generator = DynamicQuestionGenerator(df)
    profile = parse_job_profile("Senior Software Engineer with System Design")

    interview_set = generator.generate_interview(
        profile_or_query=profile,
        num_questions=3,
        round_type="balanced",
    )

    # Test Console Formatter
    text_guide = format_interview_guide(interview_set, profile=profile)
    assert "DYNAMIC INTERVIEW GUIDE" in text_guide
    assert "Target Role" in text_guide
    assert "Follow-Up Probes:" in text_guide
    assert "Q1." in text_guide

    # Test Markdown Exporter
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_guide.md"
        saved = export_markdown_interview_guide(interview_set, out_file, profile=profile)
        assert saved.is_file()
        content = saved.read_text(encoding="utf-8")
        assert "# Interview Guide: Senior Software Engineer" in content
        assert "## Candidate & Role Specification" in content
        assert "Follow-Up Probing Questions:" in content
        assert "Evaluation Signals" in content


if __name__ == "__main__":
    test_parse_job_profile()
    test_generate_evaluation_rubric()
    test_hybrid_match_ranker_scoring()
    test_hybrid_match_ranking_and_mmr()
    test_dynamic_question_generator_presets()
    test_formatting_and_markdown_export()
    print("All Day 6 semantic match ranking & dynamic question generation tests passed successfully!")
