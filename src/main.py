"""
main.py -- Entry point for the AI-Powered Interview Question Generator.

Day 1: Project skeleton                                           [done]
Day 2: Load the dataset                                           [done]
Day 3: Explore and clean the data (EDA, metrics, deduplication)   [done]
Day 4: Keyword & Skill-based Question Filtering / Categorization  [done]
Day 5: Advanced Preprocessing / TF-IDF & Embeddings Preparation   [done]
"""

from __future__ import annotations

import sys
import io

# Fix Windows console encoding (cp1252 cannot handle emoji/unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from src.data_loader import load_dataset
from src.data_cleaner import (
    explore_dataset,
    format_exploration_report,
    clean_dataset,
    save_cleaned_dataset,
)
from src.question_filter import (
    tag_skills,
    filter_questions,
    generate_interview_set,
    format_interview_set,
    QuestionFilter,
)
from src.text_embeddings import (
    SemanticQuestionSearcher,
    format_search_results,
)


def main() -> None:
    """Entry point for the Interview Question Generator."""
    print("=" * 70)
    print(">> AI-Powered Interview Question Generator")
    print("=" * 70)
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

    # -- Day 4: Skill Extraction & Categorization --
    print("[4] Auto-extracting skills & tagging questions ...")
    tagged_df = tag_skills(cleaned_df)
    saved_path = save_cleaned_dataset(tagged_df)
    print(f"    Cleaned & tagged dataset saved to: {saved_path}\n")

    # Initialize Filter engine
    qf = QuestionFilter(tagged_df)
    print("    • Available Roles      :", ", ".join(qf.get_roles()))
    print("    • Available Categories :", ", ".join(qf.get_categories()))
    print("    • Tracked Skill Topics :", ", ".join(qf.get_skills_catalog()[:5]) + " ...")
    print()

    # -- Day 4 Demo A: Filter by Specific Skill & Difficulty --
    print("[5] Demo Filter: Technical Questions with 'SQL' or 'System Design' ...")
    skill_filtered = qf.filter(
        skills=["SQL", "System Design"],
        category="Technical",
    )
    print(f"    Found {len(skill_filtered)} matching question(s):")
    for _, row in skill_filtered.iterrows():
        print(f"    - [{row['Difficulty']}] {row['Question']} (Skills: {row['Skills']})")
    print()

    # -- Day 4 Demo B: Generate Tailored Mock Interview Set --
    print("[6] Generating Tailored Mock Interview Set (Software Engineer, 4 Balanced Qs) ...")
    mock_interview = qf.generate_mock_interview(
        role="Software Engineer",
        num_questions=4,
        balance_difficulty=True,
        random_state=42,
    )
    print()
    print(format_interview_set(mock_interview))

    # -- Day 5: Advanced Preprocessing, TF-IDF & Semantic Search Engine --
    print("\n" + "=" * 70)
    print("[7] Fitting TF-IDF Vectorizer & Building Semantic Index ...")
    searcher = SemanticQuestionSearcher(tagged_df)
    vocab_size = searcher.vectorizer.vocabulary_size
    print(f"    • TF-IDF N-Gram Vocabulary Size: {vocab_size} features")
    sample_features = searcher.vectorizer.get_feature_names()[:8]
    print(f"    • Sample Feature Tokens        : {', '.join(sample_features)} ...\n")

    # -- Day 5 Demo A: Free-text Semantic Question Search --
    demo_query = "Designing scalable distributed systems with Redis caching and load balancers"
    print(f"[8] Demo Semantic Search: '{demo_query}'")
    search_results = searcher.search(demo_query, top_k=3, min_score=0.1)
    print()
    print(format_search_results(search_results, query=demo_query))
    print()

    # -- Day 5 Demo B: Similar Question Recommendation & Embedding Export --
    print("[9] Finding Questions Similar to Question #1 (OOP) & Exporting Embeddings ...")
    similar_qs = searcher.recommend_similar(question_index=6, top_k=2)
    print(f"    Target Question: {tagged_df.iloc[6]['Question']}")
    print("    Related Recommended Questions:")
    for _, sim_row in similar_qs.iterrows():
        score_pct = f"{sim_row['Similarity_Score'] * 100:.1f}%"
        print(f"    - [{score_pct}] {sim_row['Question']} (Role: {sim_row['Role']})")

    embeddings_file = searcher.export_embeddings()
    print(f"\n    Embeddings matrix and metadata exported to: {embeddings_file}")

    print("\n" + "=" * 70)
    print("[OK] Day 5 Advanced Preprocessing / TF-IDF & Embeddings Preparation complete!\n")


if __name__ == "__main__":
    main()
