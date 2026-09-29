import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.text_embeddings import (
    stem_word,
    clean_and_tokenize,
    preprocess_document,
    build_document_corpus,
    prepare_embedding_texts,
    TfidfQuestionVectorizer,
    SemanticQuestionSearcher,
    save_embeddings,
    load_embeddings,
    format_search_results,
)


def _get_sample_df() -> pd.DataFrame:
    return pd.DataFrame([
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
            "Question": "How do you resolve conflicts in a team?",
            "Ideal_Answer": "Through active listening, empathy, open communication, and finding common ground.",
            "Category": "Conflict Resolution",
            "Role": "HR",
            "Difficulty": "Medium",
            "Skills": "Soft Skills & HR",
        },
    ])


def test_stem_word():
    assert stem_word("programming") == "program"
    assert stem_word("databases") == "databas"
    assert stem_word("scalability") == "scalable"
    assert stem_word("relational") == "relate"
    assert stem_word("caching") == "cach"
    # Short words preserved
    assert stem_word("sql") == "sql"
    assert stem_word("oop") == "oop"


def test_clean_and_tokenize():
    text = "What is the difference between SQL and NoSQL databases?!"
    tokens = clean_and_tokenize(text, remove_stopwords=True)
    # Stop words like 'what', 'is', 'the', 'between', 'and' removed
    assert "sql" in tokens
    assert "nosql" in tokens
    assert "databases" in tokens
    assert "what" not in tokens
    assert "difference" in tokens

    # Test with stemming enabled
    stemmed = clean_and_tokenize(text, remove_stopwords=True, stem=True)
    assert "databas" in stemmed


def test_preprocess_document():
    raw = "Explain Object-Oriented Programming and its 4 pillars."
    res = preprocess_document(raw, stem=False, remove_stopwords=True)
    assert "object-oriented" in res
    assert "programming" in res
    assert "pillars" in res
    # "explain", "and", "its" are stopwords
    assert "explain" not in res.split()


def test_build_document_corpus():
    df = _get_sample_df()
    corpus = build_document_corpus(df, question_weight=2, skills_weight=2)
    assert len(corpus) == len(df)
    # Check that skills and question terms are present in document
    assert "object-oriented" in corpus[0]
    assert "sql" in corpus[1]
    assert "url" in corpus[2]
    assert "conflicts" in corpus[3]


def test_prepare_embedding_texts():
    df = _get_sample_df()
    prompts = prepare_embedding_texts(df)
    assert len(prompts) == len(df)
    assert prompts[0].startswith("Role: Software Engineer | Category: Technical")
    assert "Question: Explain the concept of Object-Oriented Programming." in prompts[0]


def test_tfidf_question_vectorizer():
    df = _get_sample_df()
    corpus = build_document_corpus(df)

    vectorizer = TfidfQuestionVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)

    assert vectorizer.is_fitted
    assert matrix.shape[0] == len(corpus)
    assert vectorizer.vocabulary_size > 10

    # L2 normalized check: each row vector norm should be approximately 1.0
    for row in matrix:
        norm = np.linalg.norm(row)
        assert np.isclose(norm, 1.0, atol=1e-4)

    # Check top keywords extraction
    top_kw = vectorizer.get_top_keywords(matrix[0], top_n=3)
    assert len(top_kw) > 0
    words = [k[0] for k in top_kw]
    assert any("object" in w or "programming" in w or "oop" in w for w in words)


def test_semantic_search():
    df = _get_sample_df()
    searcher = SemanticQuestionSearcher(df)

    # Query matching SQL
    results_sql = searcher.search("Tell me about relational SQL vs NoSQL schema transactions", top_k=2)
    assert not results_sql.empty
    top_hit = results_sql.iloc[0]
    assert "SQL" in top_hit["Question"]
    assert top_hit["Similarity_Score"] > 0.15

    # Query matching URL shortener system design
    results_sys = searcher.search("scalability, redis caching and system architecture", top_k=2)
    assert not results_sys.empty
    assert "URL shortener" in results_sys.iloc[0]["Question"]

    # Filter by category
    results_filtered = searcher.search(
        "conflict resolution and team communication",
        category="Conflict Resolution",
    )
    assert len(results_filtered) == 1
    assert results_filtered.iloc[0]["Category"] == "Conflict Resolution"

    # Filter by role
    results_hr = searcher.search("team", role="HR")
    assert all(r in ["HR", "General"] for r in results_hr["Role"])

    # Empty query handling
    empty_res = searcher.search("")
    assert empty_res.empty


def test_recommend_similar():
    df = _get_sample_df()
    searcher = SemanticQuestionSearcher(df)

    # Find similar questions to Question 1 (SQL and NoSQL)
    similar = searcher.recommend_similar(question_index=1, top_k=2)
    assert len(similar) <= 2
    # Ensure source question is not in the recommendations
    assert not any(q == df.iloc[1]["Question"] for q in similar["Question"])


def test_embeddings_persistence():
    df = _get_sample_df()
    searcher = SemanticQuestionSearcher(df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_embeddings.npz"
        saved_path = searcher.export_embeddings(tmp_file)
        assert saved_path.is_file()

        # Load back
        vectors, metadata, feature_names = load_embeddings(saved_path)
        assert vectors.shape == searcher.embedding_matrix.shape
        assert len(metadata) == len(df)
        assert metadata[0]["Question"] == df.iloc[0]["Question"]
        assert len(feature_names) == searcher.vectorizer.vocabulary_size


def test_format_search_results():
    df = _get_sample_df()
    searcher = SemanticQuestionSearcher(df)
    results = searcher.search("Python OOP objects", top_k=1)
    output = format_search_results(results, query="Python OOP objects")
    assert "Python OOP objects" in output
    assert "Explain the concept of Object-Oriented Programming" in output


if __name__ == "__main__":
    test_stem_word()
    test_clean_and_tokenize()
    test_preprocess_document()
    test_build_document_corpus()
    test_prepare_embedding_texts()
    test_tfidf_question_vectorizer()
    test_semantic_search()
    test_recommend_similar()
    test_embeddings_persistence()
    test_format_search_results()
    print("All Day 5 text embeddings and preprocessing tests passed successfully!")
