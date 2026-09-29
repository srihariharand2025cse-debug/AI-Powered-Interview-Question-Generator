"""
text_embeddings.py — Day 5: Advanced Preprocessing / TF-IDF & Embeddings Preparation.

Provides advanced text preprocessing, n-gram TF-IDF vectorization, dense embedding
preparation, cosine similarity ranking, keyword salience extraction, and semantic
question retrieval/recommendation.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EMBEDDINGS_DIR = _PROJECT_ROOT / "data"
_DEFAULT_EMBEDDINGS_FILE = _DEFAULT_EMBEDDINGS_DIR / "question_embeddings.npz"

# Curated English & domain filler stopwords to highlight meaningful interview terms
INTERVIEW_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves",
    # Interview conversational boilerplate
    "tell", "describe", "explain", "give", "example", "please", "would", "like",
    "someone", "used", "using", "work", "role", "previous", "experience",
}


# ---------------------------------------------------------------------------
# Advanced Text Preprocessing
# ---------------------------------------------------------------------------

def stem_word(word: str) -> str:
    """Lightweight rule-based suffix normalizer (Porter-style).

    Strips common inflectional suffixes (e.g. -ing, -ly, -ed, -es, -s, -tion)
    without requiring heavyweight external linguistic packages.
    """
    w = word.lower()
    if len(w) <= 3:
        return w

    # Step 1: Plurals and past tense
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("es") and len(w) > 4 and not w.endswith("zes"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]

    if w.endswith("eed") and len(w) > 4:
        w = w[:-1]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) >= 2 and w[-1] == w[-2] and w[-1] not in ("l", "s", "z"):
            w = w[:-1]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) >= 2 and w[-1] == w[-2] and w[-1] not in ("l", "s", "z"):
            w = w[:-1]

    # Step 2: Derivational endings
    if w.endswith("ational"):
        w = w[:-7] + "ate"
    elif w.endswith("tional"):
        w = w[:-2]
    elif w.endswith("alism"):
        w = w[:-3]
    elif w.endswith("ation"):
        w = w[:-5] + "ate"
    elif w.endswith("ment") and len(w) > 6:
        w = w[:-4]
    elif w.endswith("ability") and len(w) > 7:
        w = w[:-7] + "able"
    elif w.endswith("ibility") and len(w) > 7:
        w = w[:-7] + "ible"
    elif w.endswith("able") and len(w) > 6:
        w = w[:-4]
    elif w.endswith("ly") and len(w) > 4:
        w = w[:-2]

    return w


def clean_and_tokenize(
    text: str,
    remove_stopwords: bool = True,
    lowercase: bool = True,
    stem: bool = False,
    min_len: int = 2,
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """Tokenize and normalize text into clean words.

    Parameters
    ----------
    text : str
        Input string.
    remove_stopwords : bool, default True
        Whether to drop common stop words.
    lowercase : bool, default True
        Whether to lowercase tokens.
    stem : bool, default False
        Whether to apply rule-based suffix stemming.
    min_len : int, default 2
        Minimum character length for tokens.
    custom_stopwords : Set[str], optional
        Additional stop words to filter.

    Returns
    -------
    List[str]
        List of processed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if lowercase:
        text = text.lower()

    # Extract word tokens (allowing hyphens for compound tech terms like ci-cd or object-oriented)
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\+#\.-]+\b", text)

    stopwords = set(INTERVIEW_STOPWORDS)
    if custom_stopwords:
        stopwords.update(s.lower() for s in custom_stopwords)

    cleaned_tokens: List[str] = []
    for token in raw_tokens:
        clean_tok = token.strip(".-_")
        if len(clean_tok) < min_len:
            continue
        if remove_stopwords and clean_tok in stopwords:
            continue
        if stem:
            clean_tok = stem_word(clean_tok)
        cleaned_tokens.append(clean_tok)

    return cleaned_tokens


def preprocess_document(
    text: str,
    stem: bool = False,
    remove_stopwords: bool = True,
) -> str:
    """Preprocess text into a space-joined normalized document string.

    Parameters
    ----------
    text : str
        Raw input string.
    stem : bool, default False
        Whether to stem tokens.
    remove_stopwords : bool, default True
        Whether to filter stop words.

    Returns
    -------
    str
        Space-separated tokens.
    """
    tokens = clean_and_tokenize(text, remove_stopwords=remove_stopwords, stem=stem)
    return " ".join(tokens)


def build_document_corpus(
    df: pd.DataFrame,
    question_weight: int = 3,
    skills_weight: int = 2,
    include_answer: bool = True,
) -> List[str]:
    """Construct enriched, weighted document representations for each question.

    Assigns higher repetition weight to the Question and Skills to prioritize
    core question intent in TF-IDF and embedding vectors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns such as Question, Skills, Category, Role, Ideal_Answer.
    question_weight : int, default 3
        Number of times to repeat the Question text.
    skills_weight : int, default 2
        Number of times to repeat the Skills tags.
    include_answer : bool, default True
        Whether to include the Ideal_Answer in the document.

    Returns
    -------
    List[str]
        Enriched document strings ready for vectorization.
    """
    documents: List[str] = []

    for _, row in df.iterrows():
        q_text = str(row.get("Question", ""))
        skills_text = str(row.get("Skills", ""))
        role_text = str(row.get("Role", ""))
        cat_text = str(row.get("Category", ""))
        ans_text = str(row.get("Ideal_Answer", "")) if include_answer else ""

        # Build weighted document components
        parts: List[str] = []
        if q_text:
            parts.extend([q_text] * max(1, question_weight))
        if skills_text and skills_text.lower() != "general":
            parts.extend([skills_text] * max(1, skills_weight))
        if cat_text:
            parts.append(cat_text)
        if role_text and role_text.lower() != "general":
            parts.append(role_text)
        if ans_text:
            parts.append(ans_text)

        doc = " ".join(parts)
        documents.append(preprocess_document(doc))

    return documents


def prepare_embedding_texts(df: pd.DataFrame) -> List[str]:
    """Format dataset rows into structured metadata prompts for modern embedding models.

    Compatible with OpenAI text-embedding-3, HuggingFace, and Google Vertex AI.

    Parameters
    ----------
    df : pd.DataFrame
        Question dataset.

    Returns
    -------
    List[str]
        Structured strings for each interview question.
    """
    texts: List[str] = []
    for _, row in df.iterrows():
        role = row.get("Role", "General")
        cat = row.get("Category", "General")
        diff = row.get("Difficulty", "Medium")
        skills = row.get("Skills", "General")
        question = row.get("Question", "")
        answer = row.get("Ideal_Answer", "")

        prompt = (
            f"Role: {role} | Category: {cat} | Difficulty: {diff} | Skills: {skills} | "
            f"Question: {question} | Answer Summary: {answer}"
        )
        texts.append(prompt)
    return texts


# ---------------------------------------------------------------------------
# TF-IDF Question Vectorizer
# ---------------------------------------------------------------------------

class TfidfQuestionVectorizer:
    """TF-IDF vectorizer optimized for interview question retrieval & feature salience.

    Features:
    - Unigram and bigram tokenization (1, 2)
    - Sublinear term-frequency scaling
    - Smooth inverse document frequency
    - L2 normalization for direct cosine similarity computation
    - Out-of-vocabulary safe query transformation
    - Top salient keyword extraction per document
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        max_features: Optional[int] = 5000,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features

        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            norm="l2",
        )
        self.is_fitted: bool = False
        self._feature_names: List[str] = []

    def fit(self, documents: Sequence[str]) -> TfidfQuestionVectorizer:
        """Fit the vectorizer on a collection of preprocessed text documents."""
        self._vectorizer.fit(documents)
        self.is_fitted = True
        self._feature_names = list(self._vectorizer.get_feature_names_out())
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transform documents to normalized TF-IDF vector array."""
        if not self.is_fitted:
            raise ValueError("TfidfQuestionVectorizer has not been fitted yet.")
        sparse_mat = self._vectorizer.transform(documents)
        return sparse_mat.toarray()

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit and transform documents in one step."""
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Return list of vocabulary feature terms."""
        return self._feature_names

    @property
    def vocabulary_size(self) -> int:
        """Number of features in vocabulary."""
        return len(self._feature_names)

    def get_top_keywords(
        self,
        vector: np.ndarray,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Extract highest-scoring TF-IDF words/n-grams from a vector."""
        if not self.is_fitted:
            return []

        vec = np.asarray(vector).ravel()
        if vec.shape[0] != len(self._feature_names):
            raise ValueError("Vector dimensionality does not match vocabulary size.")

        nonzero_indices = np.where(vec > 0)[0]
        if len(nonzero_indices) == 0:
            return []

        sorted_indices = nonzero_indices[np.argsort(-vec[nonzero_indices])]
        top_indices = sorted_indices[:top_n]

        return [(self._feature_names[i], float(vec[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Embeddings Persistence & Dense Representation
# ---------------------------------------------------------------------------

def save_embeddings(
    filepath: Union[str, Path],
    vectors: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    feature_names: Optional[List[str]] = None,
) -> Path:
    """Save dense embedding matrix and optional metadata to a compressed .npz file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Target .npz file path.
    vectors : np.ndarray
        Embedding matrix of shape (n_samples, n_dimensions).
    metadata : List[Dict[str, Any]], optional
        Question metadata dicts (Role, Difficulty, Category, etc.).
    feature_names : List[str], optional
        Vocabulary terms if applicable.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_json = json.dumps(metadata) if metadata is not None else ""
    features_json = json.dumps(feature_names) if feature_names is not None else ""

    np.savez_compressed(
        path,
        vectors=vectors,
        metadata=meta_json,
        feature_names=features_json,
    )
    return path


def load_embeddings(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, Optional[List[Dict[str, Any]]], Optional[List[str]]]:
    """Load embeddings and metadata from a .npz file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to .npz archive.

    Returns
    -------
    Tuple[np.ndarray, Optional[List[Dict]], Optional[List[str]]]
        (embedding_matrix, metadata_list, feature_names)
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        vectors = data["vectors"]
        meta_str = str(data["metadata"]) if "metadata" in data else ""
        feat_str = str(data["feature_names"]) if "feature_names" in data else ""

    metadata = json.loads(meta_str) if meta_str else None
    feature_names = json.loads(feat_str) if feat_str else None
    return vectors, metadata, feature_names


# ---------------------------------------------------------------------------
# Semantic Question Searcher & Recommender Engine
# ---------------------------------------------------------------------------

class SemanticQuestionSearcher:
    """End-to-end question retrieval, semantic search, and recommendation engine.

    Combines TF-IDF vectorization with cosine similarity matching, metadata filtering,
    and keyword salience analysis.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vectorizer: Optional[TfidfQuestionVectorizer] = None,
        auto_fit: bool = True,
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.vectorizer = vectorizer or TfidfQuestionVectorizer()
        self.document_corpus: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.structured_prompts: List[str] = []

        if auto_fit and not self.df.empty:
            self.fit()

    def fit(self) -> SemanticQuestionSearcher:
        """Build document corpus, fit TF-IDF vectorizer, and compute matrix."""
        self.document_corpus = build_document_corpus(self.df)
        self.structured_prompts = prepare_embedding_texts(self.df)
        self.embedding_matrix = self.vectorizer.fit_transform(self.document_corpus)
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
        category: Optional[str] = None,
        role: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> pd.DataFrame:
        """Search questions using natural language query via vector cosine similarity.

        Parameters
        ----------
        query : str
            Free-text natural language query or requirement (e.g. "multithreading and deadlocks").
        top_k : int, default 5
            Maximum number of top questions to return.
        min_score : float, default 0.05
            Minimum cosine similarity threshold (0.0 to 1.0).
        category : str, optional
            Filter by question Category.
        role : str, optional
            Filter by target Role.
        difficulty : str, optional
            Filter by Difficulty.

        Returns
        -------
        pd.DataFrame
            Ranked questions with 'Similarity_Score' and 'Salient_Keywords'.
        """
        if self.embedding_matrix is None or not self.vectorizer.is_fitted:
            raise ValueError("SemanticQuestionSearcher has not been fitted.")

        if not query or not query.strip():
            return pd.DataFrame()

        # Preprocess and vectorize the incoming query
        clean_q = preprocess_document(query)
        if not clean_q:
            return pd.DataFrame()

        query_vec = self.vectorizer.transform([clean_q])
        scores = cosine_similarity(query_vec, self.embedding_matrix)[0]

        # Apply metadata filters
        mask = np.ones(len(self.df), dtype=bool)
        if category:
            mask &= (self.df["Category"].str.lower() == category.lower())
        if role and role.lower() != "general":
            # Match role or general questions
            role_match = (self.df["Role"].str.lower() == role.lower()) | (self.df["Role"].str.lower() == "general")
            mask &= role_match
        if difficulty:
            mask &= (self.df["Difficulty"].str.lower() == difficulty.lower())

        valid_indices = np.where(mask & (scores >= min_score))[0]
        if len(valid_indices) == 0:
            return pd.DataFrame()

        # Sort by similarity score descending
        sorted_indices = valid_indices[np.argsort(-scores[valid_indices])][:top_k]

        matched_rows: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            row_dict = self.df.iloc[idx].to_dict()
            row_dict["Similarity_Score"] = round(float(scores[idx]), 4)
            # Find salient overlapping terms between query and question
            top_kw = self.vectorizer.get_top_keywords(self.embedding_matrix[idx], top_n=3)
            row_dict["Salient_Keywords"] = ", ".join([kw[0] for kw in top_kw])
            matched_rows.append(row_dict)

        return pd.DataFrame(matched_rows)

    def recommend_similar(
        self,
        question_index: int,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """Find other interview questions most similar to a given question.

        Parameters
        ----------
        question_index : int
            Row index of the source question.
        top_k : int, default 3
            Number of similar questions to return.

        Returns
        -------
        pd.DataFrame
            Top similar questions with similarity scores.
        """
        if self.embedding_matrix is None or not (0 <= question_index < len(self.df)):
            raise ValueError(f"Invalid question index: {question_index}")

        target_vec = self.embedding_matrix[question_index:question_index + 1]
        scores = cosine_similarity(target_vec, self.embedding_matrix)[0]

        # Exclude self
        scores[question_index] = -1.0

        top_indices = np.argsort(-scores)[:top_k]
        top_indices = [idx for idx in top_indices if scores[idx] > 0]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row_dict = self.df.iloc[idx].to_dict()
            row_dict["Similarity_Score"] = round(float(scores[idx]), 4)
            results.append(row_dict)

        return pd.DataFrame(results)

    def export_embeddings(
        self,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist computed TF-IDF embedding vectors and metadata to disk.

        Parameters
        ----------
        output_file : Union[str, Path], optional
            Target .npz path. Defaults to data/question_embeddings.npz.

        Returns
        -------
        Path
            Path to saved embeddings.
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings computed to export.")

        target = Path(output_file or _DEFAULT_EMBEDDINGS_FILE)
        meta = self.df[["Question", "Category", "Role", "Difficulty", "Skills"]].to_dict(orient="records")

        return save_embeddings(
            filepath=target,
            vectors=self.embedding_matrix,
            metadata=meta,
            feature_names=self.vectorizer.get_feature_names(),
        )


# ---------------------------------------------------------------------------
# Terminal Output Formatting Helpers
# ---------------------------------------------------------------------------

def format_search_results(
    results_df: pd.DataFrame,
    query: str,
) -> str:
    """Format semantic question search results for console display."""
    if results_df.empty:
        return f"    [!] No questions found matching query: '{query}'"

    lines: List[str] = [
        f"    Query: '{query}'",
        f"    Matched: {len(results_df)} question(s) ranked by TF-IDF Cosine Relevance:",
        "    " + "-" * 62,
    ]

    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        score_pct = f"{row['Similarity_Score'] * 100:.1f}%"
        skills = row.get("Skills", "N/A")
        keywords = row.get("Salient_Keywords", "")
        lines.append(
            f"    {rank}. [{score_pct}] [{row.get('Difficulty', 'Med')}] {row['Question']}"
        )
        lines.append(
            f"       Role: {row.get('Role', 'General')} | Category: {row.get('Category', 'General')} | Skills: {skills}"
        )
        if keywords:
            lines.append(f"       Top Features: {keywords}")
        lines.append("")

    return "\n".join(lines).rstrip()
