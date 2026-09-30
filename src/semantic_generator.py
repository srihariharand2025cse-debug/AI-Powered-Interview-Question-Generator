"""
semantic_generator.py — Day 6: Semantic Match Ranking & Dynamic Question Generation.

Provides advanced hybrid semantic matching, multi-criteria relevance scoring,
Maximal Marginal Relevance (MMR) diversity reranking, dynamic candidate/job profiling,
interview stage sequencing, tailored follow-up probing, evaluation rubrics,
and interview guide generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.question_filter import extract_skills, tag_skills, SKILL_TAXONOMY
from src.text_embeddings import (
    SemanticQuestionSearcher,
    TfidfQuestionVectorizer,
    preprocess_document,
)


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

SENIORITY_LEVELS: Dict[str, Dict[str, float]] = {
    "junior": {"Easy": 0.60, "Medium": 0.35, "Hard": 0.05},
    "entry": {"Easy": 0.60, "Medium": 0.35, "Hard": 0.05},
    "intern": {"Easy": 0.70, "Medium": 0.30, "Hard": 0.00},
    "mid": {"Easy": 0.20, "Medium": 0.60, "Hard": 0.20},
    "intermediate": {"Easy": 0.20, "Medium": 0.60, "Hard": 0.20},
    "senior": {"Easy": 0.05, "Medium": 0.45, "Hard": 0.50},
    "lead": {"Easy": 0.05, "Medium": 0.35, "Hard": 0.60},
    "principal": {"Easy": 0.00, "Medium": 0.30, "Hard": 0.70},
    "staff": {"Easy": 0.00, "Medium": 0.30, "Hard": 0.70},
}

DEFAULT_DIFFICULTY_WEIGHTS: Dict[str, float] = {"Easy": 0.33, "Medium": 0.34, "Hard": 0.33}

# Stage templates mapping round types to stage sequences
INTERVIEW_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "balanced": [
        {
            "stage_name": "Warm-Up & Background",
            "category": ["Introduction", "Motivation", "Self-Assessment"],
            "difficulty": ["Easy"],
            "target_ratio": 0.20,
            "default_time_min": 7,
        },
        {
            "stage_name": "Core Technical Assessment",
            "category": ["Technical", "Process"],
            "difficulty": ["Easy", "Medium"],
            "target_ratio": 0.35,
            "default_time_min": 15,
        },
        {
            "stage_name": "System Architecture & Problem Solving",
            "category": ["System Design", "Technical"],
            "difficulty": ["Medium", "Hard"],
            "target_ratio": 0.25,
            "default_time_min": 18,
        },
        {
            "stage_name": "Behavioral & Collaboration",
            "category": ["Behavioral", "Conflict Resolution", "Process", "Career Goals"],
            "difficulty": ["Easy", "Medium", "Hard"],
            "target_ratio": 0.20,
            "default_time_min": 10,
        },
    ],
    "technical_deep_dive": [
        {
            "stage_name": "Core Fundamentals & CS Concepts",
            "category": ["Technical"],
            "difficulty": ["Easy", "Medium"],
            "target_ratio": 0.30,
            "default_time_min": 12,
        },
        {
            "stage_name": "Architecture, Scalability & Engineering",
            "category": ["System Design", "Technical"],
            "difficulty": ["Medium", "Hard"],
            "target_ratio": 0.50,
            "default_time_min": 25,
        },
        {
            "stage_name": "Engineering Practices & Code Quality",
            "category": ["Process", "Technical"],
            "difficulty": ["Medium"],
            "target_ratio": 0.20,
            "default_time_min": 10,
        },
    ],
    "behavioral_leadership": [
        {
            "stage_name": "Career Journey & Self Reflection",
            "category": ["Introduction", "Self-Assessment", "Career Goals"],
            "difficulty": ["Easy", "Medium"],
            "target_ratio": 0.25,
            "default_time_min": 10,
        },
        {
            "stage_name": "Conflict Resolution & Stakeholder Management",
            "category": ["Conflict Resolution", "Behavioral"],
            "difficulty": ["Medium", "Hard"],
            "target_ratio": 0.45,
            "default_time_min": 20,
        },
        {
            "stage_name": "Execution, Deadlines & Resilience",
            "category": ["Work Style", "Behavioral", "Experience"],
            "difficulty": ["Medium"],
            "target_ratio": 0.30,
            "default_time_min": 15,
        },
    ],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class JobProfile:
    """Structured representation of target candidate requirements."""
    role: str = "Software Engineer"
    seniority: str = "Mid"
    target_skills: List[str] = field(default_factory=list)
    raw_query: str = ""
    target_categories: List[str] = field(default_factory=list)
    difficulty_preferences: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role = self.role.strip()
        self.seniority = self.seniority.strip().capitalize()
        if not self.difficulty_preferences:
            sen_key = self.seniority.lower()
            self.difficulty_preferences = SENIORITY_LEVELS.get(
                sen_key, DEFAULT_DIFFICULTY_WEIGHTS
            ).copy()


# ---------------------------------------------------------------------------
# Profile Parser
# ---------------------------------------------------------------------------

def parse_job_profile(
    profile_input: Union[str, Dict[str, Any], JobProfile],
) -> JobProfile:
    """Parse a free-text job description, dictionary, or JobProfile instance.

    Extracts role, seniority, skills, categories, and difficulty preferences.
    """
    if isinstance(profile_input, JobProfile):
        return profile_input

    if isinstance(profile_input, dict):
        role = profile_input.get("role") or profile_input.get("Role") or "Software Engineer"
        seniority = profile_input.get("seniority") or profile_input.get("Seniority") or "Mid"
        raw_skills = profile_input.get("skills") or profile_input.get("target_skills") or []
        if isinstance(raw_skills, str):
            skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        else:
            skills = list(raw_skills)

        raw_query = profile_input.get("query") or profile_input.get("description") or ""
        categories = profile_input.get("categories") or profile_input.get("target_categories") or []
        diff_pref = profile_input.get("difficulty_preferences") or {}

        # Auto-extract any additional skills from query text
        if raw_query:
            extracted = extract_skills(raw_query)
            for sk in extracted:
                if sk not in skills:
                    skills.append(sk)

        return JobProfile(
            role=role,
            seniority=seniority,
            target_skills=skills,
            raw_query=raw_query,
            target_categories=categories,
            difficulty_preferences=diff_pref,
        )

    # Free-text input string (e.g. "Senior Backend Engineer with Python, Redis, and high concurrency")
    text = str(profile_input).strip()
    if not text:
        return JobProfile()

    # Detect Seniority
    seniority = "Mid"
    text_lower = text.lower()
    for sen in ["principal", "staff", "lead", "senior", "junior", "entry", "intern", "mid"]:
        if re.search(r"\b" + sen + r"\b", text_lower):
            seniority = sen.capitalize()
            break

    # Detect Role
    role = "Software Engineer"
    if re.search(r"\b(hr|human resources|recruiter|people ops)\b", text_lower):
        role = "HR"
    elif re.search(r"\b(data scientist|data engineer|ml engineer|machine learning)\b", text_lower):
        role = "Data Science"
    elif re.search(r"\b(software engineer|developer|backend|frontend|fullstack|devops|architect)\b", text_lower):
        role = "Software Engineer"
    elif re.search(r"\b(general|manager|operations)\b", text_lower):
        role = "General"

    # Extract target skills using taxonomy
    skills = extract_skills(text)

    # Detect categories of interest
    categories: List[str] = []
    cat_keywords = {
        "Technical": ["technical", "coding", "algorithm", "database", "python", "code"],
        "System Design": ["system design", "distributed", "scalability", "architecture", "microservices"],
        "Behavioral": ["behavioral", "culture", "star", "teamwork", "challenges"],
        "Conflict Resolution": ["conflict", "resolution", "disagreement", "stakeholder"],
        "Process": ["agile", "scrum", "ci/cd", "pipeline", "code review"],
        "Motivation": ["motivation", "passion", "drive"],
    }
    for cat, kw_list in cat_keywords.items():
        for kw in kw_list:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                if cat not in categories:
                    categories.append(cat)
                break

    return JobProfile(
        role=role,
        seniority=seniority,
        target_skills=skills,
        raw_query=text,
        target_categories=categories,
    )


# ---------------------------------------------------------------------------
# Probing & Evaluation Rubrics Generator
# ---------------------------------------------------------------------------

def generate_evaluation_rubric(
    question_row: Union[pd.Series, Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate dynamic follow-up probing questions and interviewer rubric.

    Produces tailored deep-dive prompts, key signals, and potential red flags.
    """
    row = question_row if isinstance(question_row, dict) else question_row.to_dict()
    category = str(row.get("Category", "Technical")).strip()
    skills = str(row.get("Skills", "General")).strip()
    diff = str(row.get("Difficulty", "Medium")).strip()
    q_text = str(row.get("Question", "")).lower()

    # Default probing questions
    probes: List[str] = []
    criteria: List[str] = []
    red_flags: List[str] = []

    # Domain-specific dynamic probes and rubrics
    if "System Design" in category or "system design" in skills.lower() or "shortener" in q_text:
        probes = [
            "How would you handle single points of failure and database partition tolerance?",
            "What caching eviction strategy (LRU/LFU) would you implement and why?",
            "How would you estimate read vs write throughput and storage capacity for 5 years?",
        ]
        criteria = [
            "Clearly identifies bottleneck components (Load Balancer, Caching layer, Database).",
            "Applies horizontal vs vertical scaling concepts appropriately.",
            "Considers latency, consistency trade-offs, and fault-tolerance.",
        ]
        red_flags = [
            "Proposes a single central database with no indexing or caching strategy.",
            "Cannot articulate difference between horizontal and vertical scaling.",
        ]
    elif "SQL" in skills or "database" in q_text:
        probes = [
            "Under what specific workloads would you choose NoSQL document stores over an RDBMS?",
            "Can you explain the ACID isolation levels and what dirty reads vs phantom reads are?",
            "How do B-Tree indexes improve query performance and what are their write penalties?",
        ]
        criteria = [
            "Accurately explains ACID transaction guarantees vs BASE properties.",
            "Understands query optimization, indexing strategies, and normalization.",
        ]
        red_flags = [
            "Claims NoSQL is always faster than SQL without considering relational integrity.",
            "Unaware of indexing overhead on write-heavy systems.",
        ]
    elif "Concurrency" in skills or "thread" in q_text or "process" in q_text:
        probes = [
            "How do mutexes, semaphores, and spinlocks differ in overhead and usage?",
            "Can you describe the 4 Coffman conditions required for a deadlock to occur?",
            "How does the OS scheduler manage context switching between user-level and kernel threads?",
        ]
        criteria = [
            "Demonstrates precise grasp of shared memory, race conditions, and synchronization primitives.",
            "Articulates real-world deadlock prevention techniques.",
        ]
        red_flags = [
            "Confuses thread-safe memory sharing with independent process isolation.",
            "Unaware of race condition risks in concurrent environments.",
        ]
    elif "OOP" in skills or "object-oriented" in q_text:
        probes = [
            "How do you decide between composition and class inheritance in modern architectures?",
            "Can you illustrate the Liskov Substitution Principle with a concrete anti-pattern?",
            "How does Python's method resolution order (MRO) handle multiple inheritance?",
        ]
        criteria = [
            "Defines encapsulation, abstraction, inheritance, and polymorphism with clean practical examples.",
            "Favors composition over deep inheritance hierarchies.",
        ]
        red_flags = [
            "Vague textbook definitions without practical code reasoning.",
        ]
    elif "Agile" in skills or "ci/cd" in q_text or "Process" in category:
        probes = [
            "How do you handle scope creep or high-priority interrupt tasks mid-sprint?",
            "What automated quality gates do you mandate in a production CI/CD pipeline?",
            "How do you conduct constructive code reviews without blocking development velocity?",
        ]
        criteria = [
            "Emphasizes continuous integration, automated testing, and team psychological safety.",
            "Demonstrates pragmatic balance between velocity and rigorous quality controls.",
        ]
        red_flags = [
            "Views process as rigid dogma rather than continuous feedback loop.",
            "Believes code review is purely about linting/formatting.",
        ]
    elif "Conflict" in category or "HR" in str(row.get("Role", "")) or "Behavioral" in category:
        probes = [
            "Can you walk me through the specific conversation steps you took to de-escalate that situation?",
            "If the other party had refused to compromise, what would have been your escalation path?",
            "What key lesson from that experience changed how you lead or communicate today?",
        ]
        criteria = [
            "Follows the STAR framework (Situation, Task, Action, Result).",
            "Demonstrates high emotional intelligence, empathy, and constructive conflict resolution.",
            "Takes personal accountability rather than blaming coworkers.",
        ]
        red_flags = [
            "Blames other teammates or management without self-reflection.",
            "Avoids direct conflict or claims they have never faced team disagreements.",
        ]
    else:
        # General question fallback
        probes = [
            "Can you provide a specific production example where you applied this principle?",
            "What were the trade-offs of this approach compared to alternative solutions?",
            "What would you do differently if you were starting that initiative again today?",
        ]
        criteria = [
            "Provides structured, coherent, and substantiated answers.",
            "Articulates trade-offs and shows practical hands-on experience.",
        ]
        red_flags = [
            "Superficial buzzwords without concrete technical depth.",
        ]

    # Seniority/Difficulty adjustment
    if diff == "Hard":
        time_est = 15
    elif diff == "Medium":
        time_est = 10
    else:
        time_est = 5

    return {
        "follow_up_probes": probes[:3],
        "key_criteria": criteria,
        "red_flags": red_flags,
        "estimated_minutes": time_est,
    }


# ---------------------------------------------------------------------------
# Hybrid Semantic Match Ranker
# ---------------------------------------------------------------------------

class HybridMatchRanker:
    """Multi-factor semantic match ranking engine.

    Combines:
    - Dense / TF-IDF Vector Semantic Similarity (Cosine)
    - Target Skill Overlap & Jaccard Salience
    - Role Affinity Matching
    - Seniority & Difficulty Distribution Calibration
    - Exact Keyword / Token Salience Boost
    - Maximal Marginal Relevance (MMR) for Cross-Question Diversity
    """

    def __init__(
        self,
        df: pd.DataFrame,
        searcher: Optional[SemanticQuestionSearcher] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        if "Skills" not in self.df.columns:
            self.df = tag_skills(self.df)

        self.searcher = searcher or SemanticQuestionSearcher(self.df)
        if not self.searcher.vectorizer.is_fitted:
            self.searcher.fit()

        # Configurable ranking component weights
        self.weights = {
            "semantic": 0.40,
            "skill": 0.25,
            "role": 0.15,
            "difficulty": 0.10,
            "keyword": 0.10,
        }
        if weights:
            self.weights.update(weights)

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_skill_scores(
        self,
        target_skills: Sequence[str],
    ) -> np.ndarray:
        """Compute Jaccard-style skill overlap scores across questions."""
        n_samples = len(self.df)
        if not target_skills:
            return np.zeros(n_samples, dtype=float)

        target_set = {s.strip().lower() for s in target_skills if s.strip()}
        if not target_set:
            return np.zeros(n_samples, dtype=float)

        scores = np.zeros(n_samples, dtype=float)
        for idx, row in self.df.iterrows():
            q_skills_str = str(row.get("Skills", ""))
            q_skills = {s.strip().lower() for s in q_skills_str.split(",") if s.strip()}
            q_text = str(row.get("Question", "")).lower() + " " + str(row.get("Ideal_Answer", "")).lower()

            # Direct skill match
            intersection = target_set.intersection(q_skills)
            score = len(intersection) / len(target_set)

            # Secondary text pattern scan for un-tagged target skills
            for ts in target_set:
                if ts not in q_skills and re.search(r"\b" + re.escape(ts) + r"\b", q_text):
                    score += (0.5 / len(target_set))

            scores[idx] = min(1.0, score)

        return scores

    def compute_role_scores(
        self,
        target_role: str,
    ) -> np.ndarray:
        """Compute role affinity scores (exact = 1.0, general = 0.65, mismatch = 0.15)."""
        target = target_role.strip().lower()
        scores = np.zeros(len(self.df), dtype=float)

        for idx, row in self.df.iterrows():
            q_role = str(row.get("Role", "General")).strip().lower()
            if q_role == target:
                scores[idx] = 1.0
            elif q_role == "general" or target == "general":
                scores[idx] = 0.65
            else:
                scores[idx] = 0.15

        return scores

    def compute_difficulty_scores(
        self,
        diff_preferences: Dict[str, float],
    ) -> np.ndarray:
        """Compute difficulty alignment scores based on seniority calibration."""
        scores = np.zeros(len(self.df), dtype=float)
        for idx, row in self.df.iterrows():
            q_diff = str(row.get("Difficulty", "Medium")).strip().capitalize()
            scores[idx] = diff_preferences.get(q_diff, 0.33)
        return scores

    def compute_keyword_boost(
        self,
        query: str,
    ) -> np.ndarray:
        """Compute exact keyword term matches in question and ideal answer."""
        if not query or not query.strip():
            return np.zeros(len(self.df), dtype=float)

        clean_q = preprocess_document(query)
        tokens = clean_q.split()
        if not tokens:
            return np.zeros(len(self.df), dtype=float)

        scores = np.zeros(len(self.df), dtype=float)
        for idx, row in self.df.iterrows():
            q_text = str(row.get("Question", "")).lower()
            a_text = str(row.get("Ideal_Answer", "")).lower()

            hit_count = 0
            for token in tokens:
                pattern = r"\b" + re.escape(token) + r"\b"
                if re.search(pattern, q_text):
                    hit_count += 2  # Higher boost for question hits
                elif re.search(pattern, a_text):
                    hit_count += 1

            # Normalize by token count
            scores[idx] = min(1.0, hit_count / (len(tokens) * 2))

        return scores

    def rank(
        self,
        profile_or_query: Union[str, Dict[str, Any], JobProfile],
        top_k: Optional[int] = None,
        min_score: float = 0.05,
        apply_mmr: bool = True,
        diversity_lambda: float = 0.70,
        category: Optional[str] = None,
        role: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute multi-factor hybrid scores and rank questions.

        Parameters
        ----------
        profile_or_query : Union[str, Dict[str, Any], JobProfile]
            Search query string, JD dict, or JobProfile.
        top_k : int, optional
            Limit to top K items.
        min_score : float, default 0.05
            Minimum composite score cutoff.
        apply_mmr : bool, default True
            Whether to apply Maximal Marginal Relevance to diversify candidates.
        diversity_lambda : float, default 0.70
            MMR balance parameter (1.0 = pure relevance, 0.0 = pure diversity).
        category, role, difficulty : optional metadata filters.

        Returns
        -------
        pd.DataFrame
            Ranked questions with detailed scoring breakdown.
        """
        if self.df.empty:
            return pd.DataFrame()

        profile = parse_job_profile(profile_or_query)
        query_text = profile.raw_query or f"{profile.role} {profile.seniority} {' '.join(profile.target_skills)}"

        # 1. Semantic Vector Score (Cosine Similarity)
        clean_q = preprocess_document(query_text)
        if clean_q and self.searcher.embedding_matrix is not None:
            query_vec = self.searcher.vectorizer.transform([clean_q])
            semantic_scores = cosine_similarity(query_vec, self.searcher.embedding_matrix)[0]
        else:
            semantic_scores = np.zeros(len(self.df), dtype=float)

        # 2. Skill Overlap Score
        skill_scores = self.compute_skill_scores(profile.target_skills)

        # 3. Role Affinity Score
        effective_role = role or profile.role
        role_scores = self.compute_role_scores(effective_role)

        # 4. Difficulty Calibration Score
        diff_scores = self.compute_difficulty_scores(profile.difficulty_preferences)

        # 5. Keyword Boost
        kw_scores = self.compute_keyword_boost(query_text)

        # Weighted Composite Score
        composite_scores = (
            self.weights["semantic"] * semantic_scores
            + self.weights["skill"] * skill_scores
            + self.weights["role"] * role_scores
            + self.weights["difficulty"] * diff_scores
            + self.weights["keyword"] * kw_scores
        )

        # Filter masks
        mask = composite_scores >= min_score

        if category:
            mask &= (self.df["Category"].str.lower() == category.lower())
        if difficulty:
            mask &= (self.df["Difficulty"].str.lower() == difficulty.lower())
        if role and role.lower() != "general":
            role_match = (self.df["Role"].str.lower() == role.lower()) | (self.df["Role"].str.lower() == "general")
            mask &= role_match

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame()

        # Sort indices by composite score
        sorted_indices = valid_indices[np.argsort(-composite_scores[valid_indices])].tolist()

        # Apply Maximal Marginal Relevance (MMR) if requested
        if apply_mmr and len(sorted_indices) > 1 and self.searcher.embedding_matrix is not None:
            final_indices = self._apply_mmr(
                candidate_indices=sorted_indices,
                scores=composite_scores,
                diversity_lambda=diversity_lambda,
                top_k=top_k or len(sorted_indices),
            )
        else:
            final_indices = sorted_indices[:top_k] if top_k else sorted_indices

        # Build output dataframe
        records: List[Dict[str, Any]] = []
        for rank_pos, idx in enumerate(final_indices, 1):
            row_dict = self.df.iloc[idx].to_dict()
            row_dict["Rank"] = rank_pos
            row_dict["Composite_Score"] = round(float(composite_scores[idx]), 4)
            row_dict["Semantic_Score"] = round(float(semantic_scores[idx]), 4)
            row_dict["Skill_Score"] = round(float(skill_scores[idx]), 4)
            row_dict["Role_Score"] = round(float(role_scores[idx]), 4)
            row_dict["Difficulty_Score"] = round(float(diff_scores[idx]), 4)
            row_dict["Keyword_Score"] = round(float(kw_scores[idx]), 4)

            # Salient features
            top_kw = self.searcher.vectorizer.get_top_keywords(
                self.searcher.embedding_matrix[idx], top_n=3
            )
            row_dict["Top_Keywords"] = ", ".join([k[0] for k in top_kw])
            records.append(row_dict)

        return pd.DataFrame(records)

    def _apply_mmr(
        self,
        candidate_indices: List[int],
        scores: np.ndarray,
        diversity_lambda: float,
        top_k: int,
    ) -> List[int]:
        """Maximal Marginal Relevance diversification algorithm."""
        if not candidate_indices:
            return []

        selected: List[int] = [candidate_indices[0]]
        remaining = candidate_indices[1:]
        embed_mat = self.searcher.embedding_matrix

        while remaining and len(selected) < top_k:
            selected_embeds = embed_mat[selected]
            rem_embeds = embed_mat[remaining]

            # Cross-similarity between remaining candidates and already selected questions
            cross_sims = cosine_similarity(rem_embeds, selected_embeds)
            max_sim_to_selected = np.max(cross_sims, axis=1)

            # MMR formula: lambda * Relevance - (1 - lambda) * MaxSimilarity
            rem_scores = np.array([scores[idx] for idx in remaining])
            mmr_values = (diversity_lambda * rem_scores) - ((1.0 - diversity_lambda) * max_sim_to_selected)

            best_rem_idx = int(np.argmax(mmr_values))
            selected.append(remaining[best_rem_idx])
            remaining.pop(best_rem_idx)

        return selected


# ---------------------------------------------------------------------------
# Dynamic Interview Question Generator
# ---------------------------------------------------------------------------

class DynamicQuestionGenerator:
    """End-to-end interview question generation engine.

    Orchestrates:
    - Job / Candidate Profile parsing
    - Stage-based interview structure (Warm-Up -> Core Tech -> Architecture -> Behavioral)
    - Dynamic question retrieval via HybridMatchRanker
    - Difficulty curve and time allocation
    - Automated probing questions and evaluation rubrics
    - Fallback and constraint relaxation
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ranker: Optional[HybridMatchRanker] = None,
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        if "Skills" not in self.df.columns:
            self.df = tag_skills(self.df)

        self.ranker = ranker or HybridMatchRanker(self.df)

    def generate_interview(
        self,
        profile_or_query: Union[str, Dict[str, Any], JobProfile],
        num_questions: int = 5,
        round_type: str = "balanced",
        include_rubrics: bool = True,
        diversity_lambda: float = 0.70,
    ) -> pd.DataFrame:
        """Generate a complete structured interview question set.

        Parameters
        ----------
        profile_or_query : str, dict, or JobProfile
            Candidate profile or job description.
        num_questions : int, default 5
            Total questions to produce.
        round_type : str, default 'balanced'
            One of 'balanced', 'technical_deep_dive', 'behavioral_leadership', or 'custom'.
        include_rubrics : bool, default True
            Whether to attach follow-up probes and scoring criteria.
        diversity_lambda : float, default 0.70
            MMR diversity factor.

        Returns
        -------
        pd.DataFrame
            Curated interview questions with stage sequencing and rubrics.
        """
        profile = parse_job_profile(profile_or_query)
        stages = INTERVIEW_PRESETS.get(round_type.lower(), INTERVIEW_PRESETS["balanced"])

        # Determine target question counts per stage
        selected_rows: List[Dict[str, Any]] = []
        used_indices: Set[int] = set()

        # Step 1: Allocate counts to stages
        stage_counts: List[int] = []
        allocated = 0
        for stg in stages:
            count = max(1, round(stg["target_ratio"] * num_questions))
            stage_counts.append(count)
            allocated += count

        # Adjust total to match requested num_questions
        diff = num_questions - sum(stage_counts)
        if diff != 0:
            stage_counts[0] = max(1, stage_counts[0] + diff)

        # Step 2: Sample questions per stage according to stage specifications
        for stage_idx, stg in enumerate(stages):
            stage_needed = stage_counts[stage_idx]
            stage_candidates = self._find_stage_candidates(
                profile=profile,
                allowed_categories=stg["category"],
                allowed_difficulties=stg["difficulty"],
                used_indices=used_indices,
                diversity_lambda=diversity_lambda,
            )

            # If stage candidates found, pick top matching
            for _, cand_row in stage_candidates.head(stage_needed).iterrows():
                row_dict = cand_row.to_dict()
                row_dict["Stage"] = stg["stage_name"]
                row_dict["Stage_Index"] = stage_idx + 1
                row_dict["Default_Time_Min"] = stg.get("default_time_min", 10)

                # Match original index
                orig_match = self.df[self.df["Question"] == row_dict["Question"]]
                if not orig_match.empty:
                    used_indices.add(orig_match.index[0])

                selected_rows.append(row_dict)

        # Step 3: Relaxation & Replenishment (if short of requested count)
        if len(selected_rows) < num_questions:
            needed = num_questions - len(selected_rows)
            fallback_df = self.ranker.rank(
                profile_or_query=profile,
                top_k=needed * 3,
                apply_mmr=True,
                diversity_lambda=diversity_lambda,
            )
            for _, fb_row in fallback_df.iterrows():
                if len(selected_rows) >= num_questions:
                    break
                orig_match = self.df[self.df["Question"] == fb_row["Question"]]
                if not orig_match.empty and orig_match.index[0] not in used_indices:
                    used_indices.add(orig_match.index[0])
                    fb_dict = fb_row.to_dict()
                    fb_dict["Stage"] = "Supplementary Assessment"
                    fb_dict["Stage_Index"] = len(stages) + 1
                    fb_dict["Default_Time_Min"] = 10
                    selected_rows.append(fb_dict)

        # Trim to exact count if slightly over
        selected_rows = selected_rows[:num_questions]
        result_df = pd.DataFrame(selected_rows)

        if result_df.empty:
            return result_df

        # Step 4: Attach Dynamic Follow-up Probes & Rubrics
        if include_rubrics:
            probes_col: List[str] = []
            criteria_col: List[str] = []
            red_flags_col: List[str] = []
            times_col: List[int] = []

            for _, row in result_df.iterrows():
                rubric = generate_evaluation_rubric(row)
                probes_col.append("; ".join(rubric["follow_up_probes"]))
                criteria_col.append("; ".join(rubric["key_criteria"]))
                red_flags_col.append("; ".join(rubric["red_flags"]))
                times_col.append(rubric["estimated_minutes"])

            result_df["Follow_Up_Probes"] = probes_col
            result_df["Key_Criteria"] = criteria_col
            result_df["Red_Flags"] = red_flags_col
            result_df["Estimated_Minutes"] = times_col

        # Assign final Question Order
        result_df["Question_Number"] = range(1, len(result_df) + 1)
        return result_df.reset_index(drop=True)

    def _find_stage_candidates(
        self,
        profile: JobProfile,
        allowed_categories: List[str],
        allowed_difficulties: List[str],
        used_indices: Set[int],
        diversity_lambda: float,
    ) -> pd.DataFrame:
        """Find and rank candidates matching a specific stage requirement."""
        # Query ranker for candidate pool
        ranked = self.ranker.rank(
            profile_or_query=profile,
            min_score=0.01,
            apply_mmr=True,
            diversity_lambda=diversity_lambda,
        )

        if ranked.empty:
            return pd.DataFrame()

        # Filter out already used questions
        available = []
        for _, row in ranked.iterrows():
            orig_match = self.df[self.df["Question"] == row["Question"]]
            if not orig_match.empty and orig_match.index[0] in used_indices:
                continue
            available.append(row)

        if not available:
            return pd.DataFrame()

        avail_df = pd.DataFrame(available)

        # Filter by stage categories
        cat_lower = [c.lower() for c in allowed_categories]
        cat_match = avail_df[avail_df["Category"].str.lower().isin(cat_lower)]

        if not cat_match.empty:
            # Further filter by difficulty if possible
            diff_lower = [d.lower() for d in allowed_difficulties]
            diff_match = cat_match[cat_match["Difficulty"].str.lower().isin(diff_lower)]
            if not diff_match.empty:
                return diff_match
            return cat_match

        # Soft relaxation: return top available candidates
        return avail_df


# ---------------------------------------------------------------------------
# Formatting and Export Helpers
# ---------------------------------------------------------------------------

def format_interview_guide(
    interview_df: pd.DataFrame,
    profile: Optional[JobProfile] = None,
) -> str:
    """Format an interview question set into a rich console-ready string."""
    if interview_df.empty:
        return "    [!] No interview questions could be generated."

    total_time = sum(interview_df.get("Estimated_Minutes", [10] * len(interview_df)))
    lines: List[str] = [
        "=" * 70,
        "    DYNAMIC INTERVIEW GUIDE & CANDIDATE EVALUATION PLAN",
        "=" * 70,
    ]

    if profile:
        skills_str = ", ".join(profile.target_skills) if profile.target_skills else "General"
        lines.extend([
            f"    • Target Role      : {profile.role} ({profile.seniority} Level)",
            f"    • Target Skills    : {skills_str}",
            f"    • Total Questions  : {len(interview_df)} | Est. Duration: ~{total_time} mins",
            "-" * 70,
        ])

    current_stage = ""
    for _, row in interview_df.iterrows():
        stage = row.get("Stage", "General Stage")
        if stage != current_stage:
            current_stage = stage
            lines.append(f"\n  [STAGE: {stage.upper()}]")
            lines.append("  " + "-" * 50)

        q_num = row.get("Question_Number", 1)
        score_pct = f"{row.get('Composite_Score', 0.0) * 100:.1f}%"
        diff = row.get("Difficulty", "Medium")
        mins = row.get("Estimated_Minutes", 10)

        lines.append(f"  Q{q_num}. [{diff} | ~{mins} min | Match: {score_pct}]")
        lines.append(f"      Question: {row['Question']}")
        lines.append(f"      Category: {row.get('Category', '')} | Skills: {row.get('Skills', '')}")

        # Probes
        probes = row.get("Follow_Up_Probes", "")
        if probes:
            lines.append("      Follow-Up Probes:")
            for p in probes.split("; "):
                lines.append(f"        -> {p}")

        # Criteria
        criteria = row.get("Key_Criteria", "")
        if criteria:
            lines.append(f"      Key Signal: {criteria.split('; ')[0]}")

        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def export_markdown_interview_guide(
    interview_df: pd.DataFrame,
    filepath: Union[str, Path],
    profile: Optional[JobProfile] = None,
) -> Path:
    """Export the dynamic interview guide to a clean Markdown (.md) document.

    Suitable for hiring managers, engineering interviewers, and HR panels.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    role_str = profile.role if profile else "Software Engineer"
    sen_str = profile.seniority if profile else "Mid"
    skills_str = ", ".join(profile.target_skills) if profile and profile.target_skills else "General"
    total_time = sum(interview_df.get("Estimated_Minutes", [10] * len(interview_df)))

    md: List[str] = [
        f"# Interview Guide: {sen_str} {role_str}",
        "",
        "> **Generated by AI-Powered Interview Question Generator (Day 6 Engine)**",
        "",
        "## Candidate & Role Specification",
        f"- **Role**: {role_str}",
        f"- **Seniority Level**: {sen_str}",
        f"- **Target Competencies**: {skills_str}",
        f"- **Total Questions**: {len(interview_df)}",
        f"- **Estimated Total Time**: ~{total_time} minutes",
        "",
        "---",
        "",
        "## Interview Stages & Questions",
        "",
    ]

    current_stage = ""
    for _, row in interview_df.iterrows():
        stage = row.get("Stage", "General Assessment")
        if stage != current_stage:
            current_stage = stage
            md.append(f"### Stage: {stage}\n")

        q_num = row.get("Question_Number", 1)
        diff = row.get("Difficulty", "Medium")
        mins = row.get("Estimated_Minutes", 10)
        match_pct = f"{row.get('Composite_Score', 0.0) * 100:.1f}%"

        md.append(f"#### Q{q_num}. {row['Question']}")
        md.append(f"- **Difficulty**: `{diff}` | **Time Budget**: `{mins} mins` | **Match**: `{match_pct}`")
        md.append(f"- **Category**: {row.get('Category', 'N/A')} | **Tagged Skills**: {row.get('Skills', 'N/A')}")
        md.append("")
        md.append(f"**Ideal Answer Outline:**\n> {row.get('Ideal_Answer', 'N/A')}\n")

        probes = row.get("Follow_Up_Probes", "")
        if probes:
            md.append("**Follow-Up Probing Questions:**")
            for p in probes.split("; "):
                md.append(f"- [ ] {p}")
            md.append("")

        criteria = row.get("Key_Criteria", "")
        if criteria:
            md.append("**Evaluation Signals (What to look for):**")
            for c in criteria.split("; "):
                md.append(f"- [x] {c}")
            md.append("")

        red_flags = row.get("Red_Flags", "")
        if red_flags:
            md.append("**Red Flags (Watch out for):**")
            for rf in red_flags.split("; "):
                md.append(f"- ⚠️ {rf}")
            md.append("")

        md.append("---\n")

    path.write_text("\n".join(md), encoding="utf-8")
    return path
