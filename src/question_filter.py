"""
question_filter.py — Day 4: Keyword & Skill-based Question Filtering / Categorization.

Provides robust filtering, skill extraction, keyword searching, and interview
question set generation based on job roles, technical/soft skills, difficulty,
and categories.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import pandas as pd


# ---------------------------------------------------------------------------
# Skill & Domain Taxonomy
# ---------------------------------------------------------------------------

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Python": ["python", "pandas", "numpy", "django", "flask", "fastapi"],
    "OOP": ["oop", "object oriented", "encapsulation", "inheritance", "polymorphism", "abstraction"],
    "SQL & Databases": ["sql", "nosql", "database", "postgresql", "mysql", "mongodb", "cassandra", "redis", "hbase", "acid"],
    "System Design": ["system design", "distributed system", "scalability", "load balancer", "cap theorem", "caching", "url shortener"],
    "Concurrency & OS": ["thread", "threads", "process", "processes", "race conditions", "concurrency", "synchronization", "memory"],
    "DevOps & CI/CD": ["ci/cd", "ci", "cd", "continuous integration", "continuous deployment", "docker", "kubernetes", "pipeline"],
    "Agile & Process": ["agile", "scrum", "sprint", "standup", "retrospective", "code review", "backlog"],
    "Soft Skills & HR": ["conflict", "stakeholder", "failure", "motivation", "weakness", "strength", "leadership", "communication", "time management", "deadlines"],
    "Web & Networking": ["http", "api", "rest", "redirect", "url", "hash", "base62"],
}


def extract_skills(text: str) -> List[str]:
    """Extract recognized skills and topics from a given text string.

    Parameters
    ----------
    text : str
        Input text (question, answer, job description, etc.)

    Returns
    -------
    List[str]
        List of matching canonical skill names from SKILL_TAXONOMY.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    found_skills: List[str] = []
    text_lower = text.lower()

    for canonical_skill, patterns in SKILL_TAXONOMY.items():
        for pattern in patterns:
            # Escape regex characters and match as word boundaries or exact tokens
            pattern_regex = r"(?:\b|_)" + re.escape(pattern.lower()) + r"(?:\b|_)"
            if re.search(pattern_regex, text_lower):
                if canonical_skill not in found_skills:
                    found_skills.append(canonical_skill)
                break

    return found_skills


def tag_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a DataFrame with an extracted 'Skills' column.

    Combines Question and Ideal_Answer to identify skills.

    Parameters
    ----------
    df : pd.DataFrame
        Input questions DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with an added or updated 'Skills' column.
    """
    df_copy = df.copy()

    def _get_skills_for_row(row: pd.Series) -> str:
        combined_text = f"{row.get('Question', '')} {row.get('Ideal_Answer', '')}"
        skills = extract_skills(combined_text)
        return ", ".join(skills) if skills else "General"

    df_copy["Skills"] = df_copy.apply(_get_skills_for_row, axis=1)
    return df_copy


# ---------------------------------------------------------------------------
# Filter and Matching Functions
# ---------------------------------------------------------------------------

def filter_questions(
    df: pd.DataFrame,
    role: Optional[str] = None,
    skills: Optional[Union[str, Iterable[str]]] = None,
    difficulty: Optional[Union[str, Iterable[str]]] = None,
    category: Optional[Union[str, Iterable[str]]] = None,
    keywords: Optional[Union[str, Iterable[str]]] = None,
    exclude_keywords: Optional[Union[str, Iterable[str]]] = None,
    match_all_skills: bool = False,
    include_general_role: bool = True,
) -> pd.DataFrame:
    """Filter interview questions by role, skills, difficulty, category, and keywords.

    Parameters
    ----------
    df : pd.DataFrame
        The questions DataFrame.
    role : str, optional
        Target role (e.g. 'Software Engineer', 'HR', 'General').
    skills : str or iterable of str, optional
        One or more skills/topics to match (e.g. 'Python', 'SQL', ['OOP', 'CI/CD']).
    difficulty : str or iterable of str, optional
        Difficulty level(s) (e.g. 'Easy', ['Medium', 'Hard']).
    category : str or iterable of str, optional
        Category name(s) (e.g. 'Technical', 'Behavioral').
    keywords : str or iterable of str, optional
        Free-text search keyword(s) matched against Question and Ideal_Answer.
    exclude_keywords : str or iterable of str, optional
        Exclude questions containing these keyword(s).
    match_all_skills : bool, default False
        If True, question must match ALL given skills (AND logic).
        If False, question matches if ANY given skill matches (OR logic).
    include_general_role : bool, default True
        If True, questions tagged as 'General' role are also included when filtering
        by a specific role.

    Returns
    -------
    pd.DataFrame
        Filtered questions DataFrame with a computed 'Relevance_Score' column.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()
    if "Skills" not in filtered.columns:
        filtered = tag_skills(filtered)

    # Initial score
    relevance_scores = pd.Series(1.0, index=filtered.index)

    # 1. Filter by Role
    if role:
        role_clean = str(role).strip().lower()
        if include_general_role and role_clean != "general":
            role_mask = filtered["Role"].str.lower().isin([role_clean, "general"])
        else:
            role_mask = filtered["Role"].str.lower() == role_clean
        filtered = filtered[role_mask]
        relevance_scores = relevance_scores[role_mask]

    # 2. Filter by Difficulty
    if difficulty:
        if isinstance(difficulty, str):
            diff_list = [difficulty.strip().lower()]
        else:
            diff_list = [str(d).strip().lower() for d in difficulty]
        diff_mask = filtered["Difficulty"].str.lower().isin(diff_list)
        filtered = filtered[diff_mask]
        relevance_scores = relevance_scores[diff_mask]

    # 3. Filter by Category
    if category:
        if isinstance(category, str):
            cat_list = [category.strip().lower()]
        else:
            cat_list = [str(c).strip().lower() for c in category]
        cat_mask = filtered["Category"].str.lower().isin(cat_list)
        filtered = filtered[cat_mask]
        relevance_scores = relevance_scores[cat_mask]

    # 4. Filter by Exclude Keywords
    if exclude_keywords:
        if isinstance(exclude_keywords, str):
            ex_list = [exclude_keywords.strip().lower()]
        else:
            ex_list = [str(k).strip().lower() for k in exclude_keywords if str(k).strip()]
        
        for ex in ex_list:
            ex_regex = re.escape(ex)
            q_has = filtered["Question"].str.contains(ex_regex, case=False, na=False)
            a_has = filtered["Ideal_Answer"].str.contains(ex_regex, case=False, na=False)
            not_excluded = ~(q_has | a_has)
            filtered = filtered[not_excluded]
            relevance_scores = relevance_scores[not_excluded]

    # 5. Filter / Score by Skills
    if skills:
        if isinstance(skills, str):
            skill_list = [skills.strip()]
        else:
            skill_list = [str(s).strip() for s in skills if str(s).strip()]

        if skill_list:
            skill_match_counts = pd.Series(0, index=filtered.index)
            for skill in skill_list:
                # Skill matching against Skills tag or Question/Answer text
                skill_regex = r"(?:\b|_)" + re.escape(skill.lower()) + r"(?:\b|_)"
                q_match = filtered["Question"].str.contains(skill_regex, case=False, na=False)
                a_match = filtered["Ideal_Answer"].str.contains(skill_regex, case=False, na=False)
                s_match = filtered["Skills"].str.contains(re.escape(skill), case=False, na=False)
                
                matched = q_match | a_match | s_match
                skill_match_counts += matched.astype(int)

            if match_all_skills:
                skill_mask = skill_match_counts >= len(skill_list)
            else:
                skill_mask = skill_match_counts > 0

            filtered = filtered[skill_mask]
            relevance_scores = relevance_scores[skill_mask] + (skill_match_counts[skill_mask] * 2.0)

    # 6. Filter / Score by Keywords
    if keywords:
        if isinstance(keywords, str):
            kw_list = [keywords.strip()]
        else:
            kw_list = [str(k).strip() for k in keywords if str(k).strip()]

        if kw_list:
            kw_match_counts = pd.Series(0, index=filtered.index)
            for kw in kw_list:
                kw_regex = re.escape(kw)
                q_match = filtered["Question"].str.contains(kw_regex, case=False, na=False)
                a_match = filtered["Ideal_Answer"].str.contains(kw_regex, case=False, na=False)
                matched = q_match | a_match
                # Question match is weighted more heavily
                kw_match_counts += (q_match.astype(int) * 2 + a_match.astype(int))

            kw_mask = kw_match_counts > 0
            filtered = filtered[kw_mask]
            relevance_scores = relevance_scores[kw_mask] + kw_match_counts[kw_mask]

    # Assign relevance score and sort
    filtered = filtered.copy()
    filtered["Relevance_Score"] = relevance_scores
    filtered = filtered.sort_values(by=["Relevance_Score", "Difficulty"], ascending=[False, True])

    return filtered


# ---------------------------------------------------------------------------
# Interview Set Generator
# ---------------------------------------------------------------------------

def generate_interview_set(
    df: pd.DataFrame,
    role: Optional[str] = None,
    skills: Optional[Union[str, Iterable[str]]] = None,
    difficulty: Optional[Union[str, Iterable[str]]] = None,
    category: Optional[Union[str, Iterable[str]]] = None,
    keywords: Optional[Union[str, Iterable[str]]] = None,
    num_questions: int = 5,
    balance_difficulty: bool = False,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a curated interview question set based on criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset of interview questions.
    role : str, optional
        Target role.
    skills : str or list of str, optional
        Skills to focus on.
    difficulty : str or list of str, optional
        Difficulty restriction.
    category : str or list of str, optional
        Category restriction.
    keywords : str or list of str, optional
        Search keywords.
    num_questions : int, default 5
        Target number of questions to generate.
    balance_difficulty : bool, default False
        If True, attempts to distribute questions across Easy, Medium, and Hard.
    random_state : int, optional
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Curated question set dataframe.
    """
    candidates = filter_questions(
        df=df,
        role=role,
        skills=skills,
        difficulty=difficulty,
        category=category,
        keywords=keywords,
    )

    if candidates.empty:
        return candidates

    if len(candidates) <= num_questions:
        return candidates.reset_index(drop=True)

    if balance_difficulty and not difficulty:
        # Balanced sampling across Easy, Medium, Hard
        difficulties = ["Easy", "Medium", "Hard"]
        per_diff = max(1, num_questions // len(difficulties))
        selected_dfs = []

        for diff in difficulties:
            diff_pool = candidates[candidates["Difficulty"] == diff]
            if not diff_pool.empty:
                sample_count = min(len(diff_pool), per_diff)
                selected_dfs.append(diff_pool.sample(n=sample_count, random_state=random_state))

        selected = pd.concat(selected_dfs) if selected_dfs else pd.DataFrame()
        
        # If we still need more questions to meet num_questions
        remaining_needed = num_questions - len(selected)
        if remaining_needed > 0:
            remaining_pool = candidates.drop(selected.index, errors="ignore")
            if not remaining_pool.empty:
                fill_count = min(len(remaining_pool), remaining_needed)
                fill_sample = remaining_pool.sample(n=fill_count, random_state=random_state)
                selected = pd.concat([selected, fill_sample])

        return selected.reset_index(drop=True)

    # Standard sampling based on top relevance or random sample
    # If relevance scores are diverse, sample from top candidates
    if candidates["Relevance_Score"].nunique() > 1:
        top_candidates = candidates.head(num_questions * 2)
        sampled = top_candidates.sample(n=min(len(top_candidates), num_questions), random_state=random_state)
    else:
        sampled = candidates.sample(n=min(len(candidates), num_questions), random_state=random_state)

    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting and QuestionFilter Class
# ---------------------------------------------------------------------------

def format_interview_set(questions_df: pd.DataFrame) -> str:
    """Format an interview question set into a clean, human-readable string."""
    if questions_df.empty:
        return "No questions found matching the given criteria."

    lines = [
        f"Generated Interview Question Set ({len(questions_df)} Question(s))",
        "=" * 68,
    ]

    for idx, row in questions_df.iterrows():
        q_num = idx + 1
        diff = row.get("Difficulty", "Medium")
        role = row.get("Role", "General")
        cat = row.get("Category", "General")
        skills = row.get("Skills", "General")
        q_text = row.get("Question", "")
        ans_text = row.get("Ideal_Answer", "")

        lines.append(f"\n[Question #{q_num}] [{diff}] | Role: {role} | Category: {cat}")
        lines.append(f"  Skills/Topics : {skills}")
        lines.append(f"  Question      : {q_text}")
        lines.append(f"  Ideal Answer  : {ans_text}")
        lines.append("-" * 68)

    return "\n".join(lines)


class QuestionFilter:
    """Convenience class to manage and filter a questions dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = tag_skills(df)

    def get_roles(self) -> List[str]:
        """Return a sorted list of unique roles."""
        return sorted(self.df["Role"].dropna().unique().tolist())

    def get_categories(self) -> List[str]:
        """Return a sorted list of unique categories."""
        return sorted(self.df["Category"].dropna().unique().tolist())

    def get_skills_catalog(self) -> List[str]:
        """Return list of supported skill catalog keys."""
        return list(SKILL_TAXONOMY.keys())

    def filter(
        self,
        role: Optional[str] = None,
        skills: Optional[Union[str, Iterable[str]]] = None,
        difficulty: Optional[Union[str, Iterable[str]]] = None,
        category: Optional[Union[str, Iterable[str]]] = None,
        keywords: Optional[Union[str, Iterable[str]]] = None,
        exclude_keywords: Optional[Union[str, Iterable[str]]] = None,
        match_all_skills: bool = False,
        include_general_role: bool = True,
    ) -> pd.DataFrame:
        """Filter questions using multi-criteria."""
        return filter_questions(
            df=self.df,
            role=role,
            skills=skills,
            difficulty=difficulty,
            category=category,
            keywords=keywords,
            exclude_keywords=exclude_keywords,
            match_all_skills=match_all_skills,
            include_general_role=include_general_role,
        )

    def generate_mock_interview(
        self,
        role: Optional[str] = None,
        skills: Optional[Union[str, Iterable[str]]] = None,
        difficulty: Optional[Union[str, Iterable[str]]] = None,
        category: Optional[Union[str, Iterable[str]]] = None,
        keywords: Optional[Union[str, Iterable[str]]] = None,
        num_questions: int = 5,
        balance_difficulty: bool = False,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a mock interview set."""
        return generate_interview_set(
            df=self.df,
            role=role,
            skills=skills,
            difficulty=difficulty,
            category=category,
            keywords=keywords,
            num_questions=num_questions,
            balance_difficulty=balance_difficulty,
            random_state=random_state,
        )
