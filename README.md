# AI-Powered Interview Question Generator

A simple Python project that will generate interview questions based on job role, skills, difficulty level, and question type.

## Project Structure
```
AI-Powered Interview Question Generator/
├─ data/
│   ├─ sample_questions.csv      ← 20-question starter dataset
│   └─ cleaned_questions.csv     ← Day 3 & 4: cleaned & skill-tagged dataset
├─ src/
│   ├─ __init__.py
│   ├─ main.py                   ← entry point
│   ├─ data_loader.py            ← Day 2: load & validate CSV
│   ├─ data_cleaner.py           ← Day 3: text cleaning, EDA & metrics
│   ├─ question_filter.py        ← Day 4: keyword & skill filtering / generator
│   └─ download_kaggle.py        ← optional Kaggle downloader
├─ tests/
│   ├─ test_cleaner.py           ← Day 3: test suite for cleaning & EDA
│   └─ test_filter.py            ← Day 4: test suite for filtering & skills
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Dataset

The project ships with a **20‑question sample** (`data/sample_questions.csv`).

| Column | Description |
| --- | --- |
| `Question` | The interview question |
| `Ideal_Answer` | A recommended/ideal response |
| `Category` | Topic area (Technical, Behavioral, Motivation, …) |
| `Role` | Target role (General, Software Engineer, HR, …) |
| `Difficulty` | Easy · Medium · Hard |
| `Question_Words` | (Cleaned) Word count of the question |
| `Answer_Words` | (Cleaned) Word count of the ideal answer |
| `Skills` | (Day 4) Extracted technical & soft skill tags |

### Using the full Kaggle dataset (250 K questions)

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/`
3. Run:
   ```powershell
   python -m src.download_kaggle
   ```
   The full CSV will be saved to `data/`.

## Quick Start (Windows)
```powershell
# 1️⃣ Create a virtual environment
python -m venv venv

# 2️⃣ Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the project
python -m src.main

# 5️⃣ Run unit tests
python tests/test_cleaner.py
python tests/test_filter.py
```

You should see an exploratory data analysis (EDA) report, text cleaning summary, skill tagging report, and sample filtered mock interview questions printed to the console.

---

## Progress

| Day | Task | Status |
| --- | --- | --- |
| 1 | Project skeleton | ✅ |
| 2 | Load an interview‑question dataset | ✅ |
| 3 | Explore & clean the data (EDA, text normalization, deduplication, metrics) | ✅ |
| 4 | Keyword & Skill-based Question Filtering / Categorization | ✅ |
| 5 | Advanced Preprocessing / TF-IDF & Embeddings Preparation | ⬜ |
