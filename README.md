# AI-Powered Interview Question Generator

A simple Python project that will generate interview questions based on job role, skills, difficulty level, and question type.

## Project Structure
```
AI-Powered Interview Question Generator/
├─ data/
│   └─ sample_questions.csv      ← 20-question starter dataset
├─ src/
│   ├─ __init__.py
│   ├─ main.py                   ← entry point
│   ├─ data_loader.py            ← Day 2: load & validate CSV
│   └─ download_kaggle.py        ← optional Kaggle downloader
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
```

You should see a dataset summary and sample questions printed to the console.

---

## Progress

| Day | Task | Status |
| --- | --- | --- |
| 1 | Project skeleton | ✅ |
| 2 | Load an interview‑question dataset | ✅ |
| 3‑5 | Explore and clean the data | ⬜ |
