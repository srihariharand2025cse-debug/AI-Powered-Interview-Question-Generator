"""
download_kaggle.py — Helper to download the full Kaggle dataset.

Usage
-----
1. Install the Kaggle CLI:       pip install kaggle
2. Set up your API credentials:  https://www.kaggle.com/docs/api
3. Run this script:              python -m src.download_kaggle

The dataset will be saved to data/interview_questions.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DATASET_SLUG = "aryankumardudeja/hr-interview-questions-and-ideal-answers"


def download() -> Path:
    """Download the Kaggle dataset to the data/ directory.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        print(
            "❌ kaggle package is not installed.\n"
            "   Run: pip install kaggle\n"
            "   Then configure your API token: "
            "https://www.kaggle.com/docs/api#authentication"
        )
        sys.exit(1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading dataset '{_DATASET_SLUG}' …")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        _DATASET_SLUG,
        path=str(_DATA_DIR),
        unzip=True,
    )

    # Find the downloaded CSV
    csv_files = list(_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found after download.")
        sys.exit(1)

    target = csv_files[0]
    print(f"✅ Dataset saved to {target}")
    return target


if __name__ == "__main__":
    download()
