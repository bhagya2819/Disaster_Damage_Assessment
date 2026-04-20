"""Canonical project paths resolved from the repo root.

Uses the location of this file as an anchor rather than CWD so scripts and
notebooks behave identically regardless of where they are invoked from.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("DDA_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
GT_DIR: Path = DATA_DIR / "gt"
EXTERNAL_DIR: Path = DATA_DIR / "external"

MODEL_DIR: Path = Path(os.environ.get("DDA_MODEL_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


def ensure_dirs() -> None:
    """Create all standard directories if they do not yet exist."""
    for d in (RAW_DIR, PROCESSED_DIR, GT_DIR, EXTERNAL_DIR, MODEL_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
