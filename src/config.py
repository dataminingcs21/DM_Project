"""Central configuration for the CIC-IDS2017 SVM experiments.

Exposes:
    ROOT           -- project root directory (contains src/, dataset/, configs/, ...)
    CONFIGS_DIR    -- configs/ directory
    MODELS_DIR     -- models/ directory (joblib artefacts)
    RESULTS_DIR    -- results/ directory
    TABLES_DIR     -- results/tables/ (experiments.csv lives here)
    REPORTS_DIR    -- reports/ (figures)
    CACHE_DIR      -- cache/ (cleaned datasets)

    load_config()  -- returns the run configuration dict
    ensure_dirs()  -- creates all output directories
"""
from __future__ import annotations

from pathlib import Path
import copy


# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent  # .../DM_Project/DM_Project

DATASET_DIR  = ROOT / "dataset"
CONFIGS_DIR  = ROOT / "configs"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"
TABLES_DIR   = RESULTS_DIR / "tables"
REPORTS_DIR  = ROOT / "reports"
CACHE_DIR    = ROOT / "cache"


# --------------------------------------------------------------------------- default config
_DEFAULT_CONFIG = {
    "seed": 42,
    "task": "binary",                 # binary | multiclass | family
    "imbalance": "none",              # none | class_weight | undersample | smote
    "max_rows": None,                 # global cap on rows (debug)
    "train_max_rows": None,           # cap on training rows
    "cv_max_rows": None,              # cap on rows used during tuning
    "test_size": 0.3,
    "dataset_dir": str(DATASET_DIR),
    "model": {
        "name": "linear_svm",         # linear_svm | svm_nystroem | svm_rbf
        "params": {},
    },
    "features": {
        "top_k": None,                # keep k most important features (RF importance)
        "pca": None,                  # float (variance) or int (n_components) or None
        "lda": None,                  # int (n_components) or None
    },
    "preprocessing": {
        "zero_threshold": 99.0,       # drop columns with >= X % zeros
    },
    "cv": {
        "n_splits": 3,
        "scoring": "f1",
    },
    "run_name": "run",
}


# --------------------------------------------------------------------------- API
def load_config() -> dict:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(_DEFAULT_CONFIG)


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (MODELS_DIR, RESULTS_DIR, TABLES_DIR, REPORTS_DIR, CACHE_DIR, CONFIGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
