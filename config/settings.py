"""Central configuration for the Network Attack Detection project (CIC-IDS2017).

This is the single source of truth for paths, random seed, target mode,
split ratios and thresholds.  No other module should hardcode any of them.

Notes
-----
* The project root can be overridden with the environment variable
  ``NIDS_PROJECT_ROOT`` (used by the tests to work in a temporary folder).
  It must be set *before* this module is first imported.
* Values marked "verify at Checkpoint N" are assumptions about the dataset
  that will be confirmed (or changed) once the real files have been inspected.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and directories
# ---------------------------------------------------------------------------
_ROOT_OVERRIDE = os.environ.get("NIDS_PROJECT_ROOT")
PROJECT_ROOT: Path = (
    Path(_ROOT_OVERRIDE).resolve() if _ROOT_OVERRIDE else Path(__file__).resolve().parents[1]
)

# Data flow: raw -> interim -> cleaned -> preprocessed (one direction only)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Trained models
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
TUNED_MODELS_DIR = MODELS_DIR / "tuned"

# Generated reports
REPORTS_DIR = PROJECT_ROOT / "reports"
INSPECTION_REPORT_DIR = REPORTS_DIR / "inspection"
EDA_DIR = REPORTS_DIR / "eda"
EDA_FIGURES_DIR = EDA_DIR / "figures"
EDA_TABLES_DIR = EDA_DIR / "tables"
EVALUATION_REPORT_DIR = REPORTS_DIR / "evaluation"
OPTIMIZATION_REPORT_DIR = REPORTS_DIR / "optimization"
# Data mining extensions (scripts 09-17)
MINING_REPORT_DIR = REPORTS_DIR / "mining"
EXPERIMENTS_REPORT_DIR = REPORTS_DIR / "experiments"
TEST_PREDICTIONS_DIR = EXPERIMENTS_REPORT_DIR / "test_predictions"

LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# File names inside the data folders
# ---------------------------------------------------------------------------
CLEANED_DATA_FILE = CLEANED_DATA_DIR / "cleaned.parquet"
CLEANING_REPORT_FILE = CLEANED_DATA_DIR / "cleaning_report.json"

X_TRAIN_FILE = PREPROCESSED_DIR / "X_train.npy"
X_TEST_FILE = PREPROCESSED_DIR / "X_test.npy"
Y_TRAIN_FILE = PREPROCESSED_DIR / "y_train.npy"
Y_TEST_FILE = PREPROCESSED_DIR / "y_test.npy"
PIPELINE_FILE = PREPROCESSED_DIR / "preprocessing_pipeline.joblib"
LABEL_ENCODER_FILE = PREPROCESSED_DIR / "label_encoder.joblib"
FEATURE_NAMES_FILE = PREPROCESSED_DIR / "feature_names.json"
FEATURE_SCHEMA_FILE = PREPROCESSED_DIR / "feature_schema.json"
SAMPLE_FLOWS_FILE = PREPROCESSED_DIR / "sample_flows.parquet"
PREPROCESSING_METADATA_FILE = PREPROCESSED_DIR / "preprocessing_metadata.json"

LOG_FILE_NAME = "nids.log"
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# Dataset description (verify at Checkpoint 1 on the real files)
# ---------------------------------------------------------------------------
RAW_FILE_PATTERN = "*.csv"          # only CSV files are loaded; other files are ignored
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
MALICIOUS_LABEL = "MALICIOUS"       # used only in binary mode
SOURCE_FILE_COLUMN = "source_file"  # added by the loader; metadata, never a feature
FLOAT_DTYPE = "float32"             # memory-friendly dtype for numeric columns (16 GB RAM)

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
TARGET_MODE = "multiclass"          # "multiclass" (BENIGN + each attack) or "binary"
ALLOWED_TARGET_MODES = ("multiclass", "binary")

# ---------------------------------------------------------------------------
# Reproducibility and splitting
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20                    # chosen by the user (20 % held out for testing)
CV_FOLDS = 5                        # StratifiedKFold folds (PDF uses 5 and 10)

# Rare-class handling (final decision at Checkpoint 2, after real counts are known).
# Technical minimum rows per class: a stratified split needs at least 2, and
# StratifiedKFold needs at least CV_FOLDS rows of the class in the training part.
MIN_CLASS_SAMPLES = max(2, math.ceil(CV_FOLDS / (1.0 - TEST_SIZE)))
RARE_CLASS_POLICY = "keep"          # "keep" or "merge" (merge classes below the minimum)
ALLOWED_RARE_CLASS_POLICIES = ("keep", "merge")
RARE_CLASS_MERGE_LABEL = "Other_Attack"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMPUTER_STRATEGY = "median"         # flow features are skewed; median is stable. No blanket zero-fill
SCALER_TYPE = "standard"            # "standard", "robust" or "minmax"
ALLOWED_SCALER_TYPES = ("standard", "robust", "minmax")
# Checkpoint 2 decision: signed log1p (sign(x) * log1p(|x|)) before the feature filter and the
# scaler, because the features are extremely skewed. Pipeline order: impute -> log -> filter -> scale.
LOG_TRANSFORM = True
CORRELATION_THRESHOLD = 0.95        # drop one feature of each pair above this; revisit after the EDA

# Features removed on purpose.  Format: {"column name": "technical justification"}.
# Empty until Checkpoint 1; every exclusion MUST have a written reason.
EXCLUDED_FEATURES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Cleaning (rules decided at Checkpoint 1 from the real inspection report;
# the resulting numbers are confirmed at Checkpoint 2)
# ---------------------------------------------------------------------------
# Explicit label fixes (whole label -> new label).  The character U+FFFD is
# already stored in the CSV files (the original character is lost), so it is
# replaced by a plain hyphen in the three Web Attack labels seen in the data.
LABEL_MAP: dict[str, str] = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection",
}

# Columns where a negative value is impossible.  Format: {"column": "reason"}.
# Chosen from the inspection report (negative counts per column).  The two
# Init_Win_bytes columns are NOT listed: their -1 behaves like a placeholder code.
NON_NEGATIVE_COLUMNS: dict[str, str] = {
    "Flow Duration": "a duration cannot be negative",
    "Flow Bytes/s": "a rate cannot be negative",
    "Flow Packets/s": "a rate cannot be negative",
    "Flow IAT Mean": "an inter-arrival time cannot be negative",
    "Flow IAT Max": "an inter-arrival time cannot be negative",
    "Flow IAT Min": "an inter-arrival time cannot be negative",
    "Fwd IAT Min": "an inter-arrival time cannot be negative",
    "Fwd Header Length": "a header length cannot be negative",
    "Bwd Header Length": "a header length cannot be negative",
    "Fwd Header Length.1": "a header length cannot be negative",
    "min_seg_size_forward": "a segment size cannot be negative",
}
NEGATIVE_VALUE_POLICY = "to_nan"    # "to_nan" (set to missing, imputed later) or "keep" (report only)
ALLOWED_NEGATIVE_VALUE_POLICIES = ("to_nan", "keep")

DROP_IDENTICAL_COLUMNS = True       # lossless: a column identical to an earlier one adds nothing

# Identical feature values with different labels (Checkpoint 2 decision).
CONFLICT_POLICY = "keep"            # "keep" (report only) or "drop" (remove all rows of such vectors)
ALLOWED_CONFLICT_POLICIES = ("keep", "drop")

# ---------------------------------------------------------------------------
# Training and hardware (Ryzen 7 6800H: 16 threads, 16 GB RAM, RTX 3060)
# ---------------------------------------------------------------------------
N_JOBS = max(1, (os.cpu_count() or 2) - 2)   # chosen by the user: all threads but 2
XGBOOST_DEVICE = "auto"             # "auto" (detect GPU, else CPU), "cuda" or "cpu"
ALLOWED_XGBOOST_DEVICES = ("auto", "cuda", "cpu")

# Class weighting experiment (off for the baselines; see Deliverable 6)
CLASS_WEIGHT_MODE = "none"          # "none" or "balanced" (where the algorithm supports it)
ALLOWED_CLASS_WEIGHT_MODES = ("none", "balanced")

# Models that are too slow to train on millions of rows use a class-capped
# stratified subsample.  These are STARTING VALUES, to be calibrated by timing
# at roadmap step 47.
SUBSAMPLE_SIZES: dict[str, int] = {
    "knn": 100_000,
    "svm": 50_000,
    "gradient_boosting": 100_000,
}

# ---------------------------------------------------------------------------
# Evaluation and voting
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "f1_macro"         # model comparison and GridSearchCV scoring
# Tie between classes in the majority vote:
#   "prefer_non_benign": prefer an attack class over BENIGN, then the lowest class index
#   "lowest_index":      lowest encoded class index only
TIE_BREAK_RULE = "prefer_non_benign"
ALLOWED_TIE_BREAK_RULES = ("prefer_non_benign", "lowest_index")

# ---------------------------------------------------------------------------
# Data mining extensions (unsupervised mining and extra experiments, scripts 09-17)
# None of these values changes data/preprocessed/, so they are NOT part of preprocessing_config().
# ---------------------------------------------------------------------------
MINING_SAMPLE_SIZE = 60_000         # class-capped training sample for clustering and rule mining
PROJECTION_SAMPLE_SIZE = 15_000     # class-capped sample for the PCA / t-SNE maps (t-SNE is O(n log n))
CLUSTER_K_VALUES = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
SILHOUETTE_SAMPLE_SIZE = 10_000     # silhouette is O(n^2); computed on a random subset
DBSCAN_SAMPLE_SIZE = 20_000

ANOMALY_BENIGN_TRAIN_SIZE = 300_000  # BENIGN training rows used to fit the anomaly detectors
ANOMALY_FALSE_ALARM_TARGETS = (0.001, 0.01, 0.05)  # thresholds set on held-out BENIGN training rows

RULE_BINS = 3                       # quantile bins per feature for association rules (low / mid / high)
RULE_MIN_SUPPORT = 0.002            # FP-Growth minimum support (fraction of transactions)
RULE_MIN_CONFIDENCE = 0.8
RULE_MAX_LEN = 4

IMPORTANCE_SAMPLE_SIZE = 50_000     # class-capped TEST slice for permutation importance
IMPORTANCE_REPEATS = 5

# Attack families for the coarse-label experiment (every class must be mapped).
ATTACK_FAMILIES: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_settings() -> None:
    """Fail early, with a clear message, if a setting has an invalid value."""

    def check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise ValueError(f"{name}={value!r} is invalid; allowed values: {allowed}")

    check_choice("TARGET_MODE", TARGET_MODE, ALLOWED_TARGET_MODES)
    check_choice("RARE_CLASS_POLICY", RARE_CLASS_POLICY, ALLOWED_RARE_CLASS_POLICIES)
    check_choice("SCALER_TYPE", SCALER_TYPE, ALLOWED_SCALER_TYPES)
    check_choice("XGBOOST_DEVICE", XGBOOST_DEVICE, ALLOWED_XGBOOST_DEVICES)
    check_choice("CLASS_WEIGHT_MODE", CLASS_WEIGHT_MODE, ALLOWED_CLASS_WEIGHT_MODES)
    check_choice("TIE_BREAK_RULE", TIE_BREAK_RULE, ALLOWED_TIE_BREAK_RULES)
    check_choice("NEGATIVE_VALUE_POLICY", NEGATIVE_VALUE_POLICY, ALLOWED_NEGATIVE_VALUE_POLICIES)
    check_choice("CONFLICT_POLICY", CONFLICT_POLICY, ALLOWED_CONFLICT_POLICIES)
    if not 0.0 < TEST_SIZE < 1.0:
        raise ValueError(f"TEST_SIZE must be between 0 and 1 (exclusive), got {TEST_SIZE}")
    if CV_FOLDS < 2:
        raise ValueError(f"CV_FOLDS must be at least 2, got {CV_FOLDS}")
    if not 0.0 < CORRELATION_THRESHOLD <= 1.0:
        raise ValueError(f"CORRELATION_THRESHOLD must be in (0, 1], got {CORRELATION_THRESHOLD}")
    if not isinstance(LOG_TRANSFORM, bool):
        raise ValueError(f"LOG_TRANSFORM must be True or False, got {LOG_TRANSFORM!r}")
    if N_JOBS < 1:
        raise ValueError(f"N_JOBS must be at least 1, got {N_JOBS}")
    for model_key, size in SUBSAMPLE_SIZES.items():
        if size < 1:
            raise ValueError(f"SUBSAMPLE_SIZES[{model_key!r}] must be positive, got {size}")
    for column, reason in EXCLUDED_FEATURES.items():
        if not reason.strip():
            raise ValueError(f"EXCLUDED_FEATURES[{column!r}] needs a technical justification")
    for column, reason in NON_NEGATIVE_COLUMNS.items():
        if not reason.strip():
            raise ValueError(f"NON_NEGATIVE_COLUMNS[{column!r}] needs a technical justification")
    for old, new in LABEL_MAP.items():
        if not new.strip():
            raise ValueError(f"LABEL_MAP[{old!r}] maps to an empty label")


def all_directories() -> list[Path]:
    """Every directory the project writes into (not including the source tree)."""
    return [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        CLEANED_DATA_DIR,
        PREPROCESSED_DIR,
        TRAINED_MODELS_DIR,
        TUNED_MODELS_DIR,
        INSPECTION_REPORT_DIR,
        EDA_FIGURES_DIR,
        EDA_TABLES_DIR,
        EVALUATION_REPORT_DIR,
        OPTIMIZATION_REPORT_DIR,
        MINING_REPORT_DIR,
        EXPERIMENTS_REPORT_DIR,
        TEST_PREDICTIONS_DIR,
        LOGS_DIR,
    ]


def ensure_directories() -> list[Path]:
    """Create every output directory that does not exist yet.

    Returns the list of directories that were actually created.
    """
    created: list[Path] = []
    for directory in all_directories():
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created


def preprocessing_config() -> dict:
    """Settings that change the content of ``data/preprocessed/``.

    ``preprocessing_metadata.json`` stores a hash of this dictionary, so changing any
    of these values makes saved preprocessed data (and models trained on it) stale.
    """
    return {
        "target_mode": TARGET_MODE,
        "label_column": LABEL_COLUMN,
        "benign_label": BENIGN_LABEL,
        "malicious_label": MALICIOUS_LABEL,
        "source_file_column": SOURCE_FILE_COLUMN,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "min_class_samples": MIN_CLASS_SAMPLES,
        "rare_class_policy": RARE_CLASS_POLICY,
        "rare_class_merge_label": RARE_CLASS_MERGE_LABEL,
        "imputer_strategy": IMPUTER_STRATEGY,
        "scaler_type": SCALER_TYPE,
        "log_transform": LOG_TRANSFORM,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "excluded_features": dict(sorted(EXCLUDED_FEATURES.items())),
        "float_dtype": FLOAT_DTYPE,
    }


def cleaning_config() -> dict:
    """Settings that change the content of ``data/cleaned/``.

    The cleaning script stores this dictionary (and its hash) next to the cleaned
    data, so later stages can detect cleaned data produced with other rules.
    """
    return {
        "label_column": LABEL_COLUMN,
        "source_file_column": SOURCE_FILE_COLUMN,
        "label_map": dict(sorted(LABEL_MAP.items())),
        "non_negative_columns": dict(sorted(NON_NEGATIVE_COLUMNS.items())),
        "negative_value_policy": NEGATIVE_VALUE_POLICY,
        "drop_identical_columns": DROP_IDENTICAL_COLUMNS,
        "conflict_policy": CONFLICT_POLICY,
        "min_class_samples": MIN_CLASS_SAMPLES,
    }


_validate_settings()


if __name__ == "__main__":  # python -m config.settings
    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"RAW_DATA_DIR      : {RAW_DATA_DIR}  (exists: {RAW_DATA_DIR.exists()})")
    print(f"TARGET_MODE       : {TARGET_MODE}")
    print(f"RANDOM_SEED       : {RANDOM_SEED}")
    print(f"TEST_SIZE         : {TEST_SIZE}")
    print(f"CV_FOLDS          : {CV_FOLDS}")
    print(f"MIN_CLASS_SAMPLES : {MIN_CLASS_SAMPLES}")
    print(f"N_JOBS            : {N_JOBS}")
    print(f"SCALER_TYPE       : {SCALER_TYPE}")
    print(f"XGBOOST_DEVICE    : {XGBOOST_DEVICE}")
    print(f"Directories to create: {len(all_directories())}")