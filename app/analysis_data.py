"""Cached loaders for the analysis pages (quality, EDA, statistics, benchmark).

Everything read here is committed in ``reports/`` (small files), so these pages work from a fresh clone,
without the dataset and without the trained models of the detection page:

* ``reports/app_data/analysis_sample.parquet``  class-capped sample of the cleaned data (script 19)
* ``reports/inspection/``                        facts about the raw files (script 01)
* ``reports/eda/tables/``                        EDA tables on the full data (script 02)
* ``reports/stats/``                             hypothesis tests (script 19)
* ``reports/benchmark/``                         hermetic CV benchmark + saved best pipeline (script 20)
* ``reports/evaluation/``                        full-data evaluation of the 11 models (script 06)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import settings

REPORTS = settings.REPORTS_DIR
SAMPLE_FILE = REPORTS / "app_data" / "analysis_sample.parquet"
STATS_DIR = REPORTS / "stats"
BENCH_DIR = REPORTS / "benchmark"
EDA_TABLES = settings.EDA_TABLES_DIR
INSPECTION = settings.INSPECTION_REPORT_DIR
EVAL_DIR = settings.EVALUATION_REPORT_DIR

META_COLUMNS = (settings.LABEL_COLUMN, settings.SOURCE_FILE_COLUMN)


def missing_hint(path: Path, script: str) -> None:
    try:
        shown = path.relative_to(settings.PROJECT_ROOT)
    except ValueError:  # a path configured outside the project folder
        shown = path
    st.info(f"`{shown}` was not found. Run `python scripts\\{script}` from the project folder, then reload.")


@st.cache_data(show_spinner="Loading the analysis sample ...")
def load_sample() -> pd.DataFrame | None:
    if not SAMPLE_FILE.is_file():
        return None
    return pd.read_parquet(SAMPLE_FILE)


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


@st.cache_data
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.is_file() else None


@st.cache_data
def read_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_resource(show_spinner="Loading the benchmark pipeline ...")
def load_benchmark_pipeline():
    import joblib

    model_file, meta_file = BENCH_DIR / "best_pipeline.joblib", BENCH_DIR / "best_pipeline.json"
    if not (model_file.is_file() and meta_file.is_file()):
        return None, None
    return joblib.load(model_file), json.loads(meta_file.read_text(encoding="utf-8"))
