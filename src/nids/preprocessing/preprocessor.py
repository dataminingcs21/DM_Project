"""Run the whole preprocessing: target, split, pipeline, schema, saved files.

``preprocess(df)`` takes the **cleaned** data (File 18) and returns a
:class:`PreprocessingResult` in memory.  ``save_preprocessed(result)`` writes it
to ``data/preprocessed/``.  The script ``04_preprocess.py`` (File 27) calls both.

Order of the work (no step sees the test rows before it must):

1. feature columns (``EXCLUDED_FEATURES`` removed) and the target + codec (File 19);
2. stratified split (File 20), by row positions;
3. the pipeline (File 23) is **fitted on the training rows only** and applied to
   the test rows;
4. the feature schema (File 24) is computed from the raw training rows;
5. a few real **test** flows per class are kept as examples for the interface.

Outputs are float32 (``settings.FLOAT_DTYPE``) arrays for X and int64 for y.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import settings
from nids.data import schema as data_schema
from nids.preprocessing.feature_schema import build_feature_schema, save_feature_schema
from nids.preprocessing.pipeline_builder import build_pipeline, describe_pipeline
from nids.preprocessing.splitter import split_report, stratified_split
from nids.preprocessing.target_builder import LabelCodec, build_target, make_codec
from nids.utils.io_utils import (
    config_hash,
    dump_joblib,
    read_json,
    save_array,
    write_json,
    write_parquet,
)
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.preprocessor")

SAMPLE_FLOWS_PER_CLASS = 5


@dataclass
class PreprocessingResult:
    """Everything produced by :func:`preprocess` (kept in memory until saved)."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: Pipeline
    codec: LabelCodec
    schema: dict[str, Any]
    train_idx: np.ndarray
    test_idx: np.ndarray
    sample_flows: pd.DataFrame
    report: dict[str, Any]
    split_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def feature_names(self) -> list[str]:
        return list(self.schema["model_feature_names"])


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns of the cleaned data, without the ones excluded in the settings."""
    if settings.LABEL_COLUMN not in df.columns:
        raise ValueError(f"label column '{settings.LABEL_COLUMN}' not found in the data")
    cols = [c for c in data_schema.get_feature_columns(df.columns) if c not in settings.EXCLUDED_FEATURES]
    if not cols:
        raise ValueError("no feature columns found")
    unknown = sorted(set(settings.EXCLUDED_FEATURES) - set(df.columns))
    if unknown:
        logger.warning("EXCLUDED_FEATURES names not found in the data: %s", unknown)
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns (use the cleaned data): {non_numeric[:5]}")
    return cols


def pick_sample_flows(
    raw_test: pd.DataFrame, test_target: pd.Series, per_class: int = SAMPLE_FLOWS_PER_CLASS
) -> pd.DataFrame:
    """A few real **test** flows per class (raw feature values + class name), for the interface.

    ``raw_test`` and ``test_target`` must share a 0..n-1 index (the row inside X_test).
    Deterministic (fixed seed). These are real rows of the dataset, nothing is invented.
    """
    parts = []
    for name, group in test_target.groupby(test_target, sort=True):
        take = group.sample(n=min(per_class, len(group)), random_state=settings.RANDOM_SEED).index
        parts.append(take)
    chosen = np.sort(np.concatenate([np.asarray(p) for p in parts]))
    out = raw_test.loc[chosen].copy()
    out[settings.LABEL_COLUMN] = test_target.loc[chosen].to_numpy()
    out["test_row_position"] = chosen
    return out.reset_index(drop=True)


def _to_float_array(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=settings.FLOAT_DTYPE, copy=False)


def preprocess(df: pd.DataFrame) -> PreprocessingResult:
    """Run the preprocessing on the cleaned data. Does not write anything."""
    started = time.time()
    feature_cols = select_feature_columns(df)
    logger.info("Preprocessing %d rows, %d feature columns", len(df), len(feature_cols))

    # 1. target and codec
    target, target_report = build_target(df[settings.LABEL_COLUMN])
    codec = make_codec(target)
    y_all = codec.encode(target)

    # 2. split (positions)
    train_idx, test_idx = stratified_split(target)
    split_summary, split_table = split_report(target, train_idx, test_idx)

    # 3. pipeline: fit on the training rows only
    raw_train = df[feature_cols].iloc[train_idx].astype(settings.FLOAT_DTYPE)
    pipeline = build_pipeline()
    X_train_df = pipeline.fit_transform(raw_train)
    X_train = _to_float_array(X_train_df)
    del X_train_df

    # 4. schema (raw training rows + fitted pipeline), then free the raw training copy
    feature_schema = build_feature_schema(raw_train, pipeline)
    del raw_train

    raw_test = df[feature_cols].iloc[test_idx].astype(settings.FLOAT_DTYPE).reset_index(drop=True)
    X_test = _to_float_array(pipeline.transform(raw_test))

    # 5. example flows (real test rows). ``test_row_position`` is the row inside X_test,
    #    ``cleaned_row_position`` the row inside the cleaned data.
    test_target = pd.Series(target.to_numpy()[test_idx])
    samples = pick_sample_flows(raw_test, test_target)
    samples["cleaned_row_position"] = test_idx[samples["test_row_position"].to_numpy()]
    del raw_test

    for name, array in (("X_train", X_train), ("X_test", X_test)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains missing or infinite values after preprocessing")

    y_train, y_test = y_all[train_idx], y_all[test_idx]
    class_counts = pd.DataFrame(
        {
            "class_code": np.arange(codec.n_classes),
            "class": list(codec.classes),
            "train": np.bincount(y_train, minlength=codec.n_classes),
            "test": np.bincount(y_test, minlength=codec.n_classes),
        }
    )
    report = {
        "n_rows": int(len(df)),
        "n_raw_features": len(feature_cols),
        "n_model_features": int(X_train.shape[1]),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "dtype": str(X_train.dtype),
        "target": target_report,
        "split": split_summary,
        "pipeline": describe_pipeline(pipeline),
        "class_codes": codec.to_dict(),
        "class_counts": class_counts.to_dict(orient="records"),
        "n_sample_flows": int(len(samples)),
        "seconds": round(time.time() - started, 1),
    }
    logger.info("Preprocessing done in %.1f s: X_train %s, X_test %s", report["seconds"], X_train.shape, X_test.shape)
    return PreprocessingResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, pipeline=pipeline, codec=codec,
        schema=feature_schema, train_idx=train_idx, test_idx=test_idx, sample_flows=samples, report=report,
        split_table=split_table,
    )


# ---------------------------------------------------------------------------
# Saving and freshness
# ---------------------------------------------------------------------------
def save_preprocessed(result: PreprocessingResult) -> dict[str, Any]:
    """Write every artifact to ``data/preprocessed/`` and return the metadata that was saved."""
    settings.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_array(result.X_train, settings.X_TRAIN_FILE)
    save_array(result.X_test, settings.X_TEST_FILE)
    save_array(result.y_train, settings.Y_TRAIN_FILE)
    save_array(result.y_test, settings.Y_TEST_FILE)
    dump_joblib(result.pipeline, settings.PIPELINE_FILE)
    dump_joblib(result.codec, settings.LABEL_ENCODER_FILE)
    save_feature_schema(result.schema)
    write_parquet(result.sample_flows, settings.SAMPLE_FLOWS_FILE)

    cleaning_hash = None
    if settings.CLEANING_REPORT_FILE.is_file():
        cleaning_hash = read_json(settings.CLEANING_REPORT_FILE).get("config_hash")
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(settings.preprocessing_config()),
        "config": settings.preprocessing_config(),
        "cleaned_data_config_hash": cleaning_hash,
        **result.report,
    }
    write_json(metadata, settings.PREPROCESSING_METADATA_FILE)  # written last: its presence means "complete"
    logger.info("Preprocessed data saved to %s", settings.PREPROCESSED_DIR)
    return metadata


def load_preprocessing_metadata() -> dict[str, Any]:
    return read_json(settings.PREPROCESSING_METADATA_FILE, hint="run scripts/04_preprocess.py first")


def check_preprocessed_fresh() -> tuple[bool, str]:
    """Compare the saved preprocessing with the current settings.

    Returns ``(True, "")`` when they agree, else ``(False, reason)``.  Used by the
    training stage to refuse stale data instead of silently training on it.
    """
    if not settings.PREPROCESSING_METADATA_FILE.is_file():
        return False, "no preprocessed data found; run scripts/04_preprocess.py"
    saved = load_preprocessing_metadata().get("config_hash")
    current = config_hash(settings.preprocessing_config())
    if saved != current:
        return False, "the preprocessing settings changed since the data was preprocessed; rerun scripts/04_preprocess.py"
    return True, ""