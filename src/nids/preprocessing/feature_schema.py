"""Describe the features once, so every later stage uses the same list.

The schema records, from the **training rows only**:

* the raw input columns (what the pipeline expects, in order) and their typical
  values (minimum, 1st percentile, median, 99th percentile, maximum, whether the
  values are whole numbers, how many were missing);
* which columns the models actually use after the pipeline (``used_by_models``)
  and, for the others, why they were dropped (constant, or correlated with a
  kept column);
* the final column names the models receive.

It is used by the input validator (File 56) and the Streamlit form (Files 62-63).
Nothing in it is invented: every number comes from the training data, every
"reason" from the fitted ``FeatureFilter``.  Values outside the training range
are a reason for a *warning* later, never proof of an attack.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import settings
from nids.utils.io_utils import read_json, write_json
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.feature_schema")

SCHEMA_VERSION = 1
_STAT_KEYS = ("min", "p01", "median", "p99", "max")


def _column_stats(values: np.ndarray) -> dict[str, Any]:
    """Statistics of one column over its finite values (``None`` when there are none)."""
    finite = values[np.isfinite(values)]
    stats: dict[str, Any] = {
        "n_missing_or_infinite_in_train": int(values.size - finite.size),
        "all_whole_numbers": bool(finite.size > 0 and np.all(finite == np.round(finite))),
    }
    if finite.size == 0:
        stats.update({key: None for key in _STAT_KEYS})
    else:
        p01, median, p99 = np.percentile(finite, [1, 50, 99])
        stats.update(
            {"min": float(finite.min()), "p01": float(p01), "median": float(median), "p99": float(p99), "max": float(finite.max())}
        )
    return stats


def build_feature_schema(X_train: pd.DataFrame, pipeline: Pipeline) -> dict[str, Any]:
    """Build the schema from the raw training features and the pipeline fitted on them."""
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a DataFrame with the raw feature columns")
    feature_filter = pipeline.named_steps.get("filter")
    if feature_filter is None or not hasattr(feature_filter, "support_"):
        raise ValueError("the pipeline must be fitted (it has no fitted 'filter' step)")

    raw_names = [str(c) for c in X_train.columns]
    if raw_names != [str(n) for n in feature_filter.feature_names_in_]:
        raise ValueError("X_train columns differ from the columns the pipeline was fitted on")

    model_names = [str(n) for n in pipeline.get_feature_names_out()]
    constant = set(feature_filter.constant_features_)
    correlated = feature_filter.correlated_features_

    features = []
    for position, name in enumerate(raw_names):
        used = name in set(model_names)
        if used:
            reason = None
        elif name in constant:
            reason = "constant in the training data"
        elif name in correlated:
            info = correlated[name]
            reason = f"correlated with '{info['kept_instead']}' (r = {info['correlation']})"
        else:  # cannot happen with a consistent pipeline; stated honestly if it ever does
            reason = "dropped by the pipeline (reason not recorded)"
        non_negative_reason = settings.NON_NEGATIVE_COLUMNS.get(name)
        features.append(
            {
                "name": name,
                "position": position,
                "used_by_models": used,
                "dropped_because": reason,
                "must_be_non_negative": non_negative_reason is not None,
                "non_negative_reason": non_negative_reason,
                **_column_stats(X_train[name].to_numpy(dtype="float64", copy=False)),
            }
        )

    schema = {
        "schema_version": SCHEMA_VERSION,
        "computed_on": "training rows only",
        "n_train_rows": int(len(X_train)),
        "n_raw_features": len(raw_names),
        "n_model_features": len(model_names),
        "raw_feature_names": raw_names,
        "model_feature_names": model_names,
        "features": features,
    }
    validate_schema(schema)
    logger.info("Feature schema built: %d raw features, %d used by the models", len(raw_names), len(model_names))
    return schema


def validate_schema(schema: dict[str, Any]) -> None:
    """Raise ValueError if the schema is internally inconsistent."""
    required = ("raw_feature_names", "model_feature_names", "features", "n_train_rows")
    missing = [k for k in required if k not in schema]
    if missing:
        raise ValueError(f"schema is missing keys: {missing}")
    raw, model, feats = schema["raw_feature_names"], schema["model_feature_names"], schema["features"]
    if len(set(raw)) != len(raw):
        raise ValueError("duplicate raw feature names")
    if [f["name"] for f in feats] != raw:
        raise ValueError("'features' does not match 'raw_feature_names' (names or order)")
    if not set(model) <= set(raw):
        raise ValueError(f"model features not among the raw features: {sorted(set(model) - set(raw))}")
    if [f["name"] for f in feats if f["used_by_models"]] != model:
        raise ValueError("'used_by_models' does not match 'model_feature_names'")
    if schema.get("n_raw_features") != len(raw) or schema.get("n_model_features") != len(model):
        raise ValueError("feature counts do not match the name lists")


def save_feature_schema(schema: dict[str, Any]) -> None:
    """Write ``feature_schema.json`` and ``feature_names.json`` (raw and model names)."""
    validate_schema(schema)
    write_json(schema, settings.FEATURE_SCHEMA_FILE)
    write_json(
        {"raw_feature_names": schema["raw_feature_names"], "model_feature_names": schema["model_feature_names"]},
        settings.FEATURE_NAMES_FILE,
    )


def load_feature_schema() -> dict[str, Any]:
    """Read and validate ``feature_schema.json``."""
    schema = read_json(settings.FEATURE_SCHEMA_FILE, hint="run scripts/04_preprocess.py first")
    validate_schema(schema)
    return schema


def schema_table(schema: dict[str, Any]) -> pd.DataFrame:
    """The features as a flat table (for reports and for the interface)."""
    return pd.DataFrame(schema["features"])


def check_columns(schema: dict[str, Any], columns: list[str]) -> dict[str, list[str]]:
    """Compare the columns of some data with the raw features of the schema.

    Returns ``{"missing": [...], "unexpected": [...]}``.  Columns of the schema that
    are not used by the models are reported as missing only if they are required
    by the pipeline; the pipeline fills absent values itself, so callers decide.
    """
    expected = list(schema["raw_feature_names"])
    return {
        "missing": [c for c in expected if c not in set(columns)],
        "unexpected": [c for c in columns if c not in set(expected)],
    }