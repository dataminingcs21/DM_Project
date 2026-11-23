"""Inspect the loaded dataset and report what is in it (nothing is changed).

The inspector answers the questions that must be answered *before* cleaning:
how big is the data, which classes exist and how rare are they, which columns
contain missing / infinite / negative values, which columns are constant, how
many rows are duplicated, and whether identical feature vectors carry
different labels.

``inspect_dataset`` returns ``(summary, tables)``:

* ``summary`` is a plain dictionary (safe for ``io_utils.write_json``);
* ``tables`` is a dictionary of DataFrames (``column_profile``,
  ``label_distribution``, ``label_by_file``) that the script saves as CSV.

Nothing here is specific to a column name: the columns come from the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings
from nids.data import schema
from nids.utils.logger import get_logger

logger = get_logger(__name__)


def _require_columns(df: pd.DataFrame) -> list[str]:
    if settings.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{settings.LABEL_COLUMN}' not found in the data")
    features = schema.get_feature_columns(df.columns)
    if not features:
        raise ValueError("No feature columns found in the data")
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns (load the data with loader.py): {non_numeric[:5]}")
    return features


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def column_profile(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """One row per feature: dtype, missing / infinite / negative counts, unique values, finite min / max."""
    features = features if features is not None else _require_columns(df)
    rows = []
    for col in features:
        series = df[col]
        values = series.to_numpy()
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        n_unique = int(series.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "n_inf": int(np.isinf(values).sum()),
                "n_negative": int((values < 0).sum()),
                "n_unique": n_unique,
                "is_constant": n_unique <= 1,
                "min_finite": float(finite.min()) if finite.size else np.nan,
                "max_finite": float(finite.max()) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per label with percentage, number of source files containing it, and a rarity flag."""
    label = settings.LABEL_COLUMN
    counts = df[label].value_counts(dropna=False)
    table = counts.rename_axis("label").reset_index(name="count")
    table["percent"] = (table["count"] / len(df) * 100).round(4)
    if settings.SOURCE_FILE_COLUMN in df.columns:
        n_files = df.groupby(label, dropna=False)[settings.SOURCE_FILE_COLUMN].nunique()
        table["n_files"] = table["label"].map(n_files).astype(int) if not table["label"].isna().any() else np.nan
    table["below_min_class_samples"] = table["count"] < settings.MIN_CLASS_SAMPLES
    return table


def label_by_file(df: pd.DataFrame) -> pd.DataFrame:
    """Cross table: rows per label (lines) and per source file (columns)."""
    if settings.SOURCE_FILE_COLUMN not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[settings.LABEL_COLUMN], df[settings.SOURCE_FILE_COLUMN])


# ---------------------------------------------------------------------------
# Duplicates and conflicting labels
# ---------------------------------------------------------------------------
def duplicate_statistics(df: pd.DataFrame, features: list[str] | None = None) -> dict[str, int]:
    """Count duplicated rows and feature vectors that carry more than one label.

    Duplicates are judged on features + label (``source_file`` is ignored).
    Rows are compared through a 64-bit hash of the feature values; a collision
    between two different rows is theoretically possible but negligible here.
    """
    features = features if features is not None else _require_columns(df)
    label = settings.LABEL_COLUMN

    feature_hash = pd.util.hash_pandas_object(df[features], index=False)
    full_hash = pd.util.hash_pandas_object(
        pd.DataFrame({"f": feature_hash, "l": df[label].astype("string")}), index=False
    )
    n_duplicate_rows = int(full_hash.duplicated().sum())

    labels_per_vector = df[label].astype("string").groupby(feature_hash).nunique()
    conflicting_vectors = labels_per_vector[labels_per_vector > 1]
    rows_in_conflict = int(feature_hash.isin(conflicting_vectors.index).sum())

    return {
        "n_duplicate_rows": n_duplicate_rows,
        "n_conflicting_feature_vectors": int(len(conflicting_vectors)),
        "n_rows_in_conflicting_vectors": rows_in_conflict,
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def odd_label_characters(labels: pd.Series) -> list[str]:
    """Distinct labels containing non-ASCII characters, ``?`` or the replacement character."""
    unique = labels.dropna().astype(str).unique()
    return sorted(v for v in unique if any(ord(ch) > 127 or ch == "?" for ch in v))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def inspect_dataset(
    df: pd.DataFrame, load_report: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Inspect ``df`` and return ``(summary, tables)`` (see the module docstring)."""
    features = _require_columns(df)
    logger.info("Inspecting %d rows x %d feature columns", len(df), len(features))

    profile = column_profile(df, features)
    distribution = label_distribution(df)
    crosstab = label_by_file(df)
    duplicates = duplicate_statistics(df, features)

    n_cells = len(df) * len(features)
    n_missing = int(profile["n_missing"].sum())
    n_inf = int(profile["n_inf"].sum())
    rows_with_missing_or_inf = int(
        (~np.isfinite(df[features].to_numpy(dtype="float64", copy=False))).any(axis=1).sum()
    )

    labels = df[settings.LABEL_COLUMN]
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_feature_columns": len(features),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 1),
        "n_source_files": int(df[settings.SOURCE_FILE_COLUMN].nunique())
        if settings.SOURCE_FILE_COLUMN in df.columns
        else None,
        "labels": {
            "n_classes": int(labels.nunique(dropna=True)),
            "n_missing_labels": int(labels.isna().sum()),
            "classes_below_min_samples": distribution.loc[
                distribution["below_min_class_samples"], "label"
            ].tolist(),
            "min_class_samples_rule": settings.MIN_CLASS_SAMPLES,
            "classes_in_a_single_file": distribution.loc[distribution["n_files"] == 1, "label"].tolist()
            if "n_files" in distribution
            else [],
            "labels_with_odd_characters": odd_label_characters(labels),
        },
        "values": {
            "n_cells": n_cells,
            "n_missing": n_missing,
            "n_infinite": n_inf,
            "rows_with_missing_or_infinite": rows_with_missing_or_inf,
            "columns_with_missing": profile.loc[profile["n_missing"] > 0, "column"].tolist(),
            "columns_with_infinite": profile.loc[profile["n_inf"] > 0, "column"].tolist(),
            "columns_with_negative": profile.loc[profile["n_negative"] > 0, "column"].tolist(),
            "constant_columns": profile.loc[profile["is_constant"], "column"].tolist(),
        },
        "duplicates": duplicates,
    }
    if load_report is not None:
        summary["load"] = {
            "n_files": load_report.get("n_files"),
            "encodings": {f["file"]: f["encoding"] for f in load_report.get("files", [])},
            "non_numeric_values_coerced": {
                f["file"]: f["non_numeric_values_coerced"]
                for f in load_report.get("files", [])
                if f["non_numeric_values_coerced"]
            },
        }

    tables = {"column_profile": profile, "label_distribution": distribution, "label_by_file": crosstab}
    return summary, tables