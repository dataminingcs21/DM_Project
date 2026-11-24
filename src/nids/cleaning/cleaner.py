"""Clean the loaded dataset and report exactly what was changed.

The steps, in this order (each is counted in the report, per class where it
matters):

1. drop rows with a missing label;
2. apply the explicit label fixes (``settings.LABEL_MAP``);
3. drop columns that are identical to an earlier column (lossless);
4. set impossible negative values to missing in ``settings.NON_NEGATIVE_COLUMNS``;
5. turn infinite values into missing;
6. remove exact duplicate rows (features + label), keeping the first;
7. handle feature vectors that carry more than one label (``CONFLICT_POLICY``).

What this module deliberately does **not** do:

* it does not fill missing values: the imputer is fitted on the training part
  only, later (File 23), so no information leaks from the test set;
* it does not scale, encode or select features;
* it does not cast to ``float32``: duplicates are judged on the exact
  ``float64`` values, and the cast happens when the arrays are built.

``clean_dataset`` returns ``(cleaned, report, tables)``; the input frame is not
modified.  ``report`` is JSON-safe.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from nids.data import schema
from nids.data.inspector import odd_label_characters
from nids.utils.logger import get_logger

logger = get_logger(__name__)


def _counts(labels: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in labels.value_counts().items()}


def _step(name: str, before: int, after: int, **details: Any) -> dict[str, Any]:
    return {"step": name, "rows_before": before, "rows_after": after, "rows_removed": before - after, **details}


# ---------------------------------------------------------------------------
# Individual steps (each returns the DataFrame and its report section)
# ---------------------------------------------------------------------------
def drop_missing_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = df[settings.LABEL_COLUMN].isna()
    before = len(df)
    if mask.any():
        df = df.loc[~mask]
    return df, _step("drop_missing_labels", before, len(df))


def apply_label_map(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace whole labels according to ``settings.LABEL_MAP`` and report odd labels that remain."""
    label = settings.LABEL_COLUMN
    present = _counts(df[label])
    applied = {old: present[old] for old in settings.LABEL_MAP if old in present}
    if applied:
        df[label] = df[label].replace(settings.LABEL_MAP)
    unmapped_odd = odd_label_characters(df[label])
    if unmapped_odd:
        logger.warning("Labels with odd characters that are not in LABEL_MAP: %s", unmapped_odd)
    return df, {
        "step": "apply_label_map",
        "rows_changed": int(sum(applied.values())),
        "applied": {old: {"new": settings.LABEL_MAP[old], "rows": n} for old, n in applied.items()},
        "map_entries_not_found": [old for old in settings.LABEL_MAP if old not in present],
        "odd_labels_remaining": unmapped_odd,
    }


def find_identical_columns(df: pd.DataFrame, features: list[str]) -> list[list[str]]:
    """Groups of feature columns with exactly the same values (NaN equals NaN), in column order."""
    fingerprints: dict[str, list[str]] = {}
    for col in features:
        values = np.ascontiguousarray(df[col].to_numpy(dtype="float64", copy=False))
        key = hashlib.blake2b(values.tobytes(), digest_size=16).hexdigest()
        fingerprints.setdefault(key, []).append(col)
    groups = []
    for cols in fingerprints.values():
        if len(cols) < 2:
            continue
        first = df[cols[0]].to_numpy(dtype="float64", copy=False)
        verified = [cols[0]] + [
            c for c in cols[1:] if np.array_equal(first, df[c].to_numpy(dtype="float64", copy=False), equal_nan=True)
        ]
        if len(verified) > 1:
            groups.append(verified)
    return groups


def drop_identical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = schema.get_feature_columns(df.columns)
    groups = find_identical_columns(df, features)
    to_drop = [c for group in groups for c in group[1:]] if settings.DROP_IDENTICAL_COLUMNS else []
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, {
        "step": "drop_identical_columns",
        "enabled": settings.DROP_IDENTICAL_COLUMNS,
        "identical_groups": groups,
        "columns_dropped": to_drop,
    }


def handle_negative_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Count (and, with policy ``to_nan``, blank) negative values in ``NON_NEGATIVE_COLUMNS``."""
    label = df[settings.LABEL_COLUMN]
    per_column: dict[str, int] = {}
    skipped = [c for c in settings.NON_NEGATIVE_COLUMNS if c not in df.columns]
    affected_rows = np.zeros(len(df), dtype=bool)
    for col in settings.NON_NEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col].to_numpy(dtype="float64", copy=False) < 0
        n = int(mask.sum())
        if n:
            per_column[col] = n
            affected_rows |= mask
            if settings.NEGATIVE_VALUE_POLICY == "to_nan":
                df.loc[mask, col] = np.nan
    return df, {
        "step": "handle_negative_values",
        "policy": settings.NEGATIVE_VALUE_POLICY,
        "values_per_column": per_column,
        "rows_affected": int(affected_rows.sum()),
        "rows_affected_per_class": _counts(label[affected_rows]),
        "columns_not_present": skipped,
    }


def handle_infinite_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Turn infinite values into missing values; count them per column and per class."""
    label = df[settings.LABEL_COLUMN]
    per_column: dict[str, int] = {}
    affected_rows = np.zeros(len(df), dtype=bool)
    for col in schema.get_feature_columns(df.columns):
        mask = np.isinf(df[col].to_numpy(dtype="float64", copy=False))
        n = int(mask.sum())
        if n:
            per_column[col] = n
            affected_rows |= mask
            df.loc[mask, col] = np.nan
    return df, {
        "step": "handle_infinite_values",
        "values_per_column": per_column,
        "rows_affected": int(affected_rows.sum()),
        "rows_affected_per_class": _counts(label[affected_rows]),
    }


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove rows identical in all features and label (the first is kept; ``source_file`` is ignored)."""
    subset = [*schema.get_feature_columns(df.columns), settings.LABEL_COLUMN]
    before = len(df)
    duplicated = df.duplicated(subset=subset, keep="first")
    removed_per_class = _counts(df.loc[duplicated, settings.LABEL_COLUMN])
    df = df.loc[~duplicated]
    return df, _step("remove_exact_duplicates", before, len(df), removed_per_class=removed_per_class)


def handle_conflicting_vectors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find identical feature vectors with different labels; remove them only with policy ``drop``.

    Vectors are grouped by a 64-bit hash of their feature values; a collision
    between two different vectors is theoretically possible but negligible.
    """
    features = schema.get_feature_columns(df.columns)
    label = df[settings.LABEL_COLUMN]
    vector_hash = pd.util.hash_pandas_object(df[features], index=False)
    labels_per_vector = label.astype("string").groupby(vector_hash).nunique()
    conflicting = labels_per_vector[labels_per_vector > 1].index
    mask = vector_hash.isin(conflicting).to_numpy()
    before = len(df)

    label_sets = (
        label[mask].astype(str).groupby(vector_hash[mask]).agg(lambda s: " | ".join(sorted(set(s)))) if mask.any() else pd.Series(dtype=str)
    )
    pair_counts = {k: int(v) for k, v in label_sets.value_counts().items()}
    section = {
        "policy": settings.CONFLICT_POLICY,
        "n_conflicting_vectors": int(len(conflicting)),
        "label_combinations": pair_counts,
        "rows_in_conflicting_vectors": int(mask.sum()),
        "rows_per_class": _counts(label[mask]),
    }
    if settings.CONFLICT_POLICY == "drop" and mask.any():
        df = df.loc[~mask]
    return df, _step("handle_conflicting_vectors", before, len(df), **section)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    """Run every cleaning step and return ``(cleaned, report, tables)``."""
    if settings.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{settings.LABEL_COLUMN}' not found in the data")
    non_numeric = [c for c in schema.get_feature_columns(df.columns) if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns (load the data with loader.py): {non_numeric[:5]}")

    rows_in = len(df)
    work = df.copy()
    steps: list[dict[str, Any]] = []
    counts_before: dict[str, int] = {}

    pipeline = (
        drop_missing_labels,
        apply_label_map,
        drop_identical_columns,
        handle_negative_values,
        handle_infinite_values,
        remove_exact_duplicates,
        handle_conflicting_vectors,
    )
    for step in pipeline:
        work, section = step(work)
        steps.append(section)
        if step is apply_label_map:  # class counts "before" use the fixed labels, so renamed classes line up
            counts_before = _counts(work[settings.LABEL_COLUMN])
        logger.info("%s: %s", section["step"], {k: v for k, v in section.items() if k in ("rows_removed", "rows_changed", "rows_affected", "columns_dropped") and v})

    work = work.reset_index(drop=True)
    features = schema.get_feature_columns(work.columns)
    counts_after = _counts(work[settings.LABEL_COLUMN])
    missing_per_column = {c: int(n) for c, n in work[features].isna().sum().items() if n}
    rows_with_missing = int(work[features].isna().any(axis=1).sum())

    class_table = pd.DataFrame(
        {
            "label": sorted(set(counts_before) | set(counts_after)),
        }
    )
    class_table["rows_before"] = class_table["label"].map(counts_before).fillna(0).astype(int)
    class_table["rows_after"] = class_table["label"].map(counts_after).fillna(0).astype(int)
    class_table["rows_removed"] = class_table["rows_before"] - class_table["rows_after"]
    class_table["removed_percent"] = (class_table["rows_removed"] / class_table["rows_before"].replace(0, np.nan) * 100).round(2)
    class_table["below_min_class_samples"] = class_table["rows_after"] < settings.MIN_CLASS_SAMPLES
    class_table = class_table.sort_values("rows_after", ascending=False).reset_index(drop=True)

    report = {
        "rows_in": rows_in,
        "rows_out": int(len(work)),
        "n_feature_columns_in": len(schema.get_feature_columns(df.columns)),
        "n_feature_columns_out": len(features),
        "steps": steps,
        "missing_values_after_cleaning": {
            "n_missing": int(sum(missing_per_column.values())),
            "per_column": missing_per_column,
            "rows_with_missing": rows_with_missing,
            "note": "Left on purpose: missing values are imputed on the training data only (File 23).",
        },
        "class_counts_before": counts_before,  # after the missing-label drop and the label fixes
        "class_counts_after": counts_after,
        "classes_below_min_class_samples": class_table.loc[class_table["below_min_class_samples"], "label"].tolist(),
        "min_class_samples_rule": settings.MIN_CLASS_SAMPLES,
        "config": settings.cleaning_config(),
    }
    logger.info("Cleaning finished: %d rows in, %d rows out, %d feature columns", rows_in, len(work), len(features))
    return work, report, {"class_counts": class_table}