"""Data-quality checks on any table of flows (used by the app's "Data quality" page and the notebooks).

``inspector.py`` builds the one-off inspection report of the raw files; this module gives the same kind of
answers for *any* DataFrame (a sample, an uploaded CSV): types, missing and infinite values, impossible
negatives, constant columns, duplicate rows, conflicting labels and the class balance.

Nothing here modifies the input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings


def column_quality(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing, infinite, negatives, distinct values, constant flag, min / max."""
    rows = []
    non_negative = set(getattr(settings, "NON_NEGATIVE_COLUMNS", {}))
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        values = s.to_numpy(dtype="float64", na_value=np.nan) if numeric else None
        finite = values[np.isfinite(values)] if numeric else None
        n_inf = int(np.isinf(values).sum()) if numeric else 0
        n_unique = int(s.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "kind": "numeric" if numeric else "text",
                "n_missing": int(s.isna().sum()),
                "pct_missing": float(s.isna().mean() * 100) if len(s) else 0.0,
                "n_inf": n_inf,
                "n_negative": int((finite < 0).sum()) if numeric else 0,
                "negative_impossible": col in non_negative,
                "n_unique": n_unique,
                "is_constant": n_unique <= 1,
                "min": float(finite.min()) if numeric and finite.size else np.nan,
                "max": float(finite.max()) if numeric and finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def duplicate_summary(df: pd.DataFrame, label: str | None = None) -> dict[str, Any]:
    """Exact duplicate rows, and feature vectors that appear with more than one label."""
    label = label or settings.LABEL_COLUMN
    meta = {label, settings.SOURCE_FILE_COLUMN}
    features = [c for c in df.columns if c not in meta]
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(df.duplicated(subset=[c for c in df.columns if c != settings.SOURCE_FILE_COLUMN]).sum()) if len(df) else 0,
        "n_duplicate_feature_vectors": int(df.duplicated(subset=features).sum()) if features and len(df) else 0,
    }
    if label in df.columns and features and len(df):
        per_vector = df.groupby(features, dropna=False, sort=False)[label].nunique()
        conflicting = per_vector[per_vector > 1]
        out["n_conflicting_feature_vectors"] = int(len(conflicting))
    else:
        out["n_conflicting_feature_vectors"] = None
    out["pct_duplicate_rows"] = 100.0 * out["n_duplicate_rows"] / out["n_rows"] if out["n_rows"] else 0.0
    return out


def class_balance(df: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    """Rows per class, share and imbalance ratio against the largest class."""
    label = label or settings.LABEL_COLUMN
    if label not in df.columns:
        return pd.DataFrame(columns=["class", "rows", "share_pct", "ratio_to_largest"])
    counts = df[label].astype(str).value_counts()
    table = pd.DataFrame({"class": counts.index, "rows": counts.to_numpy()})
    table["share_pct"] = 100.0 * table["rows"] / table["rows"].sum()
    table["ratio_to_largest"] = table["rows"].iloc[0] / table["rows"]
    return table


def quality_report(df: pd.DataFrame, label: str | None = None) -> dict[str, Any]:
    """Everything in one dictionary (the app renders it; tests check it)."""
    label = label or settings.LABEL_COLUMN
    columns = column_quality(df)
    dups = duplicate_summary(df, label)
    balance = class_balance(df, label)
    numeric = columns[columns["kind"] == "numeric"]
    issues = []
    if dups["n_duplicate_rows"]:
        issues.append(f"{dups['n_duplicate_rows']:,} exact duplicate rows ({dups['pct_duplicate_rows']:.2f}%)")
    if int(columns["n_missing"].sum()):
        issues.append(f"{int(columns['n_missing'].sum()):,} missing cells in {int((columns['n_missing'] > 0).sum())} column(s)")
    if int(numeric["n_inf"].sum()):
        issues.append(f"{int(numeric['n_inf'].sum()):,} infinite values in {int((numeric['n_inf'] > 0).sum())} column(s)")
    bad_neg = numeric[(numeric["n_negative"] > 0) & numeric["negative_impossible"]]
    if len(bad_neg):
        issues.append(f"impossible negative values in {len(bad_neg)} column(s): " + ", ".join(bad_neg["column"].head(5)))
    meta = columns["column"].isin([label, settings.SOURCE_FILE_COLUMN])
    constant = columns[columns["is_constant"] & ~meta]
    if len(constant):
        issues.append(f"{len(constant)} constant column(s): " + ", ".join(constant["column"].head(8)))
    text_cols = columns[(columns["kind"] == "text") & ~columns["column"].isin([label, settings.SOURCE_FILE_COLUMN])]
    if len(text_cols):
        issues.append(f"{len(text_cols)} non-numeric feature column(s): " + ", ".join(text_cols["column"].head(5)))
    if dups.get("n_conflicting_feature_vectors"):
        issues.append(f"{dups['n_conflicting_feature_vectors']:,} feature vectors carry more than one label")
    if len(balance) and balance["ratio_to_largest"].max() > 100:
        issues.append(f"strong class imbalance (largest / smallest class = {balance['ratio_to_largest'].max():,.0f})")
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_numeric": int((columns["kind"] == "numeric").sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 2**20),
        "columns": columns,
        "duplicates": dups,
        "balance": balance,
        "issues": issues,
    }
