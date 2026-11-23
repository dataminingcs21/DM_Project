"""Column and label conventions for the dataset.

This module deliberately contains **no list of feature names**.  The real
columns are read from the files at Checkpoint 1, so nothing is invented here.
What it provides instead:

* name normalization (CIC-IDS2017 headers often carry leading/trailing spaces);
* the rule for which columns are metadata, label or features;
* a comparison of one file's columns against a reference set (used by the
  loader to detect files whose structure differs);
* a numeric-coercion helper that reports how many values it could not convert.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import settings

_WHITESPACE = re.compile(r"\s+")

#: Columns added by the project (never features, never model input).
METADATA_COLUMNS: tuple[str, ...] = (settings.SOURCE_FILE_COLUMN,)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_column_name(name: Any) -> str:
    """Strip the name and collapse inner runs of whitespace into one space."""
    return _WHITESPACE.sub(" ", str(name)).strip()


def normalize_columns(columns: Iterable[Any]) -> list[str]:
    """Normalize every column name (order is kept)."""
    return [normalize_column_name(c) for c in columns]


def normalize_label_value(value: Any) -> Any:
    """Strip and collapse whitespace in a label; missing values stay missing.

    Only whitespace is touched.  Odd characters inside labels (if the real data
    has any) are reported at Checkpoint 1 and handled by an explicit decision,
    not silently here.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return value
    return _WHITESPACE.sub(" ", str(value)).strip()


def find_duplicate_columns(columns: Iterable[Any]) -> list[str]:
    """Names that appear more than once after normalization (sorted)."""
    counts = Counter(normalize_columns(columns))
    return sorted(name for name, n in counts.items() if n > 1)


# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
def get_feature_columns(columns: Iterable[Any]) -> list[str]:
    """Normalized columns that are neither the label nor metadata (order kept)."""
    excluded = {settings.LABEL_COLUMN, *METADATA_COLUMNS}
    return [c for c in normalize_columns(columns) if c not in excluded]


def has_label(columns: Iterable[Any]) -> bool:
    """True if the label column is present after normalization."""
    return settings.LABEL_COLUMN in normalize_columns(columns)


@dataclass(frozen=True)
class ColumnComparison:
    """Difference between a file's columns and a reference set."""

    missing: list[str] = field(default_factory=list)  # in reference, not in file
    extra: list[str] = field(default_factory=list)  # in file, not in reference
    same_order: bool = True

    @property
    def is_identical(self) -> bool:
        return not self.missing and not self.extra and self.same_order


def compare_columns(columns: Iterable[Any], reference: Iterable[Any]) -> ColumnComparison:
    """Compare normalized ``columns`` with the normalized ``reference``."""
    cols = normalize_columns(columns)
    ref = normalize_columns(reference)
    ref_set, col_set = set(ref), set(cols)
    return ColumnComparison(
        missing=sorted(ref_set - col_set),
        extra=sorted(col_set - ref_set),
        same_order=cols == ref if ref_set == col_set else True,
    )


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------
def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert ``columns`` to numbers; unconvertible values become NaN.

    Returns ``(df, report)`` where ``report`` maps a column name to the number
    of non-missing values that could not be converted (only columns with at
    least one such value appear).  The input frame is not modified.
    """
    out = df.copy()
    report: dict[str, int] = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        before = out[col].notna()
        converted = pd.to_numeric(out[col], errors="coerce")
        lost = int((before & converted.isna()).sum())
        if lost:
            report[col] = lost
        out[col] = converted
    return out, report