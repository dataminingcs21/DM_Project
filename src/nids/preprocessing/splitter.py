"""Stratified train/test split (Checkpoint 2 decision: option A).

A random 80/20 split that keeps each class's proportion (``stratify``), with a
fixed seed so it is reproducible.  It runs **after** duplicate removal, so
identical flows cannot end up on both sides.

The split works on **row positions** (integer arrays), not on copies of the
data: the caller takes the rows with ``df.iloc[train_idx]``.  This avoids
duplicating a 2.5-million-row table in memory.

Everything that learns from data (imputer, feature filter, scaler) is fitted on
the training rows only, afterwards (Files 22-25).  Limit to state in the
README: every attack class comes from one capture file, so this split measures
recognition of *known* attacks, not generalisation to unseen ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.splitter")


def stratified_split(
    target: pd.Series | np.ndarray,
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)``: sorted integer row positions.

    ``target`` holds one class (name or code) per row.  Raises ValueError when a
    class has fewer than 2 rows (a stratified split is then impossible) or when
    the split would leave a class out of the train or the test part.
    """
    test_size = settings.TEST_SIZE if test_size is None else test_size
    seed = settings.RANDOM_SEED if seed is None else seed
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")

    y = np.asarray(target)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("target must be a non-empty 1-D sequence")
    if pd.isna(y).any():
        raise ValueError("target contains missing values")

    counts = pd.Series(y).value_counts()
    too_small = sorted(str(c) for c, n in counts.items() if n < 2)
    if too_small:
        raise ValueError(f"classes with fewer than 2 rows cannot be split: {too_small}")

    positions = np.arange(y.size)
    train_idx, test_idx = train_test_split(positions, test_size=test_size, random_state=seed, stratify=y)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    missing_train = sorted(set(map(str, counts.index)) - set(map(str, np.unique(y[train_idx]))))
    missing_test = sorted(set(map(str, counts.index)) - set(map(str, np.unique(y[test_idx]))))
    if missing_train or missing_test:
        raise ValueError(f"classes missing from train {missing_train} or test {missing_test}")
    logger.info("Split: %d train rows, %d test rows (test_size=%.2f, seed=%d)", train_idx.size, test_idx.size, test_size, seed)
    return train_idx, test_idx


def split_report(
    target: pd.Series | np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Check the split and describe it. Returns ``(summary, per_class_table)``."""
    y = pd.Series(np.asarray(target))
    n = len(y)
    disjoint = np.intersect1d(train_idx, test_idx).size == 0
    complete = (train_idx.size + test_idx.size) == n and np.union1d(train_idx, test_idx).size == n

    table = pd.DataFrame(
        {
            "total": y.value_counts(),
            "train": y.iloc[train_idx].value_counts(),
            "test": y.iloc[test_idx].value_counts(),
        }
    ).fillna(0).astype("int64")
    table["test_percent"] = (table["test"] / table["total"] * 100).round(2)
    table = table.sort_values("total", ascending=False)
    table.index.name = "class"
    table = table.reset_index()

    summary = {
        "n_rows": int(n),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "test_fraction": round(test_idx.size / n, 5),
        "train_test_disjoint": bool(disjoint),
        "covers_all_rows": bool(complete),
        "classes_missing_from_test": table.loc[table["test"] == 0, "class"].astype(str).tolist(),
        "classes_missing_from_train": table.loc[table["train"] == 0, "class"].astype(str).tolist(),
        "classes_train_below_cv_folds": table.loc[table["train"] < settings.CV_FOLDS, "class"].astype(str).tolist(),
        "cv_folds": int(settings.CV_FOLDS),
        "test_size_setting": float(settings.TEST_SIZE),
        "random_seed": int(settings.RANDOM_SEED),
    }
    return summary, table