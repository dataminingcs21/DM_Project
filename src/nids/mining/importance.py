"""Which features drive the predictions?

* **Built-in importance** of the tree ensembles (impurity decrease for Random Forest, gain for
  XGBoost).  Fast, but biased toward features with many distinct values.
* **Permutation importance**: shuffle one feature and measure how much the macro F1 drops on held-out
  rows.  Model-agnostic and measured on data the model did not train on.  Correlated features share
  their importance (shuffling one leaves its twin), so a low score does not prove a feature is useless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("mining.importance")


def builtin_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """``feature_importances_`` of a fitted project model (or estimator), normalised to sum to 1; None if absent."""
    estimator = getattr(model, "estimator_", model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return None
    values = np.asarray(values, dtype="float64")
    total = values.sum()
    frame = pd.DataFrame({"feature": feature_names, "importance": values / total if total > 0 else values})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_table(
    model: Any, X: np.ndarray, y: np.ndarray, feature_names: list[str], n_repeats: int | None = None
) -> pd.DataFrame:
    """Mean and std of the macro F1 drop when each feature is shuffled (``n_repeats`` times)."""
    n_repeats = settings.IMPORTANCE_REPEATS if n_repeats is None else n_repeats
    estimator = getattr(model, "estimator_", model)
    result = permutation_importance(
        estimator, X, y, scoring="f1_macro", n_repeats=n_repeats, random_state=settings.RANDOM_SEED, n_jobs=1
    )
    frame = pd.DataFrame(
        {"feature": feature_names, "f1_drop_mean": result.importances_mean, "f1_drop_std": result.importances_std}
    )
    return frame.sort_values("f1_drop_mean", ascending=False).reset_index(drop=True)


def rank_agreement(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str, top: int = 10) -> dict[str, Any]:
    """How many of the top ``top`` features two rankings share, and the Spearman correlation of all ranks."""
    left = a[["feature", col_a]].rename(columns={col_a: "value_a"})
    right = b[["feature", col_b]].rename(columns={col_b: "value_b"})
    merged = left.merge(right, on="feature")
    shared = set(a["feature"].head(top)) & set(b["feature"].head(top))
    return {
        f"shared_top_{top}": sorted(shared),
        "n_shared": len(shared),
        "spearman": float(merged["value_a"].rank().corr(merged["value_b"].rank())),
    }
