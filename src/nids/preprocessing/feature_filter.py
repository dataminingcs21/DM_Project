"""Remove useless features: constant columns and highly correlated columns.

``FeatureFilter`` is a scikit-learn transformer, so it can sit inside the
preprocessing pipeline (File 23).  It **learns which columns to drop from the
training rows only** (``fit``), and ``transform`` then applies the same choice
to any data (training, test, one flow typed in the interface).

Rules (decided with the user):

1. **Constant columns** (at most one distinct finite value in the training
   rows; an all-missing column counts too) are dropped.
2. **Correlated columns**: when two columns have ``|r| >= threshold``
   (default ``settings.CORRELATION_THRESHOLD`` = 0.95), the one that comes
   **first in column order is kept** and the later one is dropped.  This is
   deterministic and gives the same result on every run.  The rule is greedy
   in column order, so the kept columns are all pairwise below the threshold.

The correlation is Pearson, computed on a random sample of at most
``max_rows`` training rows (a full 2-million-row matrix is not needed to find
|r| >= 0.95).  Missing and infinite values are ignored pair by pair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.feature_filter")

DEFAULT_MAX_ROWS = 500_000


def _as_2d_array(X: Any) -> tuple[np.ndarray, list[str] | None]:
    """Return a 2-D floating array and the column names (None for a plain array)."""
    names = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else None
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {arr.ndim} dimension(s)")
    if not np.issubdtype(arr.dtype, np.floating):
        try:
            arr = arr.astype("float64")
        except (TypeError, ValueError) as error:
            raise ValueError("X must contain only numbers") from error
    return arr, names


class FeatureFilter(BaseEstimator, TransformerMixin):
    """Drop constant and highly correlated columns (learned on the training rows)."""

    def __init__(
        self,
        correlation_threshold: float | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        random_state: int | None = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.max_rows = max_rows
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):  # noqa: ARG002 - y is accepted for pipeline compatibility, never used
        arr, names = _as_2d_array(X)
        n_rows, n_cols = arr.shape
        if n_rows < 2 or n_cols < 1:
            raise ValueError(f"X needs at least 2 rows and 1 column, got shape {arr.shape}")
        threshold = settings.CORRELATION_THRESHOLD if self.correlation_threshold is None else self.correlation_threshold
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"correlation_threshold must be in (0, 1], got {threshold}")
        if self.max_rows < 2:
            raise ValueError("max_rows must be at least 2")

        feature_names = names if names is not None else [f"x{i}" for i in range(n_cols)]

        # 1. constant columns (finite values only; one column at a time to save memory)
        constant = np.zeros(n_cols, dtype=bool)
        for j in range(n_cols):
            col = arr[:, j]
            finite = col[np.isfinite(col)]
            constant[j] = finite.size == 0 or finite.min() == finite.max()

        # 2. correlated columns (greedy, first column in order is kept)
        correlated: dict[str, dict[str, Any]] = {}
        drop_corr = np.zeros(n_cols, dtype=bool)
        candidates = np.flatnonzero(~constant)
        n_used = n_rows
        if candidates.size > 1:
            rng = np.random.default_rng(settings.RANDOM_SEED if self.random_state is None else self.random_state)
            rows = np.sort(rng.choice(n_rows, size=self.max_rows, replace=False)) if n_rows > self.max_rows else slice(None)
            sample = arr[rows][:, candidates].astype("float64")
            sample[~np.isfinite(sample)] = np.nan
            n_used = sample.shape[0]
            corr = pd.DataFrame(sample).corr().to_numpy()
            kept: list[int] = []  # positions inside `candidates`
            for pos in range(candidates.size):
                partner = next((k for k in kept if abs(corr[k, pos]) >= threshold), None)  # NaN compares False
                if partner is None:
                    kept.append(pos)
                else:
                    drop_corr[candidates[pos]] = True
                    correlated[feature_names[candidates[pos]]] = {
                        "kept_instead": feature_names[candidates[partner]],
                        "correlation": round(float(corr[partner, pos]), 6),
                    }

        support = ~(constant | drop_corr)
        if not support.any():
            raise ValueError("every column would be dropped; check the data and the threshold")

        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = n_cols
        self.threshold_ = float(threshold)
        self.n_rows_used_ = int(n_used)
        self.constant_features_ = [feature_names[j] for j in np.flatnonzero(constant)]
        self.correlated_features_ = correlated
        self.support_ = support
        logger.info(
            "Feature filter fitted: %d columns in, %d kept (%d constant, %d correlated dropped; threshold %.3f)",
            n_cols, int(support.sum()), len(self.constant_features_), len(correlated), threshold,
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X):
        check_is_fitted(self, "support_")
        arr, names = _as_2d_array(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {arr.shape[1]} columns, but the filter was fitted on {self.n_features_in_}")
        if names is not None and names != list(self.feature_names_in_):
            raise ValueError("column names or their order differ from those seen in fit")
        return arr[:, self.support_]

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_]

    # --------------------------------------------------------------- report
    @property
    def dropped_features_(self) -> list[str]:
        check_is_fitted(self, "support_")
        return [str(n) for n in self.feature_names_in_[~self.support_]]

    def report(self) -> dict[str, Any]:
        """JSON-safe description of what was dropped and why."""
        check_is_fitted(self, "support_")
        return {
            "n_features_in": int(self.n_features_in_),
            "n_features_out": int(self.support_.sum()),
            "correlation_threshold": self.threshold_,
            "rows_used_for_correlation": self.n_rows_used_,
            "constant_features": list(self.constant_features_),
            "correlated_features": dict(self.correlated_features_),
            "selected_features": [str(n) for n in self.get_feature_names_out()],
        }