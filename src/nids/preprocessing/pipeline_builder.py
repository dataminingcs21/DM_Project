"""Build the preprocessing pipeline (an unfitted scikit-learn ``Pipeline``).

Order of the steps (Checkpoint 2 decision, option A):

1. ``inf_to_nan``  turn +/- infinity into missing values (scikit-learn refuses infinity)
2. ``imputer``     fill missing values with the **training** median
3. ``log``         signed log1p: ``sign(x) * log1p(|x|)``  (skipped if ``LOG_TRANSFORM`` is False)
4. ``filter``      drop constant and highly correlated columns (File 22)
5. ``scaler``      standard scaler (or robust / min-max, from ``SCALER_TYPE``)

Every step that learns something (median, kept columns, mean and spread) learns
it in ``fit`` on the **training rows only**.  The same fitted pipeline is then
applied to the test rows and to a single flow typed in the interface.

The pipeline is built to work on a DataFrame: ``set_output(transform="pandas")``
keeps the column names from step to step, so the final column names are
available with ``pipeline.get_feature_names_out()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from config import settings
from nids.preprocessing.feature_filter import FeatureFilter
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.pipeline_builder")

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


# ---------------------------------------------------------------------------
# Small stateless transformers
# ---------------------------------------------------------------------------
class _Stateless(BaseEstimator, TransformerMixin):
    """Base for transformers that learn nothing but must remember the column names."""

    def fit(self, X, y=None):  # noqa: ARG002
        X = self._check(X)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        return self

    @staticmethod
    def _check(X):
        if getattr(X, "ndim", 2) != 2:
            raise ValueError("X must be 2-dimensional")
        return X

    def _check_transform_input(self, X):
        """Refuse a different number of columns or different column names/order (no silent mislabelling)."""
        check_is_fitted(self, "n_features_in_")
        X = self._check(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, but the pipeline was fitted on {self.n_features_in_}")
        if hasattr(X, "columns") and hasattr(self, "feature_names_in_"):
            if [str(c) for c in X.columns] != list(self.feature_names_in_):
                raise ValueError("column names or their order differ from those seen in fit")
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class InfiniteToMissing(_Stateless):
    """Replace +infinity and -infinity by NaN (the imputer then fills them)."""

    def transform(self, X):
        X = self._check_transform_input(X)
        arr = np.array(X, copy=True)  # never modify the caller's data
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype("float64")
        arr[~np.isfinite(arr)] = np.nan
        return arr


class SignedLog1p(_Stateless):
    """``sign(x) * log1p(|x|)``: compresses huge values, keeps the sign and the order, maps 0 to 0."""

    def transform(self, X):
        X = self._check_transform_input(X)
        arr = np.asarray(X)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype("float64")
        return np.sign(arr) * np.log1p(np.abs(arr))  # NaN stays NaN


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_pipeline(
    log_transform: bool | None = None,
    scaler_type: str | None = None,
    correlation_threshold: float | None = None,
) -> Pipeline:
    """Return the unfitted preprocessing pipeline (arguments default to the settings)."""
    log_transform = settings.LOG_TRANSFORM if log_transform is None else log_transform
    scaler_type = settings.SCALER_TYPE if scaler_type is None else scaler_type
    if scaler_type not in _SCALERS:
        raise ValueError(f"scaler_type must be one of {tuple(_SCALERS)}, got {scaler_type!r}")

    steps: list[tuple[str, Any]] = [
        ("inf_to_nan", InfiniteToMissing()),
        # keep_empty_features: a column that is entirely missing in training is kept (filled with 0)
        # and then removed by the filter as a constant, instead of silently changing the column set.
        ("imputer", SimpleImputer(strategy=settings.IMPUTER_STRATEGY, keep_empty_features=True)),
    ]
    if log_transform:
        steps.append(("log", SignedLog1p()))
    steps.append(("filter", FeatureFilter(correlation_threshold=correlation_threshold)))
    steps.append(("scaler", _SCALERS[scaler_type]()))

    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    logger.info("Pipeline built: %s", " -> ".join(name for name, _ in steps))
    return pipeline


def describe_pipeline(pipeline: Pipeline) -> dict[str, Any]:
    """JSON-safe description of a (fitted) pipeline for the preprocessing metadata."""
    info: dict[str, Any] = {
        "steps": [name for name, _ in pipeline.steps],
        "scaler_type": type(pipeline.named_steps["scaler"]).__name__,
        "log_transform": "log" in pipeline.named_steps,
        "imputer_strategy": pipeline.named_steps["imputer"].strategy,
    }
    feature_filter = pipeline.named_steps["filter"]
    if hasattr(feature_filter, "support_"):
        info["feature_filter"] = feature_filter.report()
    return info