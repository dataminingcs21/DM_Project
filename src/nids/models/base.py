"""Common interface of the 11 models.

Every model file (Files 30-40) is a small subclass of :class:`BaseModel` that only
says **how to build its scikit-learn / XGBoost estimator**.  Everything else is
shared here, so all models behave identically for the trainer, the evaluation,
the tuning step, the voting and the interface:

* ``fit(X, y, sample_weight=None)``: checks the data, applies class weights if
  ``CLASS_WEIGHT_MODE = "balanced"``, fits, and records the time and the number of rows;
* ``predict(X)``: integer class codes (int64);
* ``predict_scores(X)``: an ``(n_rows, n_classes)`` array, probabilities when the
  model has them, otherwise decision scores (``score_kind`` says which);
* ``build_estimator()``: a fresh **unfitted** estimator (GridSearchCV clones it);
* ``with_params(**overrides)``: a new, unfitted model of the same kind with
  other parameters (used with the tuning results);
* ``info()``: JSON-safe description (parameters, rows, seconds, weights).

Requirement on the data: class codes must be ``0..K-1`` with **every class
present** in the training data (the codes come from the label codec, File 19).
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_sample_weight

from config import model_params, settings
from nids.utils.logger import get_logger

logger = get_logger("models.base")


def check_features(X: Any, name: str = "X") -> np.ndarray:
    """2-D float array without missing or infinite values (preprocessing guarantees this)."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {arr.ndim} dimension(s)")
    if not np.issubdtype(arr.dtype, np.floating):
        try:
            arr = arr.astype("float32")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain only numbers") from error
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains missing or infinite values; use the preprocessing pipeline first")
    return arr


class BaseModel(ABC):
    """Base class of every model. Subclasses set ``key`` and implement ``_make_estimator``."""

    key: ClassVar[str]  # one of config.model_params.MODEL_KEYS
    # Numeric type the estimator must compute in. None = keep the type of the data (float32).
    # Gaussian Naive Bayes sets "float64": in float32 its log-likelihoods lose their precision.
    input_dtype: ClassVar[str | None] = None

    def __init__(self, params: dict[str, Any] | None = None, class_weight_mode: str | None = None):
        if not getattr(type(self), "key", None):
            raise TypeError(f"{type(self).__name__} must define the class attribute 'key'")
        if self.key not in model_params.MODEL_KEYS:
            raise ValueError(f"unknown model key '{self.key}'. Known: {list(model_params.MODEL_KEYS)}")
        mode = settings.CLASS_WEIGHT_MODE if class_weight_mode is None else class_weight_mode
        if mode not in settings.ALLOWED_CLASS_WEIGHT_MODES:
            raise ValueError(f"class_weight_mode must be one of {settings.ALLOWED_CLASS_WEIGHT_MODES}, got {mode!r}")
        self.params: dict[str, Any] = model_params.default_params(self.key)
        if params:
            self.params.update(params)
        self.class_weight_mode = mode
        self.estimator_: Any = None
        self.n_classes_: int | None = None
        self.n_train_rows_: int | None = None
        self.fit_seconds_: float | None = None
        self.fitted_at_: str | None = None
        self.class_weight_applied_: bool = False

    # ------------------------------------------------------------ identity
    @property
    def display_name(self) -> str:
        return model_params.DISPLAY_NAMES[self.key]

    @property
    def is_fitted(self) -> bool:
        return self.estimator_ is not None

    def __repr__(self) -> str:
        state = f"fitted on {self.n_train_rows_:,} rows" if self.is_fitted else "not fitted"
        return f"{type(self).__name__}({self.display_name}, {state})"

    def _prepare(self, X: Any, name: str = "X") -> np.ndarray:
        """Check the features and cast them to ``input_dtype`` when the model needs one."""
        X = check_features(X, name)
        return X if self.input_dtype is None else X.astype(self.input_dtype, copy=False)

    # ---------------------------------------------------------- estimator
    @abstractmethod
    def _make_estimator(self, params: dict[str, Any]) -> Any:
        """Return an unfitted estimator built from ``params`` (the only thing a subclass must write)."""

    def build_estimator(self) -> Any:
        """A fresh, unfitted estimator with the current parameters (and class weights if they apply)."""
        params = dict(self.params)
        if self.class_weight_mode == "balanced" and model_params.class_weight_mechanism(self.key) == "class_weight":
            params["class_weight"] = "balanced"
        return self._make_estimator(params)

    def with_params(self, **overrides: Any) -> "BaseModel":
        """A new unfitted model of the same kind with some parameters replaced."""
        merged = {**self.params, **overrides}
        return type(self)(params=merged, class_weight_mode=self.class_weight_mode)

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
        n_classes: int | None = None,
    ) -> "BaseModel":
        """Fit on the (already preprocessed) training data. Returns ``self``."""
        X = self._prepare(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1-dimensional with {X.shape[0]} values, got shape {y.shape}")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must hold integer class codes (use the label codec)")
        y = y.astype("int64", copy=False)

        present = np.unique(y)
        if not np.array_equal(present, np.arange(present.size)):
            raise ValueError(
                f"class codes must be 0..K-1 with every class present in the training data; found codes {present.tolist()[:20]}"
            )
        if n_classes is not None and n_classes != present.size:
            raise ValueError(f"the training data has {present.size} classes, but {n_classes} were expected (a class is missing)")

        estimator = self.build_estimator()
        accepts_weights = "sample_weight" in inspect.signature(estimator.fit).parameters
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")
        if weights is not None and (weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError("sample_weight must be finite and non-negative, one value per row")
        if weights is not None and not accepts_weights:
            raise ValueError(f"{self.display_name} does not support sample weights")

        self.class_weight_applied_ = False
        if self.class_weight_mode == "balanced":
            mechanism = model_params.class_weight_mechanism(self.key)
            if mechanism == "class_weight":
                self.class_weight_applied_ = True  # set in build_estimator
            elif mechanism == "sample_weight":
                balanced = compute_sample_weight("balanced", y)
                weights = balanced if weights is None else balanced * weights
                self.class_weight_applied_ = True
            else:
                logger.warning("%s cannot use class weights; training without them", self.display_name)

        started = time.time()
        if weights is not None:
            estimator.fit(X, y, sample_weight=weights)
        else:
            estimator.fit(X, y)
        self.fit_seconds_ = round(time.time() - started, 3)
        self.estimator_ = estimator
        self.n_classes_ = int(present.size)
        self.n_train_rows_ = int(X.shape[0])
        self.fitted_at_ = datetime.now().isoformat(timespec="seconds")
        logger.info("%s fitted on %d rows in %.1f s", self.display_name, self.n_train_rows_, self.fit_seconds_)
        return self

    # ------------------------------------------------------------ predict
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{self.display_name} is not fitted yet; call fit first")

    def predict(self, X: Any) -> np.ndarray:
        """Integer class codes, shape ``(n_rows,)``."""
        self._require_fitted()
        return np.asarray(self.estimator_.predict(self._prepare(X))).astype("int64", copy=False)

    @property
    def score_kind(self) -> str:
        """``"probability"`` or ``"decision"`` (what ``predict_scores`` returns for this model)."""
        self._require_fitted()
        return "probability" if hasattr(self.estimator_, "predict_proba") else "decision"

    def predict_scores(self, X: Any) -> np.ndarray:
        """Scores of every class, shape ``(n_rows, n_classes)``: probabilities, or decision scores if the model has none."""
        self._require_fitted()
        X = self._prepare(X)
        if hasattr(self.estimator_, "predict_proba"):
            scores = np.asarray(self.estimator_.predict_proba(X), dtype="float64")
            totals = scores.sum(axis=1, keepdims=True)
            if (totals <= 0).any() or not np.isfinite(totals).all():
                raise ValueError(f"{self.display_name} returned probabilities that cannot be normalised")
            scores = scores / totals  # rows sum to exactly 1 (ROC-AUC checks this)
        else:
            scores = np.asarray(self.estimator_.decision_function(X))
            if scores.ndim == 1:  # two classes: one score for the positive class
                scores = np.column_stack([-scores, scores])
        if scores.shape != (X.shape[0], self.n_classes_):
            raise ValueError(f"unexpected score shape {scores.shape}, expected {(X.shape[0], self.n_classes_)}")
        return scores

    # --------------------------------------------------------------- info
    def info(self) -> dict[str, Any]:
        """JSON-safe description of the model (for reports and saved metadata)."""
        info: dict[str, Any] = {
            "key": self.key,
            "display_name": self.display_name,
            "class": type(self).__name__,
            "params": {k: v for k, v in sorted(self.params.items())},
            "class_weight_mode": self.class_weight_mode,
            "class_weight_applied": self.class_weight_applied_,
            "is_fitted": self.is_fitted,
        }
        if self.is_fitted:
            info.update(
                {
                    "estimator_class": type(self.estimator_).__name__,
                    "n_classes": self.n_classes_,
                    "n_train_rows": self.n_train_rows_,
                    "fit_seconds": self.fit_seconds_,
                    "fitted_at": self.fitted_at_,
                    "score_kind": self.score_kind,
                }
            )
        return info