"""Logistic Regression (``sklearn.linear_model.LogisticRegression``).

Baseline parameters (File 28): library defaults (L2 penalty with ``C=1.0``, the
``lbfgs`` solver, one multinomial model over all classes) plus ``random_state``
and ``max_iter=1000`` (the default of 100 iterations is usually too few to
converge on 40 standardized features).

A linear model draws straight boundaries between classes, so it is expected to
trail the tree models on this data; it is fast, stable and gives usable
probabilities.  It relies on the scaler of the preprocessing pipeline: without
standardized features ``lbfgs`` converges slowly.

``info()`` adds ``n_iter`` and ``converged``.  ``converged`` is False when the
solver used all ``max_iter`` iterations, which means the result is not a true
optimum (scikit-learn would also print a ConvergenceWarning); the report must
say so instead of hiding it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from nids.models.base import BaseModel
from nids.utils.logger import get_logger

logger = get_logger("models.logistic_regression")


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""

    key = "logistic_regression"

    def _make_estimator(self, params: dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**params)

    def fit(self, X: Any, y: Any, sample_weight: Any = None, n_classes: int | None = None) -> "LogisticRegressionModel":
        super().fit(X, y, sample_weight=sample_weight, n_classes=n_classes)
        if not self.converged:
            logger.warning(
                "Logistic Regression used all %d iterations without converging: the result is not a true optimum",
                int(self.estimator_.max_iter),
            )
        return self

    @property
    def converged(self) -> bool:
        """True if the solver stopped before reaching ``max_iter`` iterations."""
        self._require_fitted()
        return bool(int(np.max(self.estimator_.n_iter_)) < int(self.estimator_.max_iter))

    def info(self) -> dict[str, Any]:
        info = super().info()
        if self.is_fitted:
            info["n_iter"] = int(np.max(self.estimator_.n_iter_))
            info["max_iter"] = int(self.estimator_.max_iter)
            info["converged"] = self.converged
        return info