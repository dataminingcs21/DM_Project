"""SGD Classifier (``sklearn.linear_model.SGDClassifier``).

Baseline parameters (File 28): library defaults (``loss="hinge"``, L2 penalty
with ``alpha=0.0001``, stochastic gradient descent one-versus-rest) plus
``random_state``, ``max_iter=1000`` and ``n_jobs``.

With the hinge loss the model is a linear support-vector machine trained by
stochastic gradient descent.  It has **no** ``predict_proba``: ``predict_scores``
returns the decision scores (one per class, not probabilities, rows do not sum
to 1), and ``score_kind`` is ``"decision"``.  The evaluation uses such scores
only for ranking-based curves.  The majority vote uses only the predicted class.

It relies on the standardized features of the pipeline.  ``info()`` adds
``n_iter`` and ``converged``; ``converged`` is False when the optimizer used
all ``max_iter`` epochs, which means the result is not a true optimum and the
report must say so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier

from nids.models.base import BaseModel
from nids.utils.logger import get_logger

logger = get_logger("models.sgd")


class SGDModel(BaseModel):
    """Linear classifier trained by stochastic gradient descent."""

    key = "sgd"

    def _make_estimator(self, params: dict[str, Any]) -> SGDClassifier:
        return SGDClassifier(**params)

    def fit(self, X: Any, y: Any, sample_weight: Any = None, n_classes: int | None = None) -> "SGDModel":
        super().fit(X, y, sample_weight=sample_weight, n_classes=n_classes)
        if not self.converged:
            logger.warning(
                "SGD Classifier used all %d epochs without converging: the result is not a true optimum",
                int(self.estimator_.max_iter),
            )
        return self

    @property
    def converged(self) -> bool:
        """True if the optimizer stopped before reaching ``max_iter`` epochs."""
        self._require_fitted()
        return bool(int(np.max(self.estimator_.n_iter_)) < int(self.estimator_.max_iter))

    def info(self) -> dict[str, Any]:
        info = super().info()
        if self.is_fitted:
            info["n_iter"] = int(np.max(self.estimator_.n_iter_))
            info["max_iter"] = int(self.estimator_.max_iter)
            info["converged"] = self.converged
        return info