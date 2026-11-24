"""Gradient Boosting (``sklearn.ensemble.GradientBoostingClassifier``).

Baseline parameters (File 28): library defaults plus ``random_state``:
``n_estimators=100`` rounds, ``learning_rate=0.1``, ``max_depth=3``.  Trees are
built one after another, each one correcting the errors of the sum of the
previous ones.

With 15 classes, **every round builds one tree per class**, so a default fit
builds 1,500 trees, one after another and on a single thread (this estimator is
not parallel).  It is by far the most expensive scikit-learn model of the 11,
which is why ``config/model_params.py`` trains it on a class-capped stratified
subsample (``SUBSAMPLE_SIZES``, currently 100,000 rows; calibrated at File 47).
The comparison table must say that it was trained on a subsample.

The model has ``predict_proba``; ``score_kind`` is ``"probability"``.  Class
weights (only in the later ``balanced`` experiment) go through ``sample_weight``.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier

from nids.models.base import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier (scikit-learn implementation)."""

    key = "gradient_boosting"

    def _make_estimator(self, params: dict[str, Any]) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(**params)