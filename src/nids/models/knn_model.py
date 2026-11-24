"""K-Nearest Neighbors (``sklearn.neighbors.KNeighborsClassifier``).

Baseline parameters (File 28): library defaults (5 neighbors, uniform weights,
Euclidean distance, algorithm chosen automatically) plus ``n_jobs``.  KNN has no
real training step: ``fit`` only stores the training rows, and every prediction
compares the new flow with all stored rows.  That is why this model trains on a
class-capped stratified **subsample** (``settings.SUBSAMPLE_SIZES["knn"]``, 100,000
rows to start with) and why predicting the whole test set takes much longer than
training.

KNN uses distances, so it relies on the scaler of the preprocessing pipeline
(the log transform and the standard scaler).  Its ``predict_proba`` is the share
of the 5 neighbors in each class: only the values 0, 0.2, 0.4, ... 1.  It cannot
use class weights (the training weights of the other models do not exist here).
"""

from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier

from nids.models.base import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier."""

    key = "knn"

    def _make_estimator(self, params: dict[str, Any]) -> KNeighborsClassifier:
        return KNeighborsClassifier(**params)