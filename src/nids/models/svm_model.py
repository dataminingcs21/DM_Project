"""Support Vector Classifier (``sklearn.svm.SVC``).

Baseline parameters (File 28): library defaults plus ``random_state``: RBF
kernel, ``C=1.0``, ``gamma="scale"``, ``probability=False``.

The RBF kernel lets the model draw curved boundaries, but the cost of training
grows roughly with the **square** of the number of rows, and prediction cost
grows with the number of support vectors.  That is why ``config/model_params.py``
trains it on a class-capped stratified subsample (``SUBSAMPLE_SIZES``, currently
50,000 rows; calibrated at File 47).  The comparison table must say that it was
trained on a subsample.

``probability=True`` is deliberately **not** set: it runs an internal 5-fold
cross-validation and multiplies the training time by about five.  The model
therefore has no probabilities: ``predict_scores`` returns the decision scores
(one per class, from the one-versus-one votes; not probabilities, rows do not
sum to 1) and ``score_kind`` is ``"decision"``.  The majority vote uses only the
predicted class.

Note: for SVC, ``predict`` follows the one-versus-one vote of the library, while
the per-class scores are derived from the same votes; in rare ties the largest
score can differ from the predicted class, so the predicted class is always the
one from ``predict``.
"""

from __future__ import annotations

from typing import Any

from sklearn.svm import SVC

from nids.models.base import BaseModel


class SVMModel(BaseModel):
    """Support Vector Classifier with an RBF kernel."""

    key = "svm"

    def _make_estimator(self, params: dict[str, Any]) -> SVC:
        return SVC(**params)