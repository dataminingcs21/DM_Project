"""Gaussian Naive Bayes (``sklearn.naive_bayes.GaussianNB``).

Baseline parameters (File 28): library defaults (``var_smoothing=1e-9``), no
other parameter.  The model assumes that, inside each class, every feature is
normally distributed and independent of the others.  Neither is true for network
flow features (many are flags or counts, and many are strongly related), so this
model is expected to be one of the weaker ones; it is fast and has no
randomness.

Things that follow from the assumptions:

* its probabilities are usually very close to 0 or 1 (overconfident);
* a feature that is constant inside a small class gets a tiny variance, which
  can make that class either too easy or too hard to predict;  ``var_smoothing``
  limits this and is searched by GridSearchCV (File 54);
* the class priors are the class frequencies of the training data (about 83 %
  BENIGN);
* it must compute in **float64** (``input_dtype``): see the comment in the class.
"""

from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import GaussianNB

from nids.models.base import BaseModel


class GaussianNBModel(BaseModel):
    """Gaussian Naive Bayes classifier."""

    key = "gaussian_nb"
    # Found on the real data: in float32 the log-likelihoods (down to -1e11 for features that are
    # constant inside a class) lose their precision, class scores become indistinguishable and the
    # probabilities do not sum to 1. Float64 is required.
    input_dtype = "float64"

    def _make_estimator(self, params: dict[str, Any]) -> GaussianNB:
        return GaussianNB(**params)