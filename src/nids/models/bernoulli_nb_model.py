"""Bernoulli Naive Bayes (``sklearn.naive_bayes.BernoulliNB``).

Baseline parameters (File 28): library defaults, no other parameter
(``alpha=1.0`` smoothing, ``binarize=0.0``).  The model works on **binary**
features: every value is turned into 1 if it is above ``binarize`` and 0
otherwise, and then each feature is modelled as a coin flip inside each class.

On this project's data that has a specific meaning.  The preprocessing pipeline
standardizes every feature (mean 0 on the training rows), so ``binarize=0.0``
turns each feature into "above or below the training average".  A large part of
the information in the values is thrown away, so this model is expected to be
among the weakest of the 11.  That is a property of the algorithm on this data,
not a defect; ``binarize`` and ``alpha`` are searched by GridSearchCV (File 54).

Its smoothing (``alpha``) prevents the zero probabilities that make Gaussian
Naive Bayes unstable, and the class priors are the class frequencies of the
training data.
"""

from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import BernoulliNB

from nids.models.base import BaseModel


class BernoulliNBModel(BaseModel):
    """Bernoulli Naive Bayes classifier."""

    key = "bernoulli_nb"

    def _make_estimator(self, params: dict[str, Any]) -> BernoulliNB:
        return BernoulliNB(**params)