"""AdaBoost (``sklearn.ensemble.AdaBoostClassifier``).

Baseline parameters (File 28): library defaults plus ``random_state``.  That
means 50 boosting rounds (``n_estimators=50``), ``learning_rate=1.0`` and the
default weak learner, a decision stump (a tree of depth 1: a single question
about a single feature).

AdaBoost trains the stumps one after another, each one giving more weight to
the rows the previous ones got wrong, and the final answer is a weighted vote of
the stumps.  With 15 classes, 50 stumps is a very small model, so this baseline
is expected to be clearly weaker than the tree ensembles; ``n_estimators``,
``learning_rate`` and the depth of the weak learner are what GridSearchCV
(File 56) can change.

**Depth option (added at File 56).**  ``base_max_depth`` (an integer, not a
scikit-learn parameter) sets the depth of the weak learner: ``base_max_depth=1``
is the default stump, ``2`` or ``4`` allow each weak learner to ask a chain of
questions.  When the option is absent the model is exactly the baseline, so the
saved baseline model and its staleness hash are unchanged.  The option is kept
as a plain integer (not a tree object) so that the saved metadata stays JSON.

The model has ``predict_proba``; ``score_kind`` is ``"probability"``.  Class
weights (only in the later ``balanced`` experiment) go through ``sample_weight``.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from nids.models.base import BaseModel


class AdaBoostModel(BaseModel):
    """AdaBoost classifier."""

    key = "adaboost"

    def _make_estimator(self, params: dict[str, Any]) -> AdaBoostClassifier:
        params = dict(params)
        depth = params.pop("base_max_depth", None)
        if depth is not None:
            if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
                raise ValueError(f"base_max_depth must be an integer >= 1, got {depth!r}")
            params["estimator"] = DecisionTreeClassifier(max_depth=depth, random_state=params.get("random_state"))
        return AdaBoostClassifier(**params)