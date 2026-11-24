"""Decision Tree (``sklearn.tree.DecisionTreeClassifier``), model 1 of the model files.

Baseline parameters (File 28): library defaults plus ``random_state``.  With the
defaults the tree grows until every leaf is pure, so on 2 million training rows
it can become large and fits the training data almost perfectly; its
``predict_proba`` is then close to hard 0/1 values.  This is a property of the
untuned baseline; GridSearchCV (File 54) searches the depth and the leaf sizes.

A tree only uses the order of each feature's values, so it is not affected by
the log transform or the scaler of the preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any

from sklearn.tree import DecisionTreeClassifier

from nids.models.base import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree classifier."""

    key = "decision_tree"

    def _make_estimator(self, params: dict[str, Any]) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**params)