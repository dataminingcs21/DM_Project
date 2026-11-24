"""Random Forest (``sklearn.ensemble.RandomForestClassifier``).

Baseline parameters (File 28): library defaults (100 trees, trees grown until the
leaves are pure) plus ``random_state`` and ``n_jobs`` (14 threads on the project
machine).  Each tree sees a bootstrap sample of the rows and a random subset of
the features at every split, so the forest is much more stable than one tree.

Cost to keep in mind: 100 fully grown trees on about 2 million training rows
take noticeably more time, memory and disk space than one tree (a saved forest
can be large).  They are measured at File 47 before the full run.  A forest only
uses the order of each feature's values, so the log transform and the scaler of
the pipeline do not change it.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from nids.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    key = "random_forest"

    def _make_estimator(self, params: dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(**params)