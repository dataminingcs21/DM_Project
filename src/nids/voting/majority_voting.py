"""Majority-vote ensemble of the supervised models.

The rule, exactly
-----------------
* Every model that takes part gives **one predicted label per flow** (``model.predict``).
* Every model has **one equal vote**.  There are no weights and **no probability
  averaging**: scores and probabilities are never used here.
* The class with the most votes wins.
* **Ties** are broken deterministically (``settings.TIE_BREAK_RULE``):

  - ``"prefer_non_benign"`` (default): among the tied classes an attack class beats
    BENIGN (an intrusion detector prefers a false alarm to a missed attack); if several
    attack classes are still tied, the one with the lowest class code wins.
  - ``"lowest_index"``: the tied class with the lowest class code wins (BENIGN is code 0).

  The same inputs therefore always give the same output.
* **Missing models are reported, never hidden.**  The vote runs on the models that are
  available; the result lists which of the expected models did not vote and why
  (not trained, failed to predict).  ``require_all=True`` refuses to vote instead.

What it does not do: train, load or score models.  ``vote_labels`` is a pure function of
predicted label arrays; ``vote_models`` additionally calls ``model.predict`` on fitted
models, isolating failures so that one broken model cannot stop the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.preprocessing.target_builder import LabelCodec
from nids.utils.logger import get_logger

logger = get_logger(__name__)


class VotingError(ValueError):
    """The vote cannot be taken (no voters, invalid predictions, missing models with ``require_all``)."""


@dataclass
class VoteResult:
    """Outcome of one vote over ``n_rows`` flows.

    ``labels``      winning class code per row (int64)
    ``counts``      votes per class, shape (n_rows, n_classes)
    ``voters``      keys of the models that voted, in the order given
    ``missing``     expected model keys that did not vote, with the reason
    ``tied``        True where two or more classes had the top vote count (before the tie-break)
    ``top_votes``   the winner's vote count per row (a measure of agreement)
    """

    labels: np.ndarray
    counts: np.ndarray
    voters: list[str]
    missing: dict[str, str] = field(default_factory=dict)
    tied: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    top_votes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="int64"))
    tie_break: str = settings.TIE_BREAK_RULE

    @property
    def n_rows(self) -> int:
        return int(self.labels.size)

    @property
    def n_voters(self) -> int:
        return len(self.voters)

    def summary(self) -> dict[str, Any]:
        """JSON-safe overview (for reports)."""
        return {
            "n_rows": self.n_rows,
            "n_voters": self.n_voters,
            "voters": list(self.voters),
            "missing": dict(self.missing),
            "tie_break_rule": self.tie_break,
            "tied_rows": int(self.tied.sum()),
            "tied_fraction": float(self.tied.mean()) if self.n_rows else 0.0,
            "unanimous_fraction": float((self.top_votes == self.n_voters).mean()) if self.n_rows else 0.0,
            "mean_top_votes": float(self.top_votes.mean()) if self.n_rows else 0.0,
        }


# ---------------------------------------------------------------------------
# The vote
# ---------------------------------------------------------------------------
def _check_prediction(key: str, pred: Any, n_classes: int, n_rows: int | None) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim != 1:
        raise VotingError(f"predictions of {key!r} must be 1-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise VotingError(f"predictions of {key!r} must be integer class codes, got dtype {arr.dtype}")
    if n_rows is not None and arr.size != n_rows:
        raise VotingError(f"{key!r} predicted {arr.size} rows but the other models predicted {n_rows}")
    if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
        raise VotingError(f"predictions of {key!r} contain class codes outside 0..{n_classes - 1}")
    return arr.astype("int64", copy=False)


def vote_labels(
    predictions: Mapping[str, Any],
    n_classes: int,
    benign_index: int | None = 0,
    tie_break: str | None = None,
    expected_keys: tuple[str, ...] | list[str] | None = None,
    missing: Mapping[str, str] | None = None,
    require_all: bool = False,
) -> VoteResult:
    """Majority vote over predicted class codes.

    ``predictions`` maps a model key to its integer label array (all of equal length).
    ``expected_keys`` defaults to the 11 mandatory models; any expected key without
    predictions is reported in ``VoteResult.missing`` (with the reason from ``missing`` if given,
    otherwise "no predictions supplied").
    """
    rule = tie_break or settings.TIE_BREAK_RULE
    if rule not in settings.ALLOWED_TIE_BREAK_RULES:
        raise VotingError(f"unknown tie-break rule {rule!r}; allowed: {list(settings.ALLOWED_TIE_BREAK_RULES)}")
    if n_classes < 2:
        raise VotingError("at least 2 classes are required")
    if benign_index is not None and not 0 <= benign_index < n_classes:
        raise VotingError(f"benign_index {benign_index} is outside 0..{n_classes - 1}")
    if not predictions:
        raise VotingError("no model supplied predictions: nothing to vote on")

    expected = list(expected_keys) if expected_keys is not None else list(model_params.MODEL_KEYS)
    reasons = dict(missing or {})
    missing_report = {k: reasons.get(k, "no predictions supplied") for k in expected if k not in predictions}
    if missing_report and require_all:
        raise VotingError("models missing from the vote: " + ", ".join(f"{k} ({v})" for k, v in missing_report.items()))

    voters = list(predictions)
    n_rows: int | None = None
    arrays = []
    for key in voters:
        arr = _check_prediction(key, predictions[key], n_classes, n_rows)
        n_rows = arr.size
        arrays.append(arr)

    counts = np.zeros((n_rows, n_classes), dtype="int64")
    rows = np.arange(n_rows)
    for arr in arrays:  # one equal vote per model
        counts[rows, arr] += 1

    top = counts.max(axis=1)
    leaders = counts == top[:, None]  # all classes sharing the top count
    tied = leaders.sum(axis=1) > 1

    candidates = leaders
    if rule == "prefer_non_benign" and benign_index is not None:
        without_benign = leaders.copy()
        without_benign[:, benign_index] = False
        has_attack = without_benign.any(axis=1)
        candidates = np.where(has_attack[:, None], without_benign, leaders)
    labels = candidates.argmax(axis=1).astype("int64")  # first True = lowest class code

    if missing_report:
        logger.warning("Majority vote without %d model(s): %s", len(missing_report), ", ".join(missing_report))
    logger.info(
        "Majority vote: %d voters, %d rows, %d tied row(s) broken by %s", len(voters), n_rows, int(tied.sum()), rule
    )
    return VoteResult(
        labels=labels, counts=counts, voters=voters, missing=missing_report, tied=tied, top_votes=top, tie_break=rule
    )


# ---------------------------------------------------------------------------
# Voting with fitted models
# ---------------------------------------------------------------------------
def collect_predictions(
    models: Mapping[str, Any], X: Any, continue_on_error: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """``model.predict(X)`` for every model; returns ``(predictions, failures)``.

    A model that raises is recorded in ``failures`` (key -> error text) and the others continue;
    with ``continue_on_error=False`` the first error is raised.
    """
    predictions: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for key, model in models.items():
        try:
            predictions[key] = np.asarray(model.predict(X))
        except Exception as exc:  # noqa: BLE001 - one broken model must not stop the others
            if not continue_on_error:
                raise
            failures[key] = f"{type(exc).__name__}: {exc}"
            logger.error("Model %s failed to predict and does not vote: %s", key, failures[key])
    return predictions, failures


def vote_models(
    models: Mapping[str, Any],
    X: Any,
    codec: LabelCodec,
    expected_keys: tuple[str, ...] | list[str] | None = None,
    not_loaded: Mapping[str, str] | None = None,
    tie_break: str | None = None,
    require_all: bool = False,
) -> VoteResult:
    """Predict with every fitted model, then vote.

    ``not_loaded`` carries the reasons for models that could not be loaded (key -> reason);
    models that fail while predicting are added to the missing list automatically.
    """
    predictions, failures = collect_predictions(models, X)
    reasons = {**(not_loaded or {}), **failures}
    return vote_labels(
        predictions,
        n_classes=codec.n_classes,
        benign_index=codec.benign_index,
        tie_break=tie_break,
        expected_keys=expected_keys,
        missing=reasons,
        require_all=require_all,
    )


# ---------------------------------------------------------------------------
# Explanation of a single vote (used by the application)
# ---------------------------------------------------------------------------
def explain_vote(
    predictions: Mapping[str, Any], result: VoteResult, codec: LabelCodec, row: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Who voted for what on one row, and the tally.

    Returns ``(voters, tally)``: ``voters`` has one line per model (``model``, ``vote``),
    ``tally`` one line per class that received a vote (``class``, ``votes``, ``winner``), most votes first.
    """
    if not 0 <= row < result.n_rows:
        raise VotingError(f"row {row} is outside 0..{result.n_rows - 1}")
    voters = pd.DataFrame(
        {
            "model": [model_params.DISPLAY_NAMES.get(k, k) for k in result.voters],
            "vote": [codec.classes[int(np.asarray(predictions[k])[row])] for k in result.voters],
        }
    )
    counts = result.counts[row]
    winner = int(result.labels[row])
    order = [i for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
    tally = pd.DataFrame(
        {
            "class": [codec.classes[i] for i in order],
            "votes": [int(counts[i]) for i in order],
            "winner": [i == winner for i in order],
        }
    )
    return voters, tally