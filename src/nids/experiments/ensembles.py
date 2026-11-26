"""Better ways to combine the eleven models than one equal vote each.

The equal-weight majority vote lets seven weak models (macro F1 0.17-0.71) outvote the three
strong tree ensembles.  The alternatives here learn how much to trust each model **from the
training data only**, using the out-of-fold (OOF) predictions of the 5-fold cross-validation
(``reports/evaluation/cross_validation/cv_<model>_oof.npy``: for every training row, the class
predicted by a copy of the model that did not train on that row).  The test set is used once,
at the end, to score every combiner.

Combiners:

* ``weighted_vote`` with one weight per model (its OOF macro F1), optionally raised to a power
  to sharpen the differences;
* ``weighted_vote`` with one weight per model **and class** (its OOF F1 on that class): a model
  that is good at PortScan but bad at Bot is trusted for PortScan votes only;
* ``best_subset``: the equal vote of the top-n models by OOF macro F1;
* **stacking**: a meta-classifier (logistic regression, or XGBoost) that takes the eleven
  predicted labels (one-hot) as input and learns which combination means which class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("experiments.ensembles")


# ---------------------------------------------------------------------------
# Weights learnt from out-of-fold predictions
# ---------------------------------------------------------------------------
def oof_model_weights(oof: dict[str, np.ndarray], y: np.ndarray, n_classes: int) -> dict[str, float]:
    """Macro F1 of every model's OOF predictions (one weight per model)."""
    labels = np.arange(n_classes)
    return {k: float(f1_score(y, p, labels=labels, average="macro", zero_division=0)) for k, p in oof.items()}


def oof_class_weights(oof: dict[str, np.ndarray], y: np.ndarray, n_classes: int) -> dict[str, np.ndarray]:
    """Per-class F1 of every model's OOF predictions, shape ``(n_classes,)`` per model."""
    labels = np.arange(n_classes)
    return {k: f1_score(y, p, labels=labels, average=None, zero_division=0).astype("float64") for k, p in oof.items()}


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------
def _break_ties(totals: np.ndarray, benign_index: int | None) -> np.ndarray:
    """Arg-max per row; among (near-)tied classes an attack beats BENIGN, then the lowest code (as in File 53)."""
    best = totals.max(axis=1, keepdims=True)
    tied = np.isclose(totals, best, rtol=0.0, atol=1e-12)
    if benign_index is not None:
        only_benign = tied[:, benign_index] & (tied.sum(axis=1) == 1)
        tied[:, benign_index] &= only_benign
    return tied.argmax(axis=1)


def weighted_vote(
    predictions: dict[str, np.ndarray],
    n_classes: int,
    weights: dict[str, float] | dict[str, np.ndarray] | None = None,
    benign_index: int | None = 0,
    power: float = 1.0,
) -> np.ndarray:
    """Class with the largest total weight. ``weights[k]`` is a scalar or a per-class array; None = equal vote."""
    if not predictions:
        raise ValueError("no predictions to vote on")
    n_rows = {len(p) for p in predictions.values()}
    if len(n_rows) != 1:
        raise ValueError(f"predictions have different lengths: {sorted(n_rows)}")
    rows = np.arange(n_rows.pop())
    totals = np.zeros((rows.size, n_classes), dtype="float64")
    for key, labels in predictions.items():
        labels = np.asarray(labels, dtype="int64")
        if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError(f"{key}: class codes outside 0..{n_classes - 1}")
        w = 1.0 if weights is None else weights[key]
        w = np.power(np.asarray(w, dtype="float64"), power)
        totals[rows, labels] += w[labels] if w.ndim == 1 else float(w)
    return _break_ties(totals, benign_index)


def best_subset(model_weights: dict[str, float], n: int) -> list[str]:
    """The ``n`` models with the largest weight (ties by key, for determinism)."""
    return [k for k, _ in sorted(model_weights.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------
def one_hot_votes(predictions: dict[str, np.ndarray], keys: list[str], n_classes: int) -> sparse.csr_matrix:
    """Sparse one-hot matrix: one block of ``n_classes`` columns per model (in ``keys`` order)."""
    n_rows = len(predictions[keys[0]])
    blocks = []
    for key in keys:
        labels = np.asarray(predictions[key], dtype="int64")
        blocks.append(sparse.csr_matrix((np.ones(n_rows, dtype="float32"), (np.arange(n_rows), labels)), shape=(n_rows, n_classes)))
    return sparse.hstack(blocks, format="csr")


def fit_stacker(kind: str, Z: sparse.csr_matrix, y: np.ndarray, n_classes: int) -> Any:
    """Meta-classifier on the one-hot votes: ``"logistic"`` or ``"xgboost"``."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=300, C=1.0, solver="saga", tol=1e-3, random_state=settings.RANDOM_SEED)
    elif kind == "xgboost":
        from xgboost import XGBClassifier

        device = "cuda" if settings.XGBOOST_DEVICE in ("auto", "cuda") and _cuda_ok() else "cpu"
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            device=device,
            num_class=n_classes,
            objective="multi:softmax",
            random_state=settings.RANDOM_SEED,
            n_jobs=settings.N_JOBS,
        )
    else:
        raise ValueError(f"unknown stacker kind {kind!r}; use 'logistic' or 'xgboost'")
    logger.info("Fitting the %s stacker on %d rows x %d columns", kind, Z.shape[0], Z.shape[1])
    return model.fit(Z, y)


def _cuda_ok() -> bool:
    try:
        import xgboost

        return bool(xgboost.build_info().get("USE_CUDA"))
    except Exception:  # pragma: no cover - depends on the machine
        return False
