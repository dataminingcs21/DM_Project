"""Evaluation metrics for the multiclass intrusion detector.

Everything here is a pure function of arrays (true labels, predicted labels,
scores), so the same code scores a model on the test set (Files 52-53), on a
cross-validation fold (File 49) or on any other rows.  Results are plain
JSON-safe dictionaries.

What is computed
----------------
* **Overall:** accuracy, balanced accuracy (mean recall over classes), macro and
  weighted precision / recall / F1, Matthews correlation coefficient.  Macro scores
  give every class the same weight, so they expose weak rare classes that accuracy
  hides: with 83% BENIGN, accuracy alone says little.
* **Per class:** precision, recall, F1 and the **support** (number of true rows).
  A class with fewer than ``LOW_SUPPORT_THRESHOLD`` true rows is flagged
  ``low_support``: its scores rest on a handful of rows and are unreliable (on this
  project's test set Heartbleed has 2 rows, SQL Injection 4, Infiltration 7).  The
  flag is reported next to every such number, never hidden.
* **Confusion matrix** (rows = true class, columns = predicted class) and its most
  frequent confusions.
* **Attack-versus-normal view**, the way an intrusion detection system is judged: a
  flow is "attack" if its class is not BENIGN.  Detection rate = attack flows predicted
  as *any* attack; false alarm rate = BENIGN flows predicted as any attack.  Getting the
  attack *type* wrong still counts as a detection here.
* **ROC-AUC, one class against the rest**, per class and macro-averaged.  Computed from
  the model's scores: probabilities for most models, decision scores for SVC and SGD
  (AUC only uses the ranking, so decision scores are valid, but they are not comparable
  as probabilities; the result records ``score_kind``).  A class that is absent from the
  true labels, or has no other-class rows, gets ``null`` and is left out of the macro mean.

Predicted labels always come from ``model.predict`` (that is what the majority vote
uses); the scores are only used for AUC.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from nids.preprocessing.target_builder import LabelCodec

LOW_SUPPORT_THRESHOLD = 30  # fewer true rows than this: the class scores are unreliable


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _check_inputs(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None, n_classes: int) -> None:
    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError(f"y_true and y_pred must be 1-D with the same length, got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no rows to evaluate")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must hold integer class codes, found {arr.dtype}")
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(f"{name} has class codes outside 0..{n_classes - 1}")
    if scores is not None:
        if scores.shape != (y_true.size, n_classes):
            raise ValueError(f"scores must have shape {(y_true.size, n_classes)}, got {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("scores contain missing or infinite values")


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
def per_class_auc(y_true: np.ndarray, scores: np.ndarray, n_classes: int) -> list[float | None]:
    """One-versus-rest ROC-AUC for every class (``None`` where it cannot be computed)."""
    aucs: list[float | None] = []
    for k in range(n_classes):
        positive = y_true == k
        if positive.sum() == 0 or positive.all():
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(positive, scores[:, k])))
    return aucs


def binary_view(cm: np.ndarray, benign_index: int | None) -> dict[str, Any] | None:
    """Attack-versus-normal numbers from a multiclass confusion matrix."""
    if benign_index is None:
        return None
    total = cm.sum()
    benign_true = cm[benign_index].sum()
    attack_true = total - benign_true
    benign_as_benign = cm[benign_index, benign_index]
    attack_as_attack = cm.sum() - cm[benign_index].sum() - (cm[:, benign_index].sum() - cm[benign_index, benign_index])
    false_alarms = benign_true - benign_as_benign
    missed_attacks = attack_true - attack_as_attack
    flagged = attack_as_attack + false_alarms

    def ratio(a: float, b: float) -> float | None:
        return float(a / b) if b > 0 else None

    detection = ratio(attack_as_attack, attack_true)
    precision = ratio(attack_as_attack, flagged)
    f1 = None if detection is None or precision is None or (detection + precision) == 0 else 2 * detection * precision / (detection + precision)
    return {
        "attack_rows": int(attack_true),
        "benign_rows": int(benign_true),
        "detection_rate": detection,  # attacks predicted as any attack
        "false_alarm_rate": ratio(false_alarms, benign_true),  # BENIGN predicted as any attack
        "attack_precision": precision,
        "attack_f1": f1,
        "missed_attacks": int(missed_attacks),
        "false_alarms": int(false_alarms),
    }


def top_confusions(cm: np.ndarray, classes: tuple[str, ...], n: int = 10) -> list[dict[str, Any]]:
    """The ``n`` most frequent wrong (true class, predicted class) pairs."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")[:n]
    rows = []
    for position in flat:
        i, j = np.unravel_index(position, off.shape)
        if off[i, j] == 0:
            break
        rows.append(
            {
                "true_class": classes[i],
                "predicted_class": classes[j],
                "rows": int(off[i, j]),
                "share_of_true_class": float(off[i, j] / cm[i].sum()) if cm[i].sum() else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    codec: LabelCodec,
    scores: np.ndarray | None = None,
    score_kind: str | None = None,
) -> dict[str, Any]:
    """All metrics for one model on one set of rows (see the module docstring)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = None if scores is None else np.asarray(scores)
    k = codec.n_classes
    _check_inputs(y_true, y_pred, scores, k)
    labels = np.arange(k)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    present = support > 0  # classes that exist in the true labels
    aucs = per_class_auc(y_true, scores, k) if scores is not None else [None] * k
    valid_aucs = [a for a in aucs if a is not None]

    per_class = []
    for i, name in enumerate(codec.classes):
        per_class.append(
            {
                "class_code": i,
                "class": name,
                "support": int(support[i]),
                "low_support": bool(support[i] < LOW_SUPPORT_THRESHOLD),
                "predicted_rows": int(cm[:, i].sum()),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "roc_auc_ovr": aucs[i],
            }
        )

    metrics: dict[str, Any] = {
        "n_rows": int(y_true.size),
        "n_classes": k,
        "score_kind": score_kind,
        "overall": {
            "accuracy": float((y_true == y_pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision[present].mean()),
            "macro_recall": float(recall[present].mean()),
            "macro_f1": float(f1[present].mean()),
            "weighted_f1": float(weighted_f1),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "roc_auc_ovr_macro": float(np.mean(valid_aucs)) if valid_aucs else None,
        },
        "macro_over_classes": int(present.sum()),
        "low_support_classes": [c["class"] for c in per_class if c["low_support"] and c["support"] > 0],
        "missing_classes": [c["class"] for c in per_class if c["support"] == 0],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(codec.classes),
        "top_confusions": top_confusions(cm, codec.classes),
        "attack_vs_normal": binary_view(cm, codec.benign_index),
    }
    return metrics


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, codec: LabelCodec, with_scores: bool = True) -> dict[str, Any]:
    """Predict with a fitted model and compute its metrics; also measures the prediction time.

    ``with_scores=False`` skips ``predict_scores`` (no AUC): used where only labels matter, since
    KNN and SVC need as long again for the scores.
    """
    X = np.asarray(X)
    started = time.time()
    y_pred = model.predict(X)
    predict_seconds = time.time() - started
    scores, score_seconds = None, None
    if with_scores:
        started = time.time()
        scores = model.predict_scores(X)
        score_seconds = time.time() - started
    result = compute_metrics(y, y_pred, codec, scores=scores, score_kind=model.score_kind if with_scores else None)
    result["model_key"] = model.key
    result["predict_seconds"] = round(predict_seconds, 3)
    result["score_seconds"] = None if score_seconds is None else round(score_seconds, 3)
    return result


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def per_class_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """The per-class results as a table (support always included; low-support rows flagged)."""
    frame = pd.DataFrame(metrics["per_class"])
    frame["note"] = np.where(frame["low_support"], "low support: unreliable", "")
    return frame[["class_code", "class", "support", "predicted_rows", "precision", "recall", "f1", "roc_auc_ovr", "note"]]


def confusion_frame(metrics: dict[str, Any], normalize: bool = False) -> pd.DataFrame:
    """The confusion matrix with class names; ``normalize=True`` divides each row by its total (row = true class)."""
    cm = np.asarray(metrics["confusion_matrix"], dtype="float64")
    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
    names = metrics["class_names"]
    return pd.DataFrame(cm if normalize else cm.astype("int64"), index=pd.Index(names, name="true"), columns=pd.Index(names, name="predicted"))