"""ROC and precision-recall curves, one class against the rest.

A confusion matrix judges the *decisions* of a model.  A curve judges its *scores*: how well the model
ranks the flows of one class above all the others, over every possible threshold.  This is how an
intrusion detection system is tuned in practice (how many attacks do we catch for how many false alarms?).

* **ROC curve** of class *k*: true positive rate against false positive rate, treating class *k* as
  "positive" and every other class as "negative".  Its area is the ROC-AUC of ``metrics.py``.
* **Precision-recall curve** of class *k*: precision against recall, with its area summarised as the
  *average precision* (AP).  With very rare classes it is more informative than the ROC curve: the false
  positive rate stays tiny when negatives are millions, even if most alerts are wrong.  The baseline of a
  random scorer is the prevalence of the class (a horizontal line).

``compute_curves`` takes the scores of any model (probabilities, or decision scores for SVC and SGD: only
the ranking matters) and returns a JSON-safe dictionary with the per-class curves (reduced to at most
``MAX_POINTS`` points for plotting) and the macro-averaged curves.  A class with no true rows, or with no
other-class rows, has no curve and is listed under ``missing_classes``; a class with a low support is flagged
(its curve rests on a handful of rows).

Macro-averaging: the ROC curves of all classes are interpolated on a common false-positive-rate grid and
their true positive rates averaged; the precision-recall curves are turned into their interpolated
precision envelope on a common recall grid and averaged.  The macro AUC and macro AP are the plain means of the
per-class values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from nids.evaluation.metrics import LOW_SUPPORT_THRESHOLD  # noqa: E402
from nids.evaluation.plots import (  # noqa: E402
    BAND, BLUE, GRID, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE, _save, _style_axes, short_class_name,
)
from nids.preprocessing.target_builder import LabelCodec  # noqa: E402

MAX_POINTS = 400
GRID_POINTS = 501
# Fixed order of series colours and a second code (line style) for overlaid models: never cycled past 4.
MODEL_STYLES = [
    ("#2a78d6", "-"),   # blue
    ("#eb6834", "--"),  # orange
    ("#1baf7a", "-."),  # aqua
    ("#6250d6", ":"),   # violet
]
MAX_OVERLAID_MODELS = len(MODEL_STYLES)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _reduce(*arrays: np.ndarray, max_points: int = MAX_POINTS) -> list[list[float]]:
    """Keep at most ``max_points`` evenly spaced points (always the first and the last) of curves of equal length."""
    n = len(arrays[0])
    if n <= max_points:
        keep = np.arange(n)
    else:
        keep = np.unique(np.concatenate([np.linspace(0, n - 1, max_points).astype(int), [0, n - 1]]))
    return [np.asarray(a)[keep].astype(float).tolist() for a in arrays]


def _interpolated_precision(recall_asc: np.ndarray, precision_asc: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolated precision at each recall in ``grid``: the best precision among the points whose recall is >= that value."""
    suffix_best = np.maximum.accumulate(precision_asc[::-1])[::-1]
    index = np.minimum(np.searchsorted(recall_asc, grid - 1e-12, side="left"), len(recall_asc) - 1)  # tolerance: float noise
    return suffix_best[index]


def compute_curves(y_true: np.ndarray, scores: np.ndarray, codec: LabelCodec, score_kind: str | None = None) -> dict[str, Any]:
    """Per-class and macro ROC / precision-recall curves for one model on one set of rows."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = codec.n_classes
    if scores.shape != (y_true.size, k):
        raise ValueError(f"scores must have shape {(y_true.size, k)}, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain missing or infinite values")
    if y_true.size == 0 or not np.issubdtype(y_true.dtype, np.integer) or y_true.min() < 0 or y_true.max() >= k:
        raise ValueError("y_true must be a non-empty vector of class codes in 0..K-1")

    fpr_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    recall_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    tprs, precisions = [], []
    per_class, missing = [], []
    for c in range(k):
        positive = y_true == c
        support = int(positive.sum())
        if support == 0 or support == positive.size:
            missing.append(codec.classes[c])
            continue
        fpr, tpr, _ = roc_curve(positive, scores[:, c])
        precision, recall, _ = precision_recall_curve(positive, scores[:, c])
        auc = float(roc_auc_score(positive, scores[:, c]))
        ap = float(average_precision_score(positive, scores[:, c]))
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        order = np.argsort(recall, kind="stable")  # recall in increasing order (scikit-learn returns it decreasing)
        precisions.append(_interpolated_precision(recall[order], precision[order], recall_grid))
        r_fpr, r_tpr = _reduce(fpr, tpr)
        r_rec, r_prec = _reduce(recall[order], precision[order])
        per_class.append(
            {
                "class_code": c,
                "class": codec.classes[c],
                "support": support,
                "low_support": bool(support < LOW_SUPPORT_THRESHOLD),
                "prevalence": support / positive.size,
                "roc_auc": auc,
                "average_precision": ap,
                "roc": {"fpr": r_fpr, "tpr": r_tpr},
                "pr": {"recall": r_rec, "precision": r_prec},
            }
        )
    if not per_class:
        raise ValueError("no class has both positive and negative rows: no curve can be drawn")
    macro_tpr = np.mean(tprs, axis=0)
    macro_precision = np.mean(precisions, axis=0)
    return {
        "n_rows": int(y_true.size),
        "score_kind": score_kind,
        "classes": per_class,
        "missing_classes": missing,
        "low_support_classes": [c["class"] for c in per_class if c["low_support"]],
        "macro": {
            "roc_auc": float(np.mean([c["roc_auc"] for c in per_class])),
            "average_precision": float(np.mean([c["average_precision"] for c in per_class])),
            "roc": {"fpr": fpr_grid.tolist(), "tpr": macro_tpr.tolist()},
            "pr": {"recall": recall_grid.tolist(), "precision": macro_precision.tolist()},
        },
    }


def curves_table(curves: dict[str, Any]) -> pd.DataFrame:
    """Per-class ROC-AUC and average precision, with the support and a note for low-support classes."""
    frame = pd.DataFrame(
        [
            {
                "class": c["class"], "support": c["support"], "prevalence": c["prevalence"],
                "roc_auc": c["roc_auc"], "average_precision": c["average_precision"],
                "note": "low support: unreliable" if c["low_support"] else "",
            }
            for c in curves["classes"]
        ]
    )
    return frame


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_class_curves(curves: dict[str, Any], path: str | Path, kind: str = "roc", title: str | None = None, ncols: int = 5) -> Path:
    """One small panel per class (ROC or precision-recall); low-support classes are shaded.

    Each panel is titled with the class, its support and its area (AUC or AP).  In ROC panels the dashed diagonal
    is a random scorer; in precision-recall panels the dashed horizontal line is the prevalence of the class.
    """
    if kind not in ("roc", "pr"):
        raise ValueError("kind must be 'roc' or 'pr'")
    classes = curves["classes"]
    n = len(classes)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.55 * nrows + 0.9), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, c in zip(axes.ravel(), classes):
        ax.set_visible(True)
        ax.set_facecolor(BAND if c["low_support"] else SURFACE)
        _style_axes(ax, "both")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9, linestyle="--")
            ax.plot(c["roc"]["fpr"], c["roc"]["tpr"], color=BLUE, linewidth=1.6)
            area = f"AUC {c['roc_auc']:.3f}"
        else:
            ax.plot([0, 1], [c["prevalence"]] * 2, color=INK_MUTED, linewidth=0.9, linestyle="--")
            ax.plot(c["pr"]["recall"], c["pr"]["precision"], color=ORANGE, linewidth=1.6)
            area = f"AP {c['average_precision']:.3f}"
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7.5)
        flag = "  low support" if c["low_support"] else ""
        ax.set_title(f"{short_class_name(c['class'])}\nn={c['support']:,}  {area}{flag}", fontsize=8.5, color=INK, loc="left")
    xlabel, ylabel = ("False positive rate", "True positive rate") if kind == "roc" else ("Recall", "Precision")
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(xlabel, fontsize=8.5, color=INK_SECONDARY)
    for row in axes:
        if row[0].get_visible():
            row[0].set_ylabel(ylabel, fontsize=8.5, color=INK_SECONDARY)
    default = "ROC curve per class (one class against the rest)" if kind == "roc" else "Precision-recall curve per class (one class against the rest)"
    fig.suptitle(title or default, x=0.01, ha="left", fontsize=12, color=INK)
    note = ("Dashed: a random scorer. Shaded panels: classes with few true rows, whose curves are unreliable."
            if kind == "roc" else
            "Dashed: the prevalence of the class (a random scorer). Shaded panels: classes with few true rows, whose curves are unreliable.")
    fig.text(0.01, -0.005, note, fontsize=8, color=INK_MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_macro_curves(
    curves_by_model: dict[str, dict[str, Any]],
    path: str | Path,
    kind: str = "roc",
    title: str | None = None,
    log_fpr: bool = True,
) -> Path:
    """Macro-averaged curves of up to 4 models on one chart (line colour and line style differ).

    ``curves_by_model`` maps a model name to its ``compute_curves`` result; the dictionary order is the legend order.
    For ROC curves ``log_fpr=True`` draws the false positive rate on a log axis, which shows the differences in the
    region that matters for an intrusion detector (few false alarms); the footnote says so.
    """
    if kind not in ("roc", "pr"):
        raise ValueError("kind must be 'roc' or 'pr'")
    if not curves_by_model:
        raise ValueError("no models to plot")
    if len(curves_by_model) > MAX_OVERLAID_MODELS:
        raise ValueError(f"at most {MAX_OVERLAID_MODELS} models can be overlaid legibly; pass the best {MAX_OVERLAID_MODELS} or use plot_class_curves per model")
    fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _style_axes(ax, "both")
    use_log = kind == "roc" and log_fpr
    if kind == "roc" and not use_log:
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9, linestyle=":")
    for (name, curves), (color, style) in zip(curves_by_model.items(), MODEL_STYLES):
        macro = curves["macro"]
        if kind == "roc":
            x, y = np.asarray(macro["roc"]["fpr"]), np.asarray(macro["roc"]["tpr"])
            if use_log:
                keep = x > 0
                x, y = x[keep], y[keep]
            label = f"{name}  (macro AUC {macro['roc_auc']:.4f})"
        else:
            x, y = np.asarray(macro["pr"]["recall"]), np.asarray(macro["pr"]["precision"])
            label = f"{name}  (macro AP {macro['average_precision']:.4f})"
        ax.plot(x, y, color=color, linestyle=style, linewidth=2.0, label=label)
    if kind == "roc":
        ax.set_xlabel("False positive rate" + (" (log scale)" if use_log else ""), color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel("True positive rate", color=INK_SECONDARY, fontsize=10)
        if use_log:
            ax.set_xscale("log")
            ax.set_xlim(1.5e-3, 1.0)
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    else:
        ax.set_xlabel("Recall", color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel("Precision", color=INK_SECONDARY, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right" if kind == "roc" else "lower left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.set_title(title or ("Macro-averaged ROC curves" if kind == "roc" else "Macro-averaged precision-recall curves"), loc="left", fontsize=12, color=INK, pad=12)
    notes = ["Macro average: every class counts equally (mean over the classes that have a curve)."]
    if use_log:
        notes.append("The false-positive-rate axis is logarithmic: it magnifies the region of few false alarms.")
    fig.text(0.01, -0.005, "\n".join(notes), fontsize=8, color=INK_MUTED, ha="left", va="top")
    return _save(fig, path)