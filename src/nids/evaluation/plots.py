"""Figures for the evaluation: confusion matrix, per-class scores, model comparison, fold spread, detection versus false alarms.

Every function takes the dictionaries / tables produced by ``metrics.py`` and
``cross_validation.py``, writes one PNG and returns its path.  Nothing is computed
here except layout, so a figure can never disagree with the tables.

Design rules (they follow the data-visualization guidelines used for this project):

* **One axis per chart**, never two scales on one plot.  Detection rate and false alarm rate, which live
  on very different scales, are two small panels side by side.
* **Colour carries a job.**  One hue for one kind of quantity: blue for the main score, orange for the
  second series where two are needed (recall against precision).  The confusion matrix uses a single
  hue from light to dark for magnitude.  Colour is never the only code: precision and recall also differ
  by marker shape, and models trained on a subsample are marked by an asterisk and a footnote.
* **Honest scales.**  Bars always start at zero.  Where a non-zero baseline is needed (detection rates that
  all sit near 100%), a dot plot is used instead of bars.
* **Reliability is drawn.**  Classes with a low support are shaded and their support is printed; a score
  from two rows is never shown as if it were as solid as one from 500,000.
* **Direct labels.**  Values are written next to the marks; grids are faint.

Files are PNG at 150 dpi.  ``matplotlib`` runs with the non-interactive ``Agg`` backend, so no window opens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

DPI = 150
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE = "#2a78d6"   # main series
ORANGE = "#eb6834"  # second series
BAND = "#efeeea"   # shading of low-support rows
_METRIC_LABELS = {"macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "mcc": "MCC"}
SEQUENTIAL = LinearSegmentedColormap.from_list("one_hue", ["#eef3fb", "#2a78d6", "#123f7a"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return target


def short_class_name(name: str) -> str:
    """Shorter class names for crowded axes."""
    return name.replace("Web Attack - ", "Web: ").replace("Sql Injection", "SQL Injection")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(metrics: dict[str, Any], path: str | Path, title: str | None = None) -> Path:
    """Row-normalised confusion matrix: each row (true class) sums to 100%.

    Cells show the percentage of that true class predicted as the column class; cells under 0.5% are left blank.
    The row labels carry the number of true rows, so a row with 2 or 9 rows is visibly a small sample.
    """
    cm = np.asarray(metrics["confusion_matrix"], dtype="float64")
    names = [short_class_name(n) for n in metrics["class_names"]]
    totals = cm.sum(axis=1)
    shares = np.divide(cm, totals[:, None], out=np.zeros_like(cm), where=totals[:, None] > 0)
    k = len(names)

    fig, ax = _new_figure(max(7.5, 0.62 * k + 2.5), max(6.5, 0.55 * k + 2.2))
    ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal")
    for i in range(k):
        for j in range(k):
            if shares[i, j] >= 0.005:
                value = shares[i, j] * 100
                text = f"{value:.0f}" if value >= 9.5 or value == 100 else f"{value:.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="white" if shares[i, j] > 0.5 else INK)
    ax.set_xticks(range(k), labels=names, rotation=55, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(range(k), labels=[f"{n}  (n={int(t):,})" for n, t in zip(names, totals)], fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Predicted class", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("True class", color=INK_SECONDARY, fontsize=10)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title or "Confusion matrix (% of each true class)", loc="left", fontsize=12, color=INK, pad=12)
    fig.text(0.01, -0.02, "Read along a row: where the flows of that true class went. The diagonal is correct. Blank = under 0.5%.",
             fontsize=8, color=INK_MUTED, ha="left")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Per-class precision and recall
# ---------------------------------------------------------------------------
def plot_per_class_metrics(metrics: dict[str, Any], path: str | Path, title: str | None = None) -> Path:
    """Precision (blue circle) and recall (orange diamond) for every class, with the support of each class.

    Rows with a low support are shaded and marked, because their scores rest on very few rows.
    """
    rows = metrics["per_class"]
    k = len(rows)
    fig, ax = _new_figure(9.5, max(4.8, 0.42 * k + 1.8))
    y = np.arange(k)[::-1]  # first class on top
    for pos, row in zip(y, rows):
        if row["low_support"]:
            ax.axhspan(pos - 0.5, pos + 0.5, color=BAND, zorder=0)
        ax.plot([row["precision"], row["recall"]], [pos, pos], color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.scatter([r["precision"] for r in rows], y, s=46, color=BLUE, marker="o", edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Precision")
    ax.scatter([r["recall"] for r in rows], y, s=50, color=ORANGE, marker="D", edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Recall")
    ax.set_yticks(y, labels=[short_class_name(r["class"]) for r in rows])
    ax.set_xlim(-0.02, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, k - 0.4)
    ax.set_xlabel("Score (higher is better)", color=INK_SECONDARY, fontsize=10)
    _style_axes(ax, "x")
    ax.grid(axis="y", visible=False)
    for pos, row in zip(y, rows):
        note = f"n={row['support']:,}" + ("  low support" if row["low_support"] else "")
        ax.text(1.015, pos, note, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5,
                color=INK_MUTED if row["low_support"] else INK_SECONDARY)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handletextpad=0.4)
    ax.set_title(title or "Precision and recall per class", loc="left", fontsize=12, color=INK, pad=28)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Model comparison (bars from zero, optional error bars)
# ---------------------------------------------------------------------------
def plot_model_comparison(
    table: pd.DataFrame,
    path: str | Path,
    value: str = "macro_f1_mean",
    error: str | None = "macro_f1_std",
    title: str | None = None,
    xlabel: str = "Macro F1",
    subsample_column: str | None = "trained_on_subsample",
    flag_column: str | None = "indistinguishable_from_best",
) -> Path:
    """Horizontal bars of one score per model, best on top, bars starting at zero.

    ``error`` draws a thin whisker (the standard deviation over the folds).  Models trained on a subsample get an
    asterisk and a footnote; models that cannot be told apart from the best (``flag_column``) are in bold.
    """
    frame = table.dropna(subset=[value]).sort_values(value, ascending=False, kind="stable").reset_index(drop=True)
    n = len(frame)
    if n == 0:
        raise ValueError("no models to plot")
    fig, ax = _new_figure(9.0, max(3.2, 0.5 * n + 1.6))
    y = np.arange(n)[::-1]
    ax.barh(y, frame[value], height=0.55, color=BLUE, zorder=2)
    if error and error in frame:
        err = frame[error].fillna(0.0).to_numpy()
        ax.errorbar(frame[value], y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=3)
    labels, any_subsample = [], False
    for _, row in frame.iterrows():
        star = bool(subsample_column and subsample_column in frame and row.get(subsample_column))
        any_subsample |= star
        labels.append(row["model"] + (" *" if star else ""))
    ax.set_yticks(y, labels=labels)
    if flag_column and flag_column in frame:
        for tick, flagged in zip(ax.get_yticklabels(), frame[flag_column]):
            if flagged:
                tick.set_fontweight("bold")
                tick.set_color(INK)
    top = float(max(frame[value].max() + (frame[error].fillna(0).max() if error and error in frame else 0), 0.05))
    ax.set_xlim(0, min(1.0, top * 1.0 + 0.16) if top < 0.9 else 1.0)
    for pos, (_, row) in zip(y, frame.iterrows()):
        spread = f" ± {row[error]:.3f}" if error and error in frame and pd.notna(row[error]) else ""
        x_text = row[value] + (row[error] if error and error in frame and pd.notna(row[error]) else 0) + 0.008
        ax.text(x_text, pos, f"{row[value]:.3f}{spread}", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    _style_axes(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or f"{xlabel} by model", loc="left", fontsize=12, color=INK, pad=12)
    notes = []
    if any_subsample:
        notes.append("* trained on a class-capped subsample (a more balanced class mix than the real traffic)")
    if flag_column and flag_column in frame and frame[flag_column].sum() > 1:
        notes.append("Bold: not distinguishable from the best model (within one standard deviation)")
    if notes:
        fig.text(0.01, -0.01, "\n".join(notes), fontsize=8, color=INK_MUTED, ha="left", va="top")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Spread over the folds
# ---------------------------------------------------------------------------
def plot_fold_spread(results: dict[str, dict[str, Any]], path: str | Path, metric: str = "macro_f1", title: str | None = None) -> Path:
    """One dot per fold for every model, with the mean marked: shows how much the score moves between splits."""
    ok = [r for r in results.values() if r.get("status") == "ok"]
    if not ok:
        raise ValueError("no successful cross-validation results to plot")
    ok.sort(key=lambda r: r["summary"][metric]["mean"], reverse=True)
    n = len(ok)
    fig, ax = _new_figure(9.0, max(3.2, 0.5 * n + 1.6))
    y = np.arange(n)[::-1]
    for pos, r in zip(y, ok):
        values = [f[metric] for f in r["folds"]]
        ax.scatter(values, [pos] * len(values), s=30, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        mean = r["summary"][metric]["mean"]
        ax.plot([mean, mean], [pos - 0.3, pos + 0.3], color=INK, linewidth=2.0, zorder=4)
        std = r["summary"][metric]["std"]
        ax.text(1.02, pos, f"{mean:.3f}" + (f" ± {std:.3f}" if std is not None else ""), transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(y, labels=[r["display_name"] + (" *" if r.get("subsampled") else "") for r in ok])
    lows = min(f[metric] for r in ok for f in r["folds"])
    ax.set_xlim(max(0.0, lows - 0.05), 1.0)
    ax.set_xlabel(f"{_METRIC_LABELS.get(metric, metric.replace('_', ' ').capitalize())} in each fold (dot) and mean (bar)", color=INK_SECONDARY, fontsize=10)
    _style_axes(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or "Spread of the score over the folds", loc="left", fontsize=12, color=INK, pad=12)
    notes = ["The axis does not start at zero: it zooms on the range of the folds."]
    if any(r.get("subsampled") for r in ok):
        notes.append("* trained on a class-capped subsample")
    fig.text(0.01, -0.01, "\n".join(notes), fontsize=8, color=INK_MUTED, ha="left", va="top")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Detection rate and false alarm rate
# ---------------------------------------------------------------------------
def plot_detection_and_false_alarms(
    table: pd.DataFrame,
    path: str | Path,
    detection: str = "detection_rate_mean",
    false_alarm: str = "false_alarm_rate_mean",
    title: str | None = None,
    subsample_column: str | None = "trained_on_subsample",
) -> Path:
    """Two dot panels side by side: attack detection rate (higher is better) and false alarm rate (lower is better).

    They are separate panels because the two rates live on very different scales (detection rates all sit near 100%).
    The models keep the order of ``table``.
    """
    frame = table.dropna(subset=[detection, false_alarm]).reset_index(drop=True)
    n = len(frame)
    if n == 0:
        raise ValueError("no models with a detection rate and a false alarm rate")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, max(3.2, 0.5 * n + 1.6)), sharey=True, facecolor=SURFACE)
    y = np.arange(n)[::-1]
    labels = [m + (" *" if subsample_column and subsample_column in frame and bool(frame.loc[i, subsample_column]) else "")
              for i, m in enumerate(frame["model"])]
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        _style_axes(ax, "x")
        ax.grid(axis="y", visible=False)
    ax1.scatter(frame[detection], y, s=44, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax2.scatter(frame[false_alarm], y, s=44, color=ORANGE, marker="D", edgecolor=SURFACE, linewidth=1.2, zorder=3)
    lo = float(frame[detection].min())
    ax1.set_xlim(max(0.0, lo - 0.03), 1.005)
    ax2.set_xlim(0, max(0.02, float(frame[false_alarm].max()) * 1.35))
    for pos, (d, f) in zip(y, zip(frame[detection], frame[false_alarm])):
        ax1.text(d, pos + 0.32, f"{d * 100:.1f}%", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
        ax2.text(f, pos + 0.32, f"{f * 100:.1f}%", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_yticks(y, labels=labels)
    ax1.set_ylim(-0.6, n - 0.2)
    ax1.set_xlabel("Attack detection rate (higher is better)", color=INK_SECONDARY, fontsize=10)
    ax2.set_xlabel("False alarm rate on normal traffic (lower is better)", color=INK_SECONDARY, fontsize=10)
    fig.suptitle(title or "Attack detection and false alarms", x=0.01, ha="left", fontsize=12, color=INK)
    notes = ["Detection: attack flows predicted as any attack (the attack type may be wrong). False alarm: BENIGN flows predicted as an attack.",
             "The left axis does not start at zero: it zooms on the range of the models."]
    if subsample_column and subsample_column in frame and frame[subsample_column].any():
        notes.append("* trained on a class-capped subsample")
    fig.text(0.01, -0.01, "\n".join(notes), fontsize=8, color=INK_MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)