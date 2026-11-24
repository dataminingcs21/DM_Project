"""EDA figures (PNG files) built from the tables of ``eda_stats.py`` and the data.

Every ``plot_*`` function saves one figure and returns its path.  Figures are
closed after saving so memory does not grow when many are produced.

The non-interactive ``Agg`` backend is selected on import: figures are only
written to files, which also works on a machine with no display.  (Streamlit
can show Agg figures, so the app is not affected.)

Plots that need many rows take a sample (default 200 000 rows) because a
histogram of 200 000 values looks the same as one of 2.8 million.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config import settings  # noqa: E402
from nids.eda.eda_stats import class_capped_sample, random_sample  # noqa: E402
from nids.utils.io_utils import ensure_dir  # noqa: E402
from nids.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DPI = 150
BAR_COLOR = "#3b6ea5"
MAX_HISTOGRAMS = 36

sns.set_theme(style="whitegrid", font_scale=0.9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fig.savefig(target, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure %s", target)
    return target


def _short_file_name(name: str) -> str:
    for suffix in (".pcap_ISCX.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _message_figure(text: str, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    return _save(fig, path)


def _horizontal_bars(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    path: str | Path,
    log_scale: bool = False,
    annotations: list[str] | None = None,
) -> Path:
    """Horizontal bar chart, first item at the top."""
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.32 * len(labels) + 1.2)))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=BAR_COLOR)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if log_scale:
        ax.set_xscale("log")
    if annotations:
        for pos, value, text in zip(positions, values, annotations):
            ax.text(value, pos, " " + text, va="center", fontsize=8)
        ax.margins(x=0.25)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Class and file structure
# ---------------------------------------------------------------------------
def plot_class_distribution(distribution: pd.DataFrame, path: str | Path) -> Path:
    """Rows per class (log scale, because the classes differ by several orders of magnitude).

    ``distribution`` is the ``label_distribution`` table from the inspector
    (columns ``label``, ``count``, ``percent``).
    """
    table = distribution.sort_values("count", ascending=False)
    notes = [f"{int(c):,} ({p:.3f}%)" for c, p in zip(table["count"], table["percent"])]
    return _horizontal_bars(
        table["label"].astype(str).tolist(),
        table["count"].astype(float).tolist(),
        "Rows per class (log scale)",
        "Number of rows",
        path,
        log_scale=True,
        annotations=notes,
    )


def plot_label_by_file(crosstab: pd.DataFrame, path: str | Path) -> Path:
    """Heatmap of rows per class (lines) and source file (columns); colour is log-scaled, numbers are the counts."""
    if crosstab.empty:
        return _message_figure("No source-file information", path)
    table = crosstab.rename(columns=_short_file_name)
    fig, ax = plt.subplots(figsize=(0.95 * table.shape[1] + 4, 0.38 * table.shape[0] + 2))
    sns.heatmap(
        np.log10(table + 1),
        annot=table.to_numpy(),
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "log10(rows + 1)"},
        linewidths=0.5,
        annot_kws={"fontsize": 7},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Rows per class and source file")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def plot_missing_and_infinite(profile: pd.DataFrame, path: str | Path) -> Path:
    """Columns that contain missing or infinite values (from the inspector's ``column_profile``)."""
    affected = profile[(profile["n_missing"] > 0) | (profile["n_inf"] > 0)]
    if affected.empty:
        return _message_figure("No missing or infinite values", path)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.5 * len(affected) + 1.2)))
    positions = np.arange(len(affected))
    ax.barh(positions - 0.2, affected["n_missing"], height=0.4, label="missing", color=BAR_COLOR)
    ax.barh(positions + 0.2, affected["n_inf"], height=0.4, label="infinite", color="#d98c3f")
    ax.set_yticks(positions)
    ax.set_yticklabels(affected["column"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of values")
    ax.set_title("Missing and infinite values per column")
    ax.legend()
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_outlier_percent(outliers: pd.DataFrame, path: str | Path, top: int = 25) -> Path:
    """Features with the largest share of values outside the 1.5 x IQR fences."""
    table = outliers.dropna(subset=["outlier_percent"]).nlargest(top, "outlier_percent")
    return _horizontal_bars(
        table["column"].tolist(),
        table["outlier_percent"].tolist(),
        f"Share of values outside the 1.5 x IQR fences (top {len(table)})",
        "Percent of finite values",
        path,
    )


def plot_skewness(descriptive: pd.DataFrame, path: str | Path, top: int = 25) -> Path:
    """Features with the largest absolute skewness."""
    table = descriptive.dropna(subset=["skewness"])
    table = table.reindex(table["skewness"].abs().sort_values(ascending=False).index).head(top)
    return _horizontal_bars(
        table["column"].tolist(),
        table["skewness"].tolist(),
        f"Skewness of the most skewed features (top {len(table)})",
        "Skewness (finite values)",
        path,
    )


# ---------------------------------------------------------------------------
# Relations between features
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(corr: pd.DataFrame, path: str | Path) -> Path:
    """Heatmap of the correlation matrix (constant columns appear blank)."""
    size = min(max(8.0, 0.2 * len(corr) + 3), 22.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.grid(False)  # no grid lines through blank (constant) cells
    ax.tick_params(axis="both", labelsize=5)
    ax.set_title("Feature correlation")
    return _save(fig, path)


def plot_top_f_statistics(separation: pd.DataFrame, path: str | Path, top: int = 20) -> Path:
    """Features ranked by the ANOVA F statistic against the class label (exploration only)."""
    table = separation.dropna(subset=["f_statistic"]).nlargest(top, "f_statistic")
    return _horizontal_bars(
        table["column"].tolist(),
        table["f_statistic"].tolist(),
        f"Features that separate the classes best (ANOVA F, top {len(table)})",
        "F statistic",
        path,
    )


def plot_class_profile(class_medians: pd.DataFrame, features: list[str], path: str | Path) -> Path:
    """Heatmap of the per-class median of ``features``, standardised across classes.

    Each column is converted to a z-score over the classes, so features on very
    different scales can be compared: red means "high for this class compared
    with the other classes", blue means "low".
    """
    table = class_medians[features]
    spread = table.std(ddof=0).replace(0, np.nan)
    z = ((table - table.mean()) / spread).fillna(0.0)
    fig, ax = plt.subplots(figsize=(0.55 * len(features) + 4, 0.38 * len(z) + 2.5))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0.4,
        cbar_kws={"label": "z-score of the class median"},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Class profile: median of each feature, compared across classes")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def plot_feature_histograms(
    df: pd.DataFrame,
    features: list[str],
    path: str | Path,
    max_rows: int = 200_000,
    ncols: int = 3,
) -> Path:
    """Histograms of ``features`` on a random sample.

    Only finite values are used and the range is limited to the 1st to 99th
    percentile so one extreme value does not flatten the picture.
    """
    if not features:
        raise ValueError("No features given")
    if len(features) > MAX_HISTOGRAMS:
        raise ValueError(f"At most {MAX_HISTOGRAMS} histograms per figure (got {len(features)})")
    sample = random_sample(df, max_rows)
    nrows = -(-len(features) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        values = sample[feature].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0 or np.ptp(values) == 0:
            ax.text(0.5, 0.5, "constant or empty", ha="center", va="center")
            ax.set_title(feature, fontsize=8)
            ax.axis("off")
            continue
        low, high = np.percentile(values, [1, 99])
        clipped = values[(values >= low) & (values <= high)]
        ax.hist(clipped if clipped.size else values, bins=50, color=BAR_COLOR)
        skew = float(pd.Series(values).skew())
        ax.set_title(f"{feature}\nskew {skew:.1f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle("Distributions (1st to 99th percentile, finite values)", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_by_class(
    df: pd.DataFrame, features: list[str], path: str | Path, cap_per_class: int = 2_000
) -> Path:
    """Box plots of ``features`` for each class (class-capped sample, symmetric-log axis, no outlier dots)."""
    if not features:
        raise ValueError("No features given")
    sample = class_capped_sample(df, cap_per_class)
    order = sample[settings.LABEL_COLUMN].value_counts().index.tolist()
    fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 0.38 * len(order) + 2), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        data = sample[[settings.LABEL_COLUMN, feature]].replace([np.inf, -np.inf], np.nan).dropna()
        sns.boxplot(data=data, x=feature, y=settings.LABEL_COLUMN, order=order, showfliers=False, color=BAR_COLOR, ax=ax)
        ax.set_xscale("symlog")
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Feature values by class (symmetric-log axis)", y=1.0)
    fig.tight_layout()
    return _save(fig, path)