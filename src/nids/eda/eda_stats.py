"""Exploratory statistics (tables only; the plots are in ``eda_plots.py``).

The EDA runs on the loaded, **uncleaned** data, so infinite values are turned
into missing values *inside the computations only*: the input DataFrame is
never modified.  Nothing here trains a model or influences preprocessing; the
results are used to take decisions (scaler, correlation threshold, which
features look uninformative) that are then written in ``config/settings.py``.

Large-data policy: statistics that are cheap per column use all rows;
statistics that are expensive (correlation, class comparison) use samples:

* ``correlation_matrix``  -> a plain random sample (default 500 000 rows);
* ``class_medians`` and ``feature_class_separation`` -> a class-capped sample
  (at most ``cap_per_class`` rows per class, all rows of rare classes).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from config import settings
from nids.data import schema
from nids.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CORRELATION_ROWS = 500_000
DEFAULT_CAP_PER_CLASS = 50_000


def _features(df: pd.DataFrame, features: list[str] | None) -> list[str]:
    if settings.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{settings.LABEL_COLUMN}' not found in the data")
    cols = features if features is not None else schema.get_feature_columns(df.columns)
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns (load the data with loader.py): {non_numeric[:5]}")
    return cols


def _finite_values(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype="float64", copy=False)
    return values[np.isfinite(values)]


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def random_sample(df: pd.DataFrame, max_rows: int, seed: int | None = None) -> pd.DataFrame:
    """Random sample of at most ``max_rows`` rows (the whole frame if it is smaller)."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=settings.RANDOM_SEED if seed is None else seed)


def class_capped_sample(df: pd.DataFrame, cap_per_class: int, seed: int | None = None) -> pd.DataFrame:
    """At most ``cap_per_class`` rows per class; classes smaller than the cap are kept whole."""
    seed = settings.RANDOM_SEED if seed is None else seed
    parts = [
        group if len(group) <= cap_per_class else group.sample(n=cap_per_class, random_state=seed)
        for _, group in df.groupby(settings.LABEL_COLUMN, sort=True)
    ]
    return pd.concat(parts)


# ---------------------------------------------------------------------------
# Per-feature statistics (all rows)
# ---------------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Per feature, on finite values: count, mean, std, min, quartiles, max, skewness."""
    rows = []
    for col in _features(df, features):
        finite = _finite_values(df[col])
        if finite.size == 0:
            rows.append({"column": col, "count": 0})
            continue
        q1, median, q3 = np.percentile(finite, [25, 50, 75])
        rows.append(
            {
                "column": col,
                "count": int(finite.size),
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                "min": float(finite.min()),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(finite.max()),
                "skewness": float(pd.Series(finite).skew()) if finite.size > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, features: list[str] | None = None, k: float = 1.5) -> pd.DataFrame:
    """Share of finite values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` per feature.

    Network traffic is heavy-tailed and attacks are often the "outliers", so
    this is a description only: nothing is removed because of it.
    """
    rows = []
    for col in _features(df, features):
        finite = _finite_values(df[col])
        if finite.size == 0:
            rows.append({"column": col, "n_outliers": 0, "outlier_percent": np.nan})
            continue
        q1, q3 = np.percentile(finite, [25, 75])
        iqr = q3 - q1
        n_out = int(((finite < q1 - k * iqr) | (finite > q3 + k * iqr)).sum())
        rows.append(
            {"column": col, "n_outliers": n_out, "outlier_percent": round(n_out / finite.size * 100, 4)}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation (sample)
# ---------------------------------------------------------------------------
def correlation_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
    max_rows: int = DEFAULT_CORRELATION_ROWS,
    seed: int | None = None,
) -> pd.DataFrame:
    """Pearson correlation between features on a random sample.

    Infinite values count as missing and each pair uses the rows where both
    values exist.  Constant columns give ``NaN`` rows and columns.
    """
    cols = _features(df, features)
    sample = random_sample(df, max_rows, seed)[cols].replace([np.inf, -np.inf], np.nan)  # sample first, then select
    logger.info("Correlation on %d rows x %d features", len(sample), len(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return sample.corr(method="pearson")


def high_correlation_pairs(corr: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Feature pairs with ``|r| >= threshold`` (default ``settings.CORRELATION_THRESHOLD``), strongest first."""
    threshold = settings.CORRELATION_THRESHOLD if threshold is None else threshold
    names = list(corr.columns)
    values = corr.to_numpy()
    rows, cols = np.triu_indices(len(names), k=1)
    r = values[rows, cols]
    keep = np.abs(r) >= threshold  # NaN compares False, so constant columns are skipped
    table = pd.DataFrame(
        {"feature_1": np.array(names)[rows[keep]], "feature_2": np.array(names)[cols[keep]], "correlation": r[keep]}
    )
    return table.reindex(table["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-class comparison (class-capped sample)
# ---------------------------------------------------------------------------
def class_medians(
    df: pd.DataFrame, features: list[str] | None = None, cap_per_class: int = DEFAULT_CAP_PER_CLASS
) -> pd.DataFrame:
    """Median of every feature per class (rows = classes, columns = features), infinite values ignored."""
    cols = _features(df, features)
    sample = class_capped_sample(df, cap_per_class)[[settings.LABEL_COLUMN, *cols]]  # sample first, then select
    sample[cols] = sample[cols].replace([np.inf, -np.inf], np.nan)
    return sample.groupby(settings.LABEL_COLUMN)[cols].median()


def feature_class_separation(
    df: pd.DataFrame, features: list[str] | None = None, cap_per_class: int = DEFAULT_CAP_PER_CLASS
) -> pd.DataFrame:
    """Exploratory ANOVA F statistic of each feature against the class label.

    A higher F means the class means differ more relative to the spread inside
    classes.  Missing and infinite values are replaced by the column median for
    this calculation only.  It is a ranking for exploration, **not** a feature
    selection step, and it only detects differences in the mean (a feature can
    separate classes in other ways).  Constant columns get ``NaN``.
    """
    cols = _features(df, features)
    sample = class_capped_sample(df, cap_per_class)[[settings.LABEL_COLUMN, *cols]]  # sample first, then select
    labels = sample[settings.LABEL_COLUMN].to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes are needed to compare features against the label")

    x = sample[cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0.0)  # a column that is entirely missing becomes 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant columns raise divide-by-zero warnings
        f_values, _ = f_classif(x.to_numpy(dtype="float64"), labels)
    table = pd.DataFrame({"column": cols, "f_statistic": f_values})
    return table.sort_values("f_statistic", ascending=False, na_position="last").reset_index(drop=True)