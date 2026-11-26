"""Formal hypothesis tests on the flow features (statistical validation of the EDA).

The EDA (``eda_stats.py``) *describes* the data; this module *tests* what the description suggests.
Five distinct tests, each answering one question the project relies on:

=====================  ==========================================================  =========================
Test                   Question (H0)                                               Effect size reported
=====================  ==========================================================  =========================
Chi-square             H0: the binned feature and the class are independent        Cramer's V
One-way ANOVA          H0: the class means of the feature are all equal            eta squared
Kruskal-Wallis         H0: the class distributions have the same location           epsilon squared
Shapiro-Wilk           H0: the feature is normally distributed (raw and log1p)     W itself
Mann-Whitney U         H0: BENIGN and attack flows have the same distribution      rank-biserial correlation
=====================  ==========================================================  =========================

Why both ANOVA and Kruskal-Wallis: Shapiro-Wilk rejects normality for every feature, so the ANOVA
assumptions do not hold; Kruskal-Wallis is the non-parametric check that the conclusion survives.

With hundreds of thousands of rows *every* p-value is tiny, so the tables also give an **effect size**
(the size of the difference, independent of n) and a **Holm-corrected** p-value per test family
(66 features tested at once).  Conclusions in the reports are drawn from the effect sizes.

Inputs are never modified; infinite values are treated as missing inside each test.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

ALPHA = 0.05
SHAPIRO_MAX_ROWS = 5000  # scipy's p-value is only accurate up to about 5000 rows
DEFAULT_BINS = 5

TESTS = ("chi2", "anova", "kruskal", "shapiro", "mannwhitney")
TEST_TITLES = {
    "chi2": "Chi-square test of independence (binned feature x class)",
    "anova": "One-way ANOVA (class means)",
    "kruskal": "Kruskal-Wallis H test (class distributions)",
    "shapiro": "Shapiro-Wilk normality test",
    "mannwhitney": "Mann-Whitney U test (BENIGN vs attack)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values[np.isfinite(values)]


def _groups(df: pd.DataFrame, feature: str, label: str, min_size: int = 2) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, group in df.groupby(label, sort=True, observed=True):
        values = _finite(group[feature]).to_numpy()
        if len(values) >= min_size:
            out[str(name)] = values
    return out


def holm(p_values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    m = int(mask.sum())
    if m == 0:
        return out
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def interpret_effect(test: str, value: float) -> str:
    """Conventional wording for an effect size (Cohen-style thresholds)."""
    if value is None or not np.isfinite(value):
        return "n/a"
    v = abs(value)
    if test == "chi2":  # Cramer's V (df > 1 thresholds)
        cuts = (0.07, 0.21, 0.35)
    elif test in ("anova", "kruskal"):  # eta / epsilon squared
        cuts = (0.01, 0.06, 0.14)
    elif test == "mannwhitney":  # rank-biserial r
        cuts = (0.1, 0.3, 0.5)
    else:
        return "n/a"
    if v < cuts[0]:
        return "negligible"
    if v < cuts[1]:
        return "small"
    if v < cuts[2]:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# The five tests (one feature each)
# ---------------------------------------------------------------------------
def chi2_independence(df: pd.DataFrame, feature: str, label: str | None = None, bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Chi-square test of independence between the class and the feature cut into quantile bins."""
    label = label or settings.LABEL_COLUMN
    values = pd.to_numeric(df[feature], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    keep = values.notna()
    values, labels = values[keep], df.loc[keep, label]
    if values.nunique() <= bins:
        binned = values.astype(str)  # few distinct values: use them as categories
    else:
        binned = pd.qcut(values, q=bins, duplicates="drop").astype(str)
    table = pd.crosstab(binned, labels)
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "chi2", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": np.nan, "n": int(keep.sum()), "note": "fewer than two categories"}
    chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    low_expected = float((expected < 5).mean())
    return {
        "test": "chi2", "feature": feature, "statistic": float(chi2), "p_value": float(p), "dof": int(dof),
        "effect_size": v, "n": int(n),
        "note": f"{table.shape[0]} bins x {table.shape[1]} classes; {low_expected:.0%} of expected counts < 5",
    }


def anova_by_class(df: pd.DataFrame, feature: str, label: str | None = None) -> dict[str, Any]:
    """One-way ANOVA across the classes, with eta squared (share of variance explained by the class)."""
    groups = _groups(df, feature, label or settings.LABEL_COLUMN)
    if len(groups) < 2:
        return {"test": "anova", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": np.nan, "n": 0, "note": "fewer than two classes"}
    arrays = list(groups.values())
    allv = np.concatenate(arrays)
    if np.ptp(allv) == 0:
        return {"test": "anova", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": len(arrays) - 1, "effect_size": 0.0, "n": len(allv), "note": "constant feature"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = stats.f_oneway(*arrays)
    grand = allv.mean()
    ss_between = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
    ss_total = ((allv - grand) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return {"test": "anova", "feature": feature, "statistic": float(f), "p_value": float(p), "dof": len(arrays) - 1, "effect_size": eta2, "n": len(allv), "note": f"{len(arrays)} classes"}


def kruskal_by_class(df: pd.DataFrame, feature: str, label: str | None = None) -> dict[str, Any]:
    """Kruskal-Wallis H test across the classes, with epsilon squared = H / (n - 1)."""
    groups = _groups(df, feature, label or settings.LABEL_COLUMN)
    if len(groups) < 2:
        return {"test": "kruskal", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": np.nan, "n": 0, "note": "fewer than two classes"}
    arrays = list(groups.values())
    n = sum(len(a) for a in arrays)
    if np.ptp(np.concatenate(arrays)) == 0:
        return {"test": "kruskal", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": len(arrays) - 1, "effect_size": 0.0, "n": n, "note": "constant feature"}
    h, p = stats.kruskal(*arrays)
    eps2 = float(h / (n - 1)) if n > 1 else np.nan
    return {"test": "kruskal", "feature": feature, "statistic": float(h), "p_value": float(p), "dof": len(arrays) - 1, "effect_size": eps2, "n": n, "note": f"{len(arrays)} classes"}


def shapiro_normality(df: pd.DataFrame, feature: str, max_rows: int = SHAPIRO_MAX_ROWS, seed: int | None = None) -> dict[str, Any]:
    """Shapiro-Wilk on a random subsample (at most ``max_rows``), on the raw values and after signed log1p."""
    values = _finite(df[feature])
    if len(values) > max_rows:
        values = values.sample(n=max_rows, random_state=settings.RANDOM_SEED if seed is None else seed)
    v = values.to_numpy()
    if len(v) < 3 or np.ptp(v) == 0:
        return {"test": "shapiro", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": np.nan, "n": len(v), "note": "constant or too few values", "statistic_log": np.nan, "p_value_log": np.nan}
    w, p = stats.shapiro(v)
    logv = np.sign(v) * np.log1p(np.abs(v))
    w_log, p_log = (stats.shapiro(logv) if np.ptp(logv) > 0 else (np.nan, np.nan))
    return {
        "test": "shapiro", "feature": feature, "statistic": float(w), "p_value": float(p), "dof": 0, "effect_size": np.nan, "n": len(v),
        "note": "W close to 1 = close to normal", "statistic_log": float(w_log), "p_value_log": float(p_log),
        "skewness": float(stats.skew(v)),
    }


def mann_whitney_benign_attack(df: pd.DataFrame, feature: str, label: str | None = None, benign: str | None = None) -> dict[str, Any]:
    """Mann-Whitney U, BENIGN flows vs all attack flows, with the rank-biserial correlation r = 1 - 2U/(n1 n2).

    r > 0 means attack flows tend to have *larger* values than BENIGN flows.
    """
    label = label or settings.LABEL_COLUMN
    benign = benign or settings.BENIGN_LABEL
    is_benign = df[label].astype(str) == benign
    a = _finite(df.loc[is_benign, feature]).to_numpy()
    b = _finite(df.loc[~is_benign, feature]).to_numpy()
    if len(a) < 1 or len(b) < 1:
        return {"test": "mannwhitney", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": np.nan, "n": len(a) + len(b), "note": "one group is empty"}
    if np.ptp(np.concatenate([a, b])) == 0:
        return {"test": "mannwhitney", "feature": feature, "statistic": np.nan, "p_value": np.nan, "dof": 0, "effect_size": 0.0, "n": len(a) + len(b), "note": "constant feature"}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = float(1.0 - 2.0 * u / (len(a) * len(b)))
    return {
        "test": "mannwhitney", "feature": feature, "statistic": float(u), "p_value": float(p), "dof": 0, "effect_size": r,
        "n": len(a) + len(b), "note": f"median BENIGN {np.median(a):.4g} vs attack {np.median(b):.4g}",
    }


_RUNNERS = {
    "chi2": chi2_independence,
    "anova": anova_by_class,
    "kruskal": kruskal_by_class,
    "shapiro": shapiro_normality,
    "mannwhitney": mann_whitney_benign_attack,
}


def run_test(test: str, df: pd.DataFrame, feature: str) -> dict[str, Any]:
    if test not in _RUNNERS:
        raise ValueError(f"unknown test {test!r}; choose from {TESTS}")
    return _RUNNERS[test](df, feature)


def run_all_tests(df: pd.DataFrame, features: list[str], tests: tuple[str, ...] = TESTS, alpha: float = ALPHA) -> dict[str, pd.DataFrame]:
    """Run every test on every feature. One table per test, sorted by effect size, with Holm p-values."""
    if settings.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{settings.LABEL_COLUMN}' not found")
    out: dict[str, pd.DataFrame] = {}
    for test in tests:
        table = pd.DataFrame([run_test(test, df, f) for f in features])
        table["p_holm"] = holm(table["p_value"])
        table["reject_h0"] = table["p_holm"] < alpha
        if test == "shapiro":
            table = table.sort_values("statistic", ascending=True, na_position="last")
        else:
            table["effect"] = [interpret_effect(test, v) for v in table["effect_size"]]
            table = table.sort_values("effect_size", key=lambda s: s.abs(), ascending=False, na_position="last")
        out[test] = table.reset_index(drop=True)
    return out


def summary_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One line per test: how many features reject H0 and how many have a large effect."""
    rows = []
    for test, table in results.items():
        tested = int(table["p_value"].notna().sum())
        rows.append(
            {
                "test": test,
                "title": TEST_TITLES[test],
                "features_tested": tested,
                "reject_h0_holm": int(table["reject_h0"].sum()),
                "large_effect": int((table.get("effect", pd.Series(dtype=str)) == "large").sum()),
                "top_feature": None if table.empty else table.iloc[0]["feature"],
            }
        )
    return pd.DataFrame(rows)
