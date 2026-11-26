"""Hermetic model benchmark: preprocessing and model in ONE scikit-learn Pipeline, cross-validated together.

The main pipeline (scripts 04-07) fits the preprocessing once on the training split and then trains the
models on the transformed arrays; cross-validation in step 06 therefore re-uses a preprocessing fitted on
all training rows (its medians, kept columns and scaler have seen every fold).  The effect is tiny for a
median / scaler on 2 million rows, but it is a leak in principle.

This module removes it.  Every candidate is::

    Pipeline([
        ("prep",  ColumnTransformer([("flow", <inf->NaN, median, log1p, filter, scaler>, FEATURES)],
                                    remainder="drop")),     # selects the features, drops Label / source_file
        ("model", <estimator>),
    ])

and ``StratifiedKFold`` clones and refits the **whole** pipeline inside each fold, so the imputer
medians, the correlation filter and the scaler are learned on the training folds only.  Raw rows go in
(even with infinite values or extra columns); predictions come out.

Candidates cover a trivial floor, two simple baselines and four ensembles (bagging and boosting),
including XGBoost and LightGBM when those libraries are installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from config import settings
from nids.preprocessing.pipeline_builder import build_pipeline

# family: "floor" (no learning), "baseline" (one simple model), "ensemble"
FAMILIES = {
    "dummy": "floor",
    "logistic_regression": "baseline",
    "decision_tree": "baseline",
    "random_forest": "ensemble",
    "extra_trees": "ensemble",
    "xgboost": "ensemble",
    "lightgbm": "ensemble",
}
DISPLAY = {
    "dummy": "Dummy (most frequent)",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _xgboost(seed: int, n_jobs: int):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)


def _lightgbm(seed: int, n_jobs: int):
    from lightgbm import LGBMClassifier

    # With learning_rate 0.1 / 63 leaves the 15-class boosting diverged on this data (macro F1 0.24 on a
    # held-out split, the tiny classes blow up the gradients).  A smaller step, smaller leaves allowed to hold
    # the rare classes (min_child_samples=5) and L2 regularisation give 0.936 on the same split.
    return LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=5, reg_lambda=1.0,
        n_jobs=n_jobs, random_state=seed, verbose=-1,
    )


_FACTORIES: dict[str, Callable[[int, int], Any]] = {
    "dummy": lambda seed, n_jobs: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda seed, n_jobs: LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    "decision_tree": lambda seed, n_jobs: DecisionTreeClassifier(random_state=seed),
    "random_forest": lambda seed, n_jobs: RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed),
    "extra_trees": lambda seed, n_jobs: ExtraTreesClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed),
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def available_models() -> list[str]:
    """Candidate keys whose library can be imported (XGBoost / LightGBM are optional)."""
    keys = []
    for key, factory in _FACTORIES.items():
        try:
            factory(0, 1)
        except ImportError:
            continue
        keys.append(key)
    return keys


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are features (Label and source_file are metadata, never features)."""
    meta = {settings.LABEL_COLUMN, settings.SOURCE_FILE_COLUMN}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def build_model_pipeline(model_key: str, features: list[str], seed: int | None = None, n_jobs: int = -1) -> Pipeline:
    """Unfitted ``Pipeline(prep=ColumnTransformer(...), model=...)`` for one candidate."""
    if model_key not in _FACTORIES:
        raise ValueError(f"unknown model {model_key!r}; choose from {tuple(_FACTORIES)}")
    seed = settings.RANDOM_SEED if seed is None else seed
    prep = ColumnTransformer(
        [("flow", build_pipeline(), list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    prep.set_output(transform="pandas")
    return Pipeline([("prep", prep), ("model", _FACTORIES[model_key](seed, n_jobs))])


def _scores(y_true: np.ndarray, y_pred: np.ndarray, benign_code: int | None) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if benign_code is not None:
        attack = y_true != benign_code
        benign = ~attack
        out["detection_rate"] = float((y_pred[attack] != benign_code).mean()) if attack.any() else np.nan
        out["false_alarm_rate"] = float((y_pred[benign] != benign_code).mean()) if benign.any() else np.nan
    return {k: float(v) for k, v in out.items()}


@dataclass
class BenchmarkResult:
    classes: list[str]
    folds: pd.DataFrame                       # one row per (model, fold)
    summary: pd.DataFrame                     # one row per model: mean and std of every score
    oof_predictions: dict[str, np.ndarray] = field(default_factory=dict)  # encoded out-of-fold predictions
    y: np.ndarray | None = None               # encoded true labels (same order as the input rows)

    def per_class_f1(self) -> pd.DataFrame:
        """Out-of-fold F1 of each class for each model (rows = classes)."""
        table = {}
        for key, pred in self.oof_predictions.items():
            table[DISPLAY.get(key, key)] = f1_score(self.y, pred, average=None, labels=range(len(self.classes)), zero_division=0)
        frame = pd.DataFrame(table, index=self.classes)
        frame.insert(0, "support", np.bincount(self.y, minlength=len(self.classes)))
        return frame

    def confusion(self, key: str) -> pd.DataFrame:
        cm = confusion_matrix(self.y, self.oof_predictions[key], labels=range(len(self.classes)))
        return pd.DataFrame(cm, index=self.classes, columns=self.classes)


def cross_validate_models(
    df: pd.DataFrame,
    models: list[str] | None = None,
    n_splits: int = 5,
    seed: int | None = None,
    n_jobs: int = -1,
    log: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Stratified k-fold CV of every candidate, the whole pipeline refitted in each fold."""
    seed = settings.RANDOM_SEED if seed is None else seed
    log = log or (lambda message: None)
    models = models or available_models()
    features = feature_columns(df)
    encoder = LabelEncoder().fit(df[settings.LABEL_COLUMN].astype(str))
    y = encoder.transform(df[settings.LABEL_COLUMN].astype(str))
    classes = [str(c) for c in encoder.classes_]
    benign_code = classes.index(settings.BENIGN_LABEL) if settings.BENIGN_LABEL in classes else None
    X = df  # the ColumnTransformer selects the feature columns itself
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))

    rows, oof = [], {}
    for key in models:
        template = build_model_pipeline(key, features, seed=seed, n_jobs=n_jobs)
        pred = np.full(len(y), -1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            pipe = clone(template)
            start = time.perf_counter()
            pipe.fit(X.iloc[train_idx], y[train_idx])
            fit_s = time.perf_counter() - start
            start = time.perf_counter()
            fold_pred = np.asarray(pipe.predict(X.iloc[test_idx])).astype(int).ravel()
            pred_s = time.perf_counter() - start
            pred[test_idx] = fold_pred
            scores = _scores(y[test_idx], fold_pred, benign_code)
            rows.append({"model": key, "display": DISPLAY.get(key, key), "family": FAMILIES.get(key, "?"), "fold": fold, **scores, "fit_seconds": fit_s, "predict_seconds": pred_s})
        oof[key] = pred
        done = [r for r in rows if r["model"] == key]
        log(f"{DISPLAY.get(key, key):<22} macro F1 {np.mean([r['macro_f1'] for r in done]):.4f} +/- {np.std([r['macro_f1'] for r in done]):.4f}")

    folds = pd.DataFrame(rows)
    metrics = [c for c in folds.columns if c not in ("model", "display", "family", "fold")]
    agg = folds.groupby(["model", "display", "family"], sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    summary = agg.reset_index().sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return BenchmarkResult(classes=classes, folds=folds, summary=summary, oof_predictions=oof, y=y)


def paired_comparison(folds: pd.DataFrame, reference: str, metric: str = "macro_f1") -> pd.DataFrame:
    """Mean paired fold difference of every model against ``reference`` (same folds for all models)."""
    wide = folds.pivot(index="fold", columns="model", values=metric)
    if reference not in wide.columns:
        raise ValueError(f"{reference!r} is not in the benchmark")
    rows = []
    for key in wide.columns:
        diff = wide[key] - wide[reference]
        rows.append({"model": key, "display": DISPLAY.get(key, key), f"mean_diff_vs_{reference}": diff.mean(), "std_diff": diff.std(ddof=1), "folds_better": int((diff > 0).sum()), "folds": len(diff)})
    return pd.DataFrame(rows).sort_values(f"mean_diff_vs_{reference}", ascending=False).reset_index(drop=True)


def holdout_scores(pipe: Pipeline, df: pd.DataFrame, encoder_classes: list[str]) -> dict[str, float]:
    """Scores of a fitted pipeline on labelled rows (labels outside the classes are refused)."""
    labels = df[settings.LABEL_COLUMN].astype(str)
    unknown = sorted(set(labels) - set(encoder_classes))
    if unknown:
        raise ValueError(f"labels unknown to the model: {unknown}")
    y = np.array([encoder_classes.index(v) for v in labels])
    pred = np.asarray(pipe.predict(df)).astype(int).ravel()
    benign = encoder_classes.index(settings.BENIGN_LABEL) if settings.BENIGN_LABEL in encoder_classes else None
    return _scores(y, pred, benign)
