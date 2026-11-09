"""Training helpers shared by the SVM (and other) experiments.

Public API:
    load_xy(cfg, rebuild=False)          -> dict of numpy arrays (X_train, X_test, y_train, y_test, feature_names)
    run_experiment(cfg, data=None)       -> {"name": str, "row": dict, "pipeline": fitted_pipeline}
    tune(cfg, search_space, n_iter, data, sample, out_file) -> (best_params, results_df)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from . import config as C
from . import preprocessing as P
from . import feature_selection as F

try:
    from imblearn.over_sampling import SMOTE  # optional
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    _HAS_IMBLEARN = False


# --------------------------------------------------------------------------- load
def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample of n rows based on Label.

    Guarantees at least 2 rows per class so downstream train_test_split(stratify=y)
    never fails.  Classes that cannot reach 2 rows are dropped entirely.
    """
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for label, group in df.groupby("Label"):
        k = max(2, int(round(len(group) * n / len(df))))
        k = min(k, len(group))
        if k < 2:
            continue
        parts.append(group.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
    if not parts:
        raise ValueError("No class has >= 2 rows after sampling; increase --max-rows.")
    return pd.concat(parts, ignore_index=True)


def _prepare_task(y_raw: np.ndarray, task: str) -> np.ndarray:
    """Convert encoded labels (0=BENIGN, 1..N=attack classes) to the task target."""
    y_raw = np.asarray(y_raw)
    if task == "binary":
        return (y_raw == 0).astype(int)          # 1 = BENIGN, 0 = attack  (matches preprocessing)
    # multiclass / family: keep as-is
    return y_raw


def load_xy(cfg: dict, rebuild: bool = False) -> dict:
    """Load, clean, encode and split the CIC-IDS2017 dataset.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test, feature_names, label_encoder(None for binary)
    """
    CACHE = C.CACHE_DIR / "clean_cache.pkl"
    if CACHE.exists() and not rebuild:
        print(f"[load_xy] loading cached dataset {CACHE}")
        df = pd.read_pickle(CACHE)
    else:
        print(f"[load_xy] reading raw CSVs from {cfg['dataset_dir']}")
        df = P.load_and_merge_datasets(cfg["dataset_dir"])
        print(f"[load_xy] cleaning ({len(df)} rows)")
        df = P.clean_data(df, zero_threshold=cfg["preprocessing"]["zero_threshold"])
        print(f"[load_xy] cleaned to {len(df)} rows")
        if cfg.get("max_rows"):
            df = _stratified_sample(df, cfg["max_rows"], cfg["seed"])
            print(f"[load_xy] capped to {len(df)} rows")
        df.to_pickle(CACHE)
        print(f"[load_xy] cached to {CACHE}")

    # encode labels: keep raw 0=BENIGN, 1..N as encoded class index
    X, y_encoded = P.encode_labels(df, binary=False)

    # split + scale
    X_tr, X_te, y_tr_raw, y_te_raw = P.split_and_scale(
        X, y_encoded,
        test_size=cfg.get("test_size", 0.3),
        random_state=cfg["seed"],
    )

    # optional feature selection / reduction BEFORE the model
    if cfg["features"].get("top_k"):
        imp = F.compute_rf_feature_importances(X_tr, y_tr_raw, n_estimators=50, random_state=cfg["seed"])
        keep = imp.head(int(cfg["features"]["top_k"])).index.tolist()
        X_tr = X_tr[keep]
        X_te = X_te[keep]
        print(f"[load_xy] kept top-{len(keep)} features")

    if cfg["features"].get("pca"):
        n_comp = cfg["features"]["pca"]
        if isinstance(n_comp, float) and 0 < n_comp < 1:
            from sklearn.decomposition import PCA
            p = PCA(n_components=n_comp, random_state=cfg["seed"])
            X_tr = pd.DataFrame(p.fit_transform(X_tr), columns=[f"PC{i+1}" for i in range(p.n_components_)])
            X_te = pd.DataFrame(p.transform(X_te), columns=[f"PC{i+1}" for i in range(p.n_components_)])
            print(f"[load_xy] PCA variance -> {p.n_components_} components")
        else:
            X_tr, X_te, _ = F.apply_pca(X_tr, X_te, n_components=int(n_comp))
            print(f"[load_xy] PCA -> {n_comp} components")

    y_tr = _prepare_task(y_tr_raw, cfg["task"])
    y_te = _prepare_task(y_te_raw, cfg["task"])

    print(f"[load_xy] train={X_tr.shape} test={X_te.shape} classes={sorted(set(y_tr.tolist()))}")
    return {
        "X_train": np.asarray(X_tr, dtype=np.float32),
        "X_test":  np.asarray(X_te, dtype=np.float32),
        "y_train": y_tr,
        "y_test":  y_te,
        "feature_names": list(X_tr.columns),
    }


# --------------------------------------------------------------------------- pipeline
def _build_estimator(cfg: dict):
    """Build the classifier (wrapped pipeline) for the requested model."""
    name = cfg["model"]["name"]
    params = cfg["model"].get("params", {}) or {}
    seed = cfg["seed"]

    if name == "linear_svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(random_state=seed, max_iter=5000, **params)

    if name == "svm_rbf":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=False, random_state=seed, **params)

    if name == "svm_nystroem":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVC
        n_comp = int(params.pop("n_components", 300))
        gamma = params.pop("gamma", None)
        nys = Nystroem(n_components=n_comp, gamma=gamma, random_state=seed)
        svc = LinearSVC(random_state=seed, max_iter=5000, **params)
        return Pipeline([("nystroem", nys), ("linearsvc", svc)])

    raise ValueError(f"Unknown model: {name}")


def _apply_imbalance(cfg: dict, estimator):
    """Wrap estimator for imbalance handling."""
    mode = cfg.get("imbalance", "none")
    if mode == "none":
        return estimator
    if mode == "class_weight":
        # LinearSVC / SVC support class_weight
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(class_weight="balanced")
            except Exception:
                pass
        return estimator
    if mode in ("undersample", "smote"):
        if not _HAS_IMBLEARN:
            print(f"[training] imblearn not installed, skipping '{mode}'")
            return estimator
        from imblearn.under_sampling import RandomUnderSampler
        sampler = SMOTE(random_state=cfg["seed"]) if mode == "smote" else RandomUnderSampler(random_state=cfg["seed"])
        return ImbPipeline([("sampler", sampler), ("clf", estimator)])
    return estimator


# --------------------------------------------------------------------------- run
def _compute_metrics(y_true, y_pred, y_score, pos_label) -> dict:
    out = {
        "recall":     recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "precision":  precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1":         f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "macro_f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        out["pr_auc"] = average_precision_score((np.asarray(y_true) == pos_label).astype(int), y_score)
    except Exception:
        out["pr_auc"] = float("nan")
    # false alarm rate = FP / N_benign (for binary: benign=1)
    if pos_label == 1:
        benign_mask = (np.asarray(y_true) == 1)
        if benign_mask.sum() > 0:
            fp = ((y_pred == 0) & (np.asarray(y_true) == 1)).sum()
            out["false_alarm_rate"] = fp / benign_mask.sum()
        else:
            out["false_alarm_rate"] = float("nan")
    else:
        out["false_alarm_rate"] = float("nan")
    return out


def _decision_scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            return p[:, -1] if p.ndim > 1 else p
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return model.predict(X)


def run_experiment(cfg: dict, data: dict | None = None) -> dict:
    """Fit one estimator and record the result row."""
    C.ensure_dirs()
    if data is None:
        data = load_xy(cfg)

    X_tr, X_te = data["X_train"], data["X_test"]
    y_tr, y_te = data["y_train"], data["y_test"]

    base = _build_estimator(cfg)
    est  = _apply_imbalance(cfg, base)

    t0 = time.perf_counter()
    est.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = est.predict(X_te)
    pred_s = time.perf_counter() - t0

    pos_label = 1 if cfg["task"] == "binary" else None
    try:
        y_score = _decision_scores(est, X_te)
    except Exception:
        y_score = y_pred

    row: dict[str, Any] = {
        "run": cfg["run_name"],
        "model": cfg["model"]["name"],
        "task": cfg["task"],
        "imbalance": cfg.get("imbalance", "none"),
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "n_features_used": int(X_tr.shape[1]),
        "fit_seconds": float(fit_s),
        "predict_seconds_per_1k": float(pred_s) * 1000.0 / max(1, X_te.shape[0]) * 1000.0,
    }
    if cfg["task"] == "binary":
        row.update(_compute_metrics(y_te, y_pred, y_score, pos_label=1))
    else:
        row["recall"]   = recall_score(y_te, y_pred, average="macro", zero_division=0)
        row["precision"]= precision_score(y_te, y_pred, average="macro", zero_division=0)
        row["f1"]       = f1_score(y_te, y_pred, average="macro", zero_division=0)
        row["macro_f1"] = row["f1"]
        row["pr_auc"]   = float("nan")
        row["false_alarm_rate"] = float("nan")

    # persist artefact
    name = cfg["run_name"]
    out = C.MODELS_DIR / f"{name}.joblib"
    try:
        dump({"estimator": est, "config": cfg, "features": data["feature_names"]}, out)
    except Exception as exc:
        print(f"[run_experiment] warning: could not save model: {exc}")

    # append to experiments.csv
    table_path = C.TABLES_DIR / "experiments.csv"
    df_row = pd.DataFrame([row])
    if table_path.exists():
        df_row.to_csv(table_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(table_path, index=False)

    return {"name": name, "row": row, "pipeline": est}


# --------------------------------------------------------------------------- tune
def tune(cfg: dict, search_space: dict, n_iter: int, data: dict, sample: int, out_file: Path) -> tuple[dict, pd.DataFrame]:
    """Randomised search over search_space. Returns (best_params, results_df)."""
    X_tr, y_tr = data["X_train"], data["y_train"]
    if sample and len(X_tr) > sample:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(X_tr), size=sample, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    base = _build_estimator(cfg)
    est  = _apply_imbalance(cfg, base)

    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    scoring = cfg["cv"]["scoring"]
    search = RandomizedSearchCV(
        est, param_distributions=search_space, n_iter=n_iter,
        scoring=scoring, cv=cv, random_state=cfg["seed"], n_jobs=-1, refit=False,
    )
    search.fit(X_tr, y_tr)

    results = pd.DataFrame(search.cv_results_)
    best = dict(search.best_params_)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(best, indent=2))
    return best, results
