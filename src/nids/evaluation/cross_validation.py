"""Stratified k-fold cross-validation of the baseline models, on the TRAINING rows only.

Why: a single train/test split gives one number per model.  Cross-validation
trains each model k times on different parts of the training data and scores it
on the part it did not see, so we also learn how much the score *moves* from one
split to the next.  The test set is never used here: it stays untouched for the
final comparison (Files 52-53).

How a model is cross-validated
------------------------------
* Folds: ``StratifiedKFold(shuffle=True, random_state=seed)`` on the class codes, so every fold has
  every class.  Every class needs at least ``n_splits`` rows (the rarest class, Heartbleed, has 9
  training rows: 5 folds work, 10 would not).
* In each fold the model is trained exactly as in ``scripts/05_train.py``: the baseline
  parameters, and for KNN, SVC and Gradient Boosting the class-capped subsample of that fold's
  training rows.  It is then scored on the held-out rows, which keep the real class mix.
* Labels only (``predict``); scores are not computed, because they are not needed here and KNN
  and SVC would take as long again.
* **Out-of-fold predictions**: every training row is predicted exactly once, by the model that
  did not see it.  Pooled over all rows they give per-class numbers with the full class counts
  (more rows for the rare classes than any single fold).

What is reported per model
--------------------------
Mean and standard deviation over the folds of accuracy, balanced accuracy, macro F1, weighted F1,
MCC, attack detection rate and false alarm rate; fit and prediction times; the pooled
out-of-fold metrics (``compute_metrics``).  **Macro F1** is the primary score for choosing between
models, because it weighs the 15 classes equally.  ``rank_models`` also marks every model whose
mean is within one standard deviation of the best: those cannot be told apart from the best on
this evidence, and the report says so instead of declaring a winner by a hair.

Results are saved per model (``cv_<key>.json`` and ``cv_<key>_oof.npy``) with a fingerprint of
the settings, so an interrupted run resumes and a run with other settings is not mixed in.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import model_params, settings
from nids.evaluation.metrics import compute_metrics
from nids.models import registry
from nids.preprocessing.target_builder import LabelCodec
from nids.training.data_access import class_capped_indices
from nids.utils.io_utils import config_hash, load_array, read_json, save_array, write_json
from nids.utils.logger import get_logger

logger = get_logger("evaluation.cross_validation")

FOLD_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc")
PRIMARY_METRIC = "macro_f1"


def cv_directory() -> Path:
    return settings.EVALUATION_REPORT_DIR / "cross_validation"


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def make_folds(y: np.ndarray, n_splits: int = 5, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds as ``(train_positions, test_positions)`` pairs (every position is in exactly one test fold)."""
    y = np.asarray(y)
    seed = settings.RANDOM_SEED if seed is None else seed
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    counts = np.bincount(y)
    too_small = {int(c): int(n) for c, n in enumerate(counts) if 0 < n < n_splits}
    if too_small or (counts == 0).any():
        raise ValueError(
            f"every class needs at least {n_splits} rows for {n_splits}-fold cross-validation; "
            f"too small: {too_small}, missing: {np.flatnonzero(counts == 0).tolist()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y)]


def fingerprint(n_rows: int, n_splits: int, seed: int, sample_size: int | None) -> str:
    """Settings that decide whether saved CV results can be reused."""
    return config_hash(
        {
            "preprocessing": config_hash(settings.preprocessing_config()),
            "model_params": config_hash(model_params.model_params_config()),
            "n_rows": int(n_rows),
            "n_splits": int(n_splits),
            "seed": int(seed),
            "sample_size": sample_size,
        }
    )


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def _fold_training_positions(key: str, y: np.ndarray, train_pos: np.ndarray, fold: int) -> np.ndarray:
    """The rows of this fold's training part that the model trains on (all, or its class-capped subsample)."""
    target = model_params.subsample_size(key)
    if target is None or target >= train_pos.size:
        return train_pos
    return train_pos[class_capped_indices(y[train_pos], target, seed=settings.RANDOM_SEED + fold)]


def cross_validate_model(
    key: str,
    X: np.ndarray,
    y: np.ndarray,
    codec: LabelCodec,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Cross-validate one model.  Returns ``(result, out_of_fold_predictions)``."""
    oof = np.full(len(y), -1, dtype="int16")
    fold_rows: list[dict[str, Any]] = []
    for number, (train_pos, test_pos) in enumerate(folds, start=1):
        use = _fold_training_positions(key, y, train_pos, number)
        model = registry.create_model(key, params=params)
        model.fit(X[use], y[use], n_classes=codec.n_classes)
        started = time.time()
        pred = model.predict(X[test_pos])
        predict_seconds = time.time() - started
        oof[test_pos] = pred
        m = compute_metrics(y[test_pos], pred, codec)
        row = {name: m["overall"][name] for name in FOLD_METRICS}
        view = m["attack_vs_normal"] or {}
        row.update(
            {
                "detection_rate": view.get("detection_rate"),
                "false_alarm_rate": view.get("false_alarm_rate"),
                "fit_seconds": model.info()["fit_seconds"],
                "predict_seconds": round(predict_seconds, 3),
                "rows_trained": int(use.size),
                "rows_scored": int(test_pos.size),
            }
        )
        fold_rows.append({"fold": number, **row})
        logger.info(
            "%s fold %d/%d: macro F1 %.4f, accuracy %.4f (fit %.1f s, %d rows)",
            model.display_name, number, len(folds), row["macro_f1"], row["accuracy"], row["fit_seconds"], use.size,
        )
    if (oof < 0).any():
        raise RuntimeError("some rows were not predicted by any fold")  # cannot happen with StratifiedKFold; a guard

    frame = pd.DataFrame(fold_rows)
    summary = {}
    for name in (*FOLD_METRICS, "detection_rate", "false_alarm_rate", "fit_seconds", "predict_seconds"):
        values = frame[name].dropna().to_numpy(dtype="float64")
        summary[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=1)) if values.size > 1 else None,
        }
    pooled = compute_metrics(y, oof.astype("int64"), codec)
    result = {
        "key": key,
        "display_name": model_params.DISPLAY_NAMES[key],
        "n_splits": len(folds),
        "n_rows": int(len(y)),
        "subsampled": model_params.subsample_size(key) is not None,
        "rows_trained_mean": float(frame["rows_trained"].mean()),
        "folds": fold_rows,
        "summary": summary,
        "pooled": pooled,
    }
    return result, oof


# ---------------------------------------------------------------------------
# Several models, saved and resumable
# ---------------------------------------------------------------------------
def _paths(key: str, directory: Path) -> tuple[Path, Path]:
    return directory / f"cv_{key}.json", directory / f"cv_{key}_oof.npy"


def run_cross_validation(
    keys: list[str] | tuple[str, ...] | None,
    X: np.ndarray,
    y: np.ndarray,
    codec: LabelCodec,
    n_splits: int = 5,
    sample_size: int | None = None,
    skip_existing: bool = True,
    directory: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Cross-validate several models (default all 11), saving each result as soon as it is done.

    ``sample_size`` cross-validates on a class-capped stratified sample of that many training rows
    (faster; the held-out folds are then more balanced than the real traffic).  A saved result is
    reused only if its fingerprint matches (same data size, folds, seed, sample and settings).
    A model that fails is recorded as ``{"status": "failed", "error": ...}`` and the others continue.
    """
    keys = list(keys) if keys else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        raise ValueError(f"unknown model key(s) {unknown}. Known: {list(model_params.MODEL_KEYS)}")
    out = Path(directory) if directory else cv_directory()
    out.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y)
    if sample_size is not None:
        idx = class_capped_indices(y, sample_size)
        X, y = X[idx], y[idx]
        logger.info("Cross-validation on a class-capped sample of %d rows", len(y))
    else:
        X = np.asarray(X)
    folds = make_folds(y, n_splits)
    stamp = fingerprint(len(y), n_splits, settings.RANDOM_SEED, sample_size)

    results: dict[str, dict[str, Any]] = {}
    for position, key in enumerate(keys, start=1):
        json_path, oof_path = _paths(key, out)
        if skip_existing and json_path.is_file() and oof_path.is_file():
            saved = read_json(json_path)
            if saved.get("fingerprint") == stamp and saved.get("status") == "ok":
                logger.info("[%d/%d] %s: reusing saved cross-validation", position, len(keys), model_params.DISPLAY_NAMES[key])
                results[key] = saved
                continue
        logger.info("[%d/%d] cross-validating %s", position, len(keys), model_params.DISPLAY_NAMES[key])
        try:
            result, oof = cross_validate_model(key, X, y, codec, folds)
        except Exception as exc:
            logger.error("%s FAILED: %s: %s", model_params.DISPLAY_NAMES[key], type(exc).__name__, exc)
            results[key] = {"key": key, "display_name": model_params.DISPLAY_NAMES[key], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            continue
        result.update({"status": "ok", "fingerprint": stamp, "sample_size": sample_size})
        save_array(oof, oof_path)
        write_json(result, json_path)  # written last: its presence marks the result as complete
        results[key] = result
    return results


def load_oof(key: str, directory: Path | str | None = None) -> np.ndarray:
    """The saved out-of-fold predictions of a model (position = row of the cross-validated data)."""
    return load_array((Path(directory) if directory else cv_directory()) / f"cv_{key}_oof.npy").astype("int64")


# ---------------------------------------------------------------------------
# Ranking and table
# ---------------------------------------------------------------------------
def rank_models(results: dict[str, dict[str, Any]], metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """One row per successful model, best first by the mean of ``metric``.

    ``indistinguishable_from_best`` is True when the mean is within one standard deviation of the best model's
    standard deviation below its mean: the evidence cannot separate such a model from the best one.
    """
    rows = []
    for key, r in results.items():
        if r.get("status") != "ok":
            continue
        s = r["summary"]
        rows.append(
            {
                "model": r["display_name"],
                "key": key,
                f"{metric}_mean": s[metric]["mean"],
                f"{metric}_std": s[metric]["std"],
                "accuracy_mean": s["accuracy"]["mean"],
                "balanced_accuracy_mean": s["balanced_accuracy"]["mean"],
                "mcc_mean": s["mcc"]["mean"],
                "detection_rate_mean": s["detection_rate"]["mean"],
                "false_alarm_rate_mean": s["false_alarm_rate"]["mean"],
                "fit_seconds_mean": s["fit_seconds"]["mean"],
                "predict_seconds_mean": s["predict_seconds"]["mean"],
                "trained_on_subsample": r["subsampled"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(f"{metric}_mean", ascending=False, kind="stable").reset_index(drop=True)
    best_mean, best_std = frame.loc[0, f"{metric}_mean"], frame.loc[0, f"{metric}_std"]
    threshold = best_mean - (0.0 if pd.isna(best_std) else best_std)
    frame["indistinguishable_from_best"] = frame[f"{metric}_mean"] >= threshold
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame