"""Shared pieces of the experiments: cached test predictions and one scoring function.

Test predictions of the saved models are expensive (KNN and SVC need minutes on 504,473 rows),
so they are computed once and cached in ``reports/experiments/test_predictions/<stage>_<key>.npy``.
A cache file is reused only when its row count matches the current test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import model_params, settings
from nids.persistence import model_store
from nids.utils.logger import get_logger

logger = get_logger("experiments.common")


def prediction_cache_file(key: str, stage: str) -> Path:
    return settings.TEST_PREDICTIONS_DIR / f"{stage}_{key}.npy"


def cached_test_predictions(
    X_test: np.ndarray, keys: tuple[str, ...] | list[str] | None = None, stage: str = "trained"
) -> dict[str, np.ndarray]:
    """Test-set class codes of every saved model in ``stage``, computed once and cached."""
    keys = tuple(model_params.MODEL_KEYS if keys is None else keys)
    out: dict[str, np.ndarray] = {}
    for key in keys:
        path = prediction_cache_file(key, stage)
        if path.is_file():
            cached = np.load(path)
            if cached.shape == (X_test.shape[0],):
                out[key] = cached
                continue
            logger.warning("Cache %s has the wrong length; recomputing", path.name)
        model = model_store.load_model(key, stage)
        logger.info("Predicting the test set with %s (%s) ...", key, stage)
        predictions = np.concatenate([model.predict(X_test[i : i + 100_000]) for i in range(0, X_test.shape[0], 100_000)])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, predictions.astype("int16"))
        out[key] = predictions.astype("int16")
    return out


def predict_batched(model: Any, X: np.ndarray, batch: int = 100_000) -> np.ndarray:
    return np.concatenate([np.asarray(model.predict(X[i : i + batch])) for i in range(0, X.shape[0], batch)]).astype("int64")


def fit_and_predict(
    key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    class_weight_mode: str | None = "none",
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a new project model (baseline parameters) and predict ``X_test``. Returns predictions and fit seconds."""
    from nids.models.registry import create_model

    model = create_model(key, class_weight_mode=class_weight_mode)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return predict_batched(model, X_test), float(model.fit_seconds_ or 0.0)


def score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, benign_index: int | None = 0, min_support: int = 30) -> dict[str, Any]:
    """Macro F1 over all classes and over classes with at least ``min_support`` true rows, plus attack detection."""
    labels = np.arange(n_classes)
    support = np.bincount(np.asarray(y_true), minlength=n_classes)
    reliable = labels[support >= min_support]
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_reliable": float(f1_score(y_true, y_pred, labels=reliable, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }
    if benign_index is not None:
        attack_true = np.asarray(y_true) != benign_index
        attack_pred = np.asarray(y_pred) != benign_index
        result["detection_rate"] = float((attack_true & attack_pred).sum() / max(attack_true.sum(), 1))
        result["false_alarm_rate"] = float((~attack_true & attack_pred).sum() / max((~attack_true).sum(), 1))
    return result


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": class_names, "support": support, "precision": precision, "recall": recall, "f1": f1})
