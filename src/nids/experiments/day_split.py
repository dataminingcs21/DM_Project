"""Split by capture file: can a detector catch attack types it has never seen?

In CIC-IDS2017 every attack class sits in exactly one of the eight capture files (one working
day or half-day).  The random 80/20 split of the main pipeline therefore always shows the
models examples of every attack, from the same session, often near-identical flows.  Here the
question is harder and closer to real use:

* **Leave one file out**: train a *binary* detector (BENIGN vs attack) on seven files, test it on
  the eighth.  The attack types of the held-out file were never seen in training (Monday has no
  attacks and is never held out).  The preprocessing pipeline is refitted on the training files
  of each fold, so nothing about the held-out file leaks into it.
* **Random-split reference**: the same model, the same rows and the same row caps, split at
  random (stratified on the class).  The gap between the two is what the random split hides.

A multiclass model cannot be tested this way (the held-out classes do not exist in training),
which is why the task is binary.  BENIGN training rows are subsampled (``benign_cap``) to keep the
run short; every attack row of the training files is kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import settings
from nids.preprocessing.pipeline_builder import build_pipeline
from nids.preprocessing.preprocessor import select_feature_columns
from nids.utils.logger import get_logger

logger = get_logger("experiments.day_split")


@dataclass
class FoldResult:
    held_out: str
    unseen_classes: list[str]
    metrics: dict[str, Any]
    per_class: pd.DataFrame


def short_name(source_file: str) -> str:
    """``Wednesday-workingHours.pcap_ISCX.csv`` -> ``Wednesday``; keeps the half-day suffix when there is one."""
    stem = source_file.split(".pcap")[0].replace("-WorkingHours", "").replace("-workingHours", "")
    return stem


def cap_benign(df: pd.DataFrame, benign_cap: int | None, seed: int) -> pd.DataFrame:
    """Keep every attack row and at most ``benign_cap`` random BENIGN rows (order preserved)."""
    is_benign = df[settings.LABEL_COLUMN].to_numpy() == settings.BENIGN_LABEL
    if benign_cap is None or is_benign.sum() <= benign_cap:
        return df
    rng = np.random.default_rng(seed)
    keep = ~is_benign
    keep[rng.choice(np.flatnonzero(is_benign), size=benign_cap, replace=False)] = True
    return df[keep]


def _prepare(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pipeline = build_pipeline().fit(train[features])
    X_train = pipeline.transform(train[features]).to_numpy(dtype="float32")
    X_test = pipeline.transform(test[features]).to_numpy(dtype="float32")
    return X_train, X_test, list(pipeline.get_feature_names_out())


def evaluate_binary(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> tuple[dict[str, Any], pd.DataFrame]:
    """Binary metrics plus the share of each class flagged as an attack."""
    labels = np.asarray(labels)
    is_attack = labels != settings.BENIGN_LABEL
    alarm = scores >= threshold
    metrics: dict[str, Any] = {
        "test_rows": int(labels.size),
        "attack_rows": int(is_attack.sum()),
        "detection_rate": float((alarm & is_attack).sum() / max(is_attack.sum(), 1)),
        "false_alarm_rate": float((alarm & ~is_attack).sum() / max((~is_attack).sum(), 1)),
        "f1_attack": float(f1_score(is_attack, alarm, zero_division=0)),
        "roc_auc": float(roc_auc_score(is_attack, scores)) if 0 < is_attack.sum() < labels.size else None,
        "average_precision": float(average_precision_score(is_attack, scores)) if is_attack.any() else None,
    }
    per_class = (
        pd.DataFrame({"class": labels, "flagged": alarm})
        .groupby("class")
        .agg(rows=("flagged", "size"), flagged_share=("flagged", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    return metrics, per_class


def leave_one_file_out(
    df: pd.DataFrame,
    make_model: Callable[[], Any],
    benign_cap: int | None = 400_000,
    seed: int | None = None,
    files: list[str] | None = None,
) -> list[FoldResult]:
    """One fold per capture file that contains attacks. ``make_model()`` returns a new binary classifier."""
    seed = settings.RANDOM_SEED if seed is None else seed
    features = select_feature_columns(df)
    src, label = settings.SOURCE_FILE_COLUMN, settings.LABEL_COLUMN
    attack_files = sorted(df.loc[df[label] != settings.BENIGN_LABEL, src].unique())
    results = []
    for held_out in files or attack_files:
        started = time.time()
        in_test = df[src].to_numpy() == held_out
        train = cap_benign(df[~in_test], benign_cap, seed)
        test = df[in_test]
        X_train, X_test, _ = _prepare(train, test, features)
        y_train = (train[label].to_numpy() != settings.BENIGN_LABEL).astype("int8")
        model = make_model().fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        metrics, per_class = evaluate_binary(scores, test[label].to_numpy())
        unseen = sorted(set(test[label]) - set(train[label]))
        metrics.update(train_rows=int(len(train)), train_attack_rows=int(y_train.sum()), seconds=round(time.time() - started, 1))
        results.append(FoldResult(short_name(held_out), unseen, metrics, per_class))
        logger.info("Held out %s: detection %.3f, false alarms %.4f, ROC-AUC %s", short_name(held_out), metrics["detection_rate"], metrics["false_alarm_rate"], metrics["roc_auc"])
    return results


def random_split_reference(
    df: pd.DataFrame, make_model: Callable[[], Any], benign_cap: int | None = 400_000, test_size: float = 0.2, seed: int | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Same detector, same caps, but a random stratified split (the optimistic setting of the main pipeline)."""
    seed = settings.RANDOM_SEED if seed is None else seed
    features = select_feature_columns(df)
    label = settings.LABEL_COLUMN
    capped = cap_benign(df, None if benign_cap is None else int(benign_cap / (1 - test_size)), seed)
    train, test = train_test_split(capped, test_size=test_size, stratify=capped[label], random_state=seed)
    X_train, X_test, _ = _prepare(train, test, features)
    y_train = (train[label].to_numpy() != settings.BENIGN_LABEL).astype("int8")
    model = make_model().fit(X_train, y_train)
    metrics, per_class = evaluate_binary(model.predict_proba(X_test)[:, 1], test[label].to_numpy())
    metrics.update(train_rows=int(len(train)))
    return metrics, per_class
