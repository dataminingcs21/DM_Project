"""Unsupervised (one-class) anomaly detection: learn normal traffic only, flag what is unusual.

This is how most real intrusion detection must work, because labelled attacks are rare and
new attacks have no labels at all.  Every detector is fitted on **BENIGN training rows only**;
attack labels are used afterwards only, to score the detectors.

Detectors (all give a score where *higher = more anomalous*):

* Isolation Forest - anomalies are isolated by fewer random splits;
* Local Outlier Factor (novelty mode) - density compared with the k nearest neighbours;
* Autoencoder - a small neural network (scikit-learn ``MLPRegressor``) trained to reproduce
  BENIGN flows; the reconstruction error of a flow is its score.

Thresholds are **not** tuned on the test set: for a target false-alarm rate ``a`` the threshold
is the ``1 - a`` quantile of the scores of *held-out BENIGN training rows* (calibration rows the
detector did not fit on).  The test set then shows the real false-alarm rate and detection rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("mining.anomaly")


# ---------------------------------------------------------------------------
# Detectors with one interface: fit(X_benign) and score(X) (higher = more anomalous)
# ---------------------------------------------------------------------------
class AnomalyDetector:
    name = "base"

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetector):
    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 200, seed: int | None = None):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=256,
            random_state=settings.RANDOM_SEED if seed is None else seed,
            n_jobs=settings.N_JOBS,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)  # sklearn: lower = more abnormal


class LOFDetector(AnomalyDetector):
    name = "Local Outlier Factor"

    def __init__(self, n_neighbors: int = 20):
        # brute force: tree indexes do not help in 40 dimensions; brute uses chunked BLAS
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, algorithm="brute", n_jobs=settings.N_JOBS)

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray, batch: int = 50_000) -> np.ndarray:
        return np.concatenate([-self.model.score_samples(X[i : i + batch]) for i in range(0, X.shape[0], batch)])


class AutoencoderDetector(AnomalyDetector):
    name = "Autoencoder"

    def __init__(self, hidden: tuple[int, ...] = (32, 16, 8, 16, 32), max_iter: int = 60, seed: int | None = None):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=settings.RANDOM_SEED if seed is None else seed,
        )

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        self.model.fit(X, X)
        return self

    def score(self, X: np.ndarray, batch: int = 100_000) -> np.ndarray:
        parts = []
        for i in range(0, X.shape[0], batch):
            chunk = X[i : i + batch]
            parts.append(np.mean((self.model.predict(chunk) - chunk) ** 2, axis=1))
        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class DetectorResult:
    name: str
    fit_rows: int
    fit_seconds: float
    score_seconds: float
    roc_auc: float
    average_precision: float
    operating_points: list[dict[str, Any]] = field(default_factory=list)
    per_class: pd.DataFrame | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fit_rows": self.fit_rows,
            "fit_seconds": self.fit_seconds,
            "score_seconds": self.score_seconds,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "operating_points": self.operating_points,
            "per_class_detection": None if self.per_class is None else self.per_class.to_dict(orient="records"),
        }


def thresholds_from_calibration(calibration_scores: np.ndarray, false_alarm_targets: tuple[float, ...]) -> dict[float, float]:
    """Threshold per target false-alarm rate: the (1 - a) quantile of BENIGN calibration scores."""
    scores = np.asarray(calibration_scores, dtype="float64")
    if scores.size == 0:
        raise ValueError("calibration scores are empty")
    for a in false_alarm_targets:
        if not 0.0 < a < 1.0:
            raise ValueError(f"false-alarm target must be in (0, 1), got {a}")
    return {a: float(np.quantile(scores, 1.0 - a)) for a in false_alarm_targets}


def operating_point(scores: np.ndarray, is_attack: np.ndarray, threshold: float) -> dict[str, float]:
    """Detection rate (recall of attacks), false-alarm rate and precision at one threshold (alarm when score > threshold)."""
    alarm = np.asarray(scores) > threshold
    is_attack = np.asarray(is_attack, dtype=bool)
    tp = int((alarm & is_attack).sum())
    fp = int((alarm & ~is_attack).sum())
    n_attack, n_benign = int(is_attack.sum()), int((~is_attack).sum())
    return {
        "threshold": float(threshold),
        "detection_rate": tp / n_attack if n_attack else float("nan"),
        "false_alarm_rate": fp / n_benign if n_benign else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "alarms": int(alarm.sum()),
    }


def per_class_detection(scores: np.ndarray, y_names: np.ndarray, thresholds: dict[float, float]) -> pd.DataFrame:
    """Share of each class flagged at every threshold (for BENIGN this is the false-alarm rate)."""
    frame = pd.DataFrame({"class": y_names, "score": scores})
    rows = []
    for cls, group in frame.groupby("class", sort=False):
        row: dict[str, Any] = {"class": cls, "test_rows": len(group), "median_score": float(group["score"].median())}
        for a, t in thresholds.items():
            row[f"flagged_at_far_{a:g}"] = float((group["score"] > t).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("test_rows", ascending=False).reset_index(drop=True)


def evaluate_detector(
    detector: AnomalyDetector,
    X_fit: np.ndarray,
    X_calibration: np.ndarray,
    X_test: np.ndarray,
    y_test_names: np.ndarray,
    false_alarm_targets: tuple[float, ...] | None = None,
) -> tuple[DetectorResult, np.ndarray]:
    """Fit on BENIGN rows, set thresholds on BENIGN calibration rows, score the test rows."""
    targets = tuple(settings.ANOMALY_FALSE_ALARM_TARGETS if false_alarm_targets is None else false_alarm_targets)
    started = time.time()
    detector.fit(X_fit)
    fit_seconds = time.time() - started
    thresholds = thresholds_from_calibration(detector.score(X_calibration), targets)
    started = time.time()
    scores = detector.score(X_test)
    score_seconds = time.time() - started
    is_attack = np.asarray(y_test_names) != settings.BENIGN_LABEL
    result = DetectorResult(
        name=detector.name,
        fit_rows=int(X_fit.shape[0]),
        fit_seconds=round(fit_seconds, 1),
        score_seconds=round(score_seconds, 1),
        roc_auc=float(roc_auc_score(is_attack, scores)),
        average_precision=float(average_precision_score(is_attack, scores)),
        operating_points=[{"target_false_alarm_rate": a, **operating_point(scores, is_attack, t)} for a, t in thresholds.items()],
        per_class=per_class_detection(scores, np.asarray(y_test_names), thresholds),
    )
    logger.info("%s: ROC-AUC %.4f, AP %.4f (fit %.0f s, score %.0f s)", detector.name, result.roc_auc, result.average_precision, fit_seconds, score_seconds)
    return result, scores
