"""Train the 11 baseline models and save them.

For each model key the trainer:

1. takes the rows the model trains on (all rows, or its class-capped subsample;
   ``data_access.training_subset``);
2. builds a fresh model from the registry (baseline parameters of File 28);
3. fits it, measures the fit time, and measures the time to predict 5,000 rows
   of its own training data (the test set is never used here);
4. saves it with ``model_store`` (stage ``"trained"``) together with an ``extra``
   record: subsample information, timings, rows used.

Rules:

* **One failure does not stop the others.**  A model that raises is recorded as
  ``failed`` with the reason and the next model is trained (``continue_on_error``).
  Nothing is hidden: the summary lists every failure.  ``KeyboardInterrupt`` still
  stops everything, and models already saved stay saved.
* **Resume.**  With ``skip_existing=True`` a model that is already saved *and fresh*
  (same preprocessing and model settings) is not retrained; a stale one is.
* **Honest numbers.**  The fit and prediction times in the summary are measured, not
  estimated; a model that did not converge is flagged.
* **No test data.**  Training uses only the training arrays.  Scores on the test set
  are the evaluation's job (Files 48-53).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import model_params, settings
from nids.models import registry
from nids.persistence import model_store
from nids.training.data_access import TrainingData, load_training_data, training_subset
from nids.utils.logger import get_logger

logger = get_logger("training.trainer")

PREDICT_TIMING_ROWS = 5000


@dataclass
class TrainResult:
    key: str
    display_name: str
    status: str  # "trained", "skipped" or "failed"
    fit_seconds: float | None = None
    predict_seconds_5000: float | None = None
    n_rows_used: int | None = None
    n_rows_available: int | None = None
    subsampled: bool | None = None
    converged: bool | None = None  # only models that report it (Logistic Regression, SGD)
    train_device: str | None = None  # XGBoost
    file_size_mb: float | None = None
    error: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSummary:
    results: list[TrainResult] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def trained(self) -> list[str]:
        return [r.key for r in self.results if r.status == "trained"]

    @property
    def skipped(self) -> list[str]:
        return [r.key for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[str]:
        return [r.key for r in self.results if r.status == "failed"]

    @property
    def not_converged(self) -> list[str]:
        return [r.key for r in self.results if r.converged is False]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": round(self.total_seconds, 1),
            "trained": self.trained,
            "skipped": self.skipped,
            "failed": self.failed,
            "not_converged": self.not_converged,
            "results": [r.to_dict() for r in self.results],
        }


def _time_prediction(model, X: np.ndarray) -> float:
    """Seconds to predict (labels and scores) up to ``PREDICT_TIMING_ROWS`` rows of the model's own training rows."""
    sample = np.asarray(X[:PREDICT_TIMING_ROWS])
    started = time.time()
    model.predict(sample)
    model.predict_scores(sample)
    return round(time.time() - started, 3)


def train_model(
    key: str,
    data: TrainingData,
    stage: str = "trained",
    save: bool = True,
    params: dict[str, Any] | None = None,
    directory: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> TrainResult:
    """Train one model on ``data`` and (by default) save it.  Exceptions are not caught here.

    ``extra`` (JSON-safe) is stored in the saved metadata next to the subsample record; the tuning step
    (File 57) uses it to keep the winning parameters and the tuning fingerprint with the tuned model.
    """
    X, y, subsample = training_subset(key, data.X, data.y)
    model = registry.create_model(key, params=params)
    logger.info("Training %s on %d of %d rows", model.display_name, subsample["n_rows"], subsample["n_rows_available"])
    model.fit(X, y, n_classes=data.n_classes)
    predict_seconds = _time_prediction(model, X)
    info = model.info()

    result = TrainResult(
        key=key,
        display_name=model.display_name,
        status="trained",
        fit_seconds=info["fit_seconds"],
        predict_seconds_5000=predict_seconds,
        n_rows_used=subsample["n_rows"],
        n_rows_available=subsample["n_rows_available"],
        subsampled=subsample["subsampled"],
        converged=info.get("converged"),
        train_device=info.get("train_device"),
    )
    if result.converged is False:
        result.note = "did not converge: the result is not a true optimum"
    if save:
        record = {
            "subsample": subsample,
            "predict_seconds_5000_train_rows": predict_seconds,
            "n_train_rows_used": subsample["n_rows"],
            **(extra or {}),
        }
        meta = model_store.save_model(model, stage=stage, extra=record, directory=directory)
        result.file_size_mb = meta["file_size_mb"]
    logger.info(
        "%s done: fit %.1f s, predicting %d rows %.2f s%s",
        model.display_name, result.fit_seconds, PREDICT_TIMING_ROWS, predict_seconds,
        f", {result.file_size_mb} MB" if result.file_size_mb is not None else "",
    )
    return result


def train_models(
    keys: list[str] | tuple[str, ...] | None = None,
    data: TrainingData | None = None,
    stage: str = "trained",
    save: bool = True,
    skip_existing: bool = False,
    continue_on_error: bool = True,
    directory: Path | str | None = None,
) -> TrainingSummary:
    """Train several models one after another (default: all 11, in the order of the project brief)."""
    keys = list(keys) if keys else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        raise ValueError(f"unknown model key(s) {unknown}. Known: {list(model_params.MODEL_KEYS)}")
    data = data if data is not None else load_training_data()
    already = model_store.list_saved(stage, directory) if skip_existing else {}

    summary = TrainingSummary()
    started = time.time()
    for position, key in enumerate(keys, start=1):
        name = model_params.DISPLAY_NAMES[key]
        if skip_existing and already.get(key, {}).get("fresh"):
            logger.info("[%d/%d] %s: already saved and fresh, skipped", position, len(keys), name)
            summary.results.append(TrainResult(key=key, display_name=name, status="skipped", note="already saved and fresh"))
            continue
        logger.info("[%d/%d] %s", position, len(keys), name)
        try:
            summary.results.append(train_model(key, data, stage=stage, save=save, directory=directory))
        except Exception as exc:  # one model failing must not hide the others
            logger.error("%s FAILED: %s: %s", name, type(exc).__name__, exc)
            summary.results.append(
                TrainResult(key=key, display_name=name, status="failed", error=f"{type(exc).__name__}: {exc}")
            )
            if not continue_on_error:
                break
    summary.total_seconds = time.time() - started
    logger.info(
        "Training finished in %.1f s: %d trained, %d skipped, %d failed",
        summary.total_seconds, len(summary.trained), len(summary.skipped), len(summary.failed),
    )
    return summary


def format_summary(summary: TrainingSummary) -> str:
    """A plain-text table of the results (the script also writes it to a report)."""
    header = f"{'model':<22}{'status':<9}{'fit s':>9}{'predict 5k s':>14}{'rows used':>12}  notes"
    lines = [header, "-" * len(header)]
    for r in summary.results:
        notes = []
        if r.subsampled:
            notes.append(f"subsample of {r.n_rows_available:,}")
        if r.note:
            notes.append(r.note)
        if r.error:
            notes.append(r.error)
        if r.train_device:
            notes.append(f"device {r.train_device}")
        if r.file_size_mb is not None:
            notes.append(f"{r.file_size_mb} MB")
        lines.append(
            f"{r.display_name:<22}{r.status:<9}"
            f"{'' if r.fit_seconds is None else f'{r.fit_seconds:.1f}':>9}"
            f"{'' if r.predict_seconds_5000 is None else f'{r.predict_seconds_5000:.2f}':>14}"
            f"{'' if r.n_rows_used is None else f'{r.n_rows_used:,}':>12}  " + "; ".join(notes)
        )
    lines.append(f"total {summary.total_seconds:.1f} s")
    return "\n".join(lines)