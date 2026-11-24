"""Hyperparameter search with ``GridSearchCV`` (File 56).

For each model a small grid (``config.model_params.TUNING_GRIDS``) is searched with
stratified k-fold cross-validation on a **class-capped sample of the TRAINING rows**.
The test set is never read here.

Why the score is "prior-corrected" macro F1
-------------------------------------------
The tuning sample keeps small classes whole and caps the large ones, so BENIGN is
about 18% of its rows instead of 83%.  Plain macro F1 measured on such folds
overstates precision (there are few BENIGN rows to be mistaken for attacks) and so
favours recall-heavy settings such as ``class_weight="balanced"``.  The correction:
every held-out row of class *c* counts ``full_count_c / sample_count_c`` times, which
is what its class would weigh in the full training data.  The weighted F1 per class is
then averaged over the classes (macro).  The weight depends only on the true class,
so it fits inside a normal scikit-learn scorer.  The plain macro F1 is recorded beside
it; the winner of each grid is chosen by the corrected score.  Both are estimates from
a few thousand rows of the large classes: noisy for the three rarest classes.

How a model is plugged into ``GridSearchCV``
--------------------------------------------
``TunableModel`` is a thin scikit-learn estimator around the project's own model
classes, so that the grid uses exactly the code that is trained later (GaussianNB's
float64 cast, XGBoost's device handling, AdaBoost's ``base_max_depth``).  Grid keys
are the parameter names of ``registry.create_model(key, params=...)``.

Honest limits (also in the doc)
-------------------------------
* The best parameters are found on at most 100,000 rows; they may not be optimal for
  2 million rows (for example, a larger ``min_samples_leaf`` may suit a smaller sample).
* Every grid contains the baseline setting.  The result states the baseline's score,
  the improvement, and whether the improvement is larger than the fold-to-fold spread of
  the paired differences ("beyond noise").  If it is not, the tuned model is not better
  in any way the folds can show.
* The winner is the best mean over the folds (first in grid order on an exact tie).
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from config import model_params, settings
from nids.evaluation.cross_validation import make_folds
from nids.models import registry
from nids.training.data_access import class_capped_indices
from nids.utils.io_utils import config_hash, read_json, write_json
from nids.utils.logger import get_logger

logger = get_logger(__name__)

SCORE = "macro_f1_corrected"
PLAIN_SCORE = "macro_f1"


class TuningError(RuntimeError):
    """The search cannot run or every grid point failed."""


_NOISY_LOGGERS = ("nids.models.base", "nids.models.xgboost")


@contextmanager
def _quiet_model_logs():
    """A grid search fits hundreds of models; their one-line INFO messages would flood the screen (warnings still show)."""
    loggers = [logging.getLogger(name) for name in _NOISY_LOGGERS]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def tuning_directory() -> Path:
    return settings.OPTIMIZATION_REPORT_DIR


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def prior_weights(y_full: np.ndarray, y_sample: np.ndarray, n_classes: int) -> np.ndarray:
    """Weight of a row of each class: ``count in the full training data / count in the tuning sample``."""
    full = np.bincount(np.asarray(y_full), minlength=n_classes).astype("float64")
    sample = np.bincount(np.asarray(y_sample), minlength=n_classes).astype("float64")
    if (sample == 0).any() and (full > 0).any():
        missing = np.flatnonzero((sample == 0) & (full > 0)).tolist()
        raise TuningError(f"the tuning sample has no rows of class code(s) {missing}")
    weights = np.divide(full, sample, out=np.zeros(n_classes), where=sample > 0)
    return weights


def corrected_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_weights: np.ndarray, n_classes: int) -> float:
    """Macro F1 over the classes present in ``y_true``, each true row weighted by ``class_weights[its class]``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(n_classes)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, sample_weight=np.asarray(class_weights)[y_true], zero_division=0
    )
    present = np.bincount(y_true, minlength=n_classes) > 0
    return float(f1[present].mean())


def make_scorers(class_weights: np.ndarray, n_classes: int) -> dict[str, Any]:
    return {
        SCORE: make_scorer(corrected_macro_f1, class_weights=np.asarray(class_weights), n_classes=n_classes),
        PLAIN_SCORE: make_scorer(corrected_macro_f1, class_weights=np.ones(n_classes), n_classes=n_classes),
    }


# ---------------------------------------------------------------------------
# The estimator that GridSearchCV sees
# ---------------------------------------------------------------------------
class TunableModel(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper around ``registry.create_model``; grid keys are the project's parameter names."""

    def __init__(self, model_key: str = "decision_tree", n_classes: int = 2, params: dict[str, Any] | None = None):
        self.model_key = model_key
        self.n_classes = n_classes
        self.params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        # ``params`` is returned as the very same object (scikit-learn's clone() checks this)
        return {"model_key": self.model_key, "n_classes": self.n_classes, "params": self.params}

    def set_params(self, **parameters: Any) -> "TunableModel":
        for name, value in parameters.items():
            if name in ("model_key", "n_classes", "params"):
                setattr(self, name, value)
            else:
                if self.params is None:
                    self.params = {}
                self.params[name] = value
        return self

    def fit(self, X: Any, y: Any) -> "TunableModel":
        with _quiet_model_logs():
            self.model_ = registry.create_model(self.model_key, params=dict(self.params or {}))
            self.model_.fit(X, y, n_classes=self.n_classes)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(X)


# ---------------------------------------------------------------------------
# Grids and baselines
# ---------------------------------------------------------------------------
def baseline_point(key: str, grid: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    """The untuned value of every parameter in the grid (what the baseline model uses)."""
    grid = grid if grid is not None else model_params.tuning_grid(key)
    defaults = registry.create_model(key).build_estimator().get_params()
    point: dict[str, Any] = {}
    for name in grid:
        if name in model_params.TUNING_BASELINE_OVERRIDES.get(key, {}):
            point[name] = model_params.TUNING_BASELINE_OVERRIDES[key][name]
        elif name in defaults:
            point[name] = defaults[name]
        else:
            raise TuningError(f"cannot determine the baseline value of {name!r} for {key!r}")
    return point


def check_grid(key: str, grid: dict[str, list[Any]] | None = None) -> None:
    """Raise if the grid does not contain the baseline setting (the search must be able to return the baseline)."""
    grid = grid if grid is not None else model_params.tuning_grid(key)
    point = baseline_point(key, grid)
    missing = {name: value for name, value in point.items() if value not in grid[name]}
    if missing:
        raise TuningError(f"the grid of {key!r} lacks the baseline value(s) {missing}")


def preflight(key: str, X: np.ndarray, y: np.ndarray, n_classes: int, grid: dict[str, list[Any]] | None = None, rows_per_class: int = 12) -> None:
    """Fit every grid point once on a tiny slice, so an invalid value is found in seconds, not after hours.

    scikit-learn validates parameters at fit time, and GridSearchCV would otherwise record an invalid grid point as a
    failed fit (a silent NaN).  Raises ``TuningError`` listing every grid point that cannot be fitted.
    """
    grid = grid if grid is not None else model_params.tuning_grid(key)
    y = np.asarray(y)
    take = np.sort(np.concatenate([np.flatnonzero(y == c)[:rows_per_class] for c in range(n_classes)]))
    Xs, ys = np.asarray(X)[take], y[take]
    names = list(grid)
    problems = []
    for values in product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        try:
            with warnings.catch_warnings(), _quiet_model_logs():
                warnings.simplefilter("ignore")  # tiny slices do not converge; only invalid values matter here
                registry.create_model(key, params=dict(params)).fit(Xs, ys, n_classes=n_classes)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{params}: {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
    if problems:
        raise TuningError(f"{len(problems)} grid point(s) of {key!r} cannot be fitted:\n  " + "\n  ".join(problems[:5]))


def n_combinations(grid: dict[str, list[Any]]) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


def _plain(value: Any) -> Any:
    """JSON-safe Python value."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fingerprint(key: str, n_rows: int, folds: int, grid: dict[str, list[Any]]) -> str:
    return config_hash(
        {
            "preprocessing": config_hash(settings.preprocessing_config()),
            "model_params": config_hash(model_params.model_params_config()),
            "tuning": config_hash(model_params.tuning_config()),
            "key": key,
            "grid": {k: [repr(v) for v in vs] for k, vs in sorted(grid.items())},
            "n_rows": int(n_rows),
            "folds": int(folds),
            "seed": settings.RANDOM_SEED,
        }
    )


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def tune_model(
    key: str,
    X: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    n_classes: int,
    folds: int | None = None,
    grid: dict[str, list[Any]] | None = None,
    outer_jobs: int | None = None,
) -> dict[str, Any]:
    """Grid-search one model on the rows ``X, y`` (already the tuning sample) and return a JSON-safe result."""
    grid = grid if grid is not None else model_params.tuning_grid(key)
    check_grid(key, grid)
    folds = folds or model_params.tuning_folds(key)
    preflight(key, X, y, n_classes, grid)
    fold_pairs = make_folds(y, folds)
    if outer_jobs is None:
        outer_jobs = 1 if key in model_params.TUNING_SEQUENTIAL else max(1, min(settings.N_JOBS, n_combinations(grid) * folds))

    search = GridSearchCV(
        TunableModel(key, n_classes, {}),
        param_grid=grid,
        scoring=make_scorers(class_weights, n_classes),
        refit=False,
        cv=fold_pairs,
        n_jobs=outer_jobs,
        error_score=np.nan,
        return_train_score=False,
    )
    started = time.time()
    search.fit(np.asarray(X), np.asarray(y))
    seconds = time.time() - started
    cvr = search.cv_results_

    mean = np.asarray(cvr[f"mean_test_{SCORE}"], dtype="float64")
    if np.isnan(mean).all():
        raise TuningError(f"every grid point of {key!r} failed")
    n_failed = int(np.isnan(mean).sum())
    best = int(np.nanargmax(mean))  # first maximum: ties resolve to the earlier grid point

    def split_scores(index: int) -> np.ndarray:
        return np.array([cvr[f"split{i}_test_{SCORE}"][index] for i in range(folds)], dtype="float64")

    point = baseline_point(key, grid)
    params_list = [dict(p) for p in cvr["params"]]
    baseline_index = next(i for i, p in enumerate(params_list) if all(p[name] == value for name, value in point.items()))
    diff = split_scores(best) - split_scores(baseline_index)
    improvement = float(diff.mean())
    spread = float(diff.std(ddof=1)) if folds > 1 else 0.0
    table = []
    for i, p in enumerate(params_list):
        table.append(
            {
                "params": {k: _plain(v) for k, v in p.items()},
                "macro_f1_corrected_mean": None if np.isnan(mean[i]) else float(mean[i]),
                "macro_f1_corrected_std": float(cvr[f"std_test_{SCORE}"][i]),
                "macro_f1_plain_mean": float(cvr[f"mean_test_{PLAIN_SCORE}"][i]),
                "macro_f1_plain_std": float(cvr[f"std_test_{PLAIN_SCORE}"][i]),
                "fit_seconds_mean": float(cvr["mean_fit_time"][i]),
                "is_baseline": i == baseline_index,
                "is_best": i == best,
            }
        )
    table.sort(key=lambda r: (-(r["macro_f1_corrected_mean"] if r["macro_f1_corrected_mean"] is not None else -1.0)))
    for rank, row in enumerate(table, start=1):
        row["rank"] = rank
    return {
        "key": key,
        "display_name": model_params.DISPLAY_NAMES[key],
        "status": "ok",
        "n_combinations": n_combinations(grid),
        "failed_combinations": n_failed,
        "folds": folds,
        "tuning_rows": int(len(y)),
        "outer_jobs": outer_jobs,
        "selection_score": SCORE,
        "best_params": {k: _plain(v) for k, v in params_list[best].items()},
        "best_score": {"corrected_mean": float(mean[best]), "corrected_std": float(cvr[f"std_test_{SCORE}"][best]),
                       "plain_mean": float(cvr[f"mean_test_{PLAIN_SCORE}"][best]), "plain_std": float(cvr[f"std_test_{PLAIN_SCORE}"][best])},
        "baseline_params": {k: _plain(v) for k, v in point.items()},
        "baseline_score": {"corrected_mean": float(mean[baseline_index]), "corrected_std": float(cvr[f"std_test_{SCORE}"][baseline_index]),
                           "plain_mean": float(cvr[f"mean_test_{PLAIN_SCORE}"][baseline_index])},
        "best_is_baseline": best == baseline_index,
        "improvement_corrected": improvement,
        "improvement_fold_spread": spread,
        "improvement_beyond_noise": bool(best != baseline_index and improvement > spread),
        "seconds": round(seconds, 1),
        "table": table,
    }


# ---------------------------------------------------------------------------
# Several models, saved and resumable
# ---------------------------------------------------------------------------
def _result_path(key: str, directory: Path) -> Path:
    return directory / f"tuning_{key}.json"


def run_tuning(
    keys: list[str] | tuple[str, ...] | None,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    sample_size: int | None = None,
    skip_existing: bool = True,
    directory: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune several models (default all 11); each result is saved as soon as it is done.

    ``X, y`` are the FULL training arrays; each model gets its own class-capped sample
    (``model_params.tuning_sample_size``).  A saved result is reused only if its fingerprint
    (data, grid, folds, seed, settings) matches.  A model that fails is recorded and the others continue.
    """
    keys = list(keys) if keys else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        raise ValueError(f"unknown model key(s) {unknown}. Known: {list(model_params.MODEL_KEYS)}")
    out = Path(directory) if directory else tuning_directory()
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)

    results: dict[str, dict[str, Any]] = {}
    for position, key in enumerate(keys, start=1):
        name = model_params.DISPLAY_NAMES[key]
        grid = model_params.tuning_grid(key)
        folds = model_params.tuning_folds(key)
        target = model_params.tuning_sample_size(key, sample_size)
        idx = class_capped_indices(y, target)
        stamp = _fingerprint(key, len(idx), folds, grid)
        path = _result_path(key, out)
        if skip_existing and path.is_file():
            saved = read_json(path)
            if saved.get("fingerprint") == stamp and saved.get("status") == "ok":
                logger.info("[%d/%d] %s: reusing the saved tuning result", position, len(keys), name)
                results[key] = saved
                continue
        logger.info("[%d/%d] tuning %s: %d grid points x %d folds on %d rows", position, len(keys), name, n_combinations(grid), folds, len(idx))
        try:
            y_sample = y[idx]
            weights = prior_weights(y, y_sample, n_classes)
            result = tune_model(key, np.asarray(X[idx]), y_sample, weights, n_classes, folds=folds, grid=grid)
        except Exception as exc:  # noqa: BLE001 - one model must not stop the others
            logger.error("%s FAILED: %s: %s", name, type(exc).__name__, exc)
            results[key] = {"key": key, "display_name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            continue
        result.update({"fingerprint": stamp, "tuned_at": datetime.now().isoformat(timespec="seconds"),
                       "class_weights_min_max": [float(weights[weights > 0].min()), float(weights.max())]})
        write_json(result, path)  # last: marks the result as complete
        results[key] = result
        b = result["best_score"]
        logger.info("%s: best %s corrected macro F1 %.4f (baseline %.4f, %s) in %.1f min",
                    name, result["best_params"], b["corrected_mean"], result["baseline_score"]["corrected_mean"],
                    "beyond fold noise" if result["improvement_beyond_noise"] else "not beyond fold noise", result["seconds"] / 60)
    return results


def load_tuning(key: str, directory: Path | str | None = None) -> dict[str, Any]:
    path = _result_path(key, Path(directory) if directory else tuning_directory())
    if not path.is_file():
        raise TuningError(f"no tuning result for {key!r} (run the tuning script first)")
    return read_json(path)


def best_params(key: str, directory: Path | str | None = None) -> dict[str, Any]:
    """The winning parameters of a tuned model, ready for ``registry.create_model(key, params=...)``."""
    result = load_tuning(key, directory)
    if result.get("status") != "ok":
        raise TuningError(f"tuning of {key!r} failed: {result.get('error')}")
    return dict(result["best_params"])


def summary_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per tuned model (the order of ``results``)."""
    rows = []
    for key, r in results.items():
        if r.get("status") != "ok":
            continue
        rows.append(
            {
                "model": r["display_name"],
                "key": key,
                "tuning_rows": r["tuning_rows"],
                "folds": r["folds"],
                "grid_points": r["n_combinations"],
                "best_params": ", ".join(f"{k}={v}" for k, v in r["best_params"].items()),
                "baseline_corrected_f1": r["baseline_score"]["corrected_mean"],
                "best_corrected_f1": r["best_score"]["corrected_mean"],
                "improvement": r["improvement_corrected"],
                "beyond_noise": r["improvement_beyond_noise"],
                "best_is_baseline": r["best_is_baseline"],
                "best_plain_f1": r["best_score"]["plain_mean"],
                "minutes": r["seconds"] / 60,
            }
        )
    return pd.DataFrame(rows)