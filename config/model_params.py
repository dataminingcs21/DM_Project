"""Default hyperparameters of the 11 models (the untuned baseline).

Decision (Checkpoint 3, question 1, option A): **library defaults**, plus only the
settings that must be fixed so that a run is reproducible and finishes:

* ``random_state`` = ``settings.RANDOM_SEED`` where the algorithm is random;
* ``n_jobs`` = ``settings.N_JOBS`` where the algorithm can use several threads;
* iteration caps where the default can stop too early to converge
  (``max_iter`` of LogisticRegression, SGDClassifier);
* ``tree_method="hist"`` for XGBoost (the fast method, also needed for the GPU).

Everything else is the default of scikit-learn 1.8.0 / XGBoost 3.1.3, so the gain
found later by GridSearchCV (File 56) is measured against a neutral baseline.
The tuning grids are further down (File 56); they do not enter the staleness hash of the trained baselines.

Models that are too slow for 2 million rows use a class-capped stratified
subsample; the sizes are in ``settings.SUBSAMPLE_SIZES`` (starting values,
calibrated by timing at File 47).
"""

from __future__ import annotations

from typing import Any

from config import settings

# Order of the mandatory list of the project brief.
MODEL_KEYS: tuple[str, ...] = (
    "knn",
    "decision_tree",
    "random_forest",
    "gaussian_nb",
    "svm",
    "bernoulli_nb",
    "logistic_regression",
    "sgd",
    "adaboost",
    "gradient_boosting",
    "xgboost",
)

DISPLAY_NAMES: dict[str, str] = {
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "SVC (RBF kernel)",
    "bernoulli_nb": "Bernoulli Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "sgd": "SGD Classifier",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

_SEED = settings.RANDOM_SEED
_JOBS = settings.N_JOBS

# key -> {parameter: value}; only the parameters that are set on purpose are listed.
_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "knn": {"n_jobs": _JOBS},                                  # n_neighbors=5, uniform weights
    "decision_tree": {"random_state": _SEED},
    "random_forest": {"random_state": _SEED, "n_jobs": _JOBS},  # n_estimators=100
    "gaussian_nb": {},                                          # var_smoothing=1e-9
    "svm": {"random_state": _SEED},                             # RBF kernel, C=1.0, no probability estimates
    "bernoulli_nb": {},                                         # binarize=0.0 (see the doc: on standardized data)
    "logistic_regression": {"random_state": _SEED, "max_iter": 1000},
    "sgd": {"random_state": _SEED, "max_iter": 1000, "n_jobs": _JOBS},  # hinge loss: no predict_proba
    "adaboost": {"random_state": _SEED},                        # n_estimators=50
    "gradient_boosting": {"random_state": _SEED},               # n_estimators=100 (one tree per class per round)
    "xgboost": {"random_state": _SEED, "n_jobs": _JOBS, "tree_method": "hist"},  # device: see settings.XGBOOST_DEVICE
}

# How each model can take class weights (used only when settings.CLASS_WEIGHT_MODE == "balanced"):
#   "class_weight"  -> the estimator has a ``class_weight`` parameter
#   "sample_weight" -> weights are passed to ``fit(..., sample_weight=...)``
#   None            -> not supported (KNN has no training weights)
CLASS_WEIGHT_SUPPORT: dict[str, str | None] = {
    "knn": None,
    "decision_tree": "class_weight",
    "random_forest": "class_weight",
    "gaussian_nb": "sample_weight",
    "svm": "class_weight",
    "bernoulli_nb": "sample_weight",
    "logistic_regression": "class_weight",
    "sgd": "class_weight",
    "adaboost": "sample_weight",
    "gradient_boosting": "sample_weight",
    "xgboost": "sample_weight",
}

# ---------------------------------------------------------------------------
# Tuning (File 56).  Grids, sample sizes and folds of the GridSearchCV step.
# They are NOT part of model_params_config(): changing a grid must not make the
# trained baselines stale (the tuning results carry their own fingerprint).
# Every grid contains the baseline setting, so the search can never return something
# that was not at least compared with the untuned model.
# ---------------------------------------------------------------------------
TUNING_SAMPLE_SIZE = 100_000   # rows of the class-capped training sample the grids are searched on
TUNING_SAMPLE_CAPS: dict[str, int] = {"knn": 50_000, "svm": 30_000, "gradient_boosting": 30_000}  # slower models: fewer rows
TUNING_FOLDS: dict[str, int] = {"gradient_boosting": 3}   # default: settings.CV_FOLDS
# Models that already use several threads (or the GPU) inside one fit: the grid points run one after another.
TUNING_SEQUENTIAL: tuple[str, ...] = ("knn", "random_forest", "sgd", "xgboost")
# Baseline values that scikit-learn / XGBoost do not report through get_params().
TUNING_BASELINE_OVERRIDES: dict[str, dict[str, Any]] = {
    "adaboost": {"base_max_depth": 1},
    "xgboost": {"max_depth": 6, "learning_rate": 0.3, "n_estimators": 100},
}

TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "knn": {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]},
    "decision_tree": {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 2, 5, 10], "class_weight": [None, "balanced"]},
    "random_forest": {
        "n_estimators": [100, 300],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced"],
    },
    "gaussian_nb": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    "svm": {"C": [1.0, 10.0], "gamma": ["scale", 0.01, 0.1], "class_weight": [None, "balanced"]},
    "bernoulli_nb": {"alpha": [0.1, 1.0, 10.0], "binarize": [0.0, 0.5, 1.0]},
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "sgd": {"loss": ["hinge", "modified_huber", "log_loss"], "alpha": [1e-6, 1e-5, 1e-4, 1e-3], "class_weight": [None, "balanced"]},
    "adaboost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0], "base_max_depth": [1, 2, 4]},
    "gradient_boosting": {"learning_rate": [0.1, 0.2], "max_depth": [3, 5]},
    "xgboost": {"max_depth": [6, 8, 10], "learning_rate": [0.1, 0.3], "n_estimators": [100, 300]},
}


def tuning_grid(model_key: str) -> dict[str, list[Any]]:
    """A copy of the tuning grid of one model."""
    _check_key(model_key)
    return {name: list(values) for name, values in TUNING_GRIDS[model_key].items()}


def tuning_sample_size(model_key: str, requested: int | None = None) -> int:
    """Rows of the tuning sample for this model: the requested (or default) size, capped for slow models."""
    _check_key(model_key)
    size = TUNING_SAMPLE_SIZE if requested is None else requested
    return min(size, TUNING_SAMPLE_CAPS.get(model_key, size))


def tuning_folds(model_key: str) -> int:
    _check_key(model_key)
    return TUNING_FOLDS.get(model_key, settings.CV_FOLDS)


def tuning_config() -> dict[str, Any]:
    """JSON-safe description of the tuning settings (part of the fingerprint of saved tuning results)."""
    return {
        "grids": {k: {n: list(v) for n, v in sorted(g.items())} for k, g in sorted(TUNING_GRIDS.items())},
        "sample_size": TUNING_SAMPLE_SIZE,
        "sample_caps": dict(sorted(TUNING_SAMPLE_CAPS.items())),
        "folds": dict(sorted(TUNING_FOLDS.items())),
        "default_folds": settings.CV_FOLDS,
        "baseline_overrides": {k: dict(sorted(v.items())) for k, v in sorted(TUNING_BASELINE_OVERRIDES.items())},
    }


# Parameters that only change speed, never the result: left out of the staleness hash.
_HARDWARE_ONLY_PARAMS = ("n_jobs",)


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------
def _check_key(model_key: str) -> None:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model '{model_key}'. Known models: {list(MODEL_KEYS)}")


def default_params(model_key: str) -> dict[str, Any]:
    """A copy of the default parameters of one model (safe to modify)."""
    _check_key(model_key)
    return dict(_DEFAULT_PARAMS[model_key])


def subsample_size(model_key: str) -> int | None:
    """Rows of the class-capped training subsample for this model, or None for the full training data."""
    _check_key(model_key)
    return settings.SUBSAMPLE_SIZES.get(model_key)


def class_weight_mechanism(model_key: str) -> str | None:
    """``"class_weight"``, ``"sample_weight"`` or None (see ``CLASS_WEIGHT_SUPPORT``)."""
    _check_key(model_key)
    return CLASS_WEIGHT_SUPPORT[model_key]


def model_params_config() -> dict[str, Any]:
    """JSON-safe description of everything here that changes what a trained model is.

    Saved next to trained models (Files 43 and 47), so models trained with other
    parameters, subsample sizes or class weighting are recognised as stale.
    ``n_jobs`` is excluded: it changes the speed, not the model.
    """
    return {
        "params": {
            key: {k: v for k, v in sorted(params.items()) if k not in _HARDWARE_ONLY_PARAMS}
            for key, params in _DEFAULT_PARAMS.items()
        },
        "subsample_sizes": dict(sorted(settings.SUBSAMPLE_SIZES.items())),
        "class_weight_mode": settings.CLASS_WEIGHT_MODE,
        "random_seed": settings.RANDOM_SEED,
    }


# ---------------------------------------------------------------------------
# Consistency (checked when the module is imported)
# ---------------------------------------------------------------------------
def _validate() -> None:
    keys = set(MODEL_KEYS)
    if len(keys) != len(MODEL_KEYS):
        raise ValueError("duplicate model keys")
    for name, mapping in (
        ("DISPLAY_NAMES", DISPLAY_NAMES),
        ("_DEFAULT_PARAMS", _DEFAULT_PARAMS),
        ("CLASS_WEIGHT_SUPPORT", CLASS_WEIGHT_SUPPORT),
    ):
        if set(mapping) != keys:
            raise ValueError(f"{name} does not list exactly the model keys (difference: {sorted(set(mapping) ^ keys)})")
    for name, mapping in (("TUNING_GRIDS", TUNING_GRIDS),):
        if set(mapping) != keys:
            raise ValueError(f"{name} does not list exactly the model keys (difference: {sorted(set(mapping) ^ keys)})")
    for key, grid in TUNING_GRIDS.items():
        if not grid or any(not values for values in grid.values()):
            raise ValueError(f"the tuning grid of {key!r} is empty or has a parameter without values")
    unknown = sorted(set(settings.SUBSAMPLE_SIZES) - keys)
    if unknown:
        raise ValueError(f"settings.SUBSAMPLE_SIZES names unknown models: {unknown}")


_validate()