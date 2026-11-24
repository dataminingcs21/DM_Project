"""Registry of the 11 models: one place that knows every model class.

The trainer, the evaluation, the tuning step, the majority vote and the
interface ask this module for a model by its key (``"random_forest"``, ...),
never by importing a model file themselves.

Model modules are imported **lazily**, on the first request for that model.  So
a problem with one model's library (for example ``xgboost`` not installed) makes
only that model unavailable instead of breaking all the others, and
``availability()`` reports it by name: the majority vote and the reports must
say which models are missing, not hide them.

Keys and their order come from ``config.model_params.MODEL_KEYS`` (the order of
the project brief).  A mismatch between this registry and ``MODEL_KEYS``, or
between a class and its registry key, is an error, not a silent difference.
"""

from __future__ import annotations

import importlib
from typing import Any

from config import model_params
from nids.models.base import BaseModel

# key -> (module, class name).  Module files follow the project naming ``<name>_model``.
_REGISTRY: dict[str, tuple[str, str]] = {
    "knn": ("nids.models.knn_model", "KNNModel"),
    "decision_tree": ("nids.models.decision_tree_model", "DecisionTreeModel"),
    "random_forest": ("nids.models.random_forest_model", "RandomForestModel"),
    "gaussian_nb": ("nids.models.gaussian_nb_model", "GaussianNBModel"),
    "svm": ("nids.models.svm_model", "SVMModel"),
    "bernoulli_nb": ("nids.models.bernoulli_nb_model", "BernoulliNBModel"),
    "logistic_regression": ("nids.models.logistic_regression_model", "LogisticRegressionModel"),
    "sgd": ("nids.models.sgd_model", "SGDModel"),
    "adaboost": ("nids.models.adaboost_model", "AdaBoostModel"),
    "gradient_boosting": ("nids.models.gradient_boosting_model", "GradientBoostingModel"),
    "xgboost": ("nids.models.xgboost_model", "XGBoostModel"),
}


class ModelUnavailableError(ImportError):
    """A registered model cannot be imported (missing library or broken file)."""


def _validate() -> None:
    if set(_REGISTRY) != set(model_params.MODEL_KEYS):
        missing = sorted(set(model_params.MODEL_KEYS) - set(_REGISTRY))
        extra = sorted(set(_REGISTRY) - set(model_params.MODEL_KEYS))
        raise ValueError(f"registry and model_params.MODEL_KEYS differ: missing {missing}, extra {extra}")


_validate()


def all_model_keys() -> tuple[str, ...]:
    """The 11 model keys, in the order of the project brief."""
    return tuple(model_params.MODEL_KEYS)


def display_name(key: str) -> str:
    _check_key(key)
    return model_params.DISPLAY_NAMES[key]


def _check_key(key: str) -> None:
    if key not in _REGISTRY:
        raise ValueError(f"unknown model key {key!r}. Known: {list(all_model_keys())}")


def get_model_class(key: str) -> type[BaseModel]:
    """The model class for ``key``.  Raises ``ValueError`` (unknown key) or ``ModelUnavailableError``."""
    _check_key(key)
    module_name, class_name = _REGISTRY[key]
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:  # missing library, empty placeholder file, wrong class name...
        raise ModelUnavailableError(
            f"model {key!r} is unavailable: {type(exc).__name__}: {exc} (expected {class_name} in {module_name})"
        ) from exc
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ModelUnavailableError(f"{module_name}.{class_name} is not a BaseModel subclass")
    if getattr(cls, "key", None) != key:
        raise ModelUnavailableError(f"{class_name}.key is {getattr(cls, 'key', None)!r}, the registry expects {key!r}")
    return cls


def create_model(key: str, params: dict[str, Any] | None = None, class_weight_mode: str | None = None) -> BaseModel:
    """A new unfitted model of the given kind (defaults of File 28 updated with ``params``)."""
    return get_model_class(key)(params=params, class_weight_mode=class_weight_mode)


def availability(keys: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """For every key: ``{"available": bool, "error": str | None, "display_name": str}``."""
    report: dict[str, dict[str, Any]] = {}
    for key in keys or all_model_keys():
        try:
            get_model_class(key)
            report[key] = {"available": True, "error": None, "display_name": display_name(key)}
        except ModelUnavailableError as exc:
            report[key] = {"available": False, "error": str(exc), "display_name": display_name(key)}
    return report


def create_all_models(
    keys: list[str] | tuple[str, ...] | None = None,
    class_weight_mode: str | None = None,
    skip_unavailable: bool = False,
) -> dict[str, BaseModel]:
    """New unfitted models for ``keys`` (default: all 11), keyed by model key, in brief order.

    An unavailable model raises ``ModelUnavailableError`` unless ``skip_unavailable`` is True,
    in which case it is left out (the caller must then report which ones are missing).
    """
    models: dict[str, BaseModel] = {}
    for key in keys or all_model_keys():
        try:
            models[key] = create_model(key, class_weight_mode=class_weight_mode)
        except ModelUnavailableError:
            if not skip_unavailable:
                raise
    return models