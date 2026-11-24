"""XGBoost (``xgboost.XGBClassifier``).

Baseline parameters (File 28): library defaults plus ``random_state``,
``n_jobs`` and ``tree_method="hist"``: 100 boosting rounds, ``max_depth=6``,
``learning_rate=0.3``, objective ``multi:softprob`` (one probability per class).

Device (``settings.XGBOOST_DEVICE``):

* ``"cpu"``  : always the CPU;
* ``"cuda"`` : the GPU; an error if no usable GPU exists (you asked for it).
  XGBoost itself only warns and trains on the CPU, so the device really used is
  read back from the fitted booster and compared with the request;
* ``"auto"`` : the GPU when a tiny test fit on it works, otherwise the CPU.
  The result of the detection is remembered for the process and logged once.

The device is chosen when the estimator is built, so it is not part of
``model_params`` (and does not make a trained model stale).  After ``fit`` the
booster is switched to the **CPU for inference**: the data given to ``predict``
lives in ordinary memory, so GPU prediction would only add a device-mismatch
warning and a copy, and a model trained on the GPU must also load on a computer
without one (the Streamlit app).  ``info()`` records the training device.

The model has ``predict_proba``; ``score_kind`` is ``"probability"``.  Class
codes must be ``0..K-1`` with every class present, which ``BaseModel`` enforces.
Class weights (only in the later ``balanced`` experiment) go through
``sample_weight``.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from config import settings
from nids.models.base import BaseModel
from nids.utils.logger import get_logger

logger = get_logger("models.xgboost")

_detected_device: str | None = None
_KNOWN_PARAMS = frozenset(XGBClassifier().get_params()) | {"device"}


def _actual_device(estimator: XGBClassifier) -> str:
    """The device a fitted XGBoost booster really uses.

    XGBoost does not fail when the GPU is missing: it prints a warning and
    silently trains on the CPU.  So the requested device is not proof; the
    booster's own configuration is.
    """
    config = json.loads(estimator.get_booster().save_config())
    return str(config["learner"]["generic_param"]["device"])


def _gpu_works() -> bool:
    """True if XGBoost can train a tiny model on the GPU."""
    try:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 4)).astype("float32")
        y = (X[:, 0] > 0).astype("int64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # XGBoost warns "No visible GPU" before falling back
            tiny = XGBClassifier(n_estimators=2, tree_method="hist", device="cuda", n_jobs=1).fit(X, y)
        return _actual_device(tiny).startswith("cuda")
    except Exception as exc:  # no GPU, no driver, or a CPU-only build: all mean "not usable"
        logger.info("XGBoost GPU not usable (%s: %s)", type(exc).__name__, str(exc).splitlines()[0][:120])
        return False


def resolve_device(setting: str | None = None) -> str:
    """The device to train on, ``"cuda"`` or ``"cpu"``, from ``settings.XGBOOST_DEVICE``."""
    global _detected_device
    setting = settings.XGBOOST_DEVICE if setting is None else setting
    if setting in ("cpu", "cuda"):
        return setting
    if setting != "auto":
        raise ValueError(f"XGBOOST_DEVICE must be one of {settings.ALLOWED_XGBOOST_DEVICES}, got {setting!r}")
    if _detected_device is None:
        _detected_device = "cuda" if _gpu_works() else "cpu"
        logger.info("XGBoost device (auto): %s", _detected_device)
    return _detected_device


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    key = "xgboost"

    def _make_estimator(self, params: dict[str, Any]) -> XGBClassifier:
        unknown = sorted(set(params) - _KNOWN_PARAMS)
        if unknown:  # XGBoost itself would only warn and ignore them
            raise TypeError(f"XGBClassifier got unknown parameter(s): {unknown}")
        # An explicit "device" in params wins over the setting.
        return XGBClassifier(**{"device": resolve_device(), **params})

    def fit(self, X: Any, y: Any, sample_weight: Any = None, n_classes: int | None = None) -> "XGBoostModel":
        super().fit(X, y, sample_weight=sample_weight, n_classes=n_classes)
        requested = str(self.estimator_.get_params().get("device") or "cpu")
        self.train_device_ = _actual_device(self.estimator_)
        if requested.startswith("cuda") and not self.train_device_.startswith("cuda"):
            if settings.XGBOOST_DEVICE == "cuda" or self.params.get("device") == "cuda":
                raise RuntimeError("XGBoost was asked to train on the GPU but fell back to the CPU: no usable GPU")
            logger.warning("XGBoost fell back to the CPU for training")
        if self.train_device_.startswith("cuda"):
            # Inference on the CPU: see the module docstring.
            self.estimator_.set_params(device="cpu")
            self.estimator_.get_booster().set_param({"device": "cpu"})
        return self

    def info(self) -> dict[str, Any]:
        info = super().info()
        if self.is_fitted:
            info["train_device"] = getattr(self, "train_device_", "unknown")
            info["n_rounds"] = int(self.estimator_.get_booster().num_boosted_rounds())
        return info