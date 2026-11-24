"""One place to read the preprocessed data for training and evaluation.

Why a separate module: the trainer (File 46), the cross-validation (49), the
tuning (54) and the evaluation (48-53) must all read exactly the same arrays,
check them the same way, and build the same subsamples.

What it does:

* ``load_training_data()`` / ``load_test_data()`` check that the preprocessed
  data exists and is **fresh** (its settings hash equals the current settings),
  then load the arrays and check that they agree with each other (rows, classes,
  feature count).  Stale or inconsistent data is an error, never a silent
  training on the wrong data.
* ``training_subset(model_key, X, y)`` gives the rows a model trains on.  Models
  without a subsample size (``settings.SUBSAMPLE_SIZES``) get **all** rows,
  without copying.  KNN, SVC and Gradient Boosting get a **class-capped stratified
  subsample** (decision of Checkpoint 3 / File 45, option A): every row of the
  small classes is kept and the large classes are capped, so that every class
  stays present (the models refuse a missing class) and the total reaches the
  target size.  The result is therefore *more balanced than the real data*; the
  comparison table must say that these models trained on such a subsample.
* The subsample is deterministic: same data and seed, same rows.

The test set is only loaded by the evaluation; no function here lets a training
or tuning step touch it by accident (``load_training_data`` never reads it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import model_params, settings
from nids.preprocessing.preprocessor import check_preprocessed_fresh, load_preprocessing_metadata
from nids.preprocessing.target_builder import LabelCodec
from nids.utils.io_utils import load_array, load_joblib
from nids.utils.logger import get_logger

logger = get_logger("training.data_access")


class DataNotReadyError(RuntimeError):
    """The preprocessed data is missing, stale or inconsistent."""


@dataclass(frozen=True)
class TrainingData:
    X: np.ndarray
    y: np.ndarray
    codec: LabelCodec
    n_classes: int
    feature_count: int

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class TestData:
    __test__ = False  # not a pytest class

    X: np.ndarray
    y: np.ndarray
    codec: LabelCodec
    n_classes: int


# ---------------------------------------------------------------------------
# Loading and checks
# ---------------------------------------------------------------------------
def check_data_ready() -> dict[str, Any]:
    """Raise ``DataNotReadyError`` unless the preprocessed data exists and is fresh; return its metadata."""
    fresh, reason = check_preprocessed_fresh()
    if not fresh:
        raise DataNotReadyError(reason)
    for path in (settings.X_TRAIN_FILE, settings.X_TEST_FILE, settings.Y_TRAIN_FILE, settings.Y_TEST_FILE, settings.LABEL_ENCODER_FILE):
        if not path.is_file():
            raise DataNotReadyError(f"missing {path.name}; rerun scripts/04_preprocess.py")
    return load_preprocessing_metadata()


def _check_arrays(X: np.ndarray, y: np.ndarray, codec: LabelCodec, name: str, expected_features: int | None) -> None:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise DataNotReadyError(f"{name}: X has shape {X.shape} and y has shape {y.shape}; they must be 2-D and 1-D with equal rows")
    if not np.issubdtype(y.dtype, np.integer):
        raise DataNotReadyError(f"{name}: y must hold integer class codes, found {y.dtype}")
    if y.size and (y.min() < 0 or y.max() >= codec.n_classes):
        raise DataNotReadyError(f"{name}: class codes outside 0..{codec.n_classes - 1}")
    if expected_features is not None and X.shape[1] != expected_features:
        raise DataNotReadyError(f"{name}: {X.shape[1]} features, the preprocessing metadata says {expected_features}")


def load_training_data(mmap: bool = False) -> TrainingData:
    """The preprocessed training arrays (never the test set).  ``mmap=True`` maps the files instead of reading them."""
    metadata = check_data_ready()
    mode = "r" if mmap else None
    X = load_array(settings.X_TRAIN_FILE, mmap_mode=mode)
    y = load_array(settings.Y_TRAIN_FILE)
    codec: LabelCodec = load_joblib(settings.LABEL_ENCODER_FILE)
    _check_arrays(X, y, codec, "training data", metadata.get("n_model_features"))
    present = np.unique(y)
    if not np.array_equal(present, np.arange(codec.n_classes)):
        raise DataNotReadyError(
            f"the training data must contain every class code 0..{codec.n_classes - 1}; found {present.tolist()}"
        )
    logger.info("Training data: %d rows, %d features, %d classes", X.shape[0], X.shape[1], codec.n_classes)
    return TrainingData(X=X, y=y, codec=codec, n_classes=codec.n_classes, feature_count=int(X.shape[1]))


def load_test_data(mmap: bool = False) -> TestData:
    """The preprocessed test arrays.  Only the evaluation steps (Files 48-53) should call this."""
    metadata = check_data_ready()
    mode = "r" if mmap else None
    X = load_array(settings.X_TEST_FILE, mmap_mode=mode)
    y = load_array(settings.Y_TEST_FILE)
    codec: LabelCodec = load_joblib(settings.LABEL_ENCODER_FILE)
    _check_arrays(X, y, codec, "test data", metadata.get("n_model_features"))
    logger.info("Test data: %d rows, %d features", X.shape[0], X.shape[1])
    return TestData(X=X, y=y, codec=codec, n_classes=codec.n_classes)


# ---------------------------------------------------------------------------
# Class-capped stratified subsample
# ---------------------------------------------------------------------------
def class_cap_for_target(counts: np.ndarray, target: int) -> tuple[int, int]:
    """The per-class cap ``c`` and the number of extra rows, so that the sample has ``target`` rows.

    Every class keeps ``min(count, c)`` rows (a class smaller than the cap is kept whole).  ``c`` is the largest
    value for which the total does not exceed ``target``; the rows still missing (fewer than the number of classes
    that are above the cap) are given one each to the largest classes.  Returns ``(cap, extra)``.
    """
    counts = np.asarray(counts, dtype="int64")
    if target < counts.size:
        raise ValueError(f"target size {target} is smaller than the number of classes ({counts.size})")
    if target >= counts.sum():
        return int(counts.max()), 0
    low, high = 1, int(counts.max())
    while low < high:  # largest cap whose total stays within the target
        mid = (low + high + 1) // 2
        if np.minimum(counts, mid).sum() <= target:
            low = mid
        else:
            high = mid - 1
    cap = low
    extra = int(target - np.minimum(counts, cap).sum())
    return cap, extra


def class_capped_indices(y: np.ndarray, target: int, seed: int | None = None) -> np.ndarray:
    """Sorted row positions of a class-capped stratified subsample of about ``target`` rows (exactly ``target`` if possible)."""
    y = np.asarray(y)
    seed = settings.RANDOM_SEED if seed is None else seed
    classes, counts = np.unique(y, return_counts=True)
    if target >= y.size:
        return np.arange(y.size)
    cap, extra = class_cap_for_target(counts, target)
    # classes above the cap that can take one more row, largest first (ties by code)
    can_grow = [i for i in np.argsort(-counts, kind="stable") if counts[i] > cap]
    bonus = set(can_grow[:extra])
    rng = np.random.default_rng(seed)
    chosen = []
    for i, (cls, count) in enumerate(zip(classes, counts)):
        positions = np.flatnonzero(y == cls)
        take = min(int(count), cap + (1 if i in bonus else 0))
        chosen.append(positions if take >= count else rng.choice(positions, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def training_subset(model_key: str, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The rows ``model_key`` trains on: all of them, or its class-capped subsample.

    Returns ``(X_used, y_used, info)``.  ``info`` is JSON-safe and goes into the saved model's metadata
    (``extra``), so the report can say which models trained on a subsample.
    """
    if model_key not in model_params.MODEL_KEYS:
        raise ValueError(f"unknown model key {model_key!r}. Known: {list(model_params.MODEL_KEYS)}")
    target = model_params.subsample_size(model_key)
    n_classes = int(y.max()) + 1
    full_counts = np.bincount(y, minlength=n_classes)
    if target is None or target >= y.shape[0]:
        info = {
            "subsampled": False,
            "requested_size": target,
            "n_rows": int(y.shape[0]),
            "n_rows_available": int(y.shape[0]),
            "class_counts": full_counts.tolist(),
            "class_counts_available": full_counts.tolist(),
            "cap_per_class": None,
        }
        return X, y, info
    idx = class_capped_indices(y, target)
    X_used, y_used = X[idx], y[idx]
    used_counts = np.bincount(y_used, minlength=n_classes)
    if (used_counts == 0).any():
        raise DataNotReadyError(f"the subsample for {model_key!r} lost a class; this should not happen")
    info = {
        "subsampled": True,
        "requested_size": int(target),
        "n_rows": int(y_used.shape[0]),
        "n_rows_available": int(y.shape[0]),
        "class_counts": used_counts.tolist(),
        "class_counts_available": full_counts.tolist(),
        "cap_per_class": int(class_cap_for_target(full_counts, target)[0]),
        "seed": settings.RANDOM_SEED,
        "method": "class-capped stratified: small classes kept whole, large classes capped",
    }
    logger.info(
        "%s trains on a class-capped subsample: %d of %d rows (cap %d per class)",
        model_key, info["n_rows"], info["n_rows_available"], info["cap_per_class"],
    )
    return X_used, y_used, info