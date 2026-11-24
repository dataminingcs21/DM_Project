"""Saving and loading trained models, each with a metadata file.

Layout (``settings.TRAINED_MODELS_DIR`` for baseline models, ``settings.TUNED_MODELS_DIR``
for tuned models)::

    artifacts/models/trained/<key>.joblib    the fitted model (a BaseModel object)
    artifacts/models/trained/<key>.json      metadata: what it is, what it was trained on, a checksum

Rules:

* **The metadata file is written last.**  A model file without metadata, or
  metadata whose checksum does not match the model file, is an interrupted or
  damaged save and is refused.  An old metadata file is deleted before a new
  model is written, so an interrupted save can never leave old metadata beside a
  new model.
* **Staleness is checked on load.**  The metadata stores the hash of the
  preprocessing settings and the hash of the model settings (``model_params_config``)
  at save time.  A model trained with other settings is refused
  (``StaleModelError``) unless ``allow_stale=True``.
* **Library versions are recorded.**  A model pickled with another scikit-learn
  or XGBoost version may load wrongly; a different version gives a warning
  (``strict_versions=True`` makes it an error).
* **Only load files made by this project.**  joblib files are Python pickles and can
  run code when loaded; never load one received from someone else.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from config import model_params, settings
from nids.models.base import BaseModel
from nids.utils.io_utils import config_hash, dump_joblib, load_joblib, read_json, write_json
from nids.utils.logger import get_logger

logger = get_logger("persistence.model_store")

STAGES = ("trained", "tuned")
_TRACKED_LIBRARIES = ("scikit-learn", "xgboost", "numpy", "joblib")


class ModelStoreError(Exception):
    """A saved model is missing, incomplete or damaged."""


class StaleModelError(ModelStoreError):
    """A saved model was trained with settings that have since changed."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def stage_dir(stage: str = "trained") -> Path:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return settings.TRAINED_MODELS_DIR if stage == "trained" else settings.TUNED_MODELS_DIR


def _check_key(key: str) -> None:
    if key not in model_params.MODEL_KEYS:
        raise ValueError(f"unknown model key {key!r}. Known: {list(model_params.MODEL_KEYS)}")


def model_file(key: str, stage: str = "trained", directory: Path | str | None = None) -> Path:
    _check_key(key)
    return (Path(directory) if directory else stage_dir(stage)) / f"{key}.joblib"


def metadata_file(key: str, stage: str = "trained", directory: Path | str | None = None) -> Path:
    _check_key(key)
    return (Path(directory) if directory else stage_dir(stage)) / f"{key}.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def current_hashes() -> dict[str, str]:
    """Fingerprints of the settings that decide whether a saved model is still valid."""
    return {
        "preprocessing_hash": config_hash(settings.preprocessing_config()),
        "model_params_hash": config_hash(model_params.model_params_config()),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_model(
    model: BaseModel,
    stage: str = "trained",
    extra: dict[str, Any] | None = None,
    compress: int = 0,
    directory: Path | str | None = None,
) -> dict[str, Any]:
    """Save a fitted model and its metadata; return the metadata.

    ``extra`` is stored as is (JSON-safe values) under ``"extra"``: the trainer puts the
    subsample size and the tuning step the search results there.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(f"save_model expects a BaseModel, got {type(model).__name__}")
    if not model.is_fitted:
        raise ModelStoreError(f"{model.display_name} is not fitted; nothing to save")
    target = model_file(model.key, stage, directory)
    meta_target = metadata_file(model.key, stage, directory)
    target.parent.mkdir(parents=True, exist_ok=True)

    if meta_target.exists():
        meta_target.unlink()  # never leave old metadata next to a model that is being replaced
    dump_joblib(model, target, compress=compress)

    metadata: dict[str, Any] = {
        "key": model.key,
        "display_name": model.display_name,
        "stage": stage,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": target.name,
        "file_size_mb": round(target.stat().st_size / 1_048_576, 2),
        "sha256": _sha256(target),
        "n_features_in": getattr(model.estimator_, "n_features_in_", None),
        **current_hashes(),
        "libraries": _library_versions(),
        "model_info": model.info(),
        "extra": extra or {},
    }
    write_json(metadata, meta_target)  # written last: it marks the save as complete
    logger.info("Saved %s (%s) to %s (%.1f MB)", model.display_name, stage, target, metadata["file_size_mb"])
    return metadata


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------
def load_metadata(key: str, stage: str = "trained", directory: Path | str | None = None) -> dict[str, Any]:
    path = metadata_file(key, stage, directory)
    if not path.is_file():
        raise ModelStoreError(
            f"no saved {stage} model for {key!r} (missing {path.name}); run the training or tuning script"
        )
    return read_json(path)


def check_fresh(metadata: dict[str, Any]) -> list[str]:
    """Reasons why a saved model is stale (empty list = still valid)."""
    now = current_hashes()
    problems = []
    if metadata.get("preprocessing_hash") != now["preprocessing_hash"]:
        problems.append("the preprocessing settings changed since this model was trained (rerun 04_preprocess.py and the training)")
    if metadata.get("model_params_hash") != now["model_params_hash"]:
        problems.append("the model settings (parameters, subsample sizes, class weights) changed since this model was trained")
    return problems


def list_saved(stage: str = "trained", directory: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """For each of the 11 keys: whether a model is saved, whether it is fresh, and why not."""
    report: dict[str, dict[str, Any]] = {}
    for key in model_params.MODEL_KEYS:
        meta_path = metadata_file(key, stage, directory)
        file_path = model_file(key, stage, directory)
        if not meta_path.is_file():
            note = "model file without metadata (interrupted save)" if file_path.is_file() else "not saved"
            report[key] = {"saved": False, "fresh": False, "problems": [note]}
            continue
        try:
            meta = read_json(meta_path)
        except ValueError as exc:
            report[key] = {"saved": False, "fresh": False, "problems": [str(exc)]}
            continue
        problems = check_fresh(meta)
        if not file_path.is_file():
            problems.insert(0, "metadata without model file")
        report[key] = {
            "saved": file_path.is_file(),
            "fresh": not problems,
            "problems": problems,
            "saved_at": meta.get("saved_at"),
            "file_size_mb": meta.get("file_size_mb"),
        }
    return report


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_model(
    key: str,
    stage: str = "trained",
    directory: Path | str | None = None,
    allow_stale: bool = False,
    verify: bool = True,
    strict_versions: bool = False,
) -> BaseModel:
    """Load a saved model after checking its metadata, checksum and freshness."""
    metadata = load_metadata(key, stage, directory)
    path = model_file(key, stage, directory)
    if not path.is_file():
        raise ModelStoreError(f"metadata exists but the model file {path.name} is missing; retrain {key!r}")
    if verify and _sha256(path) != metadata.get("sha256"):
        raise ModelStoreError(f"{path.name} does not match its checksum: the file is damaged or was replaced; retrain {key!r}")

    problems = check_fresh(metadata)
    if problems and not allow_stale:
        raise StaleModelError(f"saved {stage} model {key!r} is stale: " + "; ".join(problems))
    if problems:
        logger.warning("Loading a stale %s model %r (allow_stale=True): %s", stage, key, "; ".join(problems))

    saved_versions = metadata.get("libraries", {})
    now_versions = _library_versions()
    changed = {
        name: (saved_versions.get(name), now_versions.get(name))
        for name in _TRACKED_LIBRARIES
        if saved_versions.get(name) != now_versions.get(name)
    }
    if changed:
        message = f"{key!r} was saved with other library versions (saved, current): {changed}"
        if strict_versions:
            raise ModelStoreError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        logger.warning(message)

    model = load_joblib(path)
    if not isinstance(model, BaseModel) or model.key != key or not model.is_fitted:
        raise ModelStoreError(f"{path.name} does not contain a fitted {key!r} model")
    return model


def delete_model(key: str, stage: str = "trained", directory: Path | str | None = None) -> bool:
    """Remove a saved model and its metadata (metadata first). True if anything was removed."""
    removed = False
    for path in (metadata_file(key, stage, directory), model_file(key, stage, directory)):
        if path.exists():
            path.unlink()
            removed = True
    return removed