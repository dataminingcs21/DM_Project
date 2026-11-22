"""Safe file input/output helpers shared by the whole project.

Why this module exists
----------------------
* Every file the project writes (Parquet, JSON, NumPy arrays, joblib models) is
  written **atomically**: the data goes to a temporary ``<name>.tmp`` file that
  replaces the final file only when the write succeeded.  A crash in the middle
  of a write therefore never leaves a half-written file that a later stage
  would mistake for valid data (important for ``preprocessing_metadata.json``
  and the model files).
* Reading a missing file raises a ``FileNotFoundError`` that says which file is
  missing and, when a ``hint`` is given, which script creates it.
* JSON output is always standard JSON: NumPy numbers and arrays are converted
  and NaN / Infinity become ``null``.

Security note: ``load_joblib`` unpickles data, which can execute code.  Only
load files that this project created.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nids.utils.logger import get_logger

logger = get_logger(__name__)

PathLike = str | Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@contextmanager
def _atomic_write(path: Path) -> Iterator[Path]:
    """Yield a temporary path; move it onto ``path`` only if the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():  # still there only when the write failed
            tmp_path.unlink()


def _clean_for_json(obj: Any) -> Any:
    """Convert ``obj`` into plain Python types that ``json`` can write."""
    if obj is None or isinstance(obj, (str, bool, np.bool_)):
        return bool(obj) if isinstance(obj, np.bool_) else obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_clean_for_json(v) for v in obj), key=str)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Directories and existence checks
# ---------------------------------------------------------------------------
def ensure_dir(path: PathLike) -> Path:
    """Create the directory (and parents) if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_file(path: PathLike, hint: str | None = None) -> Path:
    """Return ``path`` if it is an existing file, otherwise raise a clear error."""
    file_path = Path(path)
    if not file_path.is_file():
        message = f"File not found: {file_path}"
        if hint:
            message += f". {hint}"
        raise FileNotFoundError(message)
    return file_path


# ---------------------------------------------------------------------------
# Parquet (cleaned data, sample flows)
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: PathLike) -> Path:
    """Write a DataFrame to Parquet (no index, snappy compression, atomic)."""
    target = Path(path)
    with _atomic_write(target) as tmp:
        df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    logger.debug("Wrote %s rows x %s columns to %s (%.1f MB)", len(df), df.shape[1], target, _size_mb(target))
    return target


def read_parquet(path: PathLike, columns: list[str] | None = None, hint: str | None = None) -> pd.DataFrame:
    """Read a Parquet file, optionally only some columns (saves memory)."""
    source = require_file(path, hint)
    return pd.read_parquet(source, columns=columns, engine="pyarrow")


# ---------------------------------------------------------------------------
# JSON (reports, schema, metadata)
# ---------------------------------------------------------------------------
def write_json(data: Any, path: PathLike, indent: int = 2) -> Path:
    """Write ``data`` as standard JSON (UTF-8, atomic).

    NumPy values are converted, NaN / Infinity are written as ``null``, sets are
    written as sorted lists and non-string dictionary keys become strings.
    """
    target = Path(path)
    text = json.dumps(_clean_for_json(data), indent=indent, ensure_ascii=False, allow_nan=False)
    with _atomic_write(target) as tmp:
        tmp.write_text(text, encoding="utf-8")
    logger.debug("Wrote JSON to %s", target)
    return target


def read_json(path: PathLike, hint: str | None = None) -> Any:
    """Read a UTF-8 JSON file."""
    source = require_file(path, hint)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {source}: {error}") from error


# ---------------------------------------------------------------------------
# NumPy arrays (X_train, y_train, ...)
# ---------------------------------------------------------------------------
def save_array(array: np.ndarray, path: PathLike) -> Path:
    """Save an array as ``.npy`` (no pickling, atomic)."""
    target = Path(path)
    with _atomic_write(target) as tmp:
        with open(tmp, "wb") as handle:  # a file handle stops NumPy from adding ".npy"
            np.save(handle, np.asarray(array), allow_pickle=False)
    logger.debug("Saved array %s %s to %s (%.1f MB)", np.asarray(array).shape, np.asarray(array).dtype, target, _size_mb(target))
    return target


def load_array(path: PathLike, mmap_mode: str | None = None, hint: str | None = None) -> np.ndarray:
    """Load a ``.npy`` array.

    ``mmap_mode="r"`` maps the file instead of reading it into RAM.  On Windows a
    mapped file cannot be overwritten until the array is released.
    """
    source = require_file(path, hint)
    return np.load(source, mmap_mode=mmap_mode, allow_pickle=False)


# ---------------------------------------------------------------------------
# joblib (fitted transformers, trained models)
# ---------------------------------------------------------------------------
def dump_joblib(obj: Any, path: PathLike, compress: int = 0) -> Path:
    """Save any picklable object with joblib (atomic).  ``compress`` is 0-9 (0 = fastest)."""
    target = Path(path)
    with _atomic_write(target) as tmp:
        joblib.dump(obj, tmp, compress=compress)
    logger.debug("Saved %s to %s (%.1f MB)", type(obj).__name__, target, _size_mb(target))
    return target


def load_joblib(path: PathLike, hint: str | None = None) -> Any:
    """Load a joblib file.  Only load files created by this project (see the module note)."""
    source = require_file(path, hint)
    return joblib.load(source)


# ---------------------------------------------------------------------------
# Configuration fingerprint
# ---------------------------------------------------------------------------
def config_hash(config: dict[str, Any], length: int = 16) -> str:
    """Short SHA-256 fingerprint of a configuration dictionary.

    The hash does not depend on the order of the keys, so the same settings
    always give the same value.  Used to detect stale preprocessed data.
    """
    payload = json.dumps(
        _clean_for_json(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]