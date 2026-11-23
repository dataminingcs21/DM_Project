"""Load the raw CIC-IDS2017 CSV files into one DataFrame.

What the loader does (and nothing more):

* finds the CSV files in the raw folder (other files are ignored);
* reads each file, trying several text encodings;
* normalizes column names and label whitespace (see ``schema.py``);
* adds the ``source_file`` metadata column;
* converts feature columns to numbers, counting values that cannot be converted;
* checks that every file has the same columns as the first one, and stops with
  a clear error if not (a silent concat would create all-NaN columns);
* returns the combined data and a report.

It does **not** clean (infinite values, missing values, duplicates): that is
the job of ``cleaning/cleaner.py`` after the data has been inspected.
Values stay in ``float64`` here; the cast to ``float32`` happens later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from nids.data import schema
from nids.utils.logger import get_logger

logger = get_logger(__name__)

#: Encodings tried in order. utf-8 first, then Windows-1252, then latin-1 (never fails).
ENCODINGS: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")


def list_raw_files(raw_dir: str | Path | None = None) -> list[Path]:
    """Sorted list of the CSV files in ``raw_dir`` (default ``settings.RAW_DATA_DIR``)."""
    directory = Path(raw_dir) if raw_dir is not None else settings.RAW_DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Raw data folder not found: {directory}. Create it and copy the CSV files into it."
        )
    files = sorted(p for p in directory.glob(settings.RAW_FILE_PATTERN) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching '{settings.RAW_FILE_PATTERN}' in {directory}. "
            "Copy the CIC-IDS2017 CSV files into this folder."
        )
    return files


def read_csv_file(path: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one CSV file and return ``(DataFrame, file_report)``.

    The frame has normalized column names, cleaned label whitespace, numeric
    feature columns and the ``source_file`` column.
    """
    file_path = Path(path)
    df: pd.DataFrame | None = None
    used_encoding = ""
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, nrows=nrows)
            used_encoding = encoding
            break
        except UnicodeDecodeError as error:
            last_error = error
    if df is None:
        raise ValueError(f"Could not decode {file_path.name} with {ENCODINGS}: {last_error}")

    df.columns = schema.normalize_columns(df.columns)
    duplicates = schema.find_duplicate_columns(df.columns)
    if duplicates:
        raise ValueError(f"{file_path.name}: duplicate column names after normalization: {duplicates}")
    if not schema.has_label(df.columns):
        raise ValueError(
            f"{file_path.name}: label column '{settings.LABEL_COLUMN}' not found. "
            f"Columns start with: {list(df.columns[:5])}"
        )

    df[settings.LABEL_COLUMN] = df[settings.LABEL_COLUMN].map(schema.normalize_label_value)
    features = schema.get_feature_columns(df.columns)
    df, coerced = schema.coerce_numeric(df, features)
    df[settings.SOURCE_FILE_COLUMN] = file_path.name

    report = {
        "file": file_path.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "encoding": used_encoding,
        "non_numeric_values_coerced": coerced,
    }
    return df, report


def load_raw_data(
    raw_dir: str | Path | None = None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load every CSV in ``raw_dir`` into one DataFrame.

    Parameters
    ----------
    raw_dir:
        Folder with the CSV files (default ``settings.RAW_DATA_DIR``).
    nrows:
        Read only the first ``nrows`` rows of each file (quick tests).

    Returns
    -------
    ``(df, report)``.  ``report`` has the number of files and rows, the feature
    count and one entry per file.  Raises ``ValueError`` if a file's columns
    differ from the first file's.
    """
    files = list_raw_files(raw_dir)
    logger.info("Loading %d CSV file(s)", len(files))

    frames: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []
    reference: list[str] | None = None

    for path in files:
        df, report = read_csv_file(path, nrows=nrows)
        if reference is None:
            reference = list(df.columns)
        else:
            comparison = schema.compare_columns(df.columns, reference)
            if comparison.missing or comparison.extra:
                raise ValueError(
                    f"{path.name}: columns differ from {files[0].name}. "
                    f"Missing: {comparison.missing}; extra: {comparison.extra}"
                )
            df = df[reference]  # same order for every file
        frames.append(df)
        file_reports.append(report)
        logger.info("  %s: %d rows, encoding %s", report["file"], report["rows"], report["encoding"])

    combined = pd.concat(frames, ignore_index=True)
    summary = {
        "n_files": len(files),
        "n_rows": int(len(combined)),
        "n_feature_columns": len(schema.get_feature_columns(combined.columns)),
        "files": file_reports,
    }
    logger.info("Loaded %d rows, %d feature columns", summary["n_rows"], summary["n_feature_columns"])
    return combined, summary