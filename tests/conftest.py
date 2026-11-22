"""Shared pytest configuration and fixtures.

Two important things happen here:

1. **The project root is redirected to a temporary folder before any project
   module is imported**, so tests can never read or overwrite the real
   ``data/``, ``artifacts/`` or ``logs/`` folders.  ``config.settings`` reads
   ``NIDS_PROJECT_ROOT`` at import time, which is why this code runs at the top
   of this file.
2. Tests marked ``@pytest.mark.real_data`` need the real CIC-IDS2017 artifacts
   and are skipped automatically when those are absent.

All fixture data is **synthetic**: the column names ("Feature A", ...) are
placeholders chosen for tests, not real dataset columns.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

# The real project root, remembered before it is redirected.
REAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Must happen before the first import of config.settings / nids.*
# .resolve() matters on Windows: the temp folder can be given with a short 8.3 name
# (C:\Users\NAME~1\...) while settings.py resolves to the long name.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="nids_tests_")).resolve()
os.environ["NIDS_PROJECT_ROOT"] = str(_TEST_ROOT)

# pytest's own temporary folders (the ``tmp_path`` fixture) normally live in
# <system temp>/pytest-of-<user>.  On some Windows machines that folder exists but
# cannot be read (created by another account or an elevated shell), and every
# test using ``tmp_path`` then fails with "Access is denied".  Giving pytest a
# fresh folder that this process has just created avoids the problem.
_PYTEST_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="nids_pytest_tmp_")).resolve()
os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_PYTEST_TEMP_ROOT)
atexit.register(shutil.rmtree, _PYTEST_TEMP_ROOT, True)  # removed when pytest exits (errors ignored)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from config import settings  # noqa: E402

SYNTHETIC_FEATURES = ("Feature A", "Feature B", "Feature C")
DEFAULT_CLASS_SIZES = {"BENIGN": 200, "DoS": 60, "PortScan": 40}


# ---------------------------------------------------------------------------
# Marker handling
# ---------------------------------------------------------------------------
def _real_artifacts_available() -> bool:
    """True when the real preprocessing metadata exists in the real project."""
    real_metadata = REAL_PROJECT_ROOT / "data" / "preprocessed" / settings.PREPROCESSING_METADATA_FILE.name
    return real_metadata.is_file()


def pytest_sessionfinish(session, exitstatus):
    """Delete the temporary project root (close the log file first: Windows locks open files)."""
    import logging
    import shutil

    nids_logger = logging.getLogger("nids")
    for handler in list(nids_logger.handlers):
        nids_logger.removeHandler(handler)
        handler.close()
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


def pytest_collection_modifyitems(config, items):
    if _real_artifacts_available():
        return
    skip = pytest.mark.skip(reason="real CIC-IDS2017 artifacts not found (run the pipeline scripts first)")
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_project_root() -> Path:
    """The temporary project root used by every test in this session."""
    assert settings.PROJECT_ROOT.resolve() == _TEST_ROOT.resolve(), (
        "settings were imported before the test root was set"
    )
    return _TEST_ROOT


@pytest.fixture(scope="session")
def real_project_root() -> Path:
    """The real project folder (for ``real_data`` tests only; never write into it)."""
    return REAL_PROJECT_ROOT


@pytest.fixture
def make_flow_frame():
    """Factory: build a small synthetic labelled DataFrame.

    ``make_flow_frame(class_sizes=None, seed=None)`` returns columns
    ``Feature A/B/C`` (class-dependent, so models can learn), ``Label`` and
    ``source_file``.  Deterministic for a given seed.
    """

    def _make(class_sizes: dict[str, int] | None = None, seed: int | None = None) -> pd.DataFrame:
        sizes = class_sizes or DEFAULT_CLASS_SIZES
        rng = np.random.default_rng(settings.RANDOM_SEED if seed is None else seed)
        frames = []
        for index, (label, size) in enumerate(sizes.items()):
            data = rng.normal(loc=index * 3.0, scale=1.0, size=(size, len(SYNTHETIC_FEATURES)))
            frame = pd.DataFrame(data, columns=list(SYNTHETIC_FEATURES))
            frame[settings.LABEL_COLUMN] = label
            frame[settings.SOURCE_FILE_COLUMN] = f"file_{index % 2}.csv"
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    return _make


@pytest.fixture
def raw_csv_dir(tmp_path: Path) -> Path:
    """A folder with two small synthetic CSV files that mimic known quirks.

    * ``a.csv`` (UTF-8): header with leading spaces, an ``Infinity`` text value,
      a non-numeric value, a label with doubled spaces.
    * ``b.csv`` (Windows-1252): different column order and an en dash in a label.
    * ``notes.complete``: a non-CSV file that must be ignored.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text(
        " Feature A, Feature B, Label\n1,Infinity,BENIGN\n2,3.5, DoS  Hulk\n3,bad,BENIGN\n",
        encoding="utf-8",
    )
    (raw / "b.csv").write_bytes(
        " Label, Feature B, Feature A\nWeb Attack \x96 XSS,1,5\nBENIGN,2,6\n".encode("latin-1")
    )
    (raw / "notes.complete").write_text("ignored")
    return raw