# `tests/conftest.py`

**Location:** `DM_Project/tests/conftest.py`
**Roadmap step:** File 09 (Stage 1, test support)

## Purpose

Shared setup for every test. pytest loads this file automatically before the tests; it does not contain tests itself.

## What it does

### 1. Protects your real data

Before any project module is imported, it creates a temporary folder (`nids_tests_...` in your system temp directory) and sets `NIDS_PROJECT_ROOT` to it. `config/settings.py` then points every path (`data/`, `artifacts/`, `reports/`, `logs/`) into that temporary folder, so **tests can never read, overwrite or delete your real data, models or logs**. The folder is deleted at the end of the test session (the log file is closed first, because Windows refuses to delete open files).

### 1b. Gives pytest its own temporary folder (added after a Windows problem)

pytest's `tmp_path` fixture normally uses `<system temp>\pytest-of-<user>`. On some Windows machines that folder exists but is not accessible (created by another account or an elevated shell), and every test that uses `tmp_path` fails with `PermissionError: [WinError 5] Access is denied: '...\Temp\pytest-of-...'`. `conftest.py` now creates a fresh folder (`nids_pytest_tmp_...` in the system temp directory), points pytest to it with the `PYTEST_DEBUG_TEMPROOT` variable, and deletes it when pytest exits. So the tests no longer depend on that locked folder.

### 2. Skips tests that need the real dataset

A test marked `@pytest.mark.real_data` is skipped automatically while `data/preprocessed/preprocessing_metadata.json` does not exist in the real project. Once you have run the pipeline, those tests run.

### 3. Fixtures

| Fixture | What it gives a test |
|---|---|
| `test_project_root` | The temporary root used by the session (it also checks that settings were not imported too early) |
| `real_project_root` | The real project folder, for `real_data` tests only. Never write into it |
| `make_flow_frame(class_sizes=None, seed=None)` | A factory that builds a synthetic labelled DataFrame: columns `Feature A/B/C` (class-dependent so models can learn), `Label`, `source_file`. Default classes: BENIGN 200, DoS 60, PortScan 40. Same seed gives the same data |
| `raw_csv_dir` | A temporary folder with two small CSV files that reproduce the loader's quirks: spaced headers, an `Infinity` value, a non-numeric value, doubled spaces in a label, a different column order, a Windows-1252 file with an en dash, plus a non-CSV file that must be ignored |

**All fixture data is synthetic.** The names `Feature A` etc. are placeholders for tests, not real dataset columns.

## Dependencies

`config/settings.py` (File 04), `numpy`, `pandas`, `pytest`. The `real_data` marker is declared in `pyproject.toml` (File 03).

## How to test

There are no tests to run yet; the first ones arrive with File 10. To check that pytest finds the file:

```powershell
python -m pytest --collect-only -q
```

**Expected output (before File 10 exists):** `no tests collected` with no error. If there is an error, read it: a `ModuleNotFoundError` means `.venv` is not active or `pip install -e .` was not run.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | `pip install -e .` not run | Run it from `DM_Project\` with `.venv` active |
| `AssertionError: settings were imported before the test root was set` | Something imported `config.settings` before pytest loaded this file (for example a `-p` plugin or a script run from `tests/`) | Run pytest from `DM_Project\` with `python -m pytest` |
| `AssertionError ... C:/Users/NAME~1/... != C:/Users/LONG NAME/...` (fixed in this version) | On Windows the temp folder can have a short 8.3 name (`NAME~1`) while settings uses the long name; paths are now resolved before comparing | Use the current `conftest.py` |
| Tests touch real folders | `NIDS_PROJECT_ROOT` was overridden in your shell | Not expected; this file sets it itself. Check with `echo $env:NIDS_PROJECT_ROOT` |
| `PermissionError: [WinError 5] Access is denied: '...\Temp\pytest-of-NAME'` (fixed in this version) | An inaccessible pytest temp folder from an earlier run or another account | Use the current `conftest.py`. Optionally delete the old folder: `Remove-Item -Recurse -Force "$env:TEMP\pytest-of-*"` |
| `PytestCacheWarning: cache could not write path ...\.pytest_cache` | The `.pytest_cache` folder in the project is not writable (for example copied from another computer) | Only a warning. To remove it: `Remove-Item -Recurse -Force .pytest_cache`, or run `python -m pytest tests -q -p no:cacheprovider` |
| `PytestUnknownMarkWarning: real_data` | `pyproject.toml` marker section missing | Re-check File 03 |
| A leftover `nids_tests_*` folder in the temp directory | A test run was killed before cleanup | Delete it by hand; harmless |

## Decisions and limitations

- **Temporary root instead of mocking paths** in each test: one rule covers all tests, including future ones.
- **`real_data` tests use `real_project_root`,** because `settings` points to the temporary root during tests.
- **Synthetic data only.** Real-data behaviour is checked by the pipeline scripts and the `real_data` tests, not by invented columns.