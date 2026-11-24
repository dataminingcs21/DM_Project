# `tests/test_model_store.py`

**Location:** `DM_Project/tests/test_model_store.py`
**Roadmap step:** File 44 (Stage 5, models)

## Purpose

Automatic tests for the model store (File 43): saving, loading, detection of damaged or interrupted saves, staleness and library-version checks.

## Why it is needed

A model store that fails silently is worse than none: the app would answer with a model that no longer matches the pipeline, or with a half-written file. These tests prove that every such case is refused with a clear reason, and that the normal case works for all 11 models.

## Dependencies

`pytest`, `numpy`, `joblib`, `config.settings`, `config.model_params`, `nids.models.registry`, `nids.persistence.model_store`. `tests/conftest.py` redirects the project root to a temporary folder, so the two tests that use the default folders never touch your real `artifacts\`.

## Data

Synthetic only: 300 rows, 5 features, 4 classes. Most tests use a small Decision Tree and pytest's temporary folders.

## What is tested (36 tests)

**Save and load (16):** a round trip for each of the 11 models (same predictions and scores after loading); metadata content (key, stage, feature count, `extra`, checksum, both hashes, library versions, the model's `info()`) and that the file on disk equals what was returned; the default `trained` and `tuned` folders; overwriting replaces the old model and its metadata; bad stage, unknown key, unfitted model and non-model object are refused; loading a model that was never saved gives a clear error.

**Damage and interruption (8):** a replaced model file and a truncated file are caught by the checksum (`verify=False` skips the check on request); a file holding another kind of model, or not a model at all, is refused; metadata without a model file; a model file without metadata (interrupted save); a save that fails while writing the metadata leaves **no old metadata** beside the new model file; a corrupt metadata file is reported by `list_saved` instead of crashing.

**Freshness and versions (9):** a fresh model loads; changing a preprocessing setting, a subsample size or the class-weight mode makes it stale (`StaleModelError` with the reason); changing the thread count does **not** (hardware only); `allow_stale=True` loads anyway; a different scikit-learn version warns, or fails with `strict_versions=True`; the same versions give no warning.

**Listing and deleting (3):** `list_saved` covers the 11 keys in order; `delete_model` removes both files and returns False the second time; the metadata is plain JSON.

## Exact command

PowerShell, from `DM_Project\`, venv active:

```powershell
python -m pytest tests -q -p no:cacheprovider
```

To run only this file: `python -m pytest tests\test_model_store.py -q -p no:cacheprovider`.

## Expected output

For this file alone: `36 passed` in a few seconds. For the whole `tests` folder: **`349 passed, 1 skipped`** (the earlier 313 passed and 1 skipped, plus these 36). The skipped test is the XGBoost no-GPU one, skipped because your GPU works. Two scikit-learn convergence warnings are expected, as before.

*Note:* in my sandbox the whole-folder run also showed a few failures in `test_loader.py` and `test_preprocessing.py`, caused only by the sandbox's different pandas version and the missing `pyarrow` package. Your venv has both, and those tests passed on your machine. This file's 36 tests pass in the sandbox.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... model_store` | placeholder empty | paste the code, check the length is not 0 |
| `PermissionError` on a temporary folder | Windows temp folder locked | `tests/conftest.py` already uses a fresh folder; close programs holding it, rerun |
| a test fails after you changed `settings.py` | a setting used by the tests was edited | restore it, or tell me which one |

## Decisions and limitations

- The tests change settings only through `monkeypatch`, so nothing leaks between tests.
- Behaviour on the real data (sizes of the saved Random Forest, time to checksum it) is measured at File 47.