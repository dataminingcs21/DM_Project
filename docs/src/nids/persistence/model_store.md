# `src/nids/persistence/model_store.py`

**Location:** `DM_Project/src/nids/persistence/model_store.py`
**Roadmap step:** File 43 (Stage 5, models)

## Purpose

Saves a fitted model to disk together with a metadata file, and loads it back after checking that it is complete, undamaged and still valid.

## Why it is needed

Training the 11 models takes a long time, tuning takes longer, and the Streamlit app must load the final models without retraining. A model file alone says nothing about what it was trained on. If the preprocessing or the model settings change later, a forgotten old model would silently give answers that no longer match the pipeline. This file makes those cases visible.

## Dependencies

`config/settings.py` (folders, preprocessing settings), `config/model_params.py` (the model settings fingerprint), `nids.models.base`, `nids.utils.io_utils` (atomic writes, hash), `nids.utils.logger`, `joblib`.

## Layout

```
artifacts/models/trained/<key>.joblib    the fitted model (baseline, File 47)
artifacts/models/trained/<key>.json      its metadata
artifacts/models/tuned/<key>.joblib      the tuned model (File 55)
artifacts/models/tuned/<key>.json
```

## Functions

| Function | What it does |
|---|---|
| `save_model(model, stage="trained", extra=None, compress=0, directory=None)` | saves a fitted model and its metadata; returns the metadata. `extra` is a free JSON-safe dictionary (the trainer puts the subsample size there, the tuning step its search results) |
| `load_model(key, stage="trained", allow_stale=False, verify=True, strict_versions=False)` | checks metadata, checksum and freshness, then loads; returns the model |
| `load_metadata(key, stage)` | the metadata only (cheap; no model loading) |
| `check_fresh(metadata)` | list of reasons why a saved model is stale (empty = valid) |
| `list_saved(stage)` | for each of the 11 keys: saved or not, fresh or not, why not |
| `delete_model(key, stage)` | removes a model and its metadata |
| `current_hashes()` | the two fingerprints used for freshness |
| `StaleModelError`, `ModelStoreError` | the errors (the first is a kind of the second) |

## What the metadata holds

Key and display name, stage, save time, file name and size, a **SHA-256 checksum** of the model file, the number of input features, the **hash of the preprocessing settings** and the **hash of the model settings** at save time, the versions of Python, scikit-learn, XGBoost, NumPy and joblib, the model's own `info()` (parameters, training rows, fit seconds, device, convergence...) and `extra`.

## Rules it enforces

- **The metadata is written last.** A model file without metadata is treated as an interrupted save. An old metadata file is deleted before a new model is written, so an interruption cannot leave old metadata beside a new model.
- **Checksum:** if the model file does not match its checksum (damaged or replaced), loading is refused. `verify=False` skips the check (it reads the whole file once, a few seconds for a large model).
- **Staleness:** if the preprocessing settings or the model settings (parameters, subsample sizes, class-weight mode, seed) differ from those at save time, `StaleModelError` is raised, naming the reason. `allow_stale=True` loads anyway and logs a warning (useful to inspect an old model, not to use it).
- **Versions:** a different scikit-learn or XGBoost version than at save time gives a warning (`strict_versions=True` makes it an error), because pickled models can load wrongly across versions.
- **Fitted models only:** saving an unfitted model is an error; loading checks that the file holds a fitted model of the right key.
- **Security note:** joblib files are Python pickles and can run code when loaded. Only load files created by this project.

## How it was checked (sandbox)

A Random Forest was saved and loaded: identical predictions, metadata complete, `check_fresh` empty. Then: a damaged file (last bytes changed) is refused by the checksum; a changed model-settings hash gives `StaleModelError` and loads only with `allow_stale=True`; a changed library version gives a warning, or an error with `strict_versions=True`; saving an unfitted model and loading a missing one both raise `ModelStoreError`; with the metadata removed `list_saved` reports "interrupted save"; `delete_model` removes both files (second call returns False); a bad stage name is rejected.

## How to test on your real data

The command trains a small Decision Tree on a class-balanced slice, saves it into a **temporary folder** (your real `artifacts\models` is not touched), loads it back, and compares predictions. PowerShell, from `DM_Project\`, venv active:

```powershell
python -c "import tempfile, numpy as np; from nids.models.decision_tree_model import DecisionTreeModel; from nids.persistence import model_store as ms; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = DecisionTreeModel().fit(X[idx], y[idx], n_classes=15); d = tempfile.mkdtemp(); meta = ms.save_model(m, directory=d, extra={'note': 'storage check'}); l = ms.load_model('decision_tree', directory=d); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; print('same predictions:', bool((l.predict(Xe) == m.predict(Xe)).all())); print('fresh problems:', ms.check_fresh(meta)); print(meta['file_size_mb'], 'MB', meta['n_features_in'], 'features', meta['libraries']['scikit-learn']); print('saved:', [k for k, v in ms.list_saved(directory=d).items() if v['saved']])"
```

**Expected output:** a log line `Decision Tree fitted on 21315 rows ...`, a log line `Saved Decision Tree (trained) to ...`, then `same predictions: True`, then `fresh problems: []`, then the file size in MB (small, below 1), `40 features` and your scikit-learn version, then `saved: ['decision_tree']`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... model_store` | code not pasted into the placeholder | paste into `src\nids\persistence\model_store.py`, check the length is not 0 |
| `ModelStoreError: ... missing <key>.json` | no model saved yet | run the training script (File 47) |
| `StaleModelError` | preprocessing or model settings changed after training | rerun the pipeline and retrain; do not use `allow_stale` for results |
| `ModelStoreError: ... checksum` | the model file was damaged or replaced | retrain that model |
| `RuntimeWarning: ... other library versions` | the venv was upgraded after training | retrain, or accept the warning knowingly |

## Decisions and limitations

- **Settings fingerprints, not data fingerprints:** a model is judged stale when settings change; rerunning preprocessing with the same settings gives the same data (fixed seed), so it does not make models stale.
- **No compression by default:** faster to write and load; `compress` is available if the Random Forest turns out large (its real size is measured at File 47).
- **Tests:** `tests/test_model_store.py` (File 44).