# `src/nids/models/xgboost_model.py`

**Location:** `DM_Project/src/nids/models/xgboost_model.py`
**Roadmap step:** File 40 (Stage 5, model 11 of 11)

## Purpose

The XGBoost model: a subclass of `BaseModel` (File 29) that builds an `xgboost.XGBClassifier`, chooses the CPU or the GPU for training, and reports which one was really used.

## Why it is needed

It is the 11th mandatory model: a fast, regularized gradient boosting implementation, usually among the strongest on tabular data, and the only one of the 11 that can use your RTX 3060.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/settings.py` (`XGBOOST_DEVICE`), `config/model_params.py` (File 28), `nids.utils.logger`, `xgboost` 3.1.3, `numpy`.

## Baseline parameters (File 28)

`{"random_state": 42, "n_jobs": 14, "tree_method": "hist"}`; otherwise library defaults: 100 rounds, `max_depth=6`, `learning_rate=0.3`, objective `multi:softprob` (one probability per class). Subsample: none (all training rows). Class weights (only in the later `balanced` experiment): balanced `sample_weight` passed to `fit`.

## Device logic (`settings.XGBOOST_DEVICE`)

| Setting | Behaviour |
|---|---|
| `"cpu"` | always the CPU |
| `"cuda"` | the GPU; **`RuntimeError`** if no usable GPU (you asked for it) |
| `"auto"` (default) | GPU if a tiny test fit really runs on it, otherwise CPU; detected once per process and logged |

**Important finding (checked in the sandbox, which has no GPU):** when a GPU is requested but missing, XGBoost 3.1.3 does **not** fail: it prints a warning and silently trains on the CPU. A naive "try the GPU" test would therefore wrongly report that a GPU exists. This file reads the device the fitted booster **actually uses** from its own configuration, and compares it with the request. `info()["train_device"]` records the real device.

**Inference always on the CPU.** After a GPU fit the booster is switched to the CPU for prediction: the data given to `predict` is in ordinary memory (so GPU prediction only adds a device-mismatch warning and a copy), and the saved model must also work on a computer without a GPU (the Streamlit app). The GPU is used for training only. The device is chosen when the estimator is built, so it is not part of `model_params`.

**I could not test the GPU path** (the sandbox has no GPU): the CPU path, the fallback detection and the `"cuda"` error are tested; the actual GPU training will be seen on your machine through the command below (it prints the device). If the GPU needs a CUDA-enabled XGBoost build that your installation lacks, `"auto"` will simply use the CPU and say so.

## Other behaviour

- **Unknown parameters raise `TypeError`.** XGBoost only warns and ignores them, which would silently break a tuning grid; this file checks every parameter name against the real `XGBClassifier` list.
- **Probabilities:** `predict_proba`; `score_kind = "probability"`, rows sum to 1.
- **Class codes** must be `0..K-1` with every class present; `BaseModel` enforces it (XGBoost fails otherwise).
- **Tuning (File 54):** `n_estimators`, `max_depth`, `learning_rate` and similar are searched.
- **Time on the full data:** not measured; measured at File 47 (with the GPU if available).

## How it was checked (sandbox, CPU only)

On standardized synthetic imbalanced data (15 classes, 30,000 rows): `predict` is int64; `predict_scores` has shape `(rows, 15)`, float64, rows sum to 1; the highest score equals the prediction; scikit-learn's multiclass ROC-AUC accepts the scores; `info()` reports the device and 100 rounds; a saved and loaded model predicts the same; `with_params(max_depth=3)` works and leaves the defaults unchanged; balanced mode is applied; an unknown parameter gives `TypeError`; `auto` resolves to `cpu` without a GPU; forcing `cuda` without a GPU gives `RuntimeError`.

## How to test on your real data

Class-balanced slice (exactly 21,315 training rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, venv active:

```powershell
python -c "import numpy as np; from nids.models.xgboost_model import XGBoostModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = XGBoostModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1))); i = m.info(); print(i['params'], 'train_device', i['train_device'], 'rounds', i['n_rounds'], i['fit_seconds'], 's')"
```

**Expected output:** a log line `XGBoost device (auto): cuda` or `cpu` (this is the new information), a line `XGBoost fitted on 21315 rows in ... s`, then `XGBoostModel(XGBoost, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float64 probability rows sum to 1: True` (the score dtype may be float32), then `{'n_jobs': 14, 'random_state': 42, 'tree_method': 'hist'} train_device ... rounds 100 ... s`. A few XGBoost warning lines may appear if it falls back; please paste everything.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: xgboost` | package missing in the venv | `pip install xgboost==3.1.3` |
| `ModuleNotFoundError ... xgboost_model` | code not pasted into the placeholder | paste into `src\nids\models\xgboost_model.py`, check the length is not 0 |
| `RuntimeError: ... fell back to the CPU` | `XGBOOST_DEVICE = "cuda"` but no usable GPU | set `"auto"` or `"cpu"` in `config\settings.py` |
| `TypeError: XGBClassifier got unknown parameter(s)` | a misspelled parameter in `params=` / `with_params` | use real `XGBClassifier` parameter names |
| `ValueError: class codes must be 0..K-1 ...` | a class missing from the slice | keep all 15 classes |

## Decisions and limitations

- **GPU for training only**, CPU for inference (portability of the saved model).
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).