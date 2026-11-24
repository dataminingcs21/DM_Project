# `src/nids/models/random_forest_model.py`

**Location:** `DM_Project/src/nids/models/random_forest_model.py`
**Roadmap step:** File 31 (Stage 5, model 2 of 11)

## Purpose

The Random Forest model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `RandomForestClassifier`.

## Why it is needed

It is one of the 11 mandatory models, and usually one of the strongest on tabular network data. It also gives a useful contrast with the single Decision Tree (File 30).

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class RandomForestModel(BaseModel):
    key = "random_forest"
    def _make_estimator(self, params):
        return RandomForestClassifier(**params)
```

## Baseline parameters (File 28)

`{"random_state": 42, "n_jobs": 14}` (14 on your machine; 16 threads minus 2), otherwise scikit-learn defaults: 100 trees, grown until the leaves are pure, a bootstrap sample of the rows per tree. Class weights (only in the later `balanced` experiment): `class_weight="balanced"`. Subsample: none (all training rows).

## Behaviour to know

- **Probabilities** are the average of the trees' votes and sum to 1 per row; they are more useful than those of a single tree.
- **Not affected by the log transform or the scaler** (only the order of values matters).
- **Same seed, same forest:** two fits with the same data give the same predictions.
- **Cost on the full data (a real risk, not yet measured):** 100 fully grown trees on about 2 million rows take much more time, memory and **disk space** than one tree. In my sandbox, on synthetic data with strongly overlapping classes (which makes trees much larger than real, well separated network flows), a forest on 100,000 rows took 87 s on a single thread and 368 MB when saved; on 200,000 rows 192 s and 710 MB. Those figures grow roughly linearly with the rows, so they are an **upper bound for the noisy synthetic case, not a forecast** for your data (the sandbox has 2 cores, your machine uses 14 threads). I will measure the real time and size at File 47 on a sample first. If the saved forest turns out too large, the options are a depth or leaf-size limit (File 54 tunes them) or a training subsample; I will ask you then, with the numbers.
- **Tuning (File 54)** will search the number of trees, the depth and the leaf sizes.

## How it was checked (sandbox)

On synthetic data with 15 imbalanced classes: 100 trees are built, `predict` gives int64, `predict_scores` has shape `(rows, 15)` with rows that sum to 1, the highest score equals the prediction, the saved/loaded (pickle) model predicts the same, two fits give the same result, `with_params(n_estimators=10, max_depth=4)` works and leaves the defaults unchanged, and balanced mode sets `class_weight="balanced"`.

## How to test on your real data

A class-balanced slice of the preprocessed training data (up to 2,000 rows per class: exactly 21,315 rows). It tests the interface, **not performance**: do not read anything into the numbers. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.random_forest_model import RandomForestModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = RandomForestModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1))); print(m.info()['params'], len(m.estimator_.estimators_), 'trees', m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `Random Forest fitted on 21315 rows in ... s`, then `RandomForestModel(Random Forest, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) probability rows sum to 1: True`, then `{'random_state': 42, 'n_jobs': 14} 100 trees ... s`. I expect a few seconds at most; I do not know the exact time on your machine.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... random_forest_model` | the code was not pasted into the placeholder, or the file is elsewhere | put it in `src\nids\models\random_forest_model.py`, check the length is not 0 |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the slice | the command keeps all 15 classes; keep them if you change it |
| `MemoryError` on the full data (later) | forest too large | see "Cost on the full data" |

## Decisions and limitations

- **Defaults on purpose:** 100 fully grown trees are the neutral baseline; tuning gains will be measurable.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).