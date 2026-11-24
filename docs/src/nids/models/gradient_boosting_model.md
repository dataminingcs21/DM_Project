# `src/nids/models/gradient_boosting_model.py`

**Location:** `DM_Project/src/nids/models/gradient_boosting_model.py`
**Roadmap step:** File 38 (Stage 5, model 9 of 11)

## Purpose

The Gradient Boosting model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `GradientBoostingClassifier`.

## Why it is needed

It is one of the 11 mandatory models: trees built one after another, each correcting the errors of the sum of the previous ones. Usually one of the strongest model families on tabular data.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class GradientBoostingModel(BaseModel):
    key = "gradient_boosting"
    def _make_estimator(self, params):
        return GradientBoostingClassifier(**params)
```

## Baseline parameters (File 28)

`{"random_state": 42}`; otherwise library defaults: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`. **Subsample: yes**, class-capped stratified, 100,000 rows (`SUBSAMPLE_SIZES`, to be calibrated at File 47). Class weights (only in the later `balanced` experiment): balanced `sample_weight` passed to `fit`.

## Behaviour to know

- **The most expensive of the scikit-learn models.** Every round builds one tree per class; with 15 classes a default fit builds **1,500 trees**, sequentially, and this estimator uses a single thread. Hence the subsample.
- **Measured in the sandbox (2 slow threads, synthetic data):** 2,000 rows took 22 s and 6,000 rows 77 s. Extrapolating, 100,000 rows would take around 20 minutes in the sandbox; your computer is probably faster, but I do not know by how much. I will measure the real time at File 47 and calibrate the subsample size then (as decided at File 28).
- **The comparison table must state** that this model, like KNN and SVC, was trained on a subsample, so its scores are not strictly comparable with models trained on all rows. This stays in the README and in the table notes.
- **Probabilities:** `predict_proba` exists; `score_kind = "probability"`, rows sum to 1.
- **Tuning (File 54)** is expensive for this model; the grid will be small and run on the subsample.
- XGBoost (File 40) is the faster gradient boosting implementation and can use the GPU; this scikit-learn version is the mandatory slower sibling.

## How it was checked (sandbox)

On standardized synthetic imbalanced data (15 classes), warnings turned into errors: `predict` is int64; `predict_scores` has shape `(rows, 15)`, float64, rows sum to 1; the highest score equals the prediction; the fitted model holds 1,500 trees (100 rounds x 15 classes); scikit-learn's multiclass ROC-AUC accepts the scores; a saved and loaded model predicts the same; `with_params(n_estimators=10)` works and leaves the defaults unchanged; balanced mode is applied; an unknown parameter gives a `TypeError`.

## How to test on your real data

Because of the cost, this check uses a **smaller** class-balanced slice (up to 500 rows per class: exactly 6,055 rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, venv active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.gradient_boosting_model import GradientBoostingModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:500] for c in range(15)])); m = GradientBoostingModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1))); print(m.info()['params'], 'trees', m.estimator_.estimators_.size, m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `Gradient Boosting fitted on 6055 rows in ... s` (tens of seconds; it is not instant), then `GradientBoostingModel(Gradient Boosting, fitted on 6,055 rows)`, then `int64 (5000,) (5000, 15) float64 probability rows sum to 1: True`, then `{'random_state': 42} trees 1500 ... s`. The fit time on 6,055 rows tells me how to scale to 100,000: please paste it.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... gradient_boosting_model` | code not pasted into the placeholder | paste into `src\nids\models\gradient_boosting_model.py`, check the length is not 0 |
| the command seems stuck | it is a sequential 1,500-tree fit | wait; it should take tens of seconds, not hours |
| `ValueError: class codes must be 0..K-1 ...` | a class missing from the slice | keep all 15 classes |

## Decisions and limitations

- **Subsample, not full data:** accepted at File 28; calibrated by timing at File 47.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).