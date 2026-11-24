# `src/nids/models/knn_model.py`

**Location:** `DM_Project/src/nids/models/knn_model.py`
**Roadmap step:** File 32 (Stage 5, model 3 of 11)

## Purpose

The K-Nearest Neighbors model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `KNeighborsClassifier`.

## Why it is needed

It is one of the 11 mandatory models, and the only distance-based one besides SVC. It is a useful check that the scaling of the preprocessing pipeline works: KNN performs badly on unscaled data.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class KNNModel(BaseModel):
    key = "knn"
    def _make_estimator(self, params):
        return KNeighborsClassifier(**params)
```

## Baseline parameters (File 28)

`{"n_jobs": 14}` and scikit-learn defaults: 5 neighbors, uniform weights, Euclidean distance. **Subsample:** class-capped stratified, 100,000 rows to start with (`settings.SUBSAMPLE_SIZES["knn"]`, Checkpoint 3, question 2); the subsample itself is drawn by the training stage (Files 45-46). **Class weights: not supported** (KNN has no training weights): in the later `balanced` experiment a warning is logged and the model trains without them; passing `sample_weight` to it raises an error.

## Behaviour to know

- **`fit` only stores the data.** Training is instant; every prediction compares the flow with all stored rows. So prediction, not training, is the slow part.
- **Probabilities are neighbor vote shares:** with 5 neighbors only the values 0, 0.2, 0.4, 0.6, 0.8 and 1 occur. They are coarse; ROC-type curves from this model are step-like.
- **Rare classes are at a disadvantage:** a rare class needs 3 of its 5 nearest neighbors to win the vote. The class-capped subsample keeps the small classes whole, which helps; their scores will still be weaker than those of the tree models, and that is a property of the algorithm.
- **Relies on the scaler:** distances are meaningful only because the pipeline applies the signed log and the standard scaler.
- **Tuning (File 54)** will search the number of neighbors and the weighting.

## Measured in the sandbox (synthetic data, one thread)

Fit on 100,000 rows: 0.006 s. Predicting 20,000 rows against the 100,000 stored rows: 3.7 s, so about **2 minutes for the whole test set** (504,473 rows) on one thread; your machine uses 14 threads. The 5-fold cross-validation inside the subsample predicts only 20,000 rows per fold, so it is cheap. These are sandbox figures on synthetic data; I will measure the real times at File 47.

## How it was checked (sandbox)

On synthetic data with 15 imbalanced classes: `predict` gives int64, `predict_scores` has shape `(rows, 15)`, rows sum to 1 and contain only the values 0, 0.2, ..., 1; the highest score equals the prediction; a saved and loaded model predicts the same; `with_params(n_neighbors=3)` works and leaves the defaults unchanged; the balanced mode logs the warning and reports `class_weight_applied = False`; `sample_weight` raises `ValueError: KNN does not support sample weights`.

## How to test on your real data

A class-balanced slice of the preprocessed training data (exactly 21,315 rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.knn_model import KNNModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = KNNModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1)), 'distinct values:', sorted(set(np.round(s.ravel(), 3).tolist()))); print(m.info()['params'], m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `KNN fitted on 21315 rows in 0.0x s`, then `KNNModel(KNN, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) probability rows sum to 1: True distinct values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, then `{'n_jobs': 14} 0.0.. s`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... knn_model` | the code was not pasted into the placeholder, or the file is elsewhere | put it in `src\nids\models\knn_model.py`, check the length is not 0 |
| `ValueError: ... does not support sample weights` | weights passed to KNN | do not pass weights to KNN |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the training slice | keep all 15 classes |
| Very slow prediction (later) | predicting a large set against a large stored set | expected; the subsample keeps it bounded |

## Decisions and limitations

- **A subsample, stated openly:** KNN's scores in the comparison table come from a model that stored 100,000 rows, not 2 million. The table will say so.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).