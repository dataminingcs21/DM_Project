# `src/nids/models/svm_model.py`

**Location:** `DM_Project/src/nids/models/svm_model.py`
**Roadmap step:** File 39 (Stage 5, model 10 of 11)

## Purpose

The Support Vector Classifier model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `SVC` with the RBF kernel.

## Why it is needed

It is one of the 11 mandatory models: a kernel method that can draw curved boundaries between classes.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class SVMModel(BaseModel):
    key = "svm"
    def _make_estimator(self, params):
        return SVC(**params)
```

## Baseline parameters (File 28)

`{"random_state": 42}`; otherwise library defaults: `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `probability=False`. **Subsample: yes**, class-capped stratified, 50,000 rows (`SUBSAMPLE_SIZES`, to be calibrated at File 47). Class weights (only in the later `balanced` experiment): `class_weight="balanced"`.

## Behaviour to know

- **Cost:** training time grows roughly with the square of the number of rows, and prediction time with the number of support vectors. That is why it trains on a subsample. In the sandbox (easy synthetic data, so optimistic), 12,000 rows trained in 0.6 s and predicting 12,000 rows took 2.5 s; real flows overlap more and need more support vectors, so I make no forecast for 50,000 real rows. I will measure at File 47.
- **The comparison table must state** that it was trained on a subsample, like KNN and Gradient Boosting.
- **No probabilities, on purpose:** `probability=True` runs an internal 5-fold cross-validation and multiplies training time by about five. So `predict_scores` returns **decision scores** (`score_kind = "decision"`), not summing to 1; the ROC-AUC for this model is computed one class against the rest (same as SGD, File 36).
- **Predicted class vs. scores:** `predict` follows the library's one-versus-one vote; the per-class scores are derived from the same votes. In rare ties the largest score can differ from the predicted class. The predicted class is always the one from `predict` (that is what the evaluation and the majority vote use).
- **Relies on the scaler:** the RBF kernel with `gamma="scale"` assumes standardized features, which the pipeline provides.
- **Input type:** scikit-learn converts the data to float64 internally, so memory use is higher than for other models during fit.
- **Tuning (File 54)** will search `C` and `gamma` on the subsample.

## How it was checked (sandbox)

On standardized synthetic data (15 classes), warnings turned into errors: `predict` is int64; `predict_scores` has shape `(rows, 15)`, float64, `decision`; here the highest score matched `predict` on 100% of the rows; a saved and loaded model predicts the same; `with_params(C=10)` works and leaves the defaults unchanged; balanced mode is applied; an unknown parameter gives a `TypeError`.

## How to test on your real data

Class-balanced slice (exactly 21,315 training rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, venv active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.svm_model import SVMModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = SVMModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); p = m.predict(Xe); print(m); print(p.dtype, p.shape, s.shape, s.dtype, m.score_kind, 'argmax matches predict:', round(float((m.estimator_.classes_[s.argmax(axis=1)] == p).mean()), 4)); print(m.info()['params'], 'support vectors', int(m.estimator_.n_support_.sum()), m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `SVC (RBF kernel) fitted on 21315 rows in ... s` (probably tens of seconds), then `SVMModel(SVC (RBF kernel), fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float32 or float64 decision argmax matches predict: 1.0` (a value just below 1.0 is possible because of ties, see above), then `{'random_state': 42} support vectors ... ... s`. Please paste the fit time and the support-vector count: they tell me how to size the 50,000-row subsample.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... svm_model` | code not pasted into the placeholder | paste into `src\nids\models\svm_model.py`, check the length is not 0 |
| the command is slow | the kernel cost grows with rows | wait; if it passes several minutes, stop it and tell me the time |
| `ValueError: class codes must be 0..K-1 ...` | a class missing from the slice | keep all 15 classes |

## Decisions and limitations

- **No probability estimates:** accepted for speed; decision scores are used instead.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).