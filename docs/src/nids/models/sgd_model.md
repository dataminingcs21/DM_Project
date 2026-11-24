# `src/nids/models/sgd_model.py`

**Location:** `DM_Project/src/nids/models/sgd_model.py`
**Roadmap step:** File 36 (Stage 5, model 7 of 11)

## Purpose

The SGD Classifier model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `SGDClassifier` and reports whether the optimizer converged.

## Why it is needed

It is one of the 11 mandatory models: a linear classifier trained by stochastic gradient descent, cheap on millions of rows.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `nids.utils.logger`, `numpy`, `scikit-learn`.

## Baseline parameters (File 28)

`{"random_state": 42, "max_iter": 1000, "n_jobs": 14}`; otherwise defaults: `loss="hinge"` (a linear support-vector machine), L2 penalty `alpha=0.0001`, one-versus-rest over the classes. Subsample: none. Class weights (only in the later `balanced` experiment): `class_weight="balanced"`.

## What this file adds to the base class

| Member | Role |
|---|---|
| `converged` | True if the optimizer stopped before using all `max_iter` epochs |
| `info()` | base description plus `n_iter`, `max_iter`, `converged` |
| `fit(...)` | base `fit`, then **logs a warning** if it did not converge |

## Behaviour to know

- **No probabilities.** The hinge loss has no `predict_proba`. `predict_scores` returns **decision scores** (one per class, not summing to 1), and `score_kind` is `"decision"`.
- **Consequence for evaluation (File 48/51):** scikit-learn's multiclass `roc_auc_score` refuses scores that are not probabilities (I checked: it raises an error). For this model, and for SVC, the ROC-AUC will be computed one class against the rest from the decision scores, class by class. The majority vote uses only the predicted class, so it is unaffected.
- **Sensitive to scaling and to the learning schedule:** it relies on the standardized features. Results can vary a little with the seed (fixed to 42).
- **Convergence:** reported, not hidden. If the cap is reached the table shows `converged = False`.
- **Tuning (File 54)** will search `alpha` (and possibly the loss).

## How it was checked (sandbox)

On standardized synthetic data (15 classes): `predict` is int64; `predict_scores` has shape `(rows, 15)` with `score_kind = "decision"`; the highest score equals the prediction; a saved and loaded model predicts the same; `with_params(alpha=0.01)` works and leaves the defaults unchanged; balanced mode sets `class_weight="balanced"`; with `max_iter=2` it reports `converged = False` and logs the warning; an unknown parameter gives a `TypeError`. Multiclass ROC-AUC on the raw scores is refused by scikit-learn (see above).

## How to test on your real data

Class-balanced slice (exactly 21,315 training rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, venv active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.sgd_model import SGDModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = SGDModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'argmax equals predict:', bool((m.estimator_.classes_[s.argmax(axis=1)] == m.predict(Xe)).all())); i = m.info(); print(i['params'], 'n_iter', i['n_iter'], 'of', i['max_iter'], 'converged', i['converged'], i['fit_seconds'], 's')"
```

**Expected output:** a log line `SGD Classifier fitted on 21315 rows in ... s` (and a warning if it did not converge), then `SGDModel(SGD Classifier, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float32 decision argmax equals predict: True`, then `{'max_iter': 1000, 'n_jobs': 14, 'random_state': 42} n_iter ... of 1000 converged ... ... s`. The score dtype may be float32 (the data's type).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... sgd_model` | code not pasted into the placeholder | paste into `src\nids\models\sgd_model.py`, check the length is not 0 |
| warning "without converging" | epoch cap reached | not an error; shown in `info()` |
| `ValueError: class codes must be 0..K-1 ...` | a class missing from the slice | keep all 15 classes |
| `TypeError: ... unexpected keyword argument` | invalid parameter | use real `SGDClassifier` parameters |

## Decisions and limitations

- **Hinge loss kept as the baseline** (File 28): it gives no probabilities; `loss="log_loss"` could be tried in tuning.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).