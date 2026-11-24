# `src/nids/models/logistic_regression_model.py`

**Location:** `DM_Project/src/nids/models/logistic_regression_model.py`
**Roadmap step:** File 35 (Stage 5, model 6 of 11)

## Purpose

The Logistic Regression model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `LogisticRegression` and reports whether the solver converged.

## Why it is needed

It is one of the 11 mandatory models: the classic linear baseline. It is fast, stable and gives usable probabilities, and it shows how far straight boundaries get on this data.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `nids.utils.logger`, `numpy`, `scikit-learn`.

## Baseline parameters (File 28)

`{"random_state": 42, "max_iter": 1000}`; otherwise scikit-learn defaults: L2 penalty with `C=1.0`, the `lbfgs` solver, one multinomial model over all classes. The default of 100 iterations is usually too few for 40 features, which is why the cap is 1000. Subsample: none (all training rows). Class weights (only in the later `balanced` experiment): `class_weight="balanced"`.

## What this file adds to the base class

| Member | Role |
|---|---|
| `converged` | True if the solver stopped before using all `max_iter` iterations |
| `info()` | the base description plus `n_iter`, `max_iter` and `converged` |
| `fit(...)` | the base `fit`, then **logs a warning** if the solver did not converge |

**Why this matters:** when `converged` is False, the model is not a true optimum: it was stopped by the iteration limit. scikit-learn prints a `ConvergenceWarning` for this; I make it part of the model's description and the log, so the report cannot hide it. Whether it matters for the scores is judged by the numbers, not assumed.

## Behaviour to know

- **Linear boundaries:** expected to trail the tree models on this data.
- **Relies on the scaler:** `lbfgs` converges slowly on unstandardized features; your pipeline standardizes them.
- **Probabilities** (softmax) are usable and sum to 1.
- **Time on the full data (a real risk, not yet measured):** every iteration passes over all rows. On synthetic data (15 classes, strongly overlapping, so harder than real flows) with 2 threads, the solver used all 1000 iterations: 100,000 rows took 31 s and 300,000 rows 109 s, and **did not converge**. Scaling linearly, 2 million rows would take about 12 minutes in that hard case. These are sandbox figures on synthetic data; I will measure the real time and the convergence at File 47. If it does not converge on the real data, the options are a higher cap, another solver, or a training subsample; I will ask you then, with the numbers.
- **Tuning (File 54)** will search the regularization strength `C`.

## How it was checked (sandbox)

On standardized synthetic data: `predict` gives int64; `predict_scores` has shape `(rows, 15)`, float64, rows sum to 1; the highest score equals the prediction; scikit-learn's multiclass ROC-AUC accepts the scores; a saved and loaded model predicts the same; `with_params(C=0.1)` works and leaves the defaults unchanged; balanced mode sets `class_weight="balanced"`; an unknown parameter gives a `TypeError`. With `max_iter=3` the model reports `converged = False`, `info()` says the same, and the log shows the warning; with the normal cap on easy data it reports `True`.

## How to test on your real data

A class-balanced slice of the preprocessed training data (exactly 21,315 rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.logistic_regression_model import LogisticRegressionModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = LogisticRegressionModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1))); i = m.info(); print(i['params'], 'n_iter', i['n_iter'], 'of', i['max_iter'], 'converged', i['converged'], i['fit_seconds'], 's')"
```

**Expected output:** a log line `Logistic Regression fitted on 21315 rows in ... s` (and possibly a warning line if it did not converge), then `LogisticRegressionModel(Logistic Regression, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float64 probability rows sum to 1: True`, then `{'max_iter': 1000, 'random_state': 42} n_iter ... of 1000 converged ... ... s`. I do not know whether it converges on your data, nor the time (probably seconds to a few tens of seconds); please paste what you see.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... logistic_regression_model` | the code was not pasted into the placeholder, or the file is elsewhere | put it in `src\nids\models\logistic_regression_model.py`, check the length is not 0 |
| `ConvergenceWarning` / log warning "without converging" | the iteration cap was reached | not an error; it is reported in `info()`; see the time section above |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the training slice | keep all 15 classes |
| `TypeError: ... unexpected keyword argument` | an invalid parameter in `params=` / `with_params` | use real `LogisticRegression` parameters |

## Decisions and limitations

- **A non-converged baseline is reported, not hidden:** the table will show `converged` for this model.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).