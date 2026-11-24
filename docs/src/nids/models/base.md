# `src/nids/models/base.py`

**Location:** `DM_Project/src/nids/models/base.py`
**Roadmap step:** File 29 (Stage 5, models)

## Purpose

Defines the **common interface of the 11 models**. Each model file (Files 30-40) will be a tiny subclass that only says how to build its estimator; everything else is shared here.

## Why it is needed

The trainer, the evaluation, the tuning step, the majority vote and the interface must treat all 11 models identically: the same way to fit, predict, get scores and describe themselves. One shared class means one place to get the rules right (data checks, class weights, timing) instead of eleven.

## Dependencies

`config/model_params.py` (File 28), `config/settings.py`, `nids.utils.logger`, `numpy`, `scikit-learn`.

## What a model file must write

```python
class DecisionTreeModel(BaseModel):
    key = "decision_tree"                      # one of model_params.MODEL_KEYS
    def _make_estimator(self, params):         # the only method to write
        return DecisionTreeClassifier(**params)
```

## The interface (`BaseModel`)

| Member | What it does |
|---|---|
| `BaseModel(params=None, class_weight_mode=None)` | parameters = defaults of File 28 updated with `params`; mode defaults to `settings.CLASS_WEIGHT_MODE` |
| `display_name`, `is_fitted`, `key` | identity and state |
| `build_estimator()` | a **fresh unfitted** estimator with the current parameters (GridSearchCV clones it); in `balanced` mode adds `class_weight="balanced"` for models that have that parameter |
| `with_params(**overrides)` | a new unfitted model of the same kind with other parameters (used with the tuning results); the original is not changed |
| `fit(X, y, sample_weight=None, n_classes=None)` | checks the data, applies class weights if asked, fits, records `fit_seconds_`, `n_train_rows_`, `fitted_at_`; returns `self` |
| `predict(X)` | integer class codes (int64) |
| `predict_scores(X)` | `(n_rows, n_classes)`: probabilities if the model has them, otherwise decision scores; for two classes the single score is turned into two columns. **Probabilities are returned as float64 and divided by their row sum**, so rows sum to 1 exactly (scikit-learn's multiclass ROC-AUC refuses rows that do not; added after a real-data finding with Gaussian NB, File 33) |
| `score_kind` | `"probability"` or `"decision"` |
| `input_dtype` (class attribute) | numeric type the estimator must compute in; `None` keeps the data's type (float32). Only Gaussian Naive Bayes sets `"float64"` (see its doc). `fit`, `predict` and `predict_scores` all go through `_prepare`, which checks the data and applies it |
| `info()` | JSON-safe description: parameters, class-weight mode and whether it was applied, rows, seconds, score kind |

## Rules enforced in `fit` (a clear `ValueError` instead of a silent problem)

- `X` is 2-D, numeric, **finite** (the pipeline guarantees this; anything else means the pipeline was skipped);
- `y` is one integer code per row; codes are `0..K-1` with **every class present** (XGBoost needs this; a gap would also shift probability columns);
- `n_classes`, when given, must equal the number of classes found (it catches a class lost in a subsample);
- `sample_weight` must be finite, non-negative and one per row; KNN does not support it (error);
- `predict` before `fit` raises `NotFittedError`.

## Class weights (only when `CLASS_WEIGHT_MODE = "balanced"`, which is the later experiment)

- models with a `class_weight` parameter (Decision Tree, Random Forest, Logistic Regression, SGD, SVC): `class_weight="balanced"` is set in `build_estimator`;
- models with `sample_weight` (Gaussian NB, Bernoulli NB, AdaBoost, Gradient Boosting, XGBoost): balanced per-row weights are computed in `fit` and multiplied with any weights you pass;
- KNN: not possible; a warning is logged and `class_weight_applied` is `False`.

In the default mode (`none`) nothing is weighted.

## How it was checked (sandbox)

Six estimator kinds (Logistic Regression, Decision Tree, KNN, SVC, Gaussian NB, SGD) were wrapped and fitted on small synthetic data: predictions are int64, scores have shape `(rows, 4)`, and the highest score equals the prediction for all of them. SVC and SGD correctly report `decision` scores, the others `probability`. Balanced mode applied for the tree and Naive Bayes and was refused with a warning for KNN. All 13 error cases raised the intended error (not fitted, NaN, text and float labels, gap in codes, wrong `n_classes`, length mismatch, KNN weights, negative weights, bad mode, missing and unknown key, abstract class). Two-class decision scores give two columns.

## How to test on your real data

The base class cannot be used alone, so this command builds a throw-away Decision Tree model on a small **class-balanced** slice of your preprocessed training data (up to 2,000 rows per class, so every class is present). It is a test of the interface, **not a result**: do not read the accuracy as the model's performance. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; from nids.models.base import BaseModel; from nids.utils.io_utils import load_array; from config import settings; T = type('T', (BaseModel,), {'key': 'decision_tree', '_make_estimator': lambda self, p: DecisionTreeClassifier(**p)}); X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = T().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, m.predict_scores(Xe).shape, m.score_kind); print(m.info()['params'], m.info()['n_train_rows'])"
```

**Expected output:** a log line `Decision Tree fitted on 21315 rows in ... s`, then `T(Decision Tree, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) probability`, then `{'random_state': 42} 21315`. The row count is exact: 2,000 for each of the 9 classes with at least 2,000 training rows (18,000), plus every row of the 6 smaller ones (Heartbleed 9, Sql Injection 17, Infiltration 29, XSS 522, Brute Force 1,176, Bot 1,562 = 3,315).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... base` | empty placeholder | paste the code, check the length is not 0 |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is absent from the training data (for example a subsample that lost a rare class) | fix the subsample (File 45 keeps small classes whole) |
| `ValueError: X contains missing or infinite values` | the preprocessing pipeline was not applied | use `X_train.npy` / the saved pipeline |
| `TypeError: ... must define the class attribute 'key'` | a model file without `key` | add `key = "..."` |
| `ValueError: ... does not support sample weights` | weights passed to KNN | do not pass weights to KNN |

## Decisions and limitations

- **Class codes must be contiguous:** the label codec guarantees it for the full data; subsampling must keep every class (small classes are kept whole).
- **Scores are not calibrated probabilities for every model:** `decision` scores (SVC, SGD) are not probabilities, and the evaluation (File 51) uses them only for ranking-based curves.
- **No saving here:** persistence is File 43 (`model_store`), which saves a model together with its description.
- **Threads:** `n_jobs` comes from `model_params`; the base class does not change it.