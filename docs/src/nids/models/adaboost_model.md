# `src/nids/models/adaboost_model.py`

**Location:** `DM_Project/src/nids/models/adaboost_model.py`
**Roadmap step:** File 37 (Stage 5, model 8 of 11)

## Purpose

The AdaBoost model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `AdaBoostClassifier`.

## Why it is needed

It is one of the 11 mandatory models: the classic boosting method, where weak learners are trained one after another and each concentrates on the rows the previous ones got wrong.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class AdaBoostModel(BaseModel):
    key = "adaboost"
    def _make_estimator(self, params):
        return AdaBoostClassifier(**params)
```

## Baseline parameters (File 28)

`{"random_state": 42}`; otherwise library defaults: `n_estimators=50`, `learning_rate=1.0`, weak learner = decision stump (a tree of depth 1). Subsample: none. Class weights (only in the later `balanced` experiment): balanced `sample_weight` passed to `fit`.

## Behaviour to know

- **Small model for 15 classes:** 50 stumps, each asking one question about one feature, is a very limited model. It is expected to be clearly weaker than Random Forest, XGBoost or Gradient Boosting. That is a property of the default, not a defect; `n_estimators`, `learning_rate` and the weak learner's depth are tuned at File 54.
- **Probabilities:** it has `predict_proba`; `score_kind` is `"probability"`, rows sum to 1.
- **Time on the full data (not yet measured):** in the sandbox, 30,000 synthetic rows took 6.6 s. Time grows at least linearly with the rows, so 2 million rows could take roughly 8 minutes or more with a single thread (AdaBoost is not parallel). I will measure the real time at File 47. If it is too slow, the options are a subsample (as for KNN, SVC and Gradient Boosting) or fewer rounds; I will ask you with the numbers.
- **Sensitive to the label noise** of rare classes: rows that are repeatedly misclassified (for example Infiltration) gain weight, so these rows can dominate later rounds.
- **scikit-learn version:** the old `algorithm` parameter is not used; the library default applies.

## How it was checked (sandbox)

On standardized synthetic imbalanced data (15 classes, 30,000 rows), warnings turned into errors: `predict` is int64; `predict_scores` has shape `(rows, 15)`, float64, rows sum to 1; the highest score equals the prediction; scikit-learn's multiclass ROC-AUC accepts the scores; a saved and loaded model predicts the same; `with_params(n_estimators=10)` works and leaves the defaults unchanged; balanced mode is applied; an unknown parameter gives a `TypeError`.

## How to test on your real data

Class-balanced slice (exactly 21,315 training rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, venv active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.adaboost_model import AdaBoostModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = AdaBoostModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1))); print(m.info()['params'], 'rounds', len(m.estimator_.estimators_), m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `AdaBoost fitted on 21315 rows in ... s` (probably 5 to 15 s), then `AdaBoostModel(AdaBoost, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float64 probability rows sum to 1: True`, then `{'random_state': 42} rounds 50 ... s`. `rounds` can be below 50 if boosting stops early (perfect fit).

## Depth option (added at File 56)
`AdaBoostModel` now accepts `base_max_depth` (an integer of at least 1) in its parameters. It builds the weak learner `DecisionTreeClassifier(max_depth=base_max_depth)` (stumps are depth 1, the baseline). Without the option nothing changes: the baseline model and its staleness hash are identical. The option is a plain integer on purpose, so the saved metadata stays JSON. Invalid values (0, negative, non-integer) raise `ValueError`. Checked in the sandbox: the registry and model-store tests still pass (197 tests), the depth option builds the expected tree, the default still gives `estimator=None`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... adaboost_model` | code not pasted into the placeholder | paste into `src\nids\models\adaboost_model.py`, check the length is not 0 |
| `ValueError: class codes must be 0..K-1 ...` | a class missing from the slice | keep all 15 classes |
| `TypeError: ... unexpected keyword argument` | invalid parameter | use real `AdaBoostClassifier` parameters |

## Decisions and limitations

- **Defaults on purpose:** the weak baseline makes the gain from tuning measurable.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).