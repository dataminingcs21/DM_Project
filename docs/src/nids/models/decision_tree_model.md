# `src/nids/models/decision_tree_model.py`

**Location:** `DM_Project/src/nids/models/decision_tree_model.py`
**Roadmap step:** File 30 (Stage 5, model 1 of 11)

## Purpose

The Decision Tree model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `DecisionTreeClassifier`.

## Why it is needed

It is one of the 11 mandatory models. It is also the fastest model to train, which makes it a good first check of the whole model stack before the slower ones.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class DecisionTreeModel(BaseModel):
    key = "decision_tree"
    def _make_estimator(self, params):
        return DecisionTreeClassifier(**params)
```

Everything else (`fit`, `predict`, `predict_scores`, `with_params`, `info`, class weights, data checks) comes from `BaseModel`.

## Baseline parameters (File 28)

`{"random_state": 42}`, otherwise scikit-learn defaults. Class weights (only in the later `balanced` experiment): `class_weight="balanced"`. Subsample: none (all training rows).

## Behaviour to know

- **No depth limit by default:** the tree grows until the leaves are pure. It fits the training data almost perfectly, so the training score means little; look at the cross-validation and test scores.
- **`predict_proba` is nearly hard 0/1** for a fully grown tree, so "probabilities" of this model are not calibrated. The majority vote uses only the predicted class, so this does not matter for the vote.
- **Not affected by the log transform or the scaler:** a tree uses only the order of the values.
- **Tuning (File 54)** will search depth and leaf sizes.

## How it was checked (sandbox)

On synthetic data with 15 imbalanced classes: the model fits, `predict` gives int64, `predict_scores` has shape `(rows, 15)` with kind `probability`, and the saved/loaded model (pickle) predicts the same. `with_params(max_depth=3)` gives a depth-3 tree and leaves the original parameters unchanged. Balanced mode sets `class_weight="balanced"`. An unknown parameter is rejected by scikit-learn with a `TypeError`.

**Timing (synthetic data, noisy and overlapping classes, so a pessimistic figure):** 200,000 rows 22 s, 500,000 rows 64 s. Real network flows are much more separable, so I expect it to be faster on your data; I will measure it at File 47 before anything long runs.

## How to test on your real data

Same idea as for File 29, now with the real class: a class-balanced slice of your preprocessed training data (up to 2,000 rows per class). It checks this file and the base interface together. It is a test of the interface, **not a result**: do not read the accuracy as the model's performance. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.decision_tree_model import DecisionTreeModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = DecisionTreeModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, m.predict_scores(Xe).shape, m.score_kind); print(m.info()['params'], m.info()['n_train_rows'], m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `Decision Tree fitted on ... rows in ... s`, then `DecisionTreeModel(Decision Tree, fitted on ... rows)` (exactly **21,315 rows**: 2,000 for each of the 9 classes that have at least 2,000 training rows = 18,000, plus every row of the 6 smaller ones: Heartbleed 9, Sql Injection 17, Infiltration 29, XSS 522, Brute Force 1,176, Bot 1,562 = 3,315), then `int64 (5000,) (5000, 15) probability`, then `{'random_state': 42} ... s`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... decision_tree_model` | empty placeholder | paste the code, check the length is not 0 |
| `ImportError ... base` | File 29 not in place | paste `models/base.py` |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the slice | the command above keeps all 15 classes; if you changed it, keep every class |
| `TypeError: ... unexpected keyword argument` | an invalid parameter passed with `params=` or `with_params` | use real `DecisionTreeClassifier` parameters |

## Decisions and limitations

- **Defaults on purpose:** a fully grown tree is the neutral baseline; the gain from tuning will be measurable.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).