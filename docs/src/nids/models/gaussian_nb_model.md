# `src/nids/models/gaussian_nb_model.py`

**Location:** `DM_Project/src/nids/models/gaussian_nb_model.py`
**Roadmap step:** File 33 (Stage 5, model 4 of 11)

## Purpose

The Gaussian Naive Bayes model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `GaussianNB`.

## Why it is needed

It is one of the 11 mandatory models. It is extremely fast, has no randomness, and gives a probabilistic baseline that is very different from trees and distance-based models.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class GaussianNBModel(BaseModel):
    key = "gaussian_nb"
    def _make_estimator(self, params):
        return GaussianNB(**params)
```

## Baseline parameters (File 28)

None: scikit-learn defaults (`var_smoothing = 1e-9`). Subsample: none (all training rows). Class weights (only in the later `balanced` experiment): balanced `sample_weight` passed to `fit`.

## A problem found on your real data (and fixed)

My first version passed your float32 data straight to scikit-learn. Your run printed `rows sum to 1: False`, which my synthetic test had not shown. I reproduced the cause in the sandbox:

- Several features are constant inside a class (for example flags). Gaussian NB gives them a tiny variance (about 1e-9), so the log-likelihood of a flow that deviates even slightly reaches -1e8 to -1e11.
- In **float32** (which scikit-learn keeps when the data is float32) such numbers have no precision left: the classes become indistinguishable, and the "probabilities" summed to 4 in my test.
- On the rows I tested, the float32 predictions disagreed with the float64 predictions in **37% of the cases**. So it was not only a cosmetic problem: the class decisions themselves were unreliable.

**Fix:** `GaussianNBModel` now declares `input_dtype = "float64"`, and the base class (File 29, updated) converts the data for this model only. I checked that, after the fix, predictions equal the float64 reference on all tested deviations and the rows sum to 1 up to 2e-16. The other models are unchanged. Memory cost: a float64 copy of the data while this model runs (about 640 MB for the full training data).

**The base class also changed (File 29):** for models that give probabilities, `predict_scores` now returns float64 values divided by the row sum, so rows sum to 1 exactly. Reason: scikit-learn's multiclass ROC-AUC (used at File 48) refuses scores whose rows do not sum to 1 within a small tolerance. The highest score and the predictions are unchanged.

## Behaviour to know

- **Strong assumptions:** inside each class every feature is normal and independent of the others. Flow features are often flags or counts and many are strongly related, so this model is expected to be one of the weaker ones. That is a property of the algorithm on this data, not a bug.
- **Overconfident probabilities:** the independence assumption multiplies 40 evidence terms, so the probabilities are usually very close to 0 or 1. In my sandbox test about half of the rows had a top probability above 0.99. The majority vote uses only the predicted class, so this does not affect the vote.
- **Features that are constant inside a small class** get a tiny variance, which can make that class too easy or too hard to predict. `var_smoothing` limits this and is searched by GridSearchCV (File 54).
- **Priors** are the class frequencies of the training data (about 83% BENIGN). In the later `balanced` experiment the sample weights make the priors equal (about 1/15 per class in my test), which changes the decisions: this is the intended effect of that experiment.
- **Not helped by the signed log in a simple way:** the log transform makes counts closer to normal, which helps this model more than it helps the trees.

## How it was checked (sandbox)

The first version was checked on synthetic data that did not reach the extreme values of the real data (see above). After the fix: on synthetic data with 15 imbalanced classes and features constant inside classes, with Python warnings turned into errors (no `RuntimeWarning`, no `NaN`): `predict` gives int64, `predict_scores` has shape `(rows, 15)`, rows sum to 1 and are finite, the highest score equals the prediction, the priors equal the class frequencies, a saved and loaded model predicts the same, `with_params(var_smoothing=1e-3)` works and leaves the defaults unchanged, balanced mode is applied, and an unknown parameter is rejected with a `TypeError`.

## How to test on your real data

A class-balanced slice of the preprocessed training data (exactly 21,315 rows). It tests the interface, **not performance**: the last number printed (share of confident rows) is a characteristic to look at, not a result. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.gaussian_nb_model import GaussianNBModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = GaussianNBModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1)), 'finite:', bool(np.isfinite(s).all())); print(m.info()['params'], m.info()['fit_seconds'], 's', '| share of rows with top probability > 0.99:', round(float((s.max(axis=1) > 0.99).mean()), 3))"
```

**Expected output:** a log line `Gaussian Naive Bayes fitted on 21315 rows in 0.0 s`, then `GaussianNBModel(Gaussian Naive Bayes, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) probability rows sum to 1: True finite: True` (**`True` is what must change from your first run, where it said `False`**), then `{} 0.0.. s | share of rows with top probability > 0.99: ...`. Your first run gave 0.968 for the last number; it may change now that the calculation is done in float64, and I do not know the new value.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... gaussian_nb_model` | the code was not pasted into the placeholder, or the file is elsewhere | put it in `src\nids\models\gaussian_nb_model.py`, check the length is not 0 |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the training slice | keep all 15 classes |
| `TypeError: ... unexpected keyword argument` | an invalid parameter in `params=` / `with_params` | use real `GaussianNB` parameters (`var_smoothing`, `priors`) |

## Decisions and limitations

- **Defaults on purpose:** the neutral baseline; the gain from `var_smoothing` will be measurable at File 55.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).