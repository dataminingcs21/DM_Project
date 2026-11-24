# `src/nids/models/bernoulli_nb_model.py`

**Location:** `DM_Project/src/nids/models/bernoulli_nb_model.py`
**Roadmap step:** File 34 (Stage 5, model 5 of 11)

## Purpose

The Bernoulli Naive Bayes model: a subclass of `BaseModel` (File 29) that builds a scikit-learn `BernoulliNB`.

## Why it is needed

It is one of the 11 mandatory models. It is extremely fast, and it shows what a model that sees only "above or below average" for each feature can do.

## Dependencies

`nids.models.base.BaseModel` (File 29), `config/model_params.py` (File 28), `scikit-learn`.

## The whole model class

```python
class BernoulliNBModel(BaseModel):
    key = "bernoulli_nb"
    def _make_estimator(self, params):
        return BernoulliNB(**params)
```

## Baseline parameters (File 28)

None: scikit-learn defaults (`alpha=1.0`, `binarize=0.0`). Subsample: none (all training rows). Class weights (only in the later `balanced` experiment): balanced `sample_weight` passed to `fit`.

## Behaviour to know

- **It works on binary features.** Each value becomes 1 if it is above `binarize` and 0 otherwise. The pipeline standardizes every feature (mean 0 on the training rows), so `binarize=0.0` means "above or below the training average". Most of the information in the values is discarded, so this model is expected to be **among the weakest of the 11**. That is a property of the algorithm on this data, not a defect. `binarize` and `alpha` are searched by GridSearchCV (File 54).
- **Smoothing:** `alpha=1.0` prevents zero probabilities, so it does not have the numerical problem found with Gaussian NB (File 33); I checked that its probabilities sum to 1 and are finite on data with huge values and features that are constant inside classes.
- **Priors** are the class frequencies of the training data. In the later `balanced` experiment the sample weights make them equal (about 1/15 per class in my test).
- **Probabilities** are also overconfident (independence assumption), but the majority vote uses only the predicted class.

## How it was checked (sandbox)

On standardized synthetic data (15 imbalanced classes, flags, constants inside classes, one feature with huge values), warnings turned into errors: `predict` gives int64; `predict_scores` has shape `(rows, 15)`, float64, finite, rows sum to 1; the highest score equals the prediction; a saved and loaded model predicts the same; `with_params(alpha=0.1, binarize=0.5)` works and leaves the defaults unchanged; balanced mode is applied; scikit-learn's multiclass ROC-AUC accepts the scores; an unknown parameter gives a `TypeError`. On the same synthetic data its training accuracy (0.82) was lower than Gaussian NB's (0.93), as expected from the binarization. That comparison is on synthetic data and says nothing about your data.

## How to test on your real data

A class-balanced slice of the preprocessed training data (exactly 21,315 rows). It tests the interface, **not performance**. PowerShell, from `DM_Project\`, `.venv` active, about 1 GB of RAM:

```powershell
python -c "import numpy as np; from nids.models.bernoulli_nb_model import BernoulliNBModel; from nids.utils.io_utils import load_array; from config import settings; X = load_array(settings.X_TRAIN_FILE, mmap_mode='r'); y = load_array(settings.Y_TRAIN_FILE); idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:2000] for c in range(15)])); m = BernoulliNBModel().fit(X[idx], y[idx], n_classes=15); Xe = load_array(settings.X_TEST_FILE, mmap_mode='r')[:5000]; s = m.predict_scores(Xe); print(m); print(m.predict(Xe).dtype, m.predict(Xe).shape, s.shape, s.dtype, m.score_kind, 'rows sum to 1:', bool(np.allclose(s.sum(axis=1), 1)), 'finite:', bool(np.isfinite(s).all())); print(m.info()['params'], 'binarize', m.estimator_.binarize, 'alpha', m.estimator_.alpha, m.info()['fit_seconds'], 's')"
```

**Expected output:** a log line `Bernoulli Naive Bayes fitted on 21315 rows in 0.0 s`, then `BernoulliNBModel(Bernoulli Naive Bayes, fitted on 21,315 rows)`, then `int64 (5000,) (5000, 15) float64 probability rows sum to 1: True finite: True`, then `{} binarize 0.0 alpha 1.0 0.0.. s`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... bernoulli_nb_model` | the code was not pasted into the placeholder, or the file is elsewhere | put it in `src\nids\models\bernoulli_nb_model.py`, check the length is not 0 |
| `ValueError: class codes must be 0..K-1 with every class present` | a class is missing from the training slice | keep all 15 classes |
| `TypeError: ... unexpected keyword argument` | an invalid parameter in `params=` / `with_params` | use real `BernoulliNB` parameters (`alpha`, `binarize`, `fit_prior`, ...) |

## Decisions and limitations

- **Defaults on purpose:** `binarize=0.0` is the library default and the neutral baseline; the gain from tuning will be measurable at File 55.
- **No separate test file:** all 11 models are tested together by `tests/test_model_registry.py` (File 42).