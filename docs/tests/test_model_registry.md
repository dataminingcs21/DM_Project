# `tests/test_model_registry.py`

**Location:** `DM_Project/tests/test_model_registry.py`
**Roadmap step:** File 42 (Stage 5, models)

## Purpose

Automatic tests for the model registry (File 41) and the common interface of all 11 models (Files 30-40), including the behaviours discovered on the real data.

## Why it is needed

The models are small files, but the trainer, the evaluation, the vote and the app all rely on the same promises: integer predictions, scores of the right shape, probabilities that sum to 1, models that can be saved and loaded, and clear errors on bad input. One parametrized test run checks those promises for every model, so a change in `base.py` or a model file that breaks one of them is caught immediately.

## Dependencies

`pytest`, `numpy`, `joblib`, `scikit-learn`, `config/model_params.py`, `nids.models.registry`, `nids.models.base`, and the 11 model files. `tests/conftest.py` redirects the project root to a temporary folder first, so the tests never touch your real `data/` or `artifacts/`.

## Data

Synthetic only: 600 rows, 6 standardized features, 4 imbalanced classes (55%, 25%, 15%, 5%) with codes 0..3, all present. The test data is easy by construction. **The tests check the interface, not performance on the real data.**

## What is tested (161 tests)

**Registry (11 tests):** the 11 keys in the order of the brief; all models available; every model created unfitted with the right key and name; `create_all_models` order; parameter override; unknown key gives `ValueError`; a missing library (`xgboost` made un-importable) makes only that model unavailable and `skip_unavailable` returns the other 10; a class whose `key` does not match its registry entry is rejected; a registry that disagrees with `MODEL_KEYS` is rejected.

**For each of the 11 models (13 tests x 11):**
- `predict` is int64 with the right shape and only valid class codes;
- `predict_scores` has shape `(rows, 4)` and is finite; SVC and SGD report `decision` scores, the other nine `probability` scores that are float64, non-negative and sum to 1, and scikit-learn's multiclass ROC-AUC accepts them;
- the highest score equals the prediction (SVC: at least 99%, because of the rare one-versus-one ties seen on the real data);
- the model beats the majority class (> 60%);
- a model saved with joblib and loaded predicts the same;
- `info()` is JSON-safe and complete;
- `predict` before `fit` raises `NotFittedError`;
- `with_params` returns a new unfitted model and does not change the original;
- an unknown parameter raises `TypeError`;
- bad data is refused: NaN, a missing class (gap in the codes), a wrong `n_classes`;
- the balanced class-weight mode is applied (KNN cannot use weights, so it reports `False`); the default mode applies none.

**Model-specific behaviour (13 tests):** KNN refuses sample weights; negative weights are refused; Gaussian NB computes in float64 and stays finite and normalised on near-constant features (the File 33 finding); Bernoulli NB stays finite with huge values; Logistic Regression and SGD report convergence correctly (`True` normally, `False` with `max_iter=1`); XGBoost device resolution (`cpu`, `cuda`, bad value), device recorded in `info()`, predictions on the CPU, and an explicit `cuda` request without a GPU raising `RuntimeError` (skipped on a computer that has a usable GPU); Decision Tree, Random Forest and XGBoost are reproducible with the fixed seed; two-class decision models return two score columns.

## Exact command

PowerShell, from `DM_Project\`, venv active:

```powershell
python -m pytest tests -q -p no:cacheprovider
```

(`-p no:cacheprovider` avoids a Windows cache-folder warning seen earlier.) To run only this file: `python -m pytest tests\test_model_registry.py -q -p no:cacheprovider`.

## Expected output

For this file alone: `161 passed` in a few seconds (about 4 s in my sandbox), plus a few **warnings that are expected**: scikit-learn convergence warnings from the `max_iter=1` tests, and XGBoost "No visible GPU" warnings on a computer without a GPU. For the whole `tests` folder: `314 passed` (153 earlier plus 161). On your computer with a GPU one test (`test_xgboost_explicit_cuda_without_gpu_is_an_error`) is **skipped**, so you may see `313 passed, 1 skipped`; that is correct.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nids` | package not installed in the venv | from `DM_Project\`: `pip install -e .` (as done earlier) |
| `ImportError` for a model | placeholder file empty | paste the code, check the length is not 0 |
| a test fails for one model only | that model file differs from the delivered code | re-copy the file |
| warnings summary | expected, see above | not an error |

## Decisions and limitations

- All 11 models are tested here instead of 11 separate test files (decided when the model files were written).
- Real-data behaviour (timings, subsample sizes, convergence on 2 million rows) is not tested here: it is measured by `scripts/05_train.py` (File 47).