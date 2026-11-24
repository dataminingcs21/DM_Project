# `src/nids/training/trainer.py`

**Location:** `DM_Project/src/nids/training/trainer.py`
**Roadmap step:** File 46 (Stage 6, training)

## Purpose

Trains the 11 baseline models one after another on the preprocessed training data, measures their fit and prediction times, and saves each one (with a record of what it trained on) through the model store.

## Why it is needed

The models, the registry, the data access and the model store are separate pieces. The trainer is the loop that joins them, with the rules that matter on a long run: one failure must not stop the rest, an interrupted run must be resumable, and the numbers in the summary must be measured.

## Dependencies

`nids.models.registry` (File 41), `nids.persistence.model_store` (File 43), `nids.training.data_access` (File 45), `config/model_params.py`, `numpy`.

## Functions and classes

| Name | What it does |
|---|---|
| `train_model(key, data, stage="trained", save=True, params=None, directory=None)` | trains one model; returns a `TrainResult`; does not catch exceptions |
| `train_models(keys=None, data=None, stage="trained", save=True, skip_existing=False, continue_on_error=True, directory=None)` | trains several (default all 11, in the order of the brief); returns a `TrainingSummary` |
| `format_summary(summary)` | a plain-text table of the results |
| `TrainResult` | key, status (`trained`, `skipped`, `failed`), fit seconds, seconds to predict 5,000 rows, rows used and available, subsampled, converged, device, file size, error, note |
| `TrainingSummary` | the results plus `trained`, `skipped`, `failed`, `not_converged`, `total_seconds`, `to_dict()` |

## What happens for each model

1. `training_subset` gives the rows: all rows, or the class-capped subsample (KNN, SVC, Gradient Boosting).
2. A fresh model is built from the registry with the baseline parameters of File 28.
3. `fit` with `n_classes` (a class lost by the subsampling would be an error).
4. The time to predict labels and scores for 5,000 rows **of its own training data** is measured. This is how the real prediction cost (important for KNN and SVC) becomes known, without touching the test set.
5. The model is saved in `artifacts\models\trained\` with an `extra` record: subsample information (rows used and available, cap, class counts), the prediction timing and the number of rows used.

## Rules

- **A failure does not hide or stop the others.** A model that raises is recorded as `failed` with `ExceptionType: message` and the loop continues (`continue_on_error=False` stops at the first failure). Ctrl+C still stops everything; models already saved stay saved.
- **Resume:** `skip_existing=True` skips a model that is already saved **and fresh** (same preprocessing and model settings). A stale model is retrained.
- **Not-converged models are flagged** in the result and the summary (Logistic Regression, SGD).
- **No test data:** only the training arrays are used.
- **Order:** the order of the project brief; the models run sequentially, each using its own threads.

## How it was checked (sandbox)

On a small synthetic set (3,000 rows, 4 classes, subsample sizes reduced to 600-900): all 11 models trained and saved, the summary table printed, no failures; the KNN metadata records 800 rows used; a second run with `skip_existing=True` skipped all 11; when `create_model` was forced to fail for SGD, SGD was recorded as `failed` with the reason while Bernoulli NB and Gaussian NB (before and after it) still trained; with `continue_on_error=False` the loop stopped after the failure; an unknown key gives `ValueError`.

## How to test on your real data

The full 11-model run belongs to `scripts/05_train.py` (File 47), because it can take a long time. This check trains the three **fastest** models on **all 2,017,889 training rows**, and saves them into a **temporary folder** (your real `artifacts\models` is untouched). It gives the first real timings. PowerShell, from `DM_Project\`, venv active, about 3 GB of RAM:

```powershell
python -c "import tempfile; from nids.training import trainer as tr; d = tempfile.mkdtemp(); s = tr.train_models(keys=['bernoulli_nb', 'gaussian_nb', 'sgd'], directory=d); print(tr.format_summary(s)); print('failed:', s.failed, 'not converged:', s.not_converged)"
```

**Expected output:** INFO log lines for each model, then a table with three rows (`Bernoulli Naive Bayes`, `Gaussian Naive Bayes`, `SGD Classifier`), each `trained`, `rows used` 2,017,889, small file sizes (below 1 MB), then `failed: [] not converged: []`. I expect seconds to about a minute in total, but I do not know the real times: please paste the table. SGD might show `not converged` if it uses all 1000 epochs on the full data; that would be reported, not hidden.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... trainer` | placeholder empty | paste the code, check the length is not 0 |
| `DataNotReadyError` | preprocessed data missing or stale | run / rerun `scripts/04_preprocess.py` |
| `failed` row with `MemoryError` | not enough RAM for that model | close other programs; tell me, and I will propose a smaller subsample |
| `ValueError: unknown model key` | a misspelled key in `keys=` | use the keys of `model_params.MODEL_KEYS` |

## Decisions and limitations

- **Baseline only:** this trains the untuned defaults. Tuned models come from `optimization/grid_search.py` (File 54).
- **Timings are for the machine and moment they were measured on.** The real time of every model is measured at File 47, and the subsample sizes may be changed then (after asking you).
- **Tests:** the trainer is exercised by the real run at File 47; the pieces it uses (registry, model store, data access) are tested separately.

## Update at File 57
`train_model(...)` has a new optional argument `extra` (a JSON-safe dictionary). It is stored in the saved metadata under `extra`, next to the subsample record. The tuning step uses it to keep the winning parameters and the tuning fingerprint with each tuned model. Without it nothing changes; `05_train.py` is unaffected. The registry, model-store and voting tests still pass (246).