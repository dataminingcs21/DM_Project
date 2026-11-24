# `scripts/05_train.py`

**Location:** `DM_Project/scripts/05_train.py`
**Roadmap step:** File 47 (Stage 6, training) - **Checkpoint 4**

## Purpose

Trains the 11 baseline models on the preprocessed training data, saves them, reloads each one as a check, and writes a training report.

## Why it is needed

This is the first time all the model code meets the real 2 million rows. It gives the real fit and prediction times, the real size of each saved model (Random Forest is the one to watch), whether Logistic Regression and SGD converge, and the trained models that the later steps (cross-validation, evaluation, tuning, the app) load.

## Dependencies

`nids.training.trainer` (46), `nids.training.data_access` (45), `nids.persistence.model_store` (43), `config/model_params.py`, `config/settings.py`. Needs the preprocessed data from `scripts/04_preprocess.py`.

## What it does

1. Loads the training data (refuses stale or inconsistent data). The test set is **not** read.
2. Prints the plan: each model and the number of rows it will train on.
3. Trains the models one after another in the order of the brief (a failed model is recorded and the others continue; `--stop-on-error` changes that).
4. Saves each model to `artifacts\models\trained\` with its metadata.
5. **Reloads every saved model** (checksum and freshness checked) and runs it on a class-balanced slice of **training** rows (up to 200 per class): the output must have the right shape and range, and the model must predict at least 2 classes. The accuracy on that slice is reported as a **smoke test only**: the models have seen those rows, so it is not a performance measure.
6. Writes `reports\training\training_report.md` and `training_summary.json`.

## Options

| Option | Meaning |
|---|---|
| `--models KEY [KEY ...]` | train only these models (keys: `knn decision_tree random_forest gaussian_nb svm bernoulli_nb logistic_regression sgd adaboost gradient_boosting xgboost`) |
| `--skip-existing` | keep models that are saved **and fresh**; a stale one is retrained. Their numbers come from their saved metadata, so the report still describes them |
| `--stop-on-error` | stop at the first failed model |
| `--dry-run` | train in memory, save nothing, print the report |
| `--sample-size N` | with `--dry-run` only: train on a class-capped sample of N rows (N at least 200), so a test cannot overwrite the real models |

A report from a run with `--models` describes only the models of that run.

## Exact commands

PowerShell, from `DM_Project\`, venv active.

**1. Quick test first** (saves nothing; about a minute or two):

```powershell
python scripts/05_train.py --dry-run --sample-size 30000
```

**2. The real run** (saves the models; this is the long one):

```powershell
python scripts/05_train.py
```

If it is interrupted (Ctrl+C, power cut), models already saved stay saved. Continue with:

```powershell
python scripts/05_train.py --skip-existing
```

## What to expect, honestly

I do **not** know the real times. What I know: your machine trained Bernoulli NB in 1.4 s, Gaussian NB in 1.1 s and SGD in 10 s on all 2 million rows (File 46), and the slice checks were fast. The slow ones are Random Forest (100 fully grown trees on 2 million rows), AdaBoost (single thread), Gradient Boosting (100,000-row subsample, 1,500 trees), Logistic Regression (up to 1000 iterations over all rows), Decision Tree, and SVC / KNN prediction. My guess for the whole run is **tens of minutes**, but that is a guess; the plan and the log lines show progress, and the report gives the measured times.

Before starting, please make sure of:
- **Disk space:** a Random Forest of 100 fully grown trees can be large (my sandbox test on 77,000 rows gave about 4 MB; I do not know the size for 2 million rows, possibly tens to hundreds of MB). Keep at least 2 GB free.
- **Memory:** close other heavy programs; Random Forest with 14 threads is the memory peak (my estimate: a few GB).
- **Power:** keep the laptop plugged in and prevent sleep during the run.

## Expected output

- The plan table (11 models and their rows: KNN 100,000, SVC 50,000, Gradient Boosting 100,000, the other eight 2,017,889).
- Log lines per model (`[3/11] Random Forest`, `... fitted on ... rows in ... s`, `Saved ... (... MB)`).
- A results table, then `Checks passed: 34/34` (3 checks per model, 1 global), and `Report written to ...\reports\training\training_report.md`. Exit code 0.
- A `NOT converged` flag in the report is **not an error**; it is information (see the report section "Not converged").
- Smoke-test accuracies should be high for most models, clearly lower for AdaBoost and Bernoulli NB (expected, weak baselines), and must not be near 1/15 = 0.07 for any model (that would mean a broken model; the "not a constant predictor" check would also fail).

## What to send me (Checkpoint 4)

Please send **`reports\training\training_report.md`** (attach it). Then I will ask you the Checkpoint 4 decisions with the real numbers: whether the subsample sizes are right, whether Logistic Regression needs a different setting, and whether to run the class-weight experiment.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `DataNotReadyError: the preprocessing settings changed ...` | `settings.py` changed after preprocessing | rerun `scripts/04_preprocess.py`, then train |
| a `failed` model with `MemoryError` | not enough RAM | close programs and rerun with `--skip-existing`; tell me which model |
| `Unknown model key(s)` | misspelled key in `--models` | use the keys listed above |
| `--sample-size ... only allowed together with --dry-run` | protection of the real models | add `--dry-run` |
| exit code 2 | a model failed or a check failed | read the "Failed models" and "Automatic checks" sections of the report |
| exit code 130 | you pressed Ctrl+C | rerun with `--skip-existing` |

## Decisions and limitations

- **Untuned baselines:** tuned models are produced later (Files 54-55).
- **Smoke-test accuracy is not performance:** the real comparison uses the test set once, in step 6.
- **Subsample sizes** (KNN 100,000, SVC 50,000, Gradient Boosting 100,000) stay as decided at File 28 until the measured times say otherwise; any change will be proposed to you with the numbers.
- **Tests:** the pieces used here are tested separately (Files 42, 44); this script is validated by its own 34 automatic checks on the real run.