# `scripts/07_optimize.py` (Checkpoint 5)

**Roadmap step:** File 57. Changed in the same step: `src/nids/optimization/grid_search.py` (quieter logs) and `src/nids/training/trainer.py` (optional `extra` argument). Replace all three code files together.

## Role
Tunes the 11 models with `GridSearchCV` (File 56) and trains each winner on the same rows as its baseline. It produces the tuned models and the report for Checkpoint 5.

## Why it is needed
It answers "how much can each model gain from tuning, and does class weighting help once the real class mix is accounted for?", using cross-validation on training rows only.

## The test set is not read
The tuned models are scored on the test set once, afterwards, by `06_evaluate.py --stage tuned` (delivered next), which puts them beside the baselines. Choosing between a tuned and a baseline model is therefore never done on the test set.

## Stages (each saves as it goes; Ctrl+C is safe, rerun continues)
1. **Grid search** per model on a class-capped sample of training rows, scored by prior-corrected macro F1 (plain macro F1 recorded beside it). Results: `reports\optimization\tuning_<model>.json`.
2. **Refit:** each winner is trained on the same rows as its baseline (all rows; for KNN, SVC and Gradient Boosting their own subsample) and saved to `artifacts\models\tuned\`. If the winner IS the baseline setting, the baseline model is saved as the tuned model without retraining. A tuned model that is already saved for the current tuning result is skipped.
3. **Checks:** every tuned model is reloaded (checksum and freshness verified), its saved parameters are compared with the winning ones, and it predicts valid classes on a slice of training rows.
4. **Report:** `reports\optimization\tuning_report.md` (**send me this one**) and `tuning_summary.json`.

## Options
`--dry-run` (plan only), `--models k1 k2 ...`, `--skip-refit`, `--rerun` (ignore saved results), `--sample-size N` (override the tuning sample; slow models stay capped).

## What the report contains
Scope; a summary table (best parameters, baseline and best corrected F1, improvement, beyond fold noise, plain F1, minutes); computed observations (which models improved beyond noise, which baselines were already best, **class weights none versus balanced for each model that has the option**, winners at the edge of their grid); the top 5 grid points per model plus the baseline; the refit table (fit seconds versus the baseline's); automatic checks; limits.

## Time (estimates, not measurements: the real numbers will be in the report)
Grid search about 30 to 40 minutes (Random Forest, KNN, SGD and XGBoost run one grid point at a time; the others in parallel). Refit about 45 to 70 minutes, dominated by AdaBoost (a deeper weak learner on 2 million rows) and Gradient Boosting (100,000 rows, deeper trees). Total roughly 1.5 to 2 hours. Keep the laptop awake and plugged in.

## Commands (PowerShell, from `DM_Project\`, venv active)
1. Plan (a few seconds, nothing runs):
```powershell
python scripts\07_optimize.py --dry-run
```
2. The real run (long; resumable):
```powershell
python scripts\07_optimize.py
```

## Expected output
Dry run: a table with one line per model (grid points, folds, tuning rows, number of fits, parallel or one by one, whether results are saved), `The test set is not read by this script.` and `Dry run: nothing was run.` Real run: for each model the grid search and the refit log (the fit messages are fewer than before), then a table of best parameters and scores, `Report: ...tuning_report.md` and `Checks passed: N/N`. Exit code 0 when no model failed and all checks passed.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `No trained baseline for [...]: run 05_train.py first` | baseline models missing or deleted | rerun `05_train.py` |
| `StaleModelError` in a check | settings changed since training | rerun `05_train.py`, then this script with `--rerun` |
| `TuningError: ... cannot be fitted` | invalid grid value for your library version | send me the message |
| `MemoryError` during a refit | very low free RAM | close programs; rerun (finished models are reused) |
| exit code 2 | a model failed or a check failed | the report's last sections say which |

## Limits (stated in the report)
Parameters are searched on at most 100,000 rows and refitted on more; the grids are small; the corrected F1 is noisy for the three rarest classes; the test set is untouched here.