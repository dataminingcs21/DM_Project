# `src/nids/evaluation/comparator.py`

**Location:** `DM_Project/src/nids/evaluation/comparator.py`  **Roadmap step:** File 52

## Role
Turns the test-set metrics of all models (plus the majority-vote ensemble) into one ranked comparison table, a per-class table and the notes that must travel with them. It only compares; it does not train, predict or vote.

## Why it is needed
The comparison table is the main result of the project. It must be built the same way for every model, rank by macro F1 (accuracy is dominated by BENIGN), and say plainly where the numbers are not comparable.

## Dependencies
`pandas`, `numpy`, `config.model_params`, `config.settings`, `nids.evaluation.metrics`, `nids.persistence.model_store`, `nids.utils.io_utils`.

## Functions
| Function | What it does |
|---|---|
| `training_info(keys=None, stage="trained", directory=None)` | reads each saved model's metadata: training rows, rows available, subsampled yes/no, fit seconds. Returns `(info, missing)` |
| `comparison_table(results, training=None)` | one row per model, sorted by macro F1 (stable). Columns: rank, model, kind (single/ensemble), macro_f1, **macro_f1_reliable** (only classes with at least 30 test rows), balanced_accuracy, accuracy, mcc, weighted_f1, detection_rate, false_alarm_rate, roc_auc_macro, score_kind, train_rows, trained_on_subsample, predict_seconds, test_rows |
| `per_class_comparison(results, metric="f1")` | classes x models; refuses models scored on different classes or rows |
| `best_model_per_class(per_class)` | best model and value for each class |
| `table_notes(frame, results, training, missing, ensemble_members)` | the honesty notes (below) |
| `to_markdown(frame, notes)` / `save_comparison(...)` | writes `model_comparison.csv/.md/.json` and `per_class_f1.csv` to `reports/evaluation/` |

`results` maps a model key to its metrics dictionary (`metrics.evaluate_model` or `compute_metrics`). The ensemble is the key `"majority_vote"`, scored with `compute_metrics` on its voted labels (no scores, so ROC-AUC shows "-"); an optional `"members"` list in it names the voters.

## Notes the table always carries
1. All rows come from the same held-out test flows, used once; ranking metric is macro F1.
2. **Subsample statement:** models trained on a class-capped subsample (KNN, SVC, Gradient Boosting) are named with their row counts, and the table says their class proportions differ from real traffic (BENIGN much rarer), so their false alarm rate and precision are not directly comparable. The Subsample column repeats this per row.
3. Low-support classes (fewer than 30 test rows) are named; `macro_f1_reliable` leaves them out.
4. ROC-AUC of decision-score models (SGD, SVC) is not a probability-based AUC.
5. Ensemble: one equal vote per model, no probability averaging, deterministic tie-break, not a trained model.
6. Missing models are listed, never dropped silently.
7. No confidence intervals: tiny differences are not evidence.

## How it was checked (sandbox)
Synthetic 4-class data (one rare class), four models saved with `model_store` and one flagged as subsampled; a simple vote added as the ensemble. Checked: sorting, the reliable macro F1 equals the hand-computed mean over classes with support >= 30, a missing model (`svm`) reported, mismatched classes or rows refused, empty input gives an empty table, JSON/CSV/Markdown written. The result differs from your real data only in numbers.

## Command (real data, training rows only; the test set is NOT used yet)
A hold-out fold of a 100,000-row class-capped sample of training rows, four fast models, then the real saved-model metadata to show which models are subsampled. PowerShell, from `DM_Project\`, about 15 s:

```powershell
python -c "import numpy as np; from nids.evaluation import comparator as C, cross_validation as cv, metrics as M; from nids.models import registry; from nids.training.data_access import load_training_data, class_capped_indices; d = load_training_data(mmap=True); idx = class_capped_indices(d.y, 100000); X, y = np.asarray(d.X[idx]), d.y[idx]; tr, te = cv.make_folds(y, 5)[0]; res = {k: M.evaluate_model(registry.create_model(k).fit(X[tr], y[tr], n_classes=d.n_classes), X[te], y[te], d.codec) for k in ['decision_tree', 'gaussian_nb', 'sgd', 'bernoulli_nb']}; info, missing = C.training_info(); print('missing:', missing); print({k: (v['n_train_rows'], v['subsampled']) for k, v in info.items() if v['subsampled']}); fr = C.comparison_table(res); print(fr[['rank', 'model', 'macro_f1', 'macro_f1_reliable', 'detection_rate', 'false_alarm_rate', 'roc_auc_macro']].round(4).to_string(index=False)); print(*C.table_notes(fr, res), sep='\n')"
```

**Expected output:** `missing: []`; a dictionary with `knn` (100000, True), `svm` (50000, True), `gradient_boosting` (100000, True); a 4-row table with Decision Tree first (macro F1 about 0.90) and `macro_f1_reliable` higher than `macro_f1` for each model (the three rare classes drag macro F1 down); then the notes (test rows, low-support classes Heartbleed, Infiltration and Web Attack - Sql Injection, and the decision-score note naming SGD Classifier). The test-rows note says "held-out test flows" even though this check uses 20,000 held-out training rows; in this check only, read it that way. No subsample note appears here because this small run passes no training info to the table.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... comparator` | placeholder empty | paste the code; Length must not be 0 |
| `ValueError: ... different test rows` | models scored on different rows | evaluate all on the same test set |
| `missing:` lists keys | model not trained or metadata deleted | rerun `05_train.py` for those |

## Limits
Macro F1 over a test set containing classes with 2, 4 and 7 rows is noisy at the bottom: read the reliable column and the per-class table beside it.