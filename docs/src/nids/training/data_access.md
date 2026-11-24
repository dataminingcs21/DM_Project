# `src/nids/training/data_access.py`

**Location:** `DM_Project/src/nids/training/data_access.py`
**Roadmap step:** File 45 (Stage 6, training)

## Purpose

The single place that reads the preprocessed arrays for training and evaluation, checks them, and decides which rows each model trains on (all rows, or a class-capped subsample).

## Why it is needed

The trainer (File 46), the cross-validation (49), the tuning (54) and the evaluation (48-53) must read exactly the same data, check it the same way and build the same subsamples. Doing this once, here, avoids eleven slightly different versions.

## Dependencies

`config/settings.py`, `config/model_params.py` (subsample sizes), `nids.preprocessing.preprocessor` (freshness check), `nids.preprocessing.target_builder` (`LabelCodec`), `nids.utils.io_utils`, `numpy`.

## Functions

| Function | What it does |
|---|---|
| `check_data_ready()` | raises `DataNotReadyError` unless the preprocessed data exists and is **fresh** (its settings hash equals the current settings); returns the preprocessing metadata |
| `load_training_data(mmap=False)` | `TrainingData(X, y, codec, n_classes, feature_count)`; checks that rows agree, `y` is integer, codes are inside `0..K-1`, the feature count matches the metadata, and **every class is present**. It never reads the test set |
| `load_test_data(mmap=False)` | `TestData(X, y, codec, n_classes)`, same checks; only the evaluation (Files 48-53) should call it |
| `class_cap_for_target(counts, target)` | the per-class cap and the extra rows needed to reach the target size |
| `class_capped_indices(y, target, seed=None)` | sorted row positions of the subsample |
| `training_subset(model_key, X, y)` | `(X_used, y_used, info)`: all rows for models without a subsample size (no copy), the subsample for KNN, SVC, Gradient Boosting. `info` is JSON-safe and is saved with the model (`extra`) |

## The subsample rule (your decision: option A)

Every class keeps `min(count, cap)` rows: **small classes are kept whole, large classes are capped**. The cap is the largest value whose total does not exceed the target; the few rows still missing are given, one each, to the largest classes, so the total is exactly the target. Rows inside a capped class are chosen at random with the project seed: same data, same rows. The positions are returned sorted.

On the class counts of your training data (from your earlier output), I computed:

| Model | Target | Cap per class | Rows |
|---|---|---|---|
| KNN | 100,000 | 18,161 | exactly 100,000 |
| Gradient Boosting | 100,000 | 18,161 | exactly 100,000 |
| SVC | 50,000 | 6,174 | exactly 50,000 |

Classes below the cap are whole: for the 100,000-row subsample that is Bot, DoS GoldenEye, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed (9), Infiltration (29), SSH-Patator, the three Web Attack classes. BENIGN, DDoS, DoS Hulk and PortScan are capped.

## An important consequence (to be stated in the report)

The subsample is **more balanced than the real traffic**. BENIGN is 83% of the real training data but only about 18% of the 100,000-row subsample. These three models therefore learn a different prior: they may predict attacks more readily than the other eight and make more false alarms on normal traffic. Their scores on the test set (which has the real class mix) are not strictly comparable with the models trained on all rows. The comparison table (File 52) will say so for each of the three.

## How it was checked (sandbox)

On a synthetic label vector with exactly your 15 training class counts: the subsample has exactly 100,000 and 50,000 rows with all 15 classes present; Heartbleed keeps all 9 rows; the result is deterministic, sorted and has no duplicates; a target above the number of rows returns every row; a target smaller than the number of classes gives a `ValueError`; `training_subset` returns the arrays themselves (no copy) for Random Forest and the subsample for KNN.

## How to test on your real data

Reads your real preprocessed arrays (memory-mapped, so little RAM) and prints what each kind of model would train on. PowerShell, from `DM_Project\`, venv active:

```powershell
python -c "from nids.training import data_access as da; d = da.load_training_data(mmap=True); print(d.n_rows, d.feature_count, d.n_classes); [print(k, (lambda r: (r[0].shape[0], r[2]['subsampled'], r[2]['cap_per_class']))(da.training_subset(k, d.X, d.y))) for k in ('knn', 'svm', 'gradient_boosting', 'random_forest')]; X, y, i = da.training_subset('knn', d.X, d.y); print(dict(zip(d.codec.classes, i['class_counts'])))"
```

**Expected output:** a log line `Training data: 2017889 rows, 40 features, 15 classes`, a line `2017889 40 15`, then:

```
knn (100000, True, 18161)
svm (50000, True, 6174)
gradient_boosting (100000, True, 18161)
random_forest (2017889, False, None)
```

(each preceded by an INFO log line about the subsample), then the class counts of the KNN subsample: BENIGN 18162, Bot 1562, DDoS 18161, DoS GoldenEye 8229, DoS Hulk 18161, DoS Slowhttptest 4182, DoS slowloris 4308, FTP-Patator 4746, Heartbleed 9, Infiltration 29, PortScan 18161, SSH-Patator 2575, Web Attack - Brute Force 1176, Web Attack - Sql Injection 17, Web Attack - XSS 522. It takes a few seconds.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... data_access` | placeholder empty | paste the code, check the length is not 0 |
| `DataNotReadyError: the preprocessing settings changed ...` | `settings.py` changed after preprocessing | rerun `scripts/04_preprocess.py`, then retrain |
| `DataNotReadyError: missing ...` | preprocessed files not found | run `scripts/04_preprocess.py` |
| `DataNotReadyError: ... every class code` | a class is missing from `y_train` | rerun preprocessing; this should not happen with your data |

## Decisions and limitations

- **Option A for the subsample** (your decision at File 45): small classes whole, large classes capped. The shift of the class prior is accepted and must be reported.
- **Sizes are starting values** (`SUBSAMPLE_SIZES`); File 47 times the models and may change them, after asking you.
- **Tests:** `tests/test_training.py` is not in the roadmap; the subsample rules are covered by the trainer's checks at File 47, and I can add a test file if you want one.