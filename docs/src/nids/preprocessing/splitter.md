# `src/nids/preprocessing/splitter.py`

**Location:** `DM_Project/src/nids/preprocessing/splitter.py`
**Roadmap step:** File 20 (Stage 4, preprocessing)

## Purpose

Splits the rows into a training part (80%) and a test part (20%), keeping each class's proportion, with a fixed seed.

## Why it is needed

Everything that learns from data (imputer, feature filter, scaler, models) must see the **training rows only**; the test rows must stay untouched until the final evaluation. This file defines which rows are which, once, for the whole project.

## Decision applied (Checkpoint 2, option A)

Stratified random split, `TEST_SIZE = 0.20`, `RANDOM_SEED = 42`, done **after** duplicate removal (identical flows cannot be on both sides).

**Limit to remember:** every attack class comes from one capture file, so the test score measures recognition of *known* attacks on unseen flows, not generalisation to new attack types. This goes into the README.

## Dependencies

`config/settings.py` (`TEST_SIZE`, `RANDOM_SEED`, `CV_FOLDS`), `nids.utils.logger`, `numpy`, `pandas`, `scikit-learn`.

## What is inside

| Function | Role |
|---|---|
| `stratified_split(target, test_size=None, seed=None)` | returns `(train_idx, test_idx)`: **sorted row positions** (integer arrays). Raises `ValueError` for a class with fewer than 2 rows, missing values, an empty target, a bad `test_size`, or a class absent from train or test |
| `split_report(target, train_idx, test_idx)` | checks the split (disjoint, covers all rows) and returns a summary dict plus a per-class table: total, train, test, test percent. Also lists classes missing from either side and classes whose train count is below the number of CV folds |

The functions return **positions, not copies of the data**: the caller takes the rows with `df.iloc[train_idx]`. This avoids duplicating a 2.5-million-row table in memory.

## How it works

1. `stratified_split` uses scikit-learn's `train_test_split` on the row positions with `stratify=target` and the seed. Same data + same seed = same split.
2. The indices are sorted (faster, cache-friendly row access; it does not change which rows are chosen).
3. It then verifies that every class appears on both sides.
4. `split_report` re-checks and describes the result, for the preprocessing report (File 27).

## How to test

Automated tests come with File 21. Quick manual check on the real data (PowerShell, from `DM_Project\`, `.venv` active):

```powershell
python -c "import pandas as pd; from nids.preprocessing.target_builder import build_target; from nids.preprocessing.splitter import stratified_split, split_report; s = pd.read_parquet('data/cleaned/cleaned.parquet', columns=['Label'])['Label']; t, _ = build_target(s); tr, te = stratified_split(t); summ, tab = split_report(t, tr, te); print(summ); print(tab.to_string())"
```

**Expected output:** a log line `Split: 2017889 train rows, 504473 test rows (test_size=0.20, seed=42)` (2,522,362 rows: about 2,017,890 / 504,472, give or take 1), the summary with `train_test_disjoint: True`, `covers_all_rows: True`, both `classes_missing_*` lists empty, `classes_train_below_cv_folds: []`, then a table of 15 classes with `test_percent` near 20 (for small classes slightly off, e.g. Heartbleed about 18%). The class table is the real information here: look at the test counts of Heartbleed, Sql Injection and Infiltration.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... splitter` | empty placeholder file | paste the code, check the length is not 0 |
| `ValueError: classes with fewer than 2 rows` | a class has 0-1 rows | fix the target (rare-class policy) |
| `ValueError: classes missing from train ... or test ...` | a very rare class and an extreme `test_size` | change `TEST_SIZE` |
| `MemoryError` | very low RAM | close programs; this step needs only the label column |

## Decisions and limitations

- **Positions instead of frames:** saves memory and lets File 25 pick the rows from the cleaned data without a second copy.
- **No validation set:** model selection uses stratified cross-validation inside the training part (Files 49, 54); the test part is used once, at the end.
- **Random split only:** a split by source file is not used (it would remove whole attack classes from training or testing).