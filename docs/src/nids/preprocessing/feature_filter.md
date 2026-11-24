# `src/nids/preprocessing/feature_filter.py`

**Location:** `DM_Project/src/nids/preprocessing/feature_filter.py`
**Roadmap step:** File 22 (Stage 4, preprocessing)

## Purpose

`FeatureFilter` removes columns that carry no extra information: **constant columns** and **highly correlated columns**. It is a scikit-learn transformer (`fit` / `transform`), so it can sit inside the preprocessing pipeline (File 23).

## Why it is needed

- Constant columns (for example `Bwd PSH Flags`) cannot help any model and break some scalers.
- Pairs of columns with |r| >= 0.95 are redundant: they slow the models down (KNN, SVC) and make linear models unstable.
- The choice must be **learned on the training rows only**; otherwise the test rows would influence which columns exist (leakage). `fit` sees training data, `transform` applies the same choice everywhere, including to one flow typed into the interface.

## Decisions applied

- Threshold `CORRELATION_THRESHOLD = 0.95` (Checkpoint 2).
- **Of two correlated columns, the first in column order is kept** (your option A). The rule is greedy in column order, so all kept columns end up pairwise below the threshold; the result is identical on every run.
- A column is constant when it has at most one distinct finite value in the training rows (an all-missing column counts as constant).

## Dependencies

`config/settings.py` (`CORRELATION_THRESHOLD`, `RANDOM_SEED`), `nids.utils.logger`, `numpy`, `pandas`, `scikit-learn`.

## What is inside

| Member | Role |
|---|---|
| `FeatureFilter(correlation_threshold=None, max_rows=500_000, random_state=None)` | the transformer; `None` means "use the settings" |
| `fit(X, y=None)` | finds constant and correlated columns. `X` is a DataFrame or a 2-D array; `y` is ignored (never used) |
| `transform(X)` | returns the kept columns as an array. Refuses a different number of columns, or (for a DataFrame) different names or order |
| `get_feature_names_out()` | names of the kept columns |
| `dropped_features_` | names of the dropped columns |
| `report()` | JSON-safe dict: counts, constant columns, and for each dropped correlated column the column kept instead and the correlation |
| attributes after fit | `constant_features_`, `correlated_features_`, `support_`, `threshold_`, `n_rows_used_`, `feature_names_in_` |

## How it works

1. Constant detection: column by column, on finite values (`inf` and missing are ignored).
2. Correlation: Pearson, on a **random sample of at most 500,000 training rows** (a full 2-million-row matrix is not needed to find |r| >= 0.95; the EDA used the same size). Missing and infinite values are ignored pair by pair. Negative correlations count (|r|).
3. Greedy rule in column order: a column is dropped if it correlates at or above the threshold with a column already kept.
4. If every column would be dropped, it raises an error instead of returning nothing.

Measured in the sandbox on random data of 2,000,000 x 66: about 8 seconds to fit.

## How to test

The automated tests come with File 26 (`tests/test_preprocessing.py`). A quick manual check on the real cleaned data, **training rows only** (PowerShell, from `DM_Project\`, `.venv` active; needs about 3 GB of RAM):

```powershell
python -c "import pandas as pd; from config import settings; from nids.data import schema; from nids.preprocessing.target_builder import build_target; from nids.preprocessing.splitter import stratified_split; from nids.preprocessing.feature_filter import FeatureFilter; df = pd.read_parquet('data/cleaned/cleaned.parquet'); t, _ = build_target(df[settings.LABEL_COLUMN]); tr, te = stratified_split(t); cols = schema.get_feature_columns(df.columns); f = FeatureFilter().fit(df.iloc[tr][cols]); r = f.report(); print(r['n_features_in'], r['n_features_out']); print('constant:', r['constant_features']); print('dropped:', {k: v['kept_instead'] for k, v in r['correlated_features'].items()})"
```

**Expected output:** a log line `Feature filter fitted: 66 columns in, N kept (C constant, K correlated dropped; threshold 0.950)`, then `66 N`, then the constant column list (I expect `Bwd PSH Flags`) and the dropped columns with the column kept instead. I do not know N yet; I will not guess. Please paste the output, because it tells us how many features the models will really use.

Note: the missing values are still present at this point (the imputer comes in File 23); the filter ignores them, so the command works.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... feature_filter` | empty placeholder file | paste the code, check the length is not 0 |
| `ValueError: X has N columns, but the filter was fitted on M` | different column set at transform time | use the same columns as in fit |
| `ValueError: column names or their order differ` | DataFrame columns reordered | keep the order from fit |
| `NotFittedError` | `transform` before `fit` | call `fit` on the training data first |
| `ValueError: every column would be dropped` | data of one value, or a threshold too low | check the data |
| `MemoryError` | too little free RAM for the command above | close programs |

## Decisions and limitations

- **Sample for the correlation:** exact on small data; on the full training data it is an estimate, very stable for |r| >= 0.95.
- **Pearson only:** it detects linear relations. Heavy skew can hide or inflate correlations; this is why the scaler decision (log transform) will be tested in File 23, and the position of the filter in the pipeline is decided there.
- **`y` is never used:** the filter does not look at the labels, so it cannot leak them.
- **Returns arrays:** the names are available through `get_feature_names_out()`.