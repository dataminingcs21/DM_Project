# `src/nids/preprocessing/pipeline_builder.py`

**Location:** `DM_Project/src/nids/preprocessing/pipeline_builder.py`
**Roadmap step:** File 23 (Stage 4, preprocessing)

Files changed together with it: `config/settings.py` (new `LOG_TRANSFORM`), `src/nids/preprocessing/target_builder.py` (a fix, see below), and the docs of those two.

## Purpose

Builds the **unfitted preprocessing pipeline**: one scikit-learn `Pipeline` that turns raw feature columns into the numbers the models receive. It is built here and fitted on the training rows in File 25.

## Order of the steps (Checkpoint 2, option A)

| # | Step name | What it does |
|---|---|---|
| 1 | `inf_to_nan` | `+inf` / `-inf` become missing (scikit-learn refuses infinity; a flow typed in the interface could contain it) |
| 2 | `imputer` | fills missing values with the **training median** (`keep_empty_features=True`: an all-missing column stays and is removed later as a constant) |
| 3 | `log` | signed `log1p`: `sign(x) * log1p(|x|)`. Keeps the order and the sign, maps 0 to 0. Skipped when `LOG_TRANSFORM = False` |
| 4 | `filter` | `FeatureFilter` (File 22): drops constant and correlated columns, now measured on the log scale |
| 5 | `scaler` | `StandardScaler` (or robust / min-max from `SCALER_TYPE`) |

Everything that learns (median, kept columns, mean and spread) learns in `fit` from the **training rows only**; the test rows and typed flows only go through `transform`.

## Dependencies

`config/settings.py` (`IMPUTER_STRATEGY`, `SCALER_TYPE`, `LOG_TRANSFORM`, `CORRELATION_THRESHOLD`), `feature_filter.py` (File 22), `scikit-learn`, `numpy`.

## What is inside

| Name | Role |
|---|---|
| `InfiniteToMissing` | stateless transformer (step 1) |
| `SignedLog1p` | stateless transformer (step 3) |
| `build_pipeline(log_transform=None, scaler_type=None, correlation_threshold=None)` | returns the unfitted pipeline; `None` means "use the settings"; rejects an unknown scaler |
| `describe_pipeline(pipeline)` | JSON-safe summary (steps, scaler, log, imputer, and the filter report once fitted) for the metadata file (File 27) |

The pipeline uses `set_output(transform="pandas")`: column names travel from step to step, so `pipeline.get_feature_names_out()` gives the final column names. The two small transformers **refuse a different number of columns, or different names or order**, instead of silently mislabelling features (I found and fixed this problem while testing: reordered columns were first accepted silently).

## Changes in other files (send them together)

1. **`config/settings.py`:** new `LOG_TRANSFORM = True` (validated as a boolean and added to `preprocessing_config()`, so changing it makes preprocessed data stale). Replace your file with the new one.
2. **`target_builder.py` (File 19), a fix on my side:** in binary mode I had used the label `ATTACK`, but your settings already define `MALICIOUS_LABEL = "MALICIOUS"`. Two sources of truth is a bug waiting to happen, so `ATTACK_LABEL` now takes its value from `settings.MALICIOUS_LABEL`. Multiclass mode (your mode) is unaffected, and the tests use the symbol, so they still pass (92 passed in my sandbox after the change).

## How it was checked (sandbox, synthetic data)

- Fit on the first 3,000 rows, transform on the other 1,000: finite output, train mean 0 and standard deviation 1, same result on a second fit and after saving with joblib.
- The input data is not modified (an `inf` stays `inf` in the original).
- A single flow with `inf` and a missing value goes through.
- The three scaler options and `LOG_TRANSFORM = False` work; the signed log keeps the order of the values.
- Time and memory (float32, 66 columns, with missing and infinite values): 250,000 rows 5.6 s / 0.96 GB peak; 500,000 rows 12.5 s / 1.73 GB peak. **By linear extrapolation** the 2,017,889 training rows would need about 50 s and roughly 7 GB at the peak; I could not measure that size (my sandbox has 7 GB). On your 16 GB machine close the browser and games for File 25.

## How to test on your real data

A light check on a **250,000-row sample of the training rows** (PowerShell, from `DM_Project\`, `.venv` active; about 1-2 GB of RAM):

```powershell
python -c "import pandas as pd; from config import settings; from nids.data import schema; from nids.preprocessing.target_builder import build_target; from nids.preprocessing.splitter import stratified_split; from nids.preprocessing.pipeline_builder import build_pipeline, describe_pipeline; df = pd.read_parquet('data/cleaned/cleaned.parquet'); t, _ = build_target(df[settings.LABEL_COLUMN]); tr, te = stratified_split(t); cols = schema.get_feature_columns(df.columns); X = df[cols].astype('float32'); p = build_pipeline(); Xt = p.fit_transform(X.iloc[tr[:250000]]); Xe = p.transform(X.iloc[te[:50000]]); r = describe_pipeline(p)['feature_filter']; print(Xt.shape, Xe.shape); print('finite:', bool(pd.DataFrame(Xt).notna().all().all()), 'train mean max abs:', round(float(Xt.mean().abs().max()), 4)); print('constant:', r['constant_features']); print('correlated dropped:', len(r['correlated_features']), 'kept:', r['n_features_out'])"
```

**Expected output:** a log line `Pipeline built: inf_to_nan -> imputer -> log -> filter -> scaler`, a line `Feature filter fitted: 66 columns in, N kept ...`, then `(250000, N) (50000, N)`, then `finite: True train mean max abs: 0.0` (about 0.0000 to 0.001), the constant column (`['Bwd PSH Flags']`) and the number of correlated columns dropped. I do not know N: with the log applied before the filter it can differ from the 45 you saw in File 22. Please paste the output.

**Correction (found when you ran it):** `tr[:250000]` is *not* a random sample. `stratified_split` returns sorted positions, so the first 250,000 training positions are the first rows of the file, mostly the earliest capture (mainly BENIGN). Your run therefore showed `Fwd URG Flags` as constant and 38 columns kept, which does **not** hold for the full training data. The final column count comes from File 25, which fits on all 2,017,889 training rows. For a random sample use `numpy.random.default_rng(0).choice(tr, 250000, replace=False)` (File 24 does).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... pipeline_builder` | empty placeholder | paste the code, check the length is not 0 |
| `AttributeError: module 'config.settings' has no attribute 'LOG_TRANSFORM'` | the old `settings.py` is still in place | replace it with the new one |
| `ValueError: column names or their order differ` | columns reordered between fit and transform | use the same columns in the same order |
| `ValueError: scaler_type must be one of ...` | typo in `SCALER_TYPE` | use `standard`, `robust` or `minmax` |
| `MemoryError` | too little free RAM | close programs; use a smaller slice in the command |

## Decisions and limitations

- **Median imputation on the raw scale, then log:** the median is preserved by an increasing transform, so the order imputer/log does not change the result.
- **Rates of zero-duration flows** (`Flow Bytes/s`, `Flow Packets/s`) receive the training median: a rate is undefined for them. This was noted at Checkpoint 2.
- **Trees ignore the log** (they only use the order of values); KNN, SVC, LogisticRegression, SGD and GaussianNB are the ones that benefit.
- **float32 data:** the pipeline keeps float32 (`FLOAT_DTYPE`); File 25 does the casting.
- **No sampling here:** the pipeline is fitted on everything it is given.