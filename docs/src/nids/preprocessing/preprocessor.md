# `src/nids/preprocessing/preprocessor.py`

**Location:** `DM_Project/src/nids/preprocessing/preprocessor.py`
**Roadmap step:** File 25 (Stage 4, preprocessing)

## Purpose

Runs the **whole preprocessing** on the cleaned data, using Files 19-24 in the right order, and saves the results to `data/preprocessed/`.

## Why it is needed

It is the one place where the leakage rules are applied together: the split comes first, and everything that learns from data (median, kept columns, scaling) learns from the training rows only. The training stage reads only what this file saves.

## Dependencies

Files 06, 19, 20, 22, 23, 24 and `config/settings.py`, `numpy`, `pandas`, `scikit-learn`.

## Order of the work (`preprocess(df)`)

1. Feature columns of the cleaned data (without `EXCLUDED_FEATURES`, currently empty) and the target + class codes (File 19).
2. Stratified 80/20 split by row positions (File 20).
3. The pipeline (File 23) is **fitted on the training rows only**, then applied to the test rows. Data is cast to float32 first.
4. The feature schema (File 24) is built from the raw training rows and the fitted pipeline.
5. A few real **test** flows per class are kept as examples for the interface (up to 5 per class, fixed seed; real rows, nothing invented).
6. Both outputs are checked: any missing or infinite value after preprocessing raises an error.

`preprocess` writes nothing. It returns a `PreprocessingResult` with `X_train`, `X_test` (float32), `y_train`, `y_test` (int64), the fitted `pipeline`, the `codec`, the `schema`, the split positions, `sample_flows`, a `report` dict and the per-class `split_table`.

## What is saved (`save_preprocessed(result)`), in `data/preprocessed/`

| File | Content |
|---|---|
| `X_train.npy`, `X_test.npy` | model inputs, float32 |
| `y_train.npy`, `y_test.npy` | class codes (BENIGN = 0) |
| `preprocessing_pipeline.joblib` | the fitted pipeline (also used by the interface) |
| `label_encoder.joblib` | the class names and codes |
| `feature_schema.json`, `feature_names.json` | File 24 |
| `sample_flows.parquet` | example real test flows: raw feature columns, `Label`, `test_row_position` (row in `X_test`), `cleaned_row_position` (row in the cleaned data). The last two columns are not features |
| `preprocessing_metadata.json` | written **last** (its presence means "complete"): time, settings and their hash, the hash of the cleaning rules, shapes, class counts, split summary, pipeline description, seconds |

## Freshness check

`check_preprocessed_fresh()` compares the hash of the saved settings with the current `preprocessing_config()` and returns `(True, "")` or `(False, reason)`. The training stage (File 45-47) uses it to refuse stale data: for example after you change `SCALER_TYPE` or `LOG_TRANSFORM`.

## How it was checked (sandbox, synthetic data)

- Train mean 0, finite outputs, input data not modified.
- A saved sample flow, sent through the **reloaded** pipeline, gives exactly the corresponding row of `X_test`: the same pipeline serves the interface.
- Sample rows equal the cleaned rows they claim to come from (positions and labels).
- Arrays and codec reload identically; changing a setting is detected as stale; missing label column and text columns are rejected.
- 400,000 x 66 rows: 7.1 s, 1.10 GB peak. **Extrapolated** to your 2,522,362 rows: about 45 s and about 6 GB at the peak (not measured at that size).

## How to test on your real data

This runs the **full** preprocessing in memory and prints a summary. **It writes nothing** (File 27 saves). Close the browser and games first (about 6-8 GB). PowerShell, from `DM_Project\`, `.venv` active:

```powershell
python -c "import pandas as pd; from nids.preprocessing.preprocessor import preprocess; r = preprocess(pd.read_parquet('data/cleaned/cleaned.parquet')); print(r.X_train.shape, r.X_test.shape, r.X_train.dtype); f = r.report['pipeline']['feature_filter']; print('constant:', f['constant_features']); print('correlated dropped:', len(f['correlated_features'])); print(pd.DataFrame(r.report['class_counts']).to_string()); print('sample flows:', r.sample_flows.shape, 'seconds:', r.report['seconds'])"
```

**Expected output:** the log lines (target built, split, pipeline built, filter fitted with 66 columns in, schema built, `Preprocessing done in ~45 s`), then the shapes `(2017889, N) (504473, N) float32` with N around 40 (I do not know it: it depends on all the training rows), the constant columns (`['Bwd PSH Flags']`), the number of correlated columns dropped, a table of the 15 classes with their train and test counts (the same counts as in File 20), and `sample flows: (71, 69) seconds: ...`: 5 flows for each of the 13 classes with at least 5 test rows, plus 4 (Sql Injection) and 2 (Heartbleed), with 66 raw columns, `Label` and the two position columns. Please paste the output.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... preprocessor` | empty placeholder | paste the code, check the length is not 0 |
| `MemoryError` or the computer freezes | less than about 7 GB free | close other programs |
| `ValueError: label column 'Label' not found` | wrong file (not the cleaned data) | use `data/cleaned/cleaned.parquet` |
| `ValueError: non-numeric feature columns` | the raw data was used | use the cleaned data |
| `ValueError: X_train contains missing or infinite values` | a bug in a step | send me the message |
| `check_preprocessed_fresh` says "settings changed" | a setting in `preprocessing_config()` changed | rerun `scripts/04_preprocess.py` |

## Decisions and limitations

- **Positions, then fit on the training rows:** the cleaned data is never copied as a whole; only the float32 training and test parts are.
- **Sample flows come from the test rows,** so they are never used to fit anything; they are for demonstrations in the interface.
- **Binary mode** works with the same code (the codec then has two classes).
- **`preprocessing_metadata.json` is written last,** so a run interrupted halfway leaves no file that claims to be complete.