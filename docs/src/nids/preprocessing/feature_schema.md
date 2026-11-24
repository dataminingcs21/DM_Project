# `src/nids/preprocessing/feature_schema.py`

**Location:** `DM_Project/src/nids/preprocessing/feature_schema.py`
**Roadmap step:** File 24 (Stage 4, preprocessing)

## Purpose

Writes down, once, **which features exist and what values they normally take**: the "identity card" of the input data. It is built from the training rows and the fitted pipeline, and saved as `feature_schema.json` and `feature_names.json` (by File 27).

## Why it is needed

- The input validator (File 56) must know the expected columns and their typical ranges to check a flow typed into the interface.
- The Streamlit form (Files 62-63) needs the names, a default value (the training median) and a sensible range for each input.
- Reports must say which features the models really use and why the others were dropped.
- Nothing is invented: every number comes from the training data, every "reason" from the fitted `FeatureFilter`.

## Dependencies

`config/settings.py` (`FEATURE_SCHEMA_FILE`, `FEATURE_NAMES_FILE`, `NON_NEGATIVE_COLUMNS`), `io_utils.py` (File 06), `pipeline_builder.py` and `feature_filter.py` (Files 22-23, through the fitted pipeline), `numpy`, `pandas`, `scikit-learn`.

## What the schema contains

Top level: `schema_version`, `computed_on` ("training rows only"), `n_train_rows`, `n_raw_features`, `n_model_features`, `raw_feature_names` (the columns the pipeline expects, in order), `model_feature_names` (the columns the models receive), and `features`.

For each raw feature:

| Field | Meaning |
|---|---|
| `name`, `position` | column name and order |
| `used_by_models` | True if it survives the pipeline |
| `dropped_because` | `None`, or "constant in the training data", or "correlated with 'X' (r = ...)" |
| `min`, `p01`, `median`, `p99`, `max` | over the finite training values (`None` if there are none) |
| `all_whole_numbers` | True if every finite training value is an integer (counts, flags) |
| `n_missing_or_infinite_in_train` | how many values were missing or infinite before imputation |
| `must_be_non_negative`, `non_negative_reason` | from `NON_NEGATIVE_COLUMNS` (Checkpoint 1) |

## Functions

| Function | Role |
|---|---|
| `build_feature_schema(X_train, pipeline)` | builds and validates the schema. `X_train` is a DataFrame of the **raw** training features; the pipeline must be fitted on the same columns |
| `validate_schema(schema)` | raises `ValueError` if names, order, counts or `used_by_models` are inconsistent |
| `save_feature_schema(schema)` / `load_feature_schema()` | write / read the two JSON files (atomic writes, validated) |
| `schema_table(schema)` | the features as a flat table |
| `check_columns(schema, columns)` | `{"missing": [...], "unexpected": [...]}` compared with the raw features |

## Why the schema lists all raw features, not only the ones used

The pipeline takes the raw columns as input. A column that the filter drops has no effect on the output, so the interface will be able to ask only for the columns with `used_by_models = True` and leave the others missing (the imputer fills them, the filter removes them). Whether the form asks for the used columns only, or for all of them, is decided at File 62. The schema supports both.

## How to test

The automated tests come with File 26. A quick check on the real data, with the pipeline fitted on a **random** sample of 250,000 training rows (PowerShell, from `DM_Project\`, `.venv` active; about 1-2 GB of RAM). Note: the statistics and the dropped columns come from this sample only; the final schema is built in File 27 from all training rows.

```powershell
python -c "import numpy as np, pandas as pd; from config import settings; from nids.data import schema; from nids.preprocessing.target_builder import build_target; from nids.preprocessing.splitter import stratified_split; from nids.preprocessing.pipeline_builder import build_pipeline; from nids.preprocessing.feature_schema import build_feature_schema, schema_table; df = pd.read_parquet('data/cleaned/cleaned.parquet'); t, _ = build_target(df[settings.LABEL_COLUMN]); tr, te = stratified_split(t); cols = schema.get_feature_columns(df.columns); rows = np.sort(np.random.default_rng(0).choice(tr, 250000, replace=False)); X = df[cols].iloc[rows].astype('float32'); p = build_pipeline().fit(X); s = build_feature_schema(X, p); print(s['n_raw_features'], s['n_model_features']); tab = schema_table(s); print(tab[['name','used_by_models','dropped_because','min','median','max']].head(8).to_string()); print('dropped:', int((~tab['used_by_models']).sum()))"
```

**Expected output:** the log lines (pipeline built, filter fitted, `Feature schema built: 66 raw features, N used by the models`), then `66 N`, a table of the first 8 features with their minimum, median and maximum, and `dropped: K` where `K = 66 - N`. I do not know N for this sample; please paste the output.

Nothing is written to disk by this command.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... feature_schema` | empty placeholder | paste the code, check the length is not 0 |
| `ValueError: X_train must be a DataFrame` | an array was passed | pass the raw feature DataFrame |
| `ValueError: X_train columns differ from the columns the pipeline was fitted on` | different columns or order | use the same columns as for `fit` |
| `ValueError: the pipeline must be fitted` | `build_pipeline()` was not fitted | call `fit` first |
| `FileNotFoundError ... feature_schema.json` when loading | File 27 not run yet | run the preprocessing script first |

## Decisions and limitations

- **Training rows only:** the ranges never include the test rows.
- **A value outside `[min, max]` is a warning later, never an error and never proof of an attack:** a real attack may be extreme.
- **Percentiles (1st and 99th)** are given because the maximum of these features is often a single extreme flow; the form can use the percentiles for its default range.
- **Infinite or missing training values are excluded from the statistics** and counted separately.