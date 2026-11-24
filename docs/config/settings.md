# `config/settings.py`

**Location:** `DM_Project/config/settings.py`
**Roadmap step:** File 04 (Stage 0, foundations)

## Purpose

The single source of truth for configuration. Every path, the random seed, the target mode, the split ratio and the thresholds are defined here, and no other module hardcodes them. It is the first Python file of the project and everything else imports it.

## Public contents

### Directories (all built with `pathlib` from `PROJECT_ROOT`)

| Constant | Folder | Used for |
|---|---|---|
| `RAW_DATA_DIR` | `data/raw/` | The 8 original CSV files |
| `INTERIM_DATA_DIR` | `data/interim/` | Per-file cleaned data |
| `CLEANED_DATA_DIR` | `data/cleaned/` | Combined cleaned dataset and cleaning report |
| `PREPROCESSED_DIR` | `data/preprocessed/` | Train/test arrays, fitted transformers, schema, metadata |
| `TRAINED_MODELS_DIR`, `TUNED_MODELS_DIR` | `artifacts/models/trained/`, `.../tuned/` | Baseline and tuned models |
| `INSPECTION_REPORT_DIR`, `EDA_FIGURES_DIR`, `EDA_TABLES_DIR`, `EVALUATION_REPORT_DIR`, `OPTIMIZATION_REPORT_DIR` | `reports/...` | Generated reports and figures |
| `LOGS_DIR` | `logs/` | Log files |

Fixed file locations (for example `CLEANED_DATA_FILE`, `X_TRAIN_FILE`, `PIPELINE_FILE`, `LABEL_ENCODER_FILE`, `FEATURE_SCHEMA_FILE`, `PREPROCESSING_METADATA_FILE`) are defined here too, so file names are never repeated in other modules.

### Settings

| Setting | Default | Meaning | Status |
|---|---|---|---|
| `TARGET_MODE` | `"multiclass"` | `"multiclass"` (BENIGN + each attack) or `"binary"` | Chosen by you |
| `LABEL_COLUMN`, `BENIGN_LABEL`, `MALICIOUS_LABEL` | `"Label"`, `"BENIGN"`, `"MALICIOUS"` | Dataset label names | Verify at Checkpoint 1 |
| `SOURCE_FILE_COLUMN` | `"source_file"` | Metadata column added by the loader; never a feature | Fixed |
| `RAW_FILE_PATTERN` | `"*.csv"` | Only CSV files are loaded; other files such as `.complete` are ignored | Fixed |
| `RANDOM_SEED` | `42` | Reproducibility | Fixed |
| `TEST_SIZE` | `0.20` | Fraction held out for testing | Chosen by you |
| `CV_FOLDS` | `5` | `StratifiedKFold` folds | Default |
| `MIN_CLASS_SAMPLES` | `7` (derived) | Technical minimum rows per class: `max(2, ceil(CV_FOLDS / (1 - TEST_SIZE)))`, so a class can appear in both splits and in every CV fold | Derived |
| `RARE_CLASS_POLICY` | `"keep"` | `"keep"` or `"merge"` (merge classes below the minimum into `RARE_CLASS_MERGE_LABEL`) | Decide at Checkpoint 2 |
| `IMPUTER_STRATEGY` | `"median"` | Missing-value strategy (never a blanket fill with zero) | Default |
| `SCALER_TYPE` | `"standard"` | `"standard"`, `"robust"` or `"minmax"` | Kept after the EDA (Checkpoint 2) |
| `LOG_TRANSFORM` | `True` | Signed `log1p` before the feature filter and the scaler (Checkpoint 2 decision, added in File 23). Part of `preprocessing_config()`, so changing it makes preprocessed data stale | Checkpoint 2 |
| `CORRELATION_THRESHOLD` | `0.95` | Correlated-feature removal threshold | Revisit after the EDA |
| `EXCLUDED_FEATURES` | `{}` | `{column: reason}`; every exclusion must have a written technical reason | Fill at Checkpoint 1 |
| `LABEL_MAP` | the three `Web Attack � X` labels mapped to `Web Attack - X` | Explicit whole-label fixes applied by the cleaner. The `�` character is stored in the CSV files themselves, so the original character is lost | Checkpoint 1 (added in File 16) |
| `NON_NEGATIVE_COLUMNS` | 11 columns, each with a reason | Columns where a negative value is impossible (durations, rates, inter-arrival times, lengths). `Init_Win_bytes_*` are deliberately not listed because their `-1` behaves like a placeholder code | Checkpoint 1; confirm at Checkpoint 2 |
| `NEGATIVE_VALUE_POLICY` | `"to_nan"` | `"to_nan"`: set those negatives to missing (imputed later on the training data). `"keep"`: count and report only | Confirm at Checkpoint 2 |
| `DROP_IDENTICAL_COLUMNS` | `True` | Drop a column whose values are identical to an earlier column (lossless) | Default |
| `CONFLICT_POLICY` | `"keep"` | Identical feature vectors with different labels: `"keep"` (report only) or `"drop"` (remove all their rows) | **Open until Checkpoint 2** |
| `N_JOBS` | all CPU threads minus 2 (14 on your machine) | Parallel workers for CPU-bound steps | Chosen by you |
| `XGBOOST_DEVICE` | `"auto"` | `"auto"` (detect the GPU, else CPU), `"cuda"` or `"cpu"` | Default |
| `CLASS_WEIGHT_MODE` | `"none"` | Class-weighting experiment; off for the baselines | Decide after Checkpoint 4 |
| `SUBSAMPLE_SIZES` | KNN 100 000, SVM 50 000, Gradient Boosting 100 000 | Training rows for slow models | **Placeholders**, calibrated at step 47 |
| `PRIMARY_METRIC` | `"f1_macro"` | Main comparison and tuning metric | Fixed (Deliverable 6) |
| `TIE_BREAK_RULE` | `"prefer_non_benign"` | Tie rule for the majority vote | Default |
| `FLOAT_DTYPE` | `"float32"` | Numeric dtype, to save RAM | Fixed |
| `LOG_LEVEL`, `LOG_FILE_NAME` | `"INFO"`, `"nids.log"` | Logging | Default |

### Functions

| Function | What it does |
|---|---|
| `ensure_directories()` | Creates every output directory that is missing. Returns the list of directories it created, so a second call returns an empty list |
| `all_directories()` | The list of directories the project writes into |
| `cleaning_config()` | (added in File 16) A dictionary of the settings that change the content of `data/cleaned/`: label map, non-negative columns, policies. The cleaning script stores it with its hash next to the cleaned data, so later stages can detect data cleaned with other rules |
| `preprocessing_config()` | A dictionary of the settings that change the content of `data/preprocessed/`. File 06 hashes it into `preprocessing_metadata.json`, so changing, for example, `SCALER_TYPE` makes the saved preprocessed data stale and the training scripts refuse to use it |

Importing the module validates every setting (`_validate_settings()`). An invalid value, such as `TARGET_MODE = "ternary"`, raises a `ValueError` that names the setting and the allowed values.

## Inputs and outputs

- **Inputs:** the optional environment variable `NIDS_PROJECT_ROOT` (a folder path). If it is set, it replaces the default project root. The tests use it to work in a temporary folder. It must be set before the module is first imported.
- **Outputs:** constants and helper functions imported by every other module. `ensure_directories()` creates folders.

## Dependencies

Python standard library only (`math`, `os`, `pathlib`). It does not import any other project file, so no circular imports are possible.

## How to run and test

From `DM_Project/` with the `.venv` active:

```powershell
python -m config.settings
python -c "import config.settings as s; print(len(s.ensure_directories()), 'folders created'); print(len(s.ensure_directories()), 'created on the second call')"
```

**Expected output of the first command** (the root path and `N_JOBS` will match your machine):

```
PROJECT_ROOT      : C:\...\DM_Project
RAW_DATA_DIR      : C:\...\DM_Project\data\raw  (exists: True)
TARGET_MODE       : multiclass
RANDOM_SEED       : 42
TEST_SIZE         : 0.2
CV_FOLDS          : 5
MIN_CLASS_SAMPLES : 7
N_JOBS            : 14
SCALER_TYPE       : standard
XGBOOST_DEVICE    : auto
Directories to create: 12
```

`N_JOBS` is 14 if Python sees 16 threads. **Expected output of the second command:** `11 folders created` (the 12 directories minus `data/raw`, which already exists) and then `0 created on the second call`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | `.venv` not active, or `pip install -e .` not run (File 03) | Activate `.venv`; run `pip install -e .` |
| `ValueError: TARGET_MODE='...' is invalid; allowed values: ...` | A typo in a setting | Use one of the allowed values listed in the message |
| `PROJECT_ROOT` points to the wrong folder | An old `NIDS_PROJECT_ROOT` environment variable is set | Clear it: `Remove-Item Env:NIDS_PROJECT_ROOT` |
| `NON_NEGATIVE_COLUMNS[...] needs a technical justification` or `LABEL_MAP[...] maps to an empty label` | An empty reason or target was entered | Write the reason / the new label |
| `NEGATIVE_VALUE_POLICY='...' is invalid` or `CONFLICT_POLICY='...' is invalid` | A typo in the policy | Use one of the allowed values in the message |
| A new `EXCLUDED_FEATURES` entry fails with "needs a technical justification" | The reason text is empty | Write the reason: `{"Destination Port": "why it is excluded"}` |
| Saved preprocessed data is reported as stale after editing this file | You changed a value included in `preprocessing_config()` | Run `python scripts/04_preprocess.py --force` (available from File 27) |

## Decisions and limitations

- **Plain Python constants,** not a settings class or a YAML file: simplest to import and fully typed by the editor. The cost is that tests must set `NIDS_PROJECT_ROOT` before the first import.
- **`MIN_CLASS_SAMPLES` is derived** from `CV_FOLDS` and `TEST_SIZE` instead of invented. With the defaults it is 7, which is a technical lower bound, not a guarantee of reliable metrics. Classes with only a handful of rows will still produce weak statistics, and the report will say so.
- **`N_JOBS`** is computed from the machine at import time. On a machine with 2 or fewer threads it falls back to 1.
- **`SUBSAMPLE_SIZES` are placeholders** until they are timed at step 47.
- **`EXCLUDED_FEATURES` is empty on purpose.** No feature is excluded without a reason, and we have not yet seen the real columns.
- **Not here:** the model hyperparameters and grids (File 28, `model_params.py`) and the metadata-column list (File 07, `schema.py`).