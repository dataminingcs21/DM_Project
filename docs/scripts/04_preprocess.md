# `scripts/04_preprocess.py`

**Location:** `DM_Project/scripts/04_preprocess.py`
**Roadmap step:** File 27 (Stage 4, **Checkpoint 3**)

## Purpose

Runs the preprocessing of File 25 on the full cleaned data and **saves everything the training stage needs** to `data/preprocessed/`. After saving, it re-reads what it wrote and runs automatic checks.

## Why it is needed

It is the only step that writes the preprocessed data. Training (Files 45-47), evaluation and the Streamlit interface all read these files, so this is the last point where a mistake can still be found cheaply.

## Dependencies

Files 06, 18 (`data/cleaned/cleaned.parquet`), 22-25, and `nids.eda.eda_stats` (File 13, only for the dry-run sample).

## What it does

1. Reads `cleaned.parquet`. It refuses to continue if the cleaning rules in `config/settings.py` changed since the cleaning (the hash in `cleaning_report.json` differs); with no cleaning report it only warns.
2. Runs `preprocess` (target, split, pipeline fitted on the training rows, schema, sample flows).
3. Saves with `save_preprocessed`; `preprocessing_metadata.json` is written last.
4. Runs 10 automatic checks (from disk): disjoint and complete split; every class on both sides; no label or `source_file` among the features; no missing or infinite values; training data centred; class codes in range; the saved arrays equal the arrays in memory; the saved label encoder equals the one in memory; the saved data is fresh for the current settings; **the saved pipeline reproduces rows of `X_test` from raw sample flows**.
5. Writes the report and two CSV files.

Exit code: 0 = done and all checks PASS; 1 = error (missing file, stale cleaning, bad option); 2 = saved, but a check failed.

## Options

| Option | Meaning |
|---|---|
| `--dry-run` | run everything in memory, write **nothing**, print the report |
| `--sample-per-class N` | use at most N rows per class (small classes kept whole). Allowed **only with `--dry-run`** (and N >= 50), so a quick test can never overwrite the real preprocessed data |

## Outputs (`data/preprocessed/`)

`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `preprocessing_pipeline.joblib`, `label_encoder.joblib`, `feature_schema.json`, `feature_names.json`, `sample_flows.parquet`, `preprocessing_metadata.json`, and (new in this file) `preprocessing_report.md` (**send me this one**), `preprocessing_class_counts.csv`, `preprocessing_features.csv`. About 0.5 GB in total.

## How to run

From `DM_Project\` with the virtual environment active (PowerShell). First a quick dry run (writes nothing, about 2-3 GB of RAM because it still reads the whole cleaned file), then the real run (about 6 GB; close the browser and games):

```powershell
python scripts/04_preprocess.py --dry-run --sample-per-class 2000
python scripts/04_preprocess.py
notepad data\preprocessed\preprocessing_report.md
```

## Expected output (real run)

```
... | INFO | nids.preprocessing.preprocessor | Preprocessing 2522362 rows, 66 feature columns
... | INFO | nids.preprocessing.splitter | Split: 2017889 train rows, 504473 test rows (test_size=0.20, seed=42)
... | INFO | nids.preprocessing.feature_filter | Feature filter fitted: 66 columns in, 40 kept (1 constant, 25 correlated dropped; threshold 0.950)
... | INFO | nids.preprocessing.preprocessor | Preprocessing done in ~30 s: X_train (2017889, 40), X_test (504473, 40)
... | INFO | nids.scripts.preprocess | Saved to ...\data\preprocessed
... | INFO | nids.scripts.preprocess | Checks passed: 10/10
```

The numbers (40 kept, 2,017,889 / 504,473) are what your File 25 run showed. Saving and the re-reading checks add some seconds. In a dry run you get 7 checks (the three disk checks are skipped) and the whole report is printed instead of saved.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `File not found: ...cleaned.parquet` | File 18 not run | `python scripts/03_clean_data.py` |
| `The cleaned data was made with other cleaning rules...` | cleaning settings changed after File 18 | rerun `scripts/03_clean_data.py` |
| `--sample-per-class ... only allowed together with --dry-run` | protection against overwriting | add `--dry-run` |
| exit code 2 and `Failed checks: [...]` | a check failed after saving | send me the log and the report |
| `MemoryError` / freeze | less than about 6 GB free | close other programs |
| `PermissionError` on an `.npy` file | file open in another program (Windows) | close it and rerun |
| `ImportError` / empty module | placeholder file not filled | paste the code, check the length is not 0 |

## Decisions and limitations

- **A dry run never writes,** and the sample option exists only in a dry run: the real outputs are always the full data.
- **The report states the checks as PASS/FAIL** so you do not have to trust my words: the check "saved pipeline reproduces `X_test` rows from raw sample flows" is the guarantee that the interface will give the same numbers as the batch preprocessing.
- **Checkpoint 3** asks you to look at three things in the report: the class counts of the rare classes, the dropped features, and that all checks read PASS.
- Running the script again replaces the preprocessed files (and the trained models become stale, a check added at File 47).