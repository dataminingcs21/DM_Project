# `scripts/03_clean_data.py`

**Location:** `DM_Project/scripts/03_clean_data.py`
**Roadmap step:** File 18 (Stage 3, **Checkpoint 2**)

## Purpose

Runs the cleaner (File 16) on the whole raw dataset and saves the result. It is the first step that writes a real data product: `data/cleaned/`.

## What it does

1. Creates the project folders if needed.
2. Loads the 8 raw CSV files (loader, File 08).
3. Runs a **diagnostic** on the raw data: how many rows have a missing or infinite value, and how many of them have `Flow Duration` = 0 (infinite rates usually come from a division by a zero duration). Nothing is changed because of it; it only goes into the report.
4. Runs `clean_dataset` (label fixes, identical columns, impossible negatives, infinite values, exact duplicates, conflicting vectors).
5. Frees the raw data, then writes the four output files.

Missing values are **left in place** on purpose: they are imputed with the training median after the split (File 23), so the test data never influences them.

## Outputs (`data/cleaned/`)

| File | Content |
|---|---|
| `cleaned.parquet` | cleaned data: feature columns (float64), `Label`, `source_file` |
| `cleaning_report.json` | every number, the cleaning rules and their hash (`config_hash`) |
| `cleaning_report.md` | the same, readable. **This is the file I need for Checkpoint 2** |
| `cleaning_class_counts.csv` | rows per class before and after |

The raw files are never modified.

## Dependencies

Files 04 (revised), 05, 06, 07, 08, 16 and `numpy`, `pandas`, `pyarrow`.

## Options

| Option | Meaning |
|---|---|
| `--nrows N` | read only the first N rows of each file (quick test) |
| `--raw-dir PATH` | another folder with CSV files (default `data/raw`) |

## How to run

From `DM_Project\` with the virtual environment active (PowerShell):

```powershell
python scripts/03_clean_data.py --nrows 5000
python scripts/03_clean_data.py
notepad data\cleaned\cleaning_report.md
```

First a quick test on a slice, then the full run. Close the browser and games before the full run: it peaks at about **6-7 GB of RAM** and should take **about 1-2 minutes** (measured on 1,000,000 rows: about 6 s and 2.7 GB; the loading time of the CSV files comes on top).

## Expected output

Last lines of the log (numbers differ with `--nrows`):

```
... | INFO | nids.scripts.clean_data | Writing the cleaned data (N rows x M columns)
... | INFO | nids.scripts.clean_data | Rows: ... in, ... out | feature columns: ... in, ... out
... | INFO | nids.scripts.clean_data | Saved to ...\data\cleaned
```

Exit code 0. On the full data I expect 2,830,743 rows in and 78 feature columns in; the rows out and the 77 (or fewer) columns out are what the report will show. I do not know the final numbers yet; I will not guess them.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ERROR ... no CSV files` | CSVs not in `data\raw` | Check the folder, or use `--raw-dir` |
| `ERROR ... columns differ` | Files with different headers in the folder | Keep only the 8 original files |
| `MemoryError` or the computer freezes | Less than about 7 GB free | Close other programs; test with `--nrows 200000` to confirm the code works |
| `PermissionError` on `cleaned.parquet` | File open in another program | Close it and rerun |
| `ImportError` / `NameError` for `clean_dataset` | `cleaner.py` is an empty placeholder | Paste the File 16 code, check the file length is not 0 |
| `ModuleNotFoundError: pyarrow` | Dependencies not installed | `pip install -e .` |

## Decisions and limitations

- **No imputation, no scaling here:** both are fitted on the training part only (Files 23, 25).
- **float64 is kept** in the parquet file (no float32 cast) so no precision is lost before the split.
- **The diagnostic is informational only.** The cleaning rules do not depend on it.
- **Checkpoint 2:** the report ends with three decisions to confirm from the real numbers: `RARE_CLASS_POLICY`, `CONFLICT_POLICY`, `NEGATIVE_VALUE_POLICY`.
- With `--nrows` the output files **replace** the full ones; rerun without it before moving on.