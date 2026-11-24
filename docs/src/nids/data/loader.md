# `src/nids/data/loader.py`

**Location:** `DM_Project/src/nids/data/loader.py`
**Roadmap step:** File 08 (Stage 1, data loading)

## Purpose

Reads the raw CIC-IDS2017 CSV files from `data/raw/` and returns one DataFrame plus a report. It loads and checks the structure only. It does **not** clean: infinite values, missing values and duplicates are counted and handled later by `cleaning/cleaner.py` (File 16), after the data has been inspected.

## Public contents

| Function | What it does |
|---|---|
| `list_raw_files(raw_dir=None)` | Sorted `.csv` files in the folder (default `settings.RAW_DATA_DIR`). Other files (for example `.complete`) are ignored. Clear error if the folder or the CSV files are missing |
| `read_csv_file(path, nrows=None)` | Reads one file and returns `(DataFrame, file_report)` |
| `load_raw_data(raw_dir=None, nrows=None)` | Loads all files and returns `(DataFrame, report)` |
| `ENCODINGS` | `("utf-8", "cp1252", "latin-1")`, tried in this order |

**Steps applied to every file**
1. Read with the first encoding that works (the one used is recorded).
2. Normalize column names (File 07); stop if two columns collide after normalization.
3. Stop if the `Label` column is missing.
4. Clean label whitespace.
5. Convert all feature columns to numbers; values that cannot be converted become `NaN` and are counted per column. Text such as `Infinity` becomes `inf` and is **not** counted here (the cleaner counts infinite values).
6. Add the `source_file` column (the file name).

`load_raw_data` then checks every file has the same columns as the first file. A file with missing or extra columns stops the load with an error naming them, because concatenating it would silently create all-`NaN` columns. Columns are put in the same order, then all files are concatenated.

**Report** (`load_raw_data` second value): `n_files`, `n_rows`, `n_feature_columns`, and one entry per file with `file`, `rows`, `columns`, `encoding`, `non_numeric_values_coerced`.

## Inputs and outputs

- **Input:** the CSV files; optional folder and `nrows` (first N rows of each file, for quick tests).
- **Output:** a DataFrame (features in `float64`, `Label`, `source_file`) and the report dictionary. Nothing is written to disk.

## Dependencies

`config/settings.py` (File 04), `nids.data.schema` (File 07), `nids.utils.logger` (File 05), `pandas`. Used by `inspector.py` (File 11) and the cleaning script.

## How to test

First put the 8 CSV files in `DM_Project\data\raw\` (move them from `dataset\` if you have not yet). From `DM_Project/` with `.venv` active (PowerShell). The first run reads about 2.8 million rows, so allow a minute or two, so use the quick `nrows` version first:

```powershell
python -c "from nids.data.loader import load_raw_data; df, r = load_raw_data(nrows=1000); print(df.shape); print(r['n_files'], r['n_rows'], r['n_feature_columns']); print(df['source_file'].unique())"
```

**Expected output:** log lines for each file, then the shape, then the file/row/feature counts and the file names. With 8 files you should see `n_files = 8` and `n_rows = 8000` (1000 rows per file, if every file has at least 1000 rows). The number of feature columns is what the real data has; **tell me the number you get** (we will confirm it at Checkpoint 1; I am not assuming it).

Full load, once the quick one works:

```powershell
python -c "from nids.data.loader import load_raw_data; df, r = load_raw_data(); print(df.shape); print(df.memory_usage(deep=True).sum()/1e9, 'GB'); print(r['files'][0])"
```

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Raw data folder not found` or `No files matching '*.csv'` | CSVs are not in `data\raw\` | Move them there from `dataset\` |
| `ValueError: <file>: columns differ from <first file>` | A file has a different structure | Send me the message; the real difference decides the fix |
| `ValueError: ... label column 'Label' not found` | The label column has another name | Check the header; update `LABEL_COLUMN` in settings if needed |
| `ValueError: ... duplicate column names after normalization` | Two headers differ only by spaces | Send me the names; we decide which to keep |
| `MemoryError` or slow machine on the full load | About 2.8 M rows x 80 columns in `float64` plus temporary copies | Close other programs (browser, games); use `nrows` for tests |
| Odd characters in labels (`?` or a different dash) | A file's label text uses a legacy encoding | Reported at Checkpoint 1; we decide the label mapping explicitly |

## Decisions and limitations

- **`float64` at load time,** `float32` later. Duplicate detection in the cleaner should compare exact values; casting to `float32` first could make different rows look identical.
- **Encoding fallback** (`utf-8`, then `cp1252`, then `latin-1`) is chosen per file. If one file mixes encodings, some characters could be wrong. The report shows which encoding each file needed.
- **Strict column check:** the first file (alphabetical) is the reference. This is deliberate: it catches structural differences rather than hiding them.
- **Whole dataset in RAM** (a few GB). Acceptable with 16 GB; if it proves too tight we will switch to per-file processing (the `data/interim/` folder exists for this).
- **Not here:** cleaning (File 16), inspection statistics (File 11).