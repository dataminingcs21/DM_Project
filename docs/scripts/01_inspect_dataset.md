# `scripts/01_inspect_dataset.py`

**Location:** `DM_Project/scripts/01_inspect_dataset.py`
**Roadmap step:** File 12 (Stage 1) **= Checkpoint 1**

## Purpose

The first script you run on the real data. It loads the 8 CSV files, inspects them and writes a report. **Nothing is cleaned or changed.** The report is the basis of the Checkpoint 1 decisions: rare classes, excluded features, odd labels, duplicates.

## What it does

1. Creates the project folders if missing (`settings.ensure_directories()`).
2. `load_raw_data` (File 08) reads and checks the CSV files.
3. `inspect_dataset` (File 11) computes the statistics.
4. Writes the results to `reports/inspection/`.

## Options

| Option | Meaning |
|---|---|
| `--nrows N` | Read only the first N rows of each file (quick test) |
| `--raw-dir PATH` | Folder with the CSV files (default `data\raw`) |

## Outputs (`reports/inspection/`)

| File | Content |
|---|---|
| `inspection_report.md` | Readable report: overview, label distribution, label x file table, values, duplicates, candidate columns to exclude, the Checkpoint 1 decision list |
| `inspection_summary.json` | The same figures in machine-readable form |
| `column_profile.csv` | One row per feature: missing / infinite / negative counts, unique values, finite min and max, constant flag |
| `label_distribution.csv` | Rows per class, percentage, number of files containing it, rarity flag |
| `label_by_file.csv` | Class x source-file counts |

**Exit code:** 0 on success; 1 (with an `ERROR` log line, no traceback) when the folder or CSV files are missing, columns differ between files, or the label column is not found.

The "candidates for `EXCLUDED_FEATURES`" section is a **name-based hint** (columns whose name contains words like port, ip, id, timestamp). It is a suggestion to discuss, not a decision.

## Dependencies

Files 04, 05, 06, 08 and 11 (and 07 through them). Run `pip install -e .` first so `config` and `nids` can be imported.

## How to run

From `DM_Project\` with `.venv` active (PowerShell):

```powershell
python scripts/01_inspect_dataset.py --nrows 2000
python scripts/01_inspect_dataset.py
```

Run the quick one first. The full run reads about 2.8 million rows; allow a few minutes and a few GB of RAM (close the browser and games).

**Expected output:** loader log lines (one per file), `Inspecting ... rows x ... feature columns`, then `Report written to ...\reports\inspection` and a final line `Rows: ... | features: ... | classes: ...`. Open `reports\inspection\inspection_report.md`.

## What I need from you (Checkpoint 1)

Send me the content of `inspection_report.md` (paste it, or send the file). I do not assume any of the numbers or class names. With it we decide, together:

1. `RARE_CLASS_POLICY` (`keep` or `merge`) for the classes below the minimum.
2. `EXCLUDED_FEATURES` (if any), each with a written reason.
3. What to do with odd characters in labels.
4. The rule for duplicates and conflicting labels (used by the cleaner, File 16).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` / `'nids'` | `.venv` not active or `pip install -e .` not run | Activate; install |
| `ERROR ... Raw data folder not found` / `No files matching '*.csv'` | CSVs not in `data\raw\` | Move them there |
| `ERROR ... columns differ from ...` | A file has another structure | Send me the message |
| `MemoryError` / the PC freezes | Full data plus temporary copies | Close other programs; keep `--nrows` for tests |
| `PermissionError` writing the report | A report file is open in Excel or an editor | Close it and rerun |

## Decisions and limitations

- **Read-only on the data;** only report files are written, so running it again simply overwrites them.
- **Errors are logged, not shown as tracebacks,** for the expected problems (missing files, structure). Unexpected errors still show a full traceback.
- **No plots here.** Visual EDA comes in Files 13–15 after Checkpoint 1.
- **Tested on synthetic CSVs only** (report content, options, error exits). It has not been run on the real files.