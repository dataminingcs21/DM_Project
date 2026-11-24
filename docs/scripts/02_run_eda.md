# `scripts/02_run_eda.py`

**Location:** `DM_Project/scripts/02_run_eda.py`
**Roadmap step:** File 15 (Stage 2, EDA)

## Purpose

Runs the whole exploratory data analysis on the raw dataset: loads the CSV files, computes the statistics (File 13), draws the figures (File 14) and writes a readable summary. It works on the **uncleaned** data and never changes it. The cleaning rules are decided afterwards, at Checkpoint 2, using this report together with the inspection report.

## What it does

1. Loads all CSV files (`load_raw_data`, File 08).
2. Computes: column profile and class tables (inspector, File 11); descriptive statistics, outlier shares, correlation matrix and highly correlated pairs, per-class medians, ANOVA F ranking (File 13).
3. Estimates how many features a simple correlation filter would drop at thresholds 0.90, 0.95, 0.98 and 0.99 (helps to choose `CORRELATION_THRESHOLD`). The real filter (File 22) is fitted on the training data only and may differ slightly.
4. Draws 11 figures (File 14). Which features appear in the histograms and box plots is chosen from the data: the 12 highest F values, and the 12 most skewed features.
5. Writes `eda_report.md`.

## Options

| Option | Meaning |
|---|---|
| `--nrows N` | Read only the first N rows of each file (quick test) |
| `--raw-dir PATH` | Folder with the CSV files (default `data\raw`) |
| `--corr-rows N` | Rows used for the correlation matrix (default 500,000) |
| `--cap-per-class N` | Maximum rows per class for class comparisons (default 50,000) |

## Outputs (`reports/eda/`)

| Location | Content |
|---|---|
| `eda_report.md` | Summary: overview, class distribution, correlated pairs, threshold sensitivity, features that separate classes, skewed features, outliers, list of figures, the decisions this EDA informs |
| `tables/` | 10 CSV tables: `column_profile`, `label_distribution`, `label_by_file`, `descriptive_statistics`, `outlier_summary`, `correlation_matrix`, `high_correlation_pairs`, `correlation_threshold_sensitivity`, `class_medians`, `feature_class_separation` |
| `figures/` | 11 PNG files: `class_distribution`, `label_by_file`, `missing_and_infinite`, `outlier_percent`, `skewness`, `correlation_heatmap`, `top_f_statistics`, `class_profile`, `histograms_top_f`, `histograms_most_skewed`, `boxplots_top_f` |

Running it again overwrites these files. Exit code 0 on success; 1 (one `ERROR` line, no traceback) for missing files, differing columns, a missing label column, or fewer than two classes in the data.

## Dependencies

Files 04, 05, 07, 08, 11, 13 and 14 (and 06 through them); `matplotlib`, `seaborn`, `scikit-learn`.

## How to run

From `DM_Project\` with `.venv` active (PowerShell). Quick test first:

```powershell
python scripts/02_run_eda.py --nrows 5000
```

Then the full run. It reads about 2.8 million rows; allow several minutes and a few GB of RAM, so **close the browser and games first**:

```powershell
python scripts/02_run_eda.py
```

**Expected output:** loader log lines, `Computing statistics`, `Correlation on ... rows x 78 features`, `Drawing figures`, then `EDA finished: 10 tables, 11 figures` and the report path. Open the results:

```powershell
notepad reports\eda\eda_report.md
Invoke-Item reports\eda\figures\
```

In sandbox tests (synthetic CSVs) it created all tables and figures, adapted to a dataset with very few features, and exited cleanly with an error line for a single-class dataset. It has not yet run on the full real data.

## What I need from you

After the full run, send me the content of `reports\eda\eda_report.md` and, if you can, the images `correlation_heatmap.png`, `class_profile.png`, `histograms_most_skewed.png` and `class_distribution.png` (screenshots are fine). The decisions it informs are the scaler type, the correlation threshold, and any suspicious features.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ERROR ... At least two classes are needed` | `--nrows` too small: the first rows of every file are one class | Use a larger `--nrows` (5000 worked in the earlier tests) or the full run |
| `ERROR ... No files matching '*.csv'` / `Raw data folder not found` | CSVs not in `data\raw\` | Move them there |
| `ERROR ... columns differ from ...` | A file has another structure | Send me the message |
| `MemoryError` / the PC freezes | Full data plus temporary copies | Close other programs; lower `--corr-rows` and `--cap-per-class` |
| `PermissionError` writing a file | A report or image is open in another program | Close it and rerun |
| `ModuleNotFoundError` | `.venv` not active or packages missing | Activate `.venv`; `python -m pip install -r requirements.txt` |

## Decisions and limitations

- **Uncleaned data on purpose:** the EDA should show the real problems (infinite values, negatives, duplicates) so the cleaning rules are justified.
- **Sampling for the expensive parts** (correlation, class comparisons); per-column statistics use every row.
- **The threshold sensitivity is an estimate** from a simple order-based rule, not the final filter.
- **The F ranking and the plots are exploration,** not feature selection, and they are not used in training.
- **The features shown in histograms and box plots are chosen by the data** (F and skewness), so a different dataset gives different figures without any code change.