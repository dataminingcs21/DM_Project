# `src/nids/eda/eda_stats.py`

**Location:** `DM_Project/src/nids/eda/eda_stats.py`
**Roadmap step:** File 13 (Stage 2, EDA)

## Purpose

Computes the exploratory statistics as tables (the plots are made in `eda_plots.py`, File 14). It runs on the loaded, **uncleaned** data. Infinite values are treated as missing *inside each calculation only*: the input DataFrame is never modified, and nothing here trains a model or feeds the preprocessing. The results inform decisions that are later written in `config/settings.py` (scaler type, correlation threshold, which features look uninformative).

## Public contents

| Function | Uses | Returns |
|---|---|---|
| `descriptive_statistics(df)` | all rows | Per feature, on finite values: `count`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, `skewness` |
| `outlier_summary(df, k=1.5)` | all rows | Per feature: `n_outliers` and `outlier_percent` outside `[Q1 - k*IQR, Q3 + k*IQR]`. A description only; nothing is removed, because attacks are often the "outliers" |
| `correlation_matrix(df, max_rows=500_000)` | random sample | Pearson correlation matrix; each pair uses the rows where both values exist; constant columns give `NaN` |
| `high_correlation_pairs(corr, threshold=None)` | the matrix | Pairs with `abs(r) >= threshold` (default `settings.CORRELATION_THRESHOLD`, 0.95), strongest first |
| `class_medians(df, cap_per_class=50_000)` | class-capped sample | Median of each feature per class (rows = classes) |
| `feature_class_separation(df, cap_per_class=50_000)` | class-capped sample | ANOVA F statistic of each feature against the class label, highest first |
| `random_sample`, `class_capped_sample` | | The two sampling helpers |

**Sampling policy.** Per-column statistics use every row. The expensive ones use samples (taken before the columns are selected, to save RAM): a plain random sample for the correlation, and a class-capped sample (at most `cap_per_class` rows per class, rare classes kept whole) for the class comparisons, so rare attacks are not lost.

**Reading `feature_class_separation` correctly.** A higher F means the class means differ more relative to the spread inside classes. It is an exploration ranking, **not** feature selection, and it only sees differences in the mean. Missing and infinite values are replaced by the column median for this calculation only. Constant columns give `NaN`. Because the sample is class-capped, rare classes weigh more than in the real data.

## Inputs and outputs

- **Input:** the DataFrame from `loader.py` (numeric features, `Label`, `source_file`). Raises `ValueError` if the label is missing, a feature is not numeric, or fewer than two classes exist for the F ranking.
- **Output:** DataFrames. Nothing is written to disk; `scripts/02_run_eda.py` (File 15) saves them.

## Dependencies

Files 04, 05 and 07; `numpy`, `pandas`, `scikit-learn` (`f_classif`).

## How to test

On a slice of the real data (from `DM_Project\`, `.venv` active, PowerShell):

```powershell
python -c "from nids.data.loader import load_raw_data; from nids.eda.eda_stats import *; df, _ = load_raw_data(nrows=5000); print(descriptive_statistics(df).head()); c = correlation_matrix(df); print(high_correlation_pairs(c).head(10)); print(feature_class_separation(df).head(10))"
```

**Expected output:** loader log lines, a descriptive table of 5 rows, a log line `Correlation on 40000 rows x 78 features`, a (possibly short) table of highly correlated pairs, and the 10 features with the highest F. I do not assume which features appear. If `Fwd Header Length` and `Fwd Header Length.1` are duplicate columns, expect them in the pairs table with correlation 1.0.

In my sandbox tests (synthetic data), a correlated pair and a class-separating feature were found as designed, the input frame was unchanged, and 400,000 rows x 78 columns took about 1 to 2 seconds per function. The full dataset has about 7 times more rows, so the per-column functions will take longer.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: Non-numeric feature columns` | Data not loaded with `loader.py` | Use `load_raw_data` |
| `ValueError: At least two classes are needed` | A slice with one class | Use more rows |
| `MemoryError` on the full dataset | Heavy operations on the whole data | Close other programs; the sampling already limits the heavy parts |
| Many `NaN` in the correlation matrix | Constant columns in the data (8 are known) | Expected; they are skipped in the pairs table |

## Decisions and limitations

- **Pearson only:** it measures linear relationships. The EDA will also show distributions, which reveal strong skew.
- **Sampled correlation:** 500,000 rows are plenty for a stable estimate on 78 features, but the values are estimates, not exact.
- **Outliers are described, not treated:** removing them would likely remove the attacks.
- **No mutual information:** it is slow on millions of rows; the F ranking is the cheap alternative, with the limitation above.
- **Tested on synthetic data only;** not yet run on the real files.