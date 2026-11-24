# `src/nids/eda/eda_plots.py`

**Location:** `DM_Project/src/nids/eda/eda_plots.py`
**Roadmap step:** File 14 (Stage 2, EDA)

## Purpose

Draws the EDA figures as PNG files from the tables of `inspector.py` (File 11) and `eda_stats.py` (File 13), and from the data. Each `plot_*` function saves **one** figure, closes it (memory does not grow) and returns its path. `scripts/02_run_eda.py` (File 15) calls them all.

The non-interactive `Agg` backend is selected on import: figures are written to files, which also works with no display. Streamlit can show Agg figures, so the app is not affected.

## Public contents

| Function | Figure | Built from |
|---|---|---|
| `plot_class_distribution(distribution, path)` | Rows per class, log scale, with counts and percentages | `label_distribution` table |
| `plot_label_by_file(crosstab, path)` | Heatmap of class x source file; colour is log-scaled, numbers are the counts | `label_by_file` table |
| `plot_missing_and_infinite(profile, path)` | Columns with missing or infinite values (a message figure if there are none) | `column_profile` table |
| `plot_outlier_percent(outliers, path, top=25)` | Features with the most values outside the 1.5 x IQR fences | `outlier_summary` |
| `plot_skewness(descriptive, path, top=25)` | Most skewed features | `descriptive_statistics` |
| `plot_correlation_heatmap(corr, path)` | Correlation matrix (constant columns appear blank) | `correlation_matrix` |
| `plot_top_f_statistics(separation, path, top=20)` | Features ranked by ANOVA F against the label | `feature_class_separation` |
| `plot_class_profile(class_medians, features, path)` | Per-class medians standardised across classes (red = high for this class compared with the others, blue = low) | `class_medians` |
| `plot_feature_histograms(df, features, path, max_rows=200_000, ncols=3)` | Histograms on a random sample, finite values only, 1st to 99th percentile; the title shows the skewness. Maximum 36 features per figure | the data |
| `plot_feature_by_class(df, features, path, cap_per_class=2_000)` | Box plots per class on a class-capped sample, symmetric-log axis, no outlier dots | the data |

**Reading the plots**
- The class chart uses a log axis because the classes differ by several orders of magnitude; bar lengths are not proportional to the row counts.
- Box plots hide outlier dots and use a symmetric-log axis so heavy-tailed features remain readable; this shows the bulk of the values, not the extremes.
- The class profile uses z-scores over the classes, so it shows what is *relatively* high or low, not actual values.

## Inputs and outputs

- **Input:** DataFrames and a target path. The path's folder is created if missing.
- **Output:** a PNG file (150 dpi). Raises `ValueError` for an empty feature list or more than 36 histograms.

## Dependencies

Files 04, 05, 06 and 13; `matplotlib`, `seaborn`, `numpy`, `pandas` (all pinned in `requirements.txt`).

## How to test

On a slice of the real data (from `DM_Project\`, `.venv` active, PowerShell):

```powershell
python -c "from nids.data.loader import load_raw_data; from nids.data.inspector import label_distribution; from nids.eda.eda_plots import plot_class_distribution, plot_feature_histograms; df, _ = load_raw_data(nrows=5000); print(plot_class_distribution(label_distribution(df), 'reports/eda/figures/test_class.png')); print(plot_feature_histograms(df, ['Destination Port', 'Flow Duration', 'Total Fwd Packets'], 'reports/eda/figures/test_hist.png'))"
```

**Expected output:** the loader log lines, then two printed file paths. Open the two PNG files in `reports\eda\figures\` to check them. Delete `test_class.png` and `test_hist.png` afterwards (File 15 makes the real ones).

In my sandbox tests all 10 figure types plus the "nothing to show" case were created from synthetic data, no figures were left open, and I looked at the images to check that labels, scales and colours are readable. The real 78-feature heatmaps have not been seen yet.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'seaborn'` | Requirements not installed | `pip install -r requirements.txt` |
| `ValueError: No features given` / `At most 36 histograms` | Wrong list | Pass 1 to 36 feature names |
| `KeyError: 'Destination Port'` (or another name) | Column not in the data | Use names from the report; they are case and space sensitive |
| `PermissionError` when saving | The PNG is open in another program | Close it and rerun |
| Blank or cut-off labels in a huge heatmap | 78 x 78 labels are small by design | Zoom in on the PNG; labels are at 150 dpi |

## Decisions and limitations

- **Static PNGs, not interactive charts:** simplest, reproducible, and they can go straight into a report.
- **Sampling for histograms and box plots:** a histogram of 200,000 values looks the same as one of 2.8 million; box plots use up to 2,000 rows per class so rare classes are not drowned out.
- **Skewed features stay hard to see in histograms** even after clipping at the 99th percentile; that is informative (it shows the skew), and the skewness number is printed in each title.
- **Plain styling:** one colour for bars, diverging colour maps only for correlation and class profile.