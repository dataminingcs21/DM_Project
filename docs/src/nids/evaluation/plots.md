# `src/nids/evaluation/plots.py`

**Location:** `DM_Project/src/nids/evaluation/plots.py`
**Roadmap step:** File 50 (Stage 7, evaluation)

## Purpose

Draws the evaluation figures from the dictionaries and tables of `metrics.py` and `cross_validation.py`. Each function writes one PNG and returns its path.

## Why it is needed

Numbers in tables are exact, but figures show patterns at a glance: which attacks are mistaken for normal traffic, which classes are weak, which models are close, and how much a score moves between folds. The figures are drawn from the same tables, so a figure can never disagree with a number in the report.

## Dependencies

`matplotlib` (Agg backend: no window opens), `numpy`, `pandas`.

## The five figures

| Function | What it shows | Input |
|---|---|---|
| `plot_confusion_matrix(metrics, path, title=None)` | where each true class went, as a percentage of that class (each row sums to 100%). The row labels carry the number of true rows | the dictionary of `compute_metrics` / `evaluate_model` |
| `plot_per_class_metrics(metrics, path, title=None)` | precision (blue circle) and recall (orange diamond) for every class, with the support printed; **low-support rows are shaded and marked** | same |
| `plot_model_comparison(table, path, value=..., error=..., ...)` | one score per model as bars from zero, best on top, with a whisker for the spread over the folds; an asterisk marks models trained on a subsample; bold marks models not distinguishable from the best | `rank_models(...)` (or the comparison table of File 52, same column names) |
| `plot_fold_spread(results, path, metric="macro_f1")` | one dot per fold for each model and the mean: how much the score moves between splits | the result of `run_cross_validation` |
| `plot_detection_and_false_alarms(table, path, ...)` | attack detection rate and false alarm rate, in **two side-by-side panels** | `rank_models(...)` |

## Design rules followed

- **One axis per chart**: detection and false alarm rates live on different scales, so they are two panels, never one chart with two axes.
- **Colour has one job**: blue for the main score, orange for the second series (recall against precision); the confusion matrix uses one hue from light to dark for magnitude. Colour is never the only code: precision and recall also differ by marker shape, and subsample models carry an asterisk and a footnote.
- **Honest scales**: bars start at zero. Where a zoomed axis is needed (detection rates near 100%, fold spreads), a dot plot is used and the figure says in a footnote that the axis does not start at zero.
- **Reliability is drawn**: low-support classes are shaded with their support printed.
- **Direct labels, faint grid, thin marks.**

## How it was checked (sandbox)

All five figures were rendered from synthetic data (15 classes with your real class proportions, 7 models cross-validated) and **looked at**: no label collisions, readable class names, the shaded low-support rows, footnotes inside the image, percentage ticks on the detection panels. Two layout problems found this way were fixed (markers clipped at the right edge of the per-class chart; decimal ticks instead of percentages on the detection panels).

## How to test on your real data

The command draws five check figures into `reports\evaluation\check\` (a folder only for checks; you can delete it). Two come from the saved Random Forest scored on a class-balanced slice of **training** rows (6,055 rows, near-perfect scores because the model has seen them: **not performance**); three come from a quick 5-fold cross-validation of four fast models on a 100,000-row class-capped sample. PowerShell, from `DM_Project\`, venv active (about 20 to 60 seconds):

```powershell
python -c "import numpy as np, tempfile; from pathlib import Path; from nids.evaluation import metrics as M, cross_validation as cv, plots as P; from nids.persistence import model_store as ms; from nids.training.data_access import load_training_data; d = load_training_data(mmap=True); out = Path('reports/evaluation/check'); idx = np.sort(np.concatenate([np.flatnonzero(d.y == c)[:500] for c in range(15)])); r = M.evaluate_model(ms.load_model('random_forest'), np.asarray(d.X[idx]), d.y[idx], d.codec); print(P.plot_confusion_matrix(r, out / 'confusion_rf_training_slice.png', title='Check: Random Forest on 6,055 TRAINING rows (not performance)')); print(P.plot_per_class_metrics(r, out / 'per_class_rf_training_slice.png', title='Check: Random Forest on 6,055 TRAINING rows (not performance)')); res = cv.run_cross_validation(['bernoulli_nb', 'gaussian_nb', 'sgd', 'decision_tree'], d.X, d.y, d.codec, sample_size=100000, directory=tempfile.mkdtemp()); t = cv.rank_models(res); print(P.plot_model_comparison(t, out / 'cv_comparison_sample.png', title='Check: macro F1, 5-fold CV on a 100,000-row class-capped sample')); print(P.plot_fold_spread(res, out / 'cv_folds_sample.png', title='Check: spread over the folds (class-capped sample)')); print(P.plot_detection_and_false_alarms(t, out / 'cv_detection_sample.png', title='Check: detection and false alarms (class-capped sample)'))"
```

**Expected output:** the INFO lines of the cross-validation and five file paths, ending in `confusion_rf_training_slice.png`, `per_class_rf_training_slice.png`, `cv_comparison_sample.png`, `cv_folds_sample.png`, `cv_detection_sample.png`. **Please open the five images and tell me whether they are readable** (class names not cut off, labels not overlapping). On the real data I expect: a nearly diagonal confusion matrix; Heartbleed, Infiltration and SQL Injection shaded as low support; Decision Tree on top of the comparison with a clear gap above the others.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... plots` | placeholder empty | paste the code, check the length is not 0 |
| `ModuleNotFoundError: matplotlib` | package missing in the venv | `pip install matplotlib` (it was installed for the EDA step) |
| `ValueError: no models to plot` | the table is empty or has no value for the score | check that the cross-validation produced results |
| `PermissionError` when saving | the PNG is open in another program | close it and rerun |

## Decisions and limitations

- **Static PNG figures** (150 dpi), used by the evaluation report (File 53). The Streamlit app (File 63) will draw its own interactive views.
- **ROC and precision-recall curves** are in `curves.py` (File 51).
- **No tests file** in the roadmap; figures are checked by looking at them, as above.