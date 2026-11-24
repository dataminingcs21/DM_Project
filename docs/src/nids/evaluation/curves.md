# `src/nids/evaluation/curves.py`

**Location:** `DM_Project/src/nids/evaluation/curves.py`
**Roadmap step:** File 51 (Stage 7, evaluation)

## Purpose

Computes and draws ROC curves and precision-recall curves, one class against the rest, from the scores of a model.

## Why it is needed

A confusion matrix judges the *decisions* of a model. A curve judges its *scores*: how well the model ranks the flows of one class above all others at every possible threshold. That is how an intrusion detector is tuned in practice (how many attacks do we catch for how many false alarms?). The precision-recall curve matters most here: for a very rare class the false positive rate stays tiny even when most alerts are wrong, but precision shows it.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scikit-learn.metrics`, `nids.evaluation.metrics` (the low-support threshold), `nids.evaluation.plots` (the shared style).

## What it computes (`compute_curves(y_true, scores, codec, score_kind=None)`)

For each class that has both positive and negative rows:
- the **ROC curve** (true positive rate against false positive rate) and its area (ROC-AUC, equal to the value of `metrics.py`);
- the **precision-recall curve** and its area, the **average precision** (AP), plus the **prevalence** of the class (the AP of a random scorer);
- the **support** and a `low_support` flag (fewer than 30 true rows).

Curves are reduced to at most 400 points for plotting. A class with no true rows, or with no other-class rows, has no curve and is listed in `missing_classes`.

**Macro curves:** the ROC curves of all classes are interpolated on a common false-positive-rate grid and their true positive rates averaged. For precision-recall, each class's curve is turned into its *interpolated precision* (the best precision among the points with a recall at least as large) on a common recall grid and averaged. The macro AUC and macro AP are the plain means over the classes. Scores can be probabilities or decision scores (SVC, SGD): only the ranking matters.

## Figures

| Function | What it shows |
|---|---|
| `plot_class_curves(curves, path, kind="roc" or "pr", title=None, ncols=5)` | one small panel per class, titled with the class, its support and its AUC or AP; low-support panels shaded; dashed line = random scorer (diagonal for ROC, prevalence for precision-recall) |
| `plot_macro_curves(curves_by_model, path, kind="roc" or "pr", title=None, log_fpr=True)` | macro-averaged curves of **at most 4 models** on one chart (colour **and** line style differ; more models are refused instead of cycling colours). ROC uses a logarithmic false-positive-rate axis by default, which magnifies the region of few false alarms (the footnote says so) |
| `curves_table(curves)` | per-class ROC-AUC and AP table with support and the low-support note |

## How it was checked (sandbox)

On synthetic data with your 15 class proportions and four models (Random Forest, SGD, Gaussian NB, SVC): the per-class AUC equals `metrics.per_class_auc`; the AP equals scikit-learn's `average_precision_score` for every class; the reduced curves have at most 400 points and start at 0 and end at 1; the result is JSON-safe; invalid inputs (wrong shape, missing values, a class absent from the rows, no valid class, too many models, bad `kind`) raise the intended error.

**A real bug found by looking at the figure and fixed:** my first macro precision-recall curve was a flat line at 1.0. The interpolated precision was computed in the wrong direction. The corrected function is checked against the definition ("best precision among the points with recall at least r") on 399 random cases, half of them with tied scores, and the curve is non-increasing in recall. The per-class precision-recall panels and the macro curves were viewed after the fix.

## How to test on your real data

A hold-out inside the **training** data (the test set is not used): a class-capped sample of 100,000 training rows, 5-fold, first fold: train on 80,000 rows, score the 20,000 held-out rows. Three models (Random Forest, SGD, Gaussian NB), four figures into `reports\evaluation\check\`. PowerShell, from `DM_Project\`, venv active (about 20 to 60 seconds):

```powershell
python -c "import numpy as np; from pathlib import Path; from nids.evaluation import curves as C, cross_validation as cv; from nids.models import registry; from nids.training.data_access import load_training_data, class_capped_indices; d = load_training_data(mmap=True); idx = class_capped_indices(d.y, 100000); X, y = np.asarray(d.X[idx]), d.y[idx]; tr, te = cv.make_folds(y, 5)[0]; out = Path('reports/evaluation/check'); allc = {}; [allc.update({n: C.compute_curves(y[te], (m := registry.create_model(k).fit(X[tr], y[tr], n_classes=15)).predict_scores(X[te]), d.codec, m.score_kind)}) for k, n in [('random_forest', 'Random Forest'), ('sgd', 'SGD'), ('gaussian_nb', 'Gaussian NB')]]; print({n: (round(c['macro']['roc_auc'], 4), round(c['macro']['average_precision'], 4)) for n, c in allc.items()}); print(C.plot_class_curves(allc['Random Forest'], out / 'curves_roc_rf_holdout.png', 'roc', title='Check: Random Forest, 20,000 held-out TRAINING rows (class-capped sample)')); print(C.plot_class_curves(allc['Random Forest'], out / 'curves_pr_rf_holdout.png', 'pr', title='Check: Random Forest, 20,000 held-out TRAINING rows (class-capped sample)')); print(C.plot_macro_curves(allc, out / 'curves_roc_macro_holdout.png', 'roc')); print(C.plot_macro_curves(allc, out / 'curves_pr_macro_holdout.png', 'pr'))"
```

**Expected output:** a dictionary with each model's `(macro ROC-AUC, macro AP)` (I do not know the numbers; expect ROC-AUC close to 1 for Random Forest and lower for SGD and Gaussian NB, and AP clearly below the AUC for the weaker models), then four file paths ending in `curves_roc_rf_holdout.png`, `curves_pr_rf_holdout.png`, `curves_roc_macro_holdout.png`, `curves_pr_macro_holdout.png`. In the per-class figures Heartbleed, Infiltration and SQL Injection should have shaded panels. **Please open the four images and tell me whether they are readable.**

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... curves` | placeholder empty | paste the code, check the length is not 0 |
| `ValueError: scores must have shape ...` | the scores do not match the number of classes | use `model.predict_scores` |
| `ValueError: at most 4 models can be overlaid ...` | too many models for one chart | pass the best four, or draw one chart per model |
| `SyntaxError` at `:=` | very old Python | not expected (Python 3.13) |

## Decisions and limitations

- **Curves of low-support classes are unreliable** (a class with 2 test rows has a staircase of 2 steps); they are shaded and labelled rather than hidden.
- **Macro precision-recall is an interpolated average**, a common convention, not the area of any single curve; the exact numbers to quote are the macro AP and the per-class APs.
- **At most 4 overlaid models** by design; the comparison of all 11 is the table and bar chart of File 52.
- **No tests file** in the roadmap; the checks above were run in the sandbox.