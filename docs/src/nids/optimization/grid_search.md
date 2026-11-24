# `src/nids/optimization/grid_search.py`

**Roadmap step:** File 56. Also changed in this step: `config/model_params.py` (tuning settings added), `src/nids/models/adaboost_model.py` (depth option). All three files must be replaced together.

## Role
Searches a small hyperparameter grid for each model with `GridSearchCV`, using stratified cross-validation on a class-capped sample of the **training** rows (the test set is never read), and returns/saves the winner, the baseline's score and the whole results table.

## Why it is needed
The baselines use library defaults (Checkpoint 3). Tuning shows how much each model can gain, and the `class_weight` options you approved at Checkpoint 4 are decided here by cross-validation, never by the test set.

## Dependencies
`numpy`, `pandas`, `scikit-learn` (`GridSearchCV`, `make_scorer`), `config.model_params`, `config.settings`, `nids.models.registry`, `nids.evaluation.cross_validation.make_folds`, `nids.training.data_access.class_capped_indices`, `nids.utils.io_utils`.

## The score: prior-corrected macro F1 (your decision)
The tuning sample keeps small classes whole and caps large ones, so BENIGN is about 18% of it (83% in real traffic). Plain macro F1 on such folds overstates precision and favours recall-heavy settings such as `class_weight="balanced"`. Each held-out row therefore counts `full_count / sample_count` of its class, the weighted F1 is computed per class and averaged. The weight depends only on the true class, so it works as a normal scorer inside `GridSearchCV`. The plain macro F1 is stored beside it. **The winner is chosen by the corrected score.** It is an estimate from a modest number of rows of the large classes, so it is noisy for the rare classes.

## What is in the module
| Name | What it does |
|---|---|
| `prior_weights(y_full, y_sample, n_classes)` | the class weights above (raises if the sample misses a class) |
| `corrected_macro_f1(y_true, y_pred, class_weights, n_classes)` | the score; with all weights 1 it equals scikit-learn's macro F1 |
| `TunableModel` | thin scikit-learn estimator around `registry.create_model`, so the grid uses exactly the code trained later (GaussianNB float64, XGBoost device, AdaBoost `base_max_depth`) |
| `baseline_point(key)` / `check_grid(key)` | the untuned values; refuses a grid that does not contain them |
| `preflight(key, X, y, n_classes, grid)` | fits every grid point once on about 12 rows per class so an invalid value (scikit-learn rejects it at fit time) is found in seconds, not as a silent NaN after hours |
| `tune_model(...)` | one grid search; returns a JSON-safe result |
| `run_tuning(keys, X, y, n_classes, sample_size=None, skip_existing=True, directory=None)` | all models, each result saved to `reports/optimization/tuning_<key>.json` as soon as it is done; resumable; a failing model is recorded and the others continue; saved results are reused only if the fingerprint (data, grid, folds, seed, settings) matches |
| `load_tuning`, `best_params`, `summary_table` | read the results (File 57 uses `best_params` to build the tuned models) |

Each result has: best parameters, best and baseline scores (corrected and plain, mean and standard deviation over the folds), `improvement_corrected` (mean of the paired fold differences best - baseline), `improvement_fold_spread` (their standard deviation), `improvement_beyond_noise` (improvement larger than that spread), the full ranked table, timings, number of failed grid points.

## How it runs
Grid points of the single-threaded models run in parallel (up to your `N_JOBS` = 14); KNN, Random Forest, SGD and XGBoost run one after another because each fit already uses all threads or the GPU. Folds are the same stratified folds for every grid point, so differences between grid points are paired.

## How it was checked (sandbox, synthetic data with your 15 class proportions)
- all 11 grids contain their baseline, and a grid without it is refused;
- the scorer equals a hand computation with class weights, and equals scikit-learn macro F1 when all weights are 1; absent classes are ignored;
- the weights multiplied by the sample counts reproduce the full class counts;
- the baseline score returned by the search equals a manual fit/score loop on the same folds (to 9 decimals); the sequential and parallel searches give identical results;
- resume works (same fingerprint reused, other sample size recomputed), unknown model keys and missing results raise;
- **a real flaw found and fixed:** my first Bernoulli NB grid used `binarize=-0.5`, which scikit-learn rejects; the search had silently recorded these points as NaN. The grid is corrected and `preflight` now makes such a mistake stop the run immediately.

## Command (real data; training rows only; about 1 to 3 minutes)
Checks that every point of all 11 grids can be fitted on your machine, then tunes the three fastest models on the real 100,000-row sample into a temporary folder (nothing in your project is overwritten). From `DM_Project\`:

```powershell
python -c "import tempfile; from nids.optimization import grid_search as G; from nids.training.data_access import load_training_data; from config import model_params; d = load_training_data(mmap=True); [G.preflight(k, d.X, d.y, d.n_classes) for k in model_params.MODEL_KEYS]; print('all 11 grids can be fitted'); res = G.run_tuning(['gaussian_nb', 'bernoulli_nb', 'decision_tree'], d.X, d.y, d.n_classes, directory=tempfile.mkdtemp()); print(G.summary_table(res).round(4).to_string(index=False))"
```

**Expected output:** `all 11 grids can be fitted`; INFO lines for the three models (5, 9 and 16 grid points x 5 folds on 100,000 rows; Decision Tree takes the longest); a table with one row per model: `best_params`, `baseline_corrected_f1`, `best_corrected_f1`, `improvement`, `beyond_noise`, `best_is_baseline`. I do not know the values; I expect Bernoulli NB to stay far below the tree models, and for the Decision Tree to show whether `class_weight` helps once the real class mix is accounted for.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... grid_search` | placeholder empty | paste the code; Length must not be 0 |
| `TuningError: ... cannot be fitted` | a grid value is invalid for your scikit-learn version | send me the message; I fix the grid |
| `TuningError: ... lacks the baseline value` | a grid was edited and lost the default | put the default back |
| `ImportError ... TUNING_GRIDS` / `AttributeError: model_params` | `config/model_params.py` not replaced | replace all three files |
| `FitFailedWarning` | a grid point failed during the search | listed in `failed_combinations`; the other points still count |

## Limits (also stated in the final README)
- Parameters are found on at most 100,000 rows (KNN 50,000; SVC and Gradient Boosting 30,000); they may not be optimal at 2 million rows.
- The grids are small on purpose; a result at the edge of a grid means a larger grid might find more.
- If `improvement_beyond_noise` is False, the folds cannot show that the tuned setting is better than the baseline.
- The corrected score estimates real-traffic macro F1 but is noisy for the three rarest classes (only a few dozen rows or fewer in the whole training set; each fold holds about a fifth of them).

## Update at File 57
- A grid search fits hundreds of models, and each fit logged one INFO line (about 300 lines for three models in your File 56 run). The search now silences the INFO lines of `nids.models.base` and `nids.models.xgboost` while it runs (warnings still show); the levels are restored afterwards. Nothing else changed in the results.