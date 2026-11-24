# `config/model_params.py`

**Location:** `DM_Project/config/model_params.py`
**Roadmap step:** File 28 (Stage 5, models)

## Purpose

Holds the **default hyperparameters** of the 11 models (the untuned baseline), their display names, which of them train on a subsample, and how each can take class weights.

## Decision applied (Checkpoint 3, question 1, option A)

Library defaults, plus only what must be fixed:
- `random_state = 42` where the algorithm is random;
- `n_jobs = settings.N_JOBS` (14 on your machine) where threads can be used;
- iteration caps where the default may stop before converging: `max_iter = 1000` for LogisticRegression and SGDClassifier;
- `tree_method = "hist"` for XGBoost (fast, and needed for the GPU).

Everything else is the default of scikit-learn 1.8.0 and XGBoost 3.1.3. This keeps the baseline neutral, so the gain found later by GridSearchCV (File 54) is measurable. The tuning grids are **not** here; they come with File 54.

## Dependencies

`config/settings.py` (`RANDOM_SEED`, `N_JOBS`, `SUBSAMPLE_SIZES`, `CLASS_WEIGHT_MODE`). No other import: the model classes themselves are imported in Files 30-40.

## The 11 models (order of the project brief)

| Key | Model | Parameters set here | Training data |
|---|---|---|---|
| `knn` | KNN | `n_jobs` | subsample (100,000 rows) |
| `decision_tree` | Decision Tree | `random_state` | all rows |
| `random_forest` | Random Forest | `random_state`, `n_jobs` (100 trees) | all rows |
| `gaussian_nb` | Gaussian Naive Bayes | none | all rows |
| `svm` | SVC, RBF kernel | `random_state` (no probability estimates) | subsample (50,000) |
| `bernoulli_nb` | Bernoulli Naive Bayes | none | all rows |
| `logistic_regression` | Logistic Regression | `random_state`, `max_iter=1000` | all rows |
| `sgd` | SGD Classifier | `random_state`, `max_iter=1000`, `n_jobs` | all rows |
| `adaboost` | AdaBoost | `random_state` (50 estimators) | all rows |
| `gradient_boosting` | Gradient Boosting | `random_state` (100 rounds, one tree per class per round) | subsample (100,000) |
| `xgboost` | XGBoost | `random_state`, `n_jobs`, `tree_method="hist"`; the device comes from `settings.XGBOOST_DEVICE` (File 40) | all rows |

The subsample sizes are the starting values in `settings.SUBSAMPLE_SIZES` (Checkpoint 3, question 2, option A); they are class-capped and stratified, and calibrated by timing at File 47. The comparison table will state which models trained on a subsample.

## Functions

| Function | Role |
|---|---|
| `default_params(key)` | a copy of the model's parameters (safe to modify); unknown key -> `ValueError` listing the known keys |
| `subsample_size(key)` | rows of the class-capped training subsample, or `None` for the full training data |
| `class_weight_mechanism(key)` | `"class_weight"`, `"sample_weight"` or `None` (KNN cannot use weights). Used only when `CLASS_WEIGHT_MODE = "balanced"` (the experiment after Checkpoint 4) |
| `model_params_config()` | JSON-safe description of everything that changes a trained model: parameters, subsample sizes, class-weight mode, seed. `n_jobs` is left out (it changes speed, not the model). Saved with trained models (Files 43, 47) so models trained with other settings are recognised as stale |

When the module is imported it checks itself: the three dictionaries list exactly the 11 keys, and `SUBSAMPLE_SIZES` names only known models.

## Things to know about the baseline (not defects, but they shape the results)

- **Bernoulli Naive Bayes** binarizes at 0 by default. On standardized data, 0 is the training mean, so each feature becomes "above or below average". It is expected to be among the weakest models. That is a property of the algorithm on this data, not a bug.
- **SGD** uses the default hinge loss, which gives no probabilities; ROC curves use its decision scores.
- **SVC** has `probability=False` (probability estimates make it about five times slower); ROC curves use its decision scores. The majority vote needs only the predicted class.
- **Gradient Boosting** builds one tree per class per round (15 classes x 100 rounds), which is why it trains on a subsample.
- **XGBoost** needs class codes 0..14 with every class present in the training data: true here (the smallest class has 9 training rows).

## How it was checked (sandbox)

All 11 estimators were built from their defaults and fitted on a small synthetic data set: every parameter name is accepted. The class-weight mechanism of each model was confirmed (`class_weight` exists as a parameter, or `fit` accepts `sample_weight`, or neither for KNN). The returned dict is a copy, and the hash description contains no `n_jobs`.

## How to test

From `DM_Project\` with the virtual environment active (PowerShell):

```powershell
python -c "from config import model_params as mp; [print(k, '|', mp.DISPLAY_NAMES[k], '|', mp.default_params(k), '|', mp.subsample_size(k), '|', mp.class_weight_mechanism(k)) for k in mp.MODEL_KEYS]"
```

**Expected output:** 11 lines, in this order: knn, decision_tree, random_forest, gaussian_nb, svm, bernoulli_nb, logistic_regression, sgd, adaboost, gradient_boosting, xgboost. For example the first is `knn | KNN | {'n_jobs': 14} | 100000 | None` and the last `xgboost | XGBoost | {'random_state': 42, 'n_jobs': 14, 'tree_method': 'hist'} | None | sample_weight`. `n_jobs` shows 14 on your machine (16 threads minus 2).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError` / empty output | the file is an empty placeholder | paste the code, check the length is not 0 |
| `ValueError: ... does not list exactly the model keys` | a model was added or renamed in one dictionary only | keep the three dictionaries in sync |
| `ValueError: settings.SUBSAMPLE_SIZES names unknown models` | a typo in `settings.SUBSAMPLE_SIZES` | use the keys from `MODEL_KEYS` |
| `ValueError: unknown model 'x'` | wrong key in a call | use a key from `MODEL_KEYS` |

## Decisions and limitations

- **Defaults, not tuned values:** results at File 47 are a baseline. A poor baseline for a model does not mean the algorithm is poor; the tuned results come at File 57.
- **Grids added at File 56:** now that real timings exist (Files 47 and 55), the tuning grids are in this file (see "Tuning settings" below). They are deliberately small.
- **The three slow models see only a subsample:** their scores are not directly comparable with models that saw all rows, and the report must say so.

## Tuning settings (added at File 56)
New names in `config/model_params.py`; **they are not part of `model_params_config()`**, so adding or changing a grid never makes your trained baselines stale.

| Name | Meaning |
|---|---|
| `TUNING_SAMPLE_SIZE = 100_000` | rows of the class-capped training sample the grids are searched on |
| `TUNING_SAMPLE_CAPS` | smaller samples for slow models: KNN 50,000, SVC 30,000, Gradient Boosting 30,000 |
| `TUNING_FOLDS` | Gradient Boosting uses 3 folds, the others `settings.CV_FOLDS` (5) |
| `TUNING_SEQUENTIAL` | KNN, Random Forest, SGD, XGBoost: grid points run one after another (the model itself already uses all threads or the GPU) |
| `TUNING_BASELINE_OVERRIDES` | baseline values libraries do not report (AdaBoost depth 1; XGBoost max_depth 6, learning_rate 0.3, n_estimators 100) |
| `TUNING_GRIDS` | one grid per model (list below) |
| `tuning_grid(key)`, `tuning_sample_size(key, requested)`, `tuning_folds(key)`, `tuning_config()` | accessors |

Grids (every grid contains the baseline setting): KNN `n_neighbors` {3,5,11,21} x `weights`; Decision Tree `criterion` x `min_samples_leaf` {1,2,5,10} x `class_weight`; Random Forest `n_estimators` {100,300} x `min_samples_leaf` {1,3} x `max_features` {sqrt,0.5} x `class_weight`; Gaussian NB `var_smoothing` 1e-9..1e-1; Bernoulli NB `alpha` {0.1,1,10} x `binarize` {0,0.5,1} (scikit-learn refuses negative `binarize`); SVC `C` {1,10} x `gamma` {scale,0.01,0.1} x `class_weight`; Logistic Regression `C` {0.01..10} x `class_weight`; SGD `loss` x `alpha` {1e-6..1e-3} x `class_weight`; AdaBoost `n_estimators` {50,100} x `learning_rate` {0.5,1} x `base_max_depth` {1,2,4}; Gradient Boosting `learning_rate` {0.1,0.2} x `max_depth` {3,5}; XGBoost `max_depth` {6,8,10} x `learning_rate` {0.1,0.3} x `n_estimators` {100,300}.