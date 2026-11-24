# `src/nids/evaluation/cross_validation.py`

**Location:** `DM_Project/src/nids/evaluation/cross_validation.py`
**Roadmap step:** File 49 (Stage 7, evaluation)

## Purpose

Cross-validates the baseline models with stratified k-fold on the **training rows only**, saves each model's result as soon as it is done, and ranks the models by macro F1 while saying which of them cannot be told apart from the best.

## Why it is needed

One train/test split gives one number per model. Cross-validation trains each model k times on different parts of the training data and scores it on the part it did not see, so you also learn how much the score moves from one split to the next. That spread is what lets us say "model A is better than B" or "they are indistinguishable". The best single model is named from these scores, **not from the test set**, which stays untouched until the final comparison (Files 52-53).

## Dependencies

`nids.evaluation.metrics` (48), `nids.models.registry` (41), `nids.training.data_access` (45, for the class-capped subsample), `config/model_params.py`, `config/settings.py`, `scikit-learn`, `pandas`.

## How a model is cross-validated

- **Folds:** `StratifiedKFold(shuffle=True, random_state=42)` on the class codes: every fold has every class. Every class needs at least `n_splits` rows; the rarest class (Heartbleed) has 9 training rows, so 5 folds work and 10 would be refused with a clear message.
- **Training in each fold** is the same as `scripts/05_train.py`: baseline parameters, and for KNN, SVC and Gradient Boosting the class-capped subsample of that fold's training part. The model is scored on the held-out rows, which keep the real class mix.
- **Labels only:** no scores are computed (not needed here, and KNN and SVC would take as long again).
- **Out-of-fold predictions:** every training row is predicted exactly once, by a model that did not see it. Pooled, they give per-class metrics on the full class counts (9, 17 and 29 for the rarest classes, more than any single fold), still flagged "low support".

## What is reported per model

Mean and standard deviation (over the folds) of accuracy, balanced accuracy, **macro F1**, weighted F1, MCC, attack detection rate and false alarm rate; fit and prediction times; the pooled out-of-fold metrics from `compute_metrics`.

**Macro F1 is the primary score** (it weighs the 15 classes equally, so it rewards handling the rare classes). `rank_models` sorts by it and adds `indistinguishable_from_best`: True when a model's mean is within one standard deviation of the best model's mean. Such models cannot be separated from the best on this evidence, and the report says so rather than crowning a winner by a hair.

## Functions

| Function | What it does |
|---|---|
| `make_folds(y, n_splits=5, seed=None)` | stratified folds (checks that every class has at least `n_splits` rows) |
| `cross_validate_model(key, X, y, codec, folds, params=None)` | one model; returns the result dictionary and the out-of-fold predictions |
| `run_cross_validation(keys, X, y, codec, n_splits=5, sample_size=None, skip_existing=True, directory=None)` | several models (default all 11), saved one by one and **resumable**; a failed model is recorded as `failed` and the others continue |
| `load_oof(key, directory=None)` | the saved out-of-fold predictions |
| `rank_models(results, metric="macro_f1")` | ranking table with the indistinguishable-from-best flag |
| `fingerprint(...)` | hash of the data size, folds, seed, sample and settings; a saved result is reused only if it matches |

Results are saved in `reports\evaluation\cross_validation\` as `cv_<key>.json` (written last) and `cv_<key>_oof.npy`.

`sample_size` cross-validates on a class-capped stratified sample (faster). The held-out folds are then more balanced than the real traffic, so such scores are not comparable with a full run.

## How it was checked (sandbox)

On synthetic data with your 15 class proportions (8,412 rows): every row is in exactly one test fold; train and test parts are disjoint; the folds are deterministic; every class is in every train and test fold; 10 folds are refused (a class has fewer than 10 rows) and so is 1 fold. Five models were cross-validated: the ranking table printed, KNN trained on exactly its subsample size in each fold (1,500) and the Decision Tree on about 6,730 rows (80%); the pooled accuracy equals the accuracy of the out-of-fold predictions; the low-support flags are the three rarest classes; a second run reused the saved results; changing the number of folds recomputed instead of reusing; `sample_size` worked; a model forced to fail was recorded as `failed` while the next one still ran; an unknown key raises `ValueError`.

## How to test on your real data

A quick run, **not** the full cross-validation: 4 fast models, 5 folds, on a class-capped sample of 100,000 training rows, saved in a temporary folder. PowerShell, from `DM_Project\`, venv active (about 1 to 3 minutes):

```powershell
python -c "import tempfile; from nids.evaluation import cross_validation as cv; from nids.training.data_access import load_training_data; d = load_training_data(mmap=True); res = cv.run_cross_validation(['bernoulli_nb', 'gaussian_nb', 'sgd', 'decision_tree'], d.X, d.y, d.codec, sample_size=100000, directory=tempfile.mkdtemp()); t = cv.rank_models(res); print(t[['rank', 'model', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'detection_rate_mean', 'false_alarm_rate_mean', 'fit_seconds_mean', 'indistinguishable_from_best']].round(4).to_string(index=False)); print('failed:', [k for k, v in res.items() if v.get('status') != 'ok']); print('pooled low-support classes:', res['decision_tree']['pooled']['low_support_classes'])"
```

**Expected output:** log lines for each fold, then a table of 4 models sorted by macro F1 (Decision Tree expected first, the two Naive Bayes models and SGD lower; I do not know the numbers), a detection rate and false alarm rate for each, `failed: []`, and `pooled low-support classes: ['Heartbleed', 'Infiltration', 'Web Attack - Sql Injection']`. These numbers come from a class-capped sample, so they are only a check of the code.

## What the full run will cost (decided at File 53)

The full 5-fold run on all 2 million training rows for the 11 models is long. From your measured fit times I estimate **about 1.5 to 2 hours** (Gradient Boosting about 40 minutes by itself, because its subsample stays 100,000 rows in every fold; KNN and SVC about 10 minutes each for their predictions). It is resumable model by model. At File 53 I will give you the choice between the full run and a sample, with these numbers.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... cross_validation` | placeholder empty | paste the code, check the length is not 0 |
| `ValueError: every class needs at least 5 rows ...` | a class is too small for the number of folds | use fewer folds, or a sample that keeps small classes whole |
| a `failed` model | an error inside that model's fold | read the error text in the result; the others are unaffected |
| results not reused | settings, folds, sample or data size changed | expected: the fingerprint differs, so it recomputes |

## Decisions and limitations

- **Training rows only:** the test set is not read.
- **Macro F1 as the primary score**, with the spread shown; rare classes make it noisy, which is why the indistinguishable flag and the pooled per-class numbers are reported next to it.
- **No tests file in the roadmap:** the checks above were run in the sandbox; the functions are exercised by the evaluation script (File 53).