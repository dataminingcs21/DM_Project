# `src/nids/prediction/predictor.py`

**Location:** `DM_Project/src/nids/prediction/predictor.py`
**Roadmap step:** File 60 (tests: File 61).

## Role
Predicts the class of one flow, or of a table of flows, with the **final set** of models. This is the single code path the Streamlit application will use:

`typed values -> input validator (File 58) -> fitted preprocessing pipeline -> every model's predict -> majority vote (File 53)`

## Why it is needed
The interface must not contain any modelling logic. Putting the whole path in one tested module means the application, the demonstration and the tests all use the same code, and a prediction typed into the application can be compared with the evaluation results.

## The final set
Follows the Checkpoint 5 evidence rule (`docs\project\checkpoint_5_decisions.md`): each model is loaded in its **tuned** version where tuning showed an improvement beyond fold noise and the tuned parameters differ from the baseline, otherwise as its **baseline**. The decision is read from the tuning record `07_optimize.py` stored in the tuned model's metadata (`extra.tuning`); it never uses a test score. For your run this gives 7 tuned (KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting) and 4 baseline (Random Forest, XGBoost, Logistic Regression, Gaussian NB).

`Predictor.load("baseline")` or `Predictor.load("tuned")` load all 11 models in one version (used for comparison and testing).

## What it does when something is wrong
| Situation | Behaviour |
|---|---|
| input has an error (File 58) | the flow is **not** predicted; `FlowPrediction.label` is `None` and `validation` lists the findings |
| a model cannot be loaded (missing, damaged, stale) | it does not vote; it is **not replaced** by another version; it is listed with the reason in `Predictor.not_loaded` and in every result's `missing` |
| a model fails while predicting | the others vote; the failure is listed as missing |
| every model failed or none loaded | `PredictorError` |
| `require_all=True` and any model not loaded | `PredictorError` |
| preprocessing settings changed since the data was preprocessed | `PredictorError` with the reason |
| saved pipeline and `feature_schema.json` disagree | `PredictorError` |

## Dependencies
`config.settings`, `config.model_params`, `nids.persistence.model_store`, `nids.prediction.input_validator`, `nids.preprocessing` (schema, `check_preprocessed_fresh`, `LabelCodec`), `nids.voting.majority_voting`, `nids.utils.io_utils`. Needs the files written by `04_preprocess.py` (pipeline, label encoder, schema, sample flows) and the saved models of `05_train.py` and `07_optimize.py`.

## Main parts
| Part | What it does |
|---|---|
| `Predictor.load(selection="final", require_all=False)` | loads pipeline, codec, schema and the 11 models (checksum and freshness checked by the model store) |
| `predict_flow(values, missing="error")` | one flow -> `FlowPrediction`: `label`, `code`, `votes` (one line per model: model, version, vote), `tally`, `agreement` ("9 of 11 models voted for X", with the tie rule named if tied), `missing`, `validation` |
| `predict_frame(df, missing="error")` | a table -> `BatchPrediction`: one line per input row in input order, `prediction` (empty for refused rows), `top_votes`, `n_voters`, `tied`, and one column per voting model |
| `model_table()` | per model: version used and why, loaded or not and why not, training rows, subsample flag, tuned parameters, file size |
| `final_set()` | the evidence rule as a function |
| `load_sample_flows()` | the real test flows `04_preprocess.py` saved (raw features, class, `test_row_position`) |

## Command (PowerShell, from `DM_Project\`, venv active)
```powershell
python -m nids.prediction.predictor
```
Loading the 11 models takes roughly 10 to 60 seconds (checksums of the files, a 122 MB Random Forest, KNN and SVC).

## Expected output
1. A table of the 11 models: `version` tuned for KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting and baseline for Random Forest, Gaussian NB, Logistic Regression, XGBoost; `loaded` True for all; training rows (KNN 100,000, SVC 50,000, Gradient Boosting 100,000 and `trained_on_subsample` True for those three).
2. `Typical flow (per-feature training medians; not a real flow): <class> - N of 11 models voted for <class>`. The class is whatever the models say about a median flow; it is not meaningful on its own.
3. `Sample flows: N real test flows, N predicted, 0 refused by the input check.` (up to 5 per class; Heartbleed has only 2 test rows). A refused sample would mean a cleaned test row has a missing value where the training data had none.
4. `final vote equals the true class on X of N` (an illustration on a few flows, **not** a metric).
5. `check 1, typed-flow path vs saved X_test rows: max difference 0.00e+00, 0 of N values differ -> OK (identical)`: the typed-flow path (validator + pipeline) reproduces the saved preprocessed test matrix exactly. The predictor casts the raw values to float32 before the pipeline, exactly as `04_preprocess.py` did; with float64 the values differ by about 1e-6, which was enough to flip one flow's vote in the first real run (70 of 71 identical).
6. `check 2, final vote vs saved evaluation vote (majority_vote_final): N of N identical -> OK`: the predictor gives exactly the vote saved by `06_evaluate.py --stage tuned` on the same test rows.

If check 2 shows N of N identical the application's prediction equals the evaluation's, row for row.

Exit code 0 when all models loaded and the checks pass; 2 otherwise; 1 when the artifacts are missing.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `the preprocessing settings changed ...` | settings changed after `04_preprocess.py` | rerun `04_preprocess.py` and the later steps |
| `NOT LOADED: {...}` | a model file is missing, damaged or stale | the reason is printed; rerun `05_train.py` or `07_optimize.py` for that model |
| `check 2 skipped` | `06_evaluate.py --stage tuned` has not been run | run it |
| `MISMATCH` in a check | files from different runs are mixed | send me the output |

## Limits
- The answer is the majority of 11 models, several of them weak (Gaussian NB, Bernoulli NB). Agreement is shown with every answer; a low agreement means the models disagree, not that the flow is an attack.
- Rare classes (Heartbleed, Infiltration, SQL Injection) have 2 to 7 test rows: the system's behaviour on them is not reliably known.
- Web Attack XSS and Brute Force are often confused; the vote does not separate them reliably.
- KNN, SVC and Gradient Boosting were trained on class-capped subsamples (shifted class prior).
- Predictions are only as good as the typed values: the input check catches impossible values, not unrealistic combinations.