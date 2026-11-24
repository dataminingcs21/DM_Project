# `tests/test_predictor.py`

**Roadmap step:** File 61. Tests for `src/nids/prediction/predictor.py` (File 60). Synthetic data only; it never reads your dataset, your saved models or your reports.

## How it is built
Once per test run a small complete project is created in a temporary folder: preprocessing with the project's own `preprocess` and `save_preprocessed` (4 classes, 640 flows, 6 raw features of which 4 are used by the models), 11 baseline models and 11 tuned models with tuning records (KNN and Decision Tree "improved beyond noise", Gaussian NB "baseline already best", the rest "within noise"). The `settings` paths are redirected to that folder while this file runs, so no other test and no real file is touched. Tests that damage files (delete, corrupt, make stale) work on a copy.

The main tests compare the predictor with an independent plain-Python computation of the written rule: the pipeline, each model's own `predict`, count the votes, apply the tie rule.

## What it checks (60 tests)
| Group | Checks |
|---|---|
| Evidence rule | tuned only for the models with a beyond-noise tuning gain; all 11 covered; subset of keys; no tuned model means baseline; the decision table (beyond noise, within noise, same as baseline, empty record); the decisions equal those of `06_evaluate.py` |
| Loading | all 11 loaded in the final versions; the tuned models really carry the tuned parameters; `baseline` and `tuned` selections; unknown selection; the model table; a missing model is reported and **not replaced** by another version; a damaged file (checksum); a stale model; `require_all`; no model at all; constructor without models; stale preprocessing; missing pipeline file; pipeline and schema disagreeing; a model with another number of classes |
| One flow | typical flow; votes table (11 lines, version per model); tally and the agreement sentence; JSON-safe summary; the label equals the written rule on every sample flow; refused input is not predicted; impossible port refused; warnings pass through; missing-value policy; same answer whatever the key order; input dict not modified |
| Votes with controlled models | clear majority; a tie prefers the attack and the answer says so; a failing model is listed and the others vote; models that did not load are listed; every model failing raises |
| Table of flows | columns and order; equals the single-flow answers; equals the written rule; refused rows keep their place and index; a column error refuses everything and runs no model; no rows; JSON-safe summary; a failing model in a batch |
| Same arithmetic as preprocessing | the typed-flow path reproduces the saved test matrix **bit for bit** (regression test for the float32 problem found in the first real run); float64 input would differ; the validator's own frame is not changed by the cast |
| Sample flows | columns and classes; every sample passes the input check; missing file |
| Demonstration | exit 0 with both checks OK; a different saved vote is a MISMATCH (exit 2); check 2 skipped without the saved vote; a model that did not load is a failure (exit 2); missing artifacts (exit 1) |

## Command (PowerShell, from `DM_Project\`)
```powershell
python -m pytest tests\test_predictor.py -q
```
**Expected:** `60 passed` in a few seconds (about 6 s in my sandbox).

Full suite (optional): `python -m pytest -q` (earlier 483 passed, 1 skipped; now expected 543 passed, 1 skipped).

## Sandbox check beyond passing
I broke the predictor seven ways, one at a time: no float32 cast before the pipeline, tuned version always used, missing models not reported, batch results written to the wrong rows, invalid input predicted anyway, a model silently replaced by its baseline, and the tie wording removed. The tests failed each time (3, 9, 1, 1, 3, 5 and 1 tests); the original passes. In the full sandbox run 543 tests passed and one failed: `tests/test_loader.py::...test_counts_unconvertible_values_and_keeps_input`, an older test of File 14 that passes on your machine (483 passed); it depends on the pandas version of my sandbox.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nids.prediction.predictor` | File 60 not pasted (or not the corrected version) | paste it |
| `test_typed_flow_path_reproduces_the_saved_test_matrix_exactly` fails | the float32 cast is missing in `Predictor._features` | use the corrected File 60 |
| `ImportError ... pyarrow` | pyarrow missing in the venv | `pip install pyarrow` |