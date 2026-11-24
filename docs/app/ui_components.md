# `app/ui_components.py`

**Location:** `DM_Project/app/ui_components.py`
**Roadmap step:** File 62. The main page (File 63) composes these pieces; the integration test (File 64) runs the finished app in a headless Streamlit session and also unit-tests the plain functions below.

## Role
The reusable parts of the Streamlit application, so that `streamlit_app.py` stays short. Your two design choices are built in: three ways to enter flows (manual form, real examples, CSV upload) and the 40 form fields grouped by name prefix.

## Two kinds of code, kept apart
**Plain functions (no Streamlit calls, testable without a browser)**
| Function | What it does |
|---|---|
| `group_features(names)` | groups the 40 field names by the first word of the name; a first word shared by fewer than 3 features goes to `Other`; the rule is printed on the page (`GROUPING_RULE`) |
| `format_value`, `field_help`, `field_key` | how a number appears in a text box (whole numbers without decimals, others with up to 10 significant digits) and the tooltip of each field (training range, median, whole-number and non-negative rules, the port limit) |
| `median_values(schema)`, `sample_counts`, `sample_values` | the starting values: training medians, or the n-th real example flow of a class |
| `values_from_state` | the typed values (as text) of every field |
| `read_uploaded_csv(file, max_rows=50,000)` | reads a CSV; strips spaces around column names; sets the `Label` column aside (never used for prediction); reads at most 50,000 rows and says so; plain `ValueError` messages for empty or unreadable files |
| `results_table(batch)`, `to_csv_bytes` | result per row with 1-based row numbers and the reason a row was refused |
| `agreement_band`, `agreement_short`, `is_benign` | plain wording for how many models agree: all / more than half / only the most votes |
| `load_class_reliability()`, `load_model_scores()`, `reliability_lines(label, ...)` | read what `06_evaluate.py --stage tuned` saved (`test_tuned\majority_vote_final.json`, `model_comparison_tuned.json`) and turn it into sentences such as "found 98.5% of the 391 'Bot' flows (recall)". Every number about model quality comes from these files, none is typed into the code |

**Rendering functions** (`render_form`, `render_findings`, `render_prediction`, `render_batch_result`, `render_status`, `render_models_panel`, `render_limits`) and cached loaders (`get_predictor`, `get_samples`, `get_reliability`, `get_model_scores`: the models are loaded once per server process).

## Design decisions
- **Text boxes, not number boxes.** Every field is a text box. A number box would block typing text or a huge value in the browser, and the input check (File 58) would never see it. With text boxes everything typed reaches the validator, which explains the problem in plain words.
- **No Streamlit form wrapper**, so that "Load this real example" fills the fields immediately.
- **The check comes first.** A flow with an error is not predicted; the page shows the findings.
- **Nothing is hidden.** Each model's vote and version, the tally, the agreement, models that did not vote and why, and what the test set says about the predicted class are shown with every answer.
- **Limits are on the page** (`GENERAL_LIMITS`): a prediction is not proof, one dataset and a random split, subsample-trained models, hand-typed values.

## Dependencies
`streamlit` (1.52.2 as pinned), `pandas`, `config.settings`, `config.model_params`, `nids.prediction.input_validator`, `nids.prediction.predictor`, `nids.utils.io_utils`.

## Command (PowerShell, from `DM_Project\`, venv active)
No browser is needed to check this file:
```powershell
python app\ui_components.py
```
(The browser application itself is File 63.)

## Expected output
Streamlit may print `missing ScriptRunContext` or `No runtime found` warnings when imported outside the browser; they are harmless. Then:
- `40 feature fields in N groups.` followed by the grouping rule and one line per group listing its fields (the group names come from your real feature names);
- the first five fields at the training median, as typed in the form;
- `Real example flows per class (71 in all): ...` (up to 5 per class);
- `Test-set results for the class panel: found for 15 classes` and `Model scores for the models tab: found (11 models, 3 votes)`;
- a CSV reading demo line: columns `['Destination Port', 'Flow Duration']` and the note that the `Label` column was ignored.

## What I checked in the sandbox
A headless Streamlit session (`streamlit.testing`) with these components over a small synthetic project: the page loads without exceptions; the form has one text box per field; "Predict" on the medians gives a result with metrics and tables; loading a real example of a class fills the fields and predicts it; `-5` in Flow Duration and `abc` give the plain-language error and no prediction; a table with two refused rows shows 4 predicted, 2 refused and the findings sorted by row; the models tab and limits render.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `Cannot read the feature schema` | preprocessing not run | run `scripts\04_preprocess.py` |
| `Real example flows: not available` | `sample_flows.parquet` missing | run `scripts\04_preprocess.py` |
| `... not found (run 06_evaluate.py --stage tuned)` | the tuned evaluation has not been run | run it; the app still works without these panels |
| `ModuleNotFoundError: streamlit` | not installed in the venv | `pip install -r requirements.txt` |

## Limits
- The CSV reader accepts only CSV files and reads at most 50,000 rows per upload (about 10 seconds of prediction on the full final set).
- The grouping is mechanical (first word); it does not describe what a feature means.
- The panels about model quality describe the held-out test flows of the dataset, not any particular flow typed by the user.