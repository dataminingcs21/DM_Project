# `tests/test_input_validator.py`

**Roadmap step:** File 59. Tests for `src/nids/prediction/input_validator.py` (File 58). Synthetic data only; it never reads your dataset, your saved schema or your models. The schema is built by the project's own `build_pipeline` and `build_feature_schema` from a small synthetic table (3,000 rows, using feature names that appear in the settings, such as `Destination Port` and `Flow Duration`), so the tests follow the real schema format.

## What it checks (85 tests)
| Group | Checks |
|---|---|
| Schema assumptions | the synthetic schema has the model and unused features, the non-negative flags and the infinite values in training that the other tests rely on |
| Valid flow | typical flow ok; frame has all raw columns in pipeline order and float64; unused columns NaN; values kept exactly; numeric text with spaces and `1e3`; integers and numpy numbers; column order irrelevant; unused columns given are info only; text in an unused column ignored |
| Errors | not a number (8 kinds of bad text/objects, booleans, a huge integer); blank required value (6 kinds of blank); required feature not given; empty flow; negative in a non-negative feature; negative allowed elsewhere; infinity where training never had it (also with `missing="median"`) |
| Protocol limits | ports 65,536, 655,360, -1 and 1e9 are errors with the 16-bit reason; ports 0, 80, 443, 65,535 are valid; a fractional port is only a warning; a limit for an unknown feature is ignored; a limit can be added |
| Warnings | outside the training range (value kept, "not evidence of an attack"); range edges are inside; fraction in a whole-number feature; infinite or blank where training also had them; unexpected column with and without a suggestion; warnings do not invalidate the row |
| Missing policy | `median` fills and names it; a column not given; text still rejected; unknown policy refused |
| Tables | clean table; only bad rows invalid and numbered from 1; index kept; numeric and nullable integer columns; boolean column; no rows; duplicate column; missing column; 2,000 findings counted but only 500 listed; input not modified; wrong input types |
| Results and helpers | JSON-safe summary; no row number for a single flow; issues table; long values shortened; `field_specs`; `typical_flow`; `required_features`; default schema read from disk; the demonstration runs and reports a missing schema |
| With the pipeline | giving only the 40 model features gives exactly the same pipeline output as giving all raw columns; output columns are the model features; an infinite value seen in training is imputed exactly like a blank |

## Command (PowerShell, from `DM_Project\`)
```powershell
python -m pytest tests\test_input_validator.py -q
```
**Expected:** `85 passed` in about a second.

Full suite (optional): `python -m pytest -q` (earlier 398 passed, 1 skipped; now expected 483 passed, 1 skipped).

## Sandbox check beyond passing
I broke the validator six ways, one at a time: negative check off, protocol limit off, blank values accepted, range warning off, the "seen in training" exception ignored, and unused columns made required. The tests failed each time; the original passes.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nids.prediction.input_validator` | File 58 not pasted | paste it |
| `ImportError ... tests` | wrong folder | run from `DM_Project\` |
| a test fails with a different `n_missing_or_infinite_in_train` | `feature_schema.py` changed | send me the output |