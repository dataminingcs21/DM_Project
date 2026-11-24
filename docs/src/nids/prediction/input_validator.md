# `src/nids/prediction/input_validator.py`

**Location:** `DM_Project/src/nids/prediction/input_validator.py`
**Roadmap step:** File 58 (tests: File 59). Updated once after your first run (protocol limit for ports).

## Role
Checks a flow typed into the application (or a table of flows uploaded as CSV) before it reaches the models, and turns it into the table the preprocessing pipeline expects. It decides only from `data\preprocessed\feature_schema.json`, which was built from the training rows. It never transforms, scales or predicts, and it never repairs a value silently.

## Why it is needed
The models were trained on cleaned, numeric features. A typed value that is text, negative where the cleaning step treated negatives as impossible, or missing would otherwise be turned into a number by the pipeline without anyone being told, and the prediction would look as reliable as any other.

## Dependencies
`numpy`, `pandas`, `nids.preprocessing.feature_schema` (`load_feature_schema`), `nids.utils.logger`. Needs `04_preprocess.py` to have written `feature_schema.json`.

## What counts as what
| Level | Situation | Code |
|---|---|---|
| error | feature used by the models missing, blank or not given | `missing`, `missing_column` |
| error | value is not a plain number (text, a boolean, `1,000`, `1_000`) | `not_a_number` |
| error | negative value in a feature the cleaning step treats as non-negative (durations, rates, header lengths, segment size) | `negative` |
| error | infinity in a feature whose training values were never infinite | `non_finite` |
| error | value breaks a protocol limit in `HARD_LIMITS`: a `Destination Port` outside 0 to 65535 (a TCP/UDP port is a 16-bit number) | `impossible_value` |
| error | no rows, or the same column twice | `no_rows`, `duplicate_column` |
| warning | value outside the minimum to maximum seen in training (the models extrapolate; **not evidence of an attack**) | `out_of_range` |
| warning | fractional value in a feature whose training values were all whole numbers | `not_whole` |
| warning | missing or infinite value in a feature where the training data also had such values; the pipeline uses the training median, as in training | `missing_like_training`, `non_finite_like_training` |
| warning | `missing="median"`: a missing value filled by the training median | `missing_filled`, `missing_column_filled` |
| warning | column that is not a known feature (with a close-name suggestion) | `unexpected_column` |
| info | columns given that the models do not use (constant or correlated with a kept column); ignored | `unused_given` |

Only the features the models use (`model_feature_names`, 40 in your run) are required. The other raw columns do not influence the models; they are left empty and the pipeline imputes and drops them. This was checked: passing only the model features gives exactly the same pipeline output as passing all raw columns.

## Protocol limits (`HARD_LIMITS`)
A short table inside the module: `{feature: (lowest, highest, reason)}`. A limit belongs there only when it is true by definition (a port is 16 bits), never because of what the training data contained. Today it holds one entry, `Destination Port`. It is not part of the cleaning or preprocessing settings, so changing it never makes saved data or models stale. This was added after the first demonstration showed a port of 655,360 producing only a range warning.

## Functions
| Function | What it does |
|---|---|
| `validate_flow(values, schema=None, missing="error")` | one flow as `{feature name: value}` -> `ValidationResult` |
| `validate_frame(data, schema=None, missing="error")` | a table of flows (one row per flow) -> `ValidationResult` |
| `ValidationResult` | `frame` (all raw columns in pipeline order, float64, NaN where empty or invalid), `valid_mask`, `valid_frame`, `ok`, `n_valid`, `issues`, `errors`, `warnings`, `counts`, `summary()` |
| `issues_table(result)` | the findings as a table for the interface |
| `field_specs(schema)` | for each model feature: min, 1st percentile, median, 99th percentile, max, whole-number flag, non-negative flag (for building the form) |
| `typical_flow(schema)` | every model feature at its training median: a valid starting point for a form (a per-feature median, not a real flow) |
| `python -m nids.prediction.input_validator` | demonstration on your real schema |

`row` in a finding is 1-based (1 is the first row given); it is empty for a single flow. At most 500 findings are listed; the counts per code are always complete (`summary()["issues_not_listed"]` says how many were not listed).

## How it works
1. Column checks: duplicates, unknown columns, required columns not given.
2. Each column is read as numbers: numeric columns directly; text columns cell by cell (`"  12.5 "` is accepted, `"abc"`, `"1,000"`, `"1_000"`, booleans are not). Blank, `None`, `NaN` and the text `nan` are "missing".
3. Per required feature: missing / infinite handling (see the table), negative check, range and whole-number warnings.
4. The result frame has every raw column in the pipeline's order; invalid cells are NaN and the row is marked invalid in `valid_mask`.

## Command (PowerShell, from `DM_Project\`, venv active)
```powershell
python -m nids.prediction.input_validator
```

## Expected output
A first line with the schema size (`Schema: ... raw features, 40 used by the models, ...`), how many used features must be non-negative, how many are whole-number features, then eight cases, each with `ok=...`, the error and warning counts and the findings:
the typical flow is `ok=True, errors=0, warnings=0`; text in a field, a field left out, a negative duration and an impossible port are `ok=False` with one error; a field left out with `missing='median'` and a value ten times the training maximum are `ok=True` with one warning; a port of 65,536 is `ok=False` with the error `... is impossible: a TCP/UDP port is a 16-bit number (0 to 65535)`; a misspelled extra column is `ok=True` with one warning and a suggestion. The value ten times the training maximum is now tried on the first feature that has no protocol limit (not `Destination Port`).

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `... feature_schema.json ... run scripts/04_preprocess.py first` | preprocessing not run | run `04_preprocess.py` |
| `ModuleNotFoundError: nids` | package not installed in the venv | run from `DM_Project\` with the venv active (the editable install `pip install -e .` described in `docs\project\pyproject.md`) |
| typical flow shows warnings | a median fell outside its own min to max, or whole-number rule | send me the output |

## Limits
- Ranges come from the training rows. A value inside the range can still be an unusual flow, and a value outside it is not proof of an attack.
- The validator checks each feature on its own; it does not check that a combination of values is physically possible (for example packets, bytes and duration agreeing).
- Features that were infinite or missing in training are accepted with a warning and replaced by the training median.