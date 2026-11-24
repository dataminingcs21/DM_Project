# `tests/test_preprocessing.py`

**Location:** `DM_Project/tests/test_preprocessing.py`
**Roadmap step:** File 26 (Stage 4, tests)

## Purpose

Automated tests for the whole preprocessing stage: the feature filter (File 22), the pipeline (File 23), the feature schema (File 24) and the preprocessor (File 25). They use only synthetic data, run in about a second and never touch your real data (a fixture clears the temporary `preprocessed` folder before and after each test).

## What is tested (61 tests)

**Feature filter (14)**
- constant and all-missing columns are dropped; of two correlated columns the first is kept; the column order decides; negative correlation counts; the greedy rule compares with the *kept* columns only;
- threshold from the argument and from the settings; missing and infinite values are ignored; `transform` returns the kept columns; plain arrays work;
- other column count, names or order are refused; not fitted / bad input / all columns dropped raise errors; the labels `y` are never used; the sample is reproducible; the report is JSON-safe.

**Signed log (2) and pipeline (13)**
- sign, zero, order and missing values of `sign(x) * log1p(|x|)`; huge values are compressed;
- step order; log switched off by argument and by setting; unknown scaler rejected;
- the training output is centred, scaled and finite; output names match; **missing values get the training median**; infinity is treated as missing; the input is not modified;
- **test rows cannot change what was learned** (extreme test values leave the fit unchanged); other columns refused; min-max and robust scalers; the pipeline survives joblib; the description is JSON-safe.

**Feature schema (10)**
- structure and counts; reasons for dropped features (constant / correlated with X); statistics come from the training data; missing and infinite values are counted, not used; an all-missing column has no statistics; whole-number flag; non-negative columns come from the settings;
- the validator detects four kinds of inconsistency; bad inputs rejected; save and load round trip (both JSON files); the table and the column check.

**Preprocessor (14)**
- shapes, dtypes (float32 / int64); the split is exactly File 20's split; training data centred, everything finite; codes decode back to the original labels, BENIGN is 0; constant and correlated columns removed;
- the label and `source_file` are never features;
- **test rows cannot influence the training output** (absurd test values give an identical `X_train`);
- the input is not modified; **sample flows are real test rows and reproduce their `X_test` row through the pipeline**; sample picking is deterministic and capped; `EXCLUDED_FEATURES` are removed; binary mode works; bad inputs rejected; the report is JSON-safe and consistent.

**Saving and freshness (6)**
- every file is written and reloads identically; the saved pipeline reproduces the saved arrays; fresh after saving; stale when a setting changes; not fresh without metadata; the metadata file is written last.

## Dependencies

Files 04, 05, 06, 09 (`conftest.py`), 19-25 and `pytest`, `numpy`, `pandas`, `scikit-learn`, `joblib`.

## How to run

From `DM_Project\` with the virtual environment active (PowerShell):

```powershell
python -m pytest tests -q
```

**Expected output:**

```
...............................................................................
153 passed in 3.xx s
```

(92 earlier tests + 61 new.) Only this file: `python -m pytest tests/test_preprocessing.py -q` gives `61 passed`.

I checked that the tests can fail. Six deliberate breakages each made tests fail: keeping the last instead of the first correlated column (12 failed), filling missing values with a constant instead of the median (1), removing the sign from the log (6), fitting the pipeline on all rows (2), not dropping constants (13), and taking sample flows from the wrong rows (1). I restored the code afterwards and the full suite passed again.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError` for a `nids.preprocessing...` module | a placeholder file is still empty | paste the code of Files 19-25, check the length is not 0 |
| `AttributeError: ... no attribute 'LOG_TRANSFORM'` or `'MALICIOUS_LABEL'` | old `settings.py` | replace it with the one from File 23 |
| `test_binary_mode` fails | old `target_builder.py` (label `ATTACK`) | replace it with the File 23 version |
| A test fails after you change a setting | some tests pin the current settings (threshold 0.95, 5 sample flows per class) | tell me which setting you changed |

## Decisions and limitations

- **Small synthetic data:** each test builds a few hundred rows, so a failure points at one rule. The real data was checked by running Files 22-25 yourself (66 -> 40 columns; 2,017,889 / 504,473 rows).
- **The preprocessor result is built once** for the file (a module-level fixture) to keep the run fast; tests that change a setting build their own.
- **No test of memory or speed:** measured by hand (File 25 on your data: 28.7 s).