# `tests/test_target_and_split.py`

**Location:** `DM_Project/tests/test_target_and_split.py`
**Roadmap step:** File 21 (Stage 4, tests)

## Purpose

Automated tests for the target builder (`target_builder.py`, File 19) and the splitter (`splitter.py`, File 20). They use only synthetic labels, so they run in about half a second and never touch your real data.

## What is tested (38 tests)

**Target builder (14)**
- multiclass with policy `keep` leaves labels unchanged; classes below the minimum are flagged;
- policy `merge` combines rare classes into `Other_Attack`, never merges BENIGN, keeps the original counts in the report, and is refused when the merged class would still be too small;
- binary mode (BENIGN vs MALICIOUS, the label from `settings.MALICIOUS_LABEL`), including that it ignores the rare-class policy;
- defaults come from `settings`; the index is preserved; the input is not modified;
- invalid mode, policy, missing labels and data without BENIGN are rejected; the report is JSON-safe.

**Label codec (9)**
- BENIGN is index 0, the rest alphabetical; no BENIGN index when BENIGN is absent;
- encode/decode round trip; encoding ignores the row index; unknown names and out-of-range codes are rejected (never guessed); duplicate or empty class lists are rejected; the codec survives joblib; `to_dict`.

**Splitter (10)**
- train and test are disjoint and cover every row; every class is on both sides; proportions follow `test_size`; the test fraction follows the setting;
- same seed gives the same split, another seed another split; indices are sorted integers; numeric codes are accepted;
- a class with one row, a bad `test_size`, an empty target and missing values are rejected; works with the shared synthetic frame.

**Split report (5)**
- confirms a valid split; detects overlap and gaps; lists classes missing from a side; flags classes whose train count is below the CV folds; the summary is JSON-safe.

## Dependencies

Files 04, 05, 09 (`conftest.py`), 19, 20, and `pytest`, `numpy`, `pandas`, `joblib`.

## How to run

From `DM_Project\` with the virtual environment active (PowerShell):

```powershell
python -m pytest tests -q
```

**Expected output:**

```
........................................................................ [ 78%]
....................                                                     [100%]
92 passed in 1.xx s
```

(54 earlier tests + 38 new.) Only this file: `python -m pytest tests/test_target_and_split.py -q` gives `38 passed`.

I checked that the tests can fail: removing the stratification, changing the class order of the codec, and letting BENIGN be merged each made tests fail.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError` for `nids.preprocessing.splitter` or `target_builder` | an empty placeholder file | paste the code from File 19 / 20, check the length is not 0 |
| `ModuleNotFoundError: nids` | virtual environment not active or `pip install -e .` not run | activate; install |
| `ModuleNotFoundError: joblib` | dependencies not installed | `pip install -e .` |
| A test fails after you changed `TEST_SIZE`, `MIN_CLASS_SAMPLES` or `CV_FOLDS` | some tests use the settings values | tell me which setting you changed |

## Decisions and limitations

- **Small synthetic labels:** a failure points at one rule. Real-data behaviour was checked by the commands of Files 19 and 20 (15 classes, 2,017,889 / 504,473 rows).
- **The test of the CV warning uses a 4-row class** (about 3 rows remain in train, fewer than 5 folds).
- **No test of the random choice itself:** only its properties (reproducible, stratified, disjoint, complete).