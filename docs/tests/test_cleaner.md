# `tests/test_cleaner.py`

**Location:** `DM_Project/tests/test_cleaner.py`
**Roadmap step:** File 17 (Stage 3, tests)

## Purpose

Automated tests for the cleaner (`nids/cleaning/cleaner.py`, File 16) and for the cleaning settings added to `config/settings.py`. They use only synthetic data. The settings that name real dataset columns (`LABEL_MAP`, `NON_NEGATIVE_COLUMNS`) are replaced by synthetic names for the tests (an automatic fixture), so the tests check the cleaning **logic**, not the real column list. They run in under a second and never touch your real data.

## What is tested (27 tests)

- **Labels:** rows with a missing label are dropped; the label map replaces whole labels and reports them; unused map entries are listed; an odd label that is not in the map is reported and left unchanged.
- **Identical columns:** found, first one kept, `NaN` equals `NaN`; columns that differ slightly are kept; the switch `DROP_IDENTICAL_COLUMNS` turns dropping off (the groups are still reported).
- **Negative values:** only the listed columns are blanked; counts per column and per class; absent listed columns are reported; policy `keep` only reports; zero is not negative.
- **Infinite values:** `inf` and `-inf` become missing, counted per column and per class. Also pinned: `-inf` in a *non-negative* column is counted by the negatives step (which runs first).
- **Duplicates:** exact duplicates removed and the first kept; `source_file` is ignored; the same features with another label are **not** duplicates; `inf`, a blanked negative and `NaN` all count as equal (this proves that the order of the steps is right).
- **Conflicting vectors:** policy `keep` reports vectors, rows and label combinations (`BENIGN | DoS`); policy `drop` removes every row of such vectors; the no-conflict case.
- **Whole cleaner:** the input is not modified and the index is reset; missing values are left for the imputer; the report is JSON-safe and consistent; the class table uses the fixed labels; classes below the minimum are flagged; non-numeric features and a missing label column are rejected.
- **Real settings:** every `LABEL_MAP` key contains `�` and no value does; every non-negative column has a reason; the `Init_Win_bytes_*` columns are not listed; `cleaning_config()` is JSON-safe.

## Dependencies

Files 04 (revised), 05, 07, 09, 11, 16 and `pytest`, `numpy`, `pandas`.

## How to run

From `DM_Project/` with `.venv` active (PowerShell):

```powershell
python -m pytest tests -q
```

**Expected output:**

```
......................................................                   [100%]
54 passed in 0.xx s
```

(27 tests from `test_loader.py` and 27 from this file.) To run only this file: `python -m pytest tests/test_cleaner.py -q`.

I checked that the tests can fail: moving the duplicate removal before the infinite/negative steps, making duplicates ignore the label, and filling missing values with 0 inside the cleaner each made tests fail.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` / `'config'` | `.venv` not active or `pip install -e .` not run | Activate; install |
| `ImportError` for `nids.cleaning.cleaner` or `cannot import name ...` | `cleaner.py` is still an empty placeholder | Paste the code from File 16 and save (check that the file length is not 0) |
| `AttributeError: module 'config.settings' has no attribute 'LABEL_MAP'` (or `NON_NEGATIVE_COLUMNS`, ...) | The old `settings.py` is still in place | Replace it with the revised one from File 16 |
| `test_defaults` fails | A setting was edited (empty reason, or a `LABEL_MAP` entry without `�`) | Read the assertion; restore the value |
| One test fails after you changed the cleaning rules | The test pins the old rule | Tell me which rule you changed; the test may need to change |

## Decisions and limitations

- **Synthetic columns, not the real list:** tests stay valid if the real list changes. The real list is checked by the dedicated `TestDefaultSettings` test and by the full run on the real data (File 18).
- **Small frames on purpose:** each test builds a few rows so a failure points at one rule.
- **No test of memory or speed:** those were measured by hand (1,000,000 rows x 78 columns: about 6 s and 2.7 GB).
- **No test on the real dataset:** the real behaviour is checked by `scripts/03_clean_data.py` and the numbers at Checkpoint 2.