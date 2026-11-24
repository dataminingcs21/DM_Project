# `tests/test_loader.py`

**Location:** `DM_Project/tests/test_loader.py`
**Roadmap step:** File 10 (Stage 1, tests)

## Purpose

Automated tests for the two data-loading files: `nids/data/schema.py` (File 07) and `nids/data/loader.py` (File 08). They use only the synthetic fixtures from `conftest.py` (File 09), so they run in well under a second and never touch your real data.

## What is tested (27 tests)

**`schema.py`**
- Column names are stripped and inner spaces collapsed; labels likewise; missing labels stay missing.
- Duplicates are found after normalization.
- Feature columns exclude the label and `source_file`.
- `compare_columns` reports identical, missing/extra and different-order cases.
- `coerce_numeric` counts unconvertible values, turns `"Infinity"` into `inf` (not `NaN`), skips numeric columns and does not modify its input.

**`loader.py`**
- `list_raw_files`: only CSVs, sorted; clear errors for a missing or empty folder.
- `read_csv_file`: normalized columns, cleaned labels, `source_file` added, coerced-value count, `Infinity` kept for the cleaner, Windows-1252 file decoded (en dash in a label), `nrows`, missing label column, duplicate columns.
- `load_raw_data`: files combined with values aligned by column name even when column order differs, report content, `nrows` per file, error on differing columns, error with no files, and the default folder is `settings.RAW_DATA_DIR`.

## Dependencies

Files 04, 05, 07, 08 and 09, plus `pytest`, `numpy`, `pandas`.

## How to run

From `DM_Project/` with `.venv` active (PowerShell):

```powershell
python -m pytest tests/test_loader.py -q
```

**Expected output:**

```
...........................                                              [100%]
27 passed in 0.xx s
```

For more detail: `python -m pytest tests/test_loader.py -v`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` or `'config'` | `.venv` not active or `pip install -e .` not run | Activate `.venv`; run it |
| `fixture 'raw_csv_dir' not found` | `tests/conftest.py` missing or the test is run from another folder | Run from `DM_Project\` and check File 09 |
| Fails with `UnicodeDecodeError` or a wrong label in the Windows-1252 test | Your Python could not decode with `cp1252` (very unusual) | Send me the traceback |
| Tests pass here but the real data fails to load | Tests use synthetic data | Run the real loader command from File 08 and send me the message |

## Decisions and limitations

- **Synthetic data only.** These tests prove the logic, not that your 8 real files have a consistent structure; that is checked when you run the loader on them (and at Checkpoint 1).
- **Schema and loader share one test file** because they are small and used together; a separate `test_schema.py` would not add clarity.
- **The column-reordering step in `load_raw_data` cannot be detected by a test.** `pandas.concat` already aligns columns by name, so the explicit reordering is only a safeguard. The test checks the behaviour that matters: values from a file with a different column order land in the right columns.