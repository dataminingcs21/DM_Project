# `src/nids/data/schema.py`

**Location:** `DM_Project/src/nids/data/schema.py`
**Roadmap step:** File 07 (Stage 1, data loading)

## Purpose

Defines the column and label conventions used by the loader, inspector, cleaner and preprocessing: how names are normalized, which columns are label / metadata / features, and how to compare the columns of two files. CIC-IDS2017 headers are known to be inconsistent in spacing, so every stage must use the same normalization.

**It contains no list of feature names.** The real columns are read from the data at Checkpoint 1; nothing is invented.

## Public contents

| Name | What it does |
|---|---|
| `METADATA_COLUMNS` | `("source_file",)`, taken from settings. Never a feature |
| `normalize_column_name(name)` / `normalize_columns(columns)` | Strip the name and collapse inner whitespace runs into one space |
| `normalize_label_value(value)` | Same whitespace cleanup for a label; `NaN`/`None` stay missing. Touches whitespace only |
| `find_duplicate_columns(columns)` | Names that repeat after normalization (for example `"A"` and `" A"`) |
| `get_feature_columns(columns)` | Normalized columns minus the label and metadata, order kept |
| `has_label(columns)` | Whether `Label` is present after normalization |
| `compare_columns(columns, reference)` | Returns `ColumnComparison(missing, extra, same_order)` with an `is_identical` property |
| `coerce_numeric(df, columns)` | Converts columns to numbers; returns `(new_df, report)` where `report` counts values that could not be converted (they become `NaN`). The input is not modified |

## Inputs and outputs

- **Inputs:** column names, label values or a DataFrame.
- **Outputs:** cleaned names/lists, a comparison object, or a converted copy plus a report. No files are read or written.

## Dependencies

`config/settings.py` (File 04: `LABEL_COLUMN`, `SOURCE_FILE_COLUMN`) and `pandas`. It does not use the logger or `io_utils`.

## How to test

From `DM_Project/` with `.venv` active (PowerShell):

```powershell
python -c "from nids.data.schema import *; print(normalize_columns([' Flow Duration', 'Label ', 'a   b'])); print(get_feature_columns([' Flow Duration', ' Label', 'source_file', 'X'])); print(compare_columns(['a','b','z'], ['a','b','c']))"
```

**Expected output:**

```
['Flow Duration', 'Label', 'a b']
['Flow Duration', 'X']
ColumnComparison(missing=['c'], extra=['z'], same_order=True)
```

(The automated tests arrive with `tests/test_loader.py`, File 10.)

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` or `'config'` | `.venv` not active or `pip install -e .` not run | Activate `.venv`; `pip install -e .` |
| `KeyError` in `coerce_numeric` | A listed column is not in the DataFrame | Normalize the DataFrame's columns first, then pass names from `get_feature_columns` |
| `has_label` returns `False` for a real file | The label column has another name than `settings.LABEL_COLUMN` | Check the real header at Checkpoint 1 and update `LABEL_COLUMN` |

## Decisions and limitations

- **No hardcoded feature list:** avoids inventing columns; the loader compares each file against the first file's columns instead.
- **Label cleanup is whitespace only.** If the real labels contain odd characters (some copies of CIC-IDS2017 do), that is reported at Checkpoint 1 and fixed by an explicit decision, not by a silent replacement.
- **`coerce_numeric` converts text such as `"Infinity"` to `inf`, not `NaN`.** Handling of infinite values belongs to the cleaner (File 16), which counts them.
- **`same_order` is only meaningful when both column sets are identical;** otherwise `missing`/`extra` already describe the difference.