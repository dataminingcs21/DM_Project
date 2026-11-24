# `src/nids/cleaning/cleaner.py`

**Location:** `DM_Project/src/nids/cleaning/cleaner.py`
**Roadmap step:** File 16 (Stage 3, cleaning). It also required a revision of `config/settings.py` (new cleaning settings, see `docs/config/settings.md`).

## Purpose

Cleans the loaded dataset by explicit, counted rules and returns a report of exactly what changed. The rules come from the real inspection report (Checkpoint 1). The numbers they produce are confirmed at **Checkpoint 2**, after `scripts/03_clean_data.py` (File 18) runs them on your data.

## What it does (in this order)

| # | Step | Rule | Counted |
|---|---|---|---|
| 1 | `drop_missing_labels` | Remove rows with no label | rows removed |
| 2 | `apply_label_map` | Replace whole labels from `settings.LABEL_MAP` (the three `Web Attack � X` labels become `Web Attack - X`). Warns about any other label with odd characters | rows changed, map entries not found, odd labels left |
| 3 | `drop_identical_columns` | Drop a feature column whose values are exactly equal to an earlier column (`NaN` equals `NaN`). Lossless | groups found, columns dropped |
| 4 | `handle_negative_values` | In `NON_NEGATIVE_COLUMNS`, set negatives to missing (policy `to_nan`) or only count them (`keep`). Listed columns that are absent (for example a dropped duplicate) are reported, not an error | values per column, rows affected, **rows affected per class** |
| 5 | `handle_infinite_values` | Turn `inf` and `-inf` into missing | values per column, rows affected, **per class** |
| 6 | `remove_exact_duplicates` | Remove rows identical in all features and label, keeping the first; `source_file` is ignored | rows removed, **per class** |
| 7 | `handle_conflicting_vectors` | Find identical feature vectors that carry more than one label. Policy `keep` only reports; `drop` removes all their rows | vectors, rows, **label combinations** (for example `BENIGN \| DoS`), rows per class |

The order matters: infinite and impossible values are made missing **before** duplicates are judged, so two rows that differ only by `inf` versus a blanked value count as the same row.

## What it deliberately does not do

- **No imputation.** Missing values stay missing; the imputer is fitted on the training part only (File 23), so nothing leaks from the test set. The report lists the missing values that remain.
- **No scaling, encoding or feature selection.**
- **No cast to `float32`.** Duplicates are judged on exact `float64` values; the cast happens when the arrays are built.

## Public contents

| Function | Returns |
|---|---|
| `clean_dataset(df)` | `(cleaned, report, tables)`. The input frame is not modified. `tables["class_counts"]` has, per class: `rows_before` (after the missing-label drop and the label fixes), `rows_after`, `rows_removed`, `removed_percent`, `below_min_class_samples` |
| the seven step functions | each returns `(df, report_section)`; they change the frame they receive, so call them through `clean_dataset` |
| `find_identical_columns(df, features)` | groups of identical columns |

The report (JSON-safe) contains `rows_in`, `rows_out`, feature counts in and out, one section per step, the remaining missing values, the class counts before and after, the classes below `MIN_CLASS_SAMPLES`, and the cleaning `config` used.

## Inputs and outputs

- **Input:** the DataFrame from `loader.py`. `ValueError` if the label is missing or a feature is not numeric.
- **Output:** the cleaned frame (index reset), the report and the class table. Nothing is written to disk; `scripts/03_clean_data.py` (File 18) saves them.

## Dependencies

Files 04 (revised), 05, 07 and 11 (`odd_label_characters`); `numpy`, `pandas`.

## How to test

The automated tests come with File 17 (`tests/test_cleaner.py`) and the full run with File 18. In the sandbox I tested synthetic data with every problem injected (missing label, odd label, identical column, negatives, `inf`, duplicate, conflicting vector): each was found, counted and handled as described, the input was unchanged, both policies of `NEGATIVE_VALUE_POLICY` and `CONFLICT_POLICY` worked, and 1,000,000 rows x 78 columns took about 6 seconds and 2.7 GB. It has not yet run on the real data.

Expected size on the real data: roughly 3 times that, so about 15 to 20 seconds and 6 to 7 GB of RAM peak (the input and a working copy both live in memory). Close other programs before File 18.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: Non-numeric feature columns` | Data not loaded with `loader.py` | Use `load_raw_data` |
| `ValueError: Label column ... not found` | Wrong frame | Use the loader output |
| `MemoryError` | Not enough free RAM for the data plus the working copy | Close other programs |
| Warning `Labels with odd characters that are not in LABEL_MAP` | A label with a strange character that has no entry | Add it to `LABEL_MAP` after checking the real text |
| A `NON_NEGATIVE_COLUMNS` name listed under `columns_not_present` | The column was renamed, dropped as identical, or is absent | Expected for `Fwd Header Length.1` once it is dropped as an identical column |

## Decisions and limitations

- **Negatives become missing, rows are kept.** The rows affected are few (the largest group was about 0.1% of the data in the inspection), and dropping rows could remove rare-class rows. The per-class counts in the report let you check this at Checkpoint 2.
- **`Init_Win_bytes_*` keep their `-1`.** It behaves like a placeholder code (minimum exactly -1, maximum 65,535), not a measurement error.
- **Duplicates are removed before the split,** so identical rows can never sit in both the training and the test part. Which duplicate is kept is arbitrary (the first), and `source_file` then names its file.
- **Conflicting vectors:** `CONFLICT_POLICY = "keep"` for now; the label combinations in the report show which classes are confused by identical traffic.
- **Hash-based grouping for conflicts:** vectors are compared through a 64-bit hash of their values; a false match is theoretically possible and negligible. Exact duplicate removal uses pandas' exact comparison, not a hash.
- **Class counts "before" are taken after the label fixes,** so a renamed class lines up with its counts after cleaning.