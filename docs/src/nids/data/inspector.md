# `src/nids/data/inspector.py`

**Location:** `DM_Project/src/nids/data/inspector.py`
**Roadmap step:** File 11 (Stage 1, data loading)

## Purpose

Describes the loaded dataset **without changing it**. It answers what must be known before cleaning (Checkpoint 1): size, classes and their rarity, missing / infinite / negative values per column, constant columns, duplicated rows, and identical feature vectors that carry different labels. It never assumes a column name; everything comes from the data.

## Public contents

| Function | What it returns |
|---|---|
| `inspect_dataset(df, load_report=None)` | `(summary, tables)`. `summary` is a JSON-safe dictionary; `tables` has three DataFrames. `load_report` (from the loader) adds encodings and coerced-value counts |
| `column_profile(df)` | One row per feature: `dtype`, `n_missing`, `n_inf`, `n_negative`, `n_unique`, `is_constant`, `min_finite`, `max_finite` (min and max ignore `inf`/`NaN`) |
| `label_distribution(df)` | Per label: `count`, `percent`, `n_files` (how many source files contain it), `below_min_class_samples` |
| `label_by_file(df)` | Cross table of label x source file |
| `duplicate_statistics(df)` | `n_duplicate_rows`, `n_conflicting_feature_vectors`, `n_rows_in_conflicting_vectors` |
| `odd_label_characters(labels)` | Labels containing non-ASCII characters or `?` |

**`summary` contents:** `n_rows`, `n_feature_columns`, `memory_mb`, `n_source_files`; `labels` (class count, missing labels, classes below `MIN_CLASS_SAMPLES`, classes present in a single file, labels with odd characters); `values` (missing and infinite counts, rows affected, and the lists of columns with missing / infinite / negative values and constant columns); `duplicates`; and `load` when a loader report is given.

**Why these checks matter**
- *Classes in a single file* decide whether the scenario-aware (grouped by file) split is possible for them.
- *Negative values* are only counted, not judged: some features legitimately can be negative. The report shows where; Checkpoint 1 decides.
- *Conflicting feature vectors* (same numbers, different label) cannot be learned by any model and put a ceiling on accuracy; the cleaner decides what to do with them.
- Duplicates are judged on features + label; `source_file` is ignored.

## Inputs and outputs

- **Input:** the DataFrame from `loader.py` (numeric features, `Label`, `source_file`).
- **Output:** the returned `summary` and `tables`. Nothing is written; `scripts/01_inspect_dataset.py` (File 12) saves them. It raises `ValueError` if the label column is missing or a feature column is not numeric.

## Dependencies

Files 04 (`settings`), 05 (`logger`), 07 (`schema`); `numpy`, `pandas`. It is used by File 12.

## How to test

It is tested through File 12 on your real data. A quick check on a small slice (from `DM_Project/`, `.venv` active, PowerShell):

```powershell
python -c "from nids.data.loader import load_raw_data; from nids.data.inspector import inspect_dataset; df, r = load_raw_data(nrows=2000); s, t = inspect_dataset(df, r); print(s['n_rows'], s['n_feature_columns']); print(t['label_distribution'])"
```

**Expected output:** an `Inspecting ...` log line, then `16000 <N>` (8 files x 2000 rows, if each file has that many) and a label table. The numbers and class names are whatever the real files contain; I am not assuming them.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: Label column 'Label' not found` | The frame was not produced by the loader | Use `load_raw_data` |
| `ValueError: Non-numeric feature columns` | Data not loaded through the loader | Use `load_raw_data` (it converts to numbers) |
| `MemoryError` on the full dataset | Duplicate and finite checks make temporary copies (a few GB on top of the data) | Close other programs; this is the heaviest step before training |
| Very slow full run | Per-column unique counts on ~2.8 M rows | Normal: expect roughly a minute |

## Decisions and limitations

- **Read-only.** Fixing anything is the cleaner's job (File 16), after the figures here are reviewed at Checkpoint 1.
- **Hash-based duplicate detection:** rows are compared by a 64-bit hash of their values (fast and memory-friendly). A false match between two different rows is theoretically possible but negligible.
- **Exact duplicates only,** compared on `float64` values as loaded. Near-duplicates are not detected.
- **`n_files` shows where a label occurs, not how the attack was generated;** it is a hint for the scenario split, not proof of independence.
- **Tested on synthetic data with injected problems** (inf, NaN, negative, constant column, duplicate, conflicting labels, rare classes, odd label character); all were reported correctly. It has not yet seen the real files.