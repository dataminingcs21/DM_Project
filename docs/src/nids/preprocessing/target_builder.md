# `src/nids/preprocessing/target_builder.py`

**Location:** `DM_Project/src/nids/preprocessing/target_builder.py`
**Roadmap step:** File 19 (Stage 4, preprocessing)

## Purpose

Turns the cleaned `Label` column into the **target** the models learn to predict, and gives a fixed mapping between class names and integer codes.

## Why it is needed

- The splitter (File 20), the models and the evaluation all need the same integer codes.
- The target mode (`multiclass` or `binary`) and the rare-class rule must live in one place, so changing a setting changes everything consistently.
- The code order must be fixed: the majority-vote tie rule prefers an attack over BENIGN and then the lowest index.

## Dependencies

`config/settings.py` (`TARGET_MODE`, `RARE_CLASS_POLICY`, `RARE_CLASS_MERGE_LABEL`, `MIN_CLASS_SAMPLES`, `BENIGN_LABEL`), `nids.utils.logger`, `numpy`, `pandas`.

## What is inside

| Name | Role |
|---|---|
| `ATTACK_LABEL` | `settings.MALICIOUS_LABEL` (`"MALICIOUS"`), the positive class in binary mode |
| `LabelCodec` | immutable mapping names <-> integers: `encode`, `decode`, `n_classes`, `benign_index`, `to_dict`. Saved with joblib as `label_encoder.joblib` (File 27) |
| `to_binary(labels)` | BENIGN stays, everything else becomes MALICIOUS |
| `merge_rare_classes(labels, ...)` | merges classes below `MIN_CLASS_SAMPLES` into `Other_Attack` (BENIGN never merged) |
| `build_target(labels, mode=None, rare_policy=None)` | applies the mode and the policy; returns `(target, report)` |
| `make_codec(target)` | creates the codec: **BENIGN = 0, the other classes alphabetically** |

## How it works

1. `build_target` checks the mode and the policy, rejects missing labels and data without BENIGN.
2. Binary mode: `to_binary`. Multiclass with policy `merge`: `merge_rare_classes`. Multiclass with `keep` (**your Checkpoint 2 decision**): the labels are unchanged.
3. The report lists counts before and after, merged classes and classes still below the minimum (a warning is logged for those).
4. `make_codec` fixes the order. `encode` refuses unknown names instead of guessing; `decode` refuses out-of-range codes.

Nothing is learned from the data (class names are not a leak), so this step can run before the split.

## Your current settings

`TARGET_MODE = multiclass`, `RARE_CLASS_POLICY = keep`: the target keeps the **15 classes**, and `classes_below_min_class_samples` is empty (the smallest class, Heartbleed, has 11 rows; the minimum is 7).

## How to test

The automated tests come with File 21. Quick manual check (PowerShell, from `DM_Project\`, `.venv` active):

```powershell
python -c "import pandas as pd; from nids.preprocessing.target_builder import build_target, make_codec; s = pd.read_parquet('data/cleaned/cleaned.parquet', columns=['Label'])['Label']; t, r = build_target(s); c = make_codec(t); print(r['n_classes'], r['n_rows']); print(c.classes[:3], c.benign_index)"
```

**Expected output:** the log line `Target built: mode=multiclass, 15 classes, 2522362 rows`, then `15 2522362`, then `('BENIGN', 'Bot', 'DDoS') 0`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ImportError ... target_builder` | empty placeholder file | paste the code, check the length is not 0 |
| `FileNotFoundError ... cleaned.parquet` | File 18 not run on the full data | run `python scripts/03_clean_data.py` |
| `ValueError: no 'BENIGN' rows found` | wrong column or `--nrows` slice without BENIGN | check the input |
| `ValueError: the merged class ... would have only N rows` | policy `merge` with too few rare rows | keep the policy `keep` or change the settings |
| `ValueError: unknown class name(s)` | a name not in the codec was encoded | rebuild the codec from the same target |

## Decisions and limitations

- **Own `LabelCodec` instead of scikit-learn's `LabelEncoder`:** the class order must not be alphabetical (BENIGN first); a plain class guarantees it and stays easy to save.
- **Merging is refused when it does not help** (the merged class would still be too small) rather than silently producing an unusable class.
- **Binary mode ignores the rare-class policy** (only two classes exist).