# scripts/08_make_comparison_files.py

## 1. Role
Makes files you can use to compare **true labels** with the **app's predictions**: a random sample of real flows, an answers file with the dataset's own labels, and the predictions of the final model set.

## 2. Why it is needed
You asked for real examples and their real answers. The only real, labelled traffic in this project is CIC-IDS2017. To keep the test fair, the script uses only the **held-out test split** (rows no model trained on). The flows are therefore real and unseen, but they are **from CIC-IDS2017**, not from another network. Nothing is invented: no flow, no label.

## 3. Dependencies
- `data/cleaned/cleaned.parquet` (script 03), `data/preprocessed/y_test.npy` and `label_encoder.joblib` (script 04)
- the final model set (scripts 05 and 07) through `nids.prediction.predictor`
- `nids.preprocessing.splitter.stratified_split`, `target_builder` (same split as script 04)

## 4. Code
`scripts/08_make_comparison_files.py` (complete file delivered with this document).

## 5. How it works
1. Loads the cleaned data and recomputes the stratified 80/20 split with the same settings and seed as script 04.
2. **Checks** that the recomputed test labels equal the saved `y_test.npy`. If not, it stops and asks you to rerun script 04. This proves the flows are held-out rows.
3. Draws the flows at random (`--n`, default 300) or a fixed number per class (`--per-class N`). The draw seed (`--seed`, default 2026) is separate from the split seed.
4. Writes the input CSV with only the 40 model features, no label, every digit kept, so the app reads exactly the same numbers.
5. Predicts the written file with the final set (the same code path as the app's upload tab), builds the comparison and the summary.
6. Reads the written CSV back and checks that the predictions are unchanged.

## 6. Command
```powershell
python scripts\08_make_comparison_files.py
python scripts\08_make_comparison_files.py --per-class 20     # every class gets flows
python scripts\08_make_comparison_files.py --n 1000 --seed 7
```

## 7. Expected output
A line with flows / correct / accuracy, a per-class table, four `[OK]` checks and the folder `reports\comparison\`:

| File | Content |
|---|---|
| `comparison_input.csv` | flows, 40 features, no label: upload it in the app |
| `comparison_answers.csv` | `row`, `true_label`, `cleaned_row_position` |
| `comparison_predictions.csv` | `row`, `prediction`, `top_votes`, `n_voters`, `tied`, one column per model |
| `comparison.csv` | true and predicted label side by side with `correct` |
| `comparison_per_class.csv`, `comparison_summary.md` | per-class results, mistakes, cautions |

Numbers cannot be predicted here. With a purely random draw, rare classes are usually missing; use `--per-class` to test them. Heartbleed, Infiltration and SQL Injection have only 2, 7 and 4 test rows in the whole dataset, so their results are not reliable whatever the sample shows. Accuracy on the held-out split is optimistic anyway: the split is random, so flows of one attack session appear on both sides.

## 8. Common errors
- *The recomputed test split does not equal the saved y_test.npy*: the cleaned data or the settings changed after script 04. Rerun scripts 03 and 04 (and then 05, 07).
- *models could not be loaded*: run scripts 05 and 07 first.
- `FileNotFoundError` for `cleaned.parquet`: run script 03.
- Memory: the script loads the cleaned data once (as script 04 does); close other programs if it fails.# scripts/08_make_comparison_files.py

## 1. Role
Makes files you can use to compare **true labels** with the **app's predictions**: a random sample of real flows, an answers file with the dataset's own labels, and the predictions of the final model set.

## 2. Why it is needed
You asked for real examples and their real answers. The only real, labelled traffic in this project is CIC-IDS2017. To keep the test fair, the script uses only the **held-out test split** (rows no model trained on). The flows are therefore real and unseen, but they are **from CIC-IDS2017**, not from another network. Nothing is invented: no flow, no label.

## 3. Dependencies
- `data/cleaned/cleaned.parquet` (script 03), `data/preprocessed/y_test.npy` and `label_encoder.joblib` (script 04)
- the final model set (scripts 05 and 07) through `nids.prediction.predictor`
- `nids.preprocessing.splitter.stratified_split`, `target_builder` (same split as script 04)

## 4. Code
`scripts/08_make_comparison_files.py` (complete file delivered with this document).

## 5. How it works
1. Loads the cleaned data and recomputes the stratified 80/20 split with the same settings and seed as script 04.
2. **Checks** that the recomputed test labels equal the saved `y_test.npy`. If not, it stops and asks you to rerun script 04. This proves the flows are held-out rows.
3. Draws the flows at random (`--n`, default 300) or a fixed number per class (`--per-class N`). The draw seed (`--seed`, default 2026) is separate from the split seed.
4. Writes the input CSV with only the 40 model features, no label, every digit kept, so the app reads exactly the same numbers.
5. Predicts the written file with the final set (the same code path as the app's upload tab), builds the comparison and the summary.
6. Reads the written CSV back and checks that the predictions are unchanged.

## 6. Command
```powershell
python scripts\08_make_comparison_files.py
python scripts\08_make_comparison_files.py --per-class 20     # every class gets flows
python scripts\08_make_comparison_files.py --n 1000 --seed 7
```

## 7. Expected output
A line with flows / correct / accuracy, a per-class table, four `[OK]` checks and the folder `reports\comparison\`:

| File | Content |
|---|---|
| `comparison_input.csv` | flows, 40 features, no label: upload it in the app |
| `comparison_answers.csv` | `row`, `true_label`, `cleaned_row_position` |
| `comparison_predictions.csv` | `row`, `prediction`, `top_votes`, `n_voters`, `tied`, one column per model |
| `comparison.csv` | true and predicted label side by side with `correct` |
| `comparison_per_class.csv`, `comparison_summary.md` | per-class results, mistakes, cautions |

Numbers cannot be predicted here. With a purely random draw, rare classes are usually missing; use `--per-class` to test them. Heartbleed, Infiltration and SQL Injection have only 2, 7 and 4 test rows in the whole dataset, so their results are not reliable whatever the sample shows. Accuracy on the held-out split is optimistic anyway: the split is random, so flows of one attack session appear on both sides.

## 8. Common errors
- *The recomputed test split does not equal the saved y_test.npy*: the cleaned data or the settings changed after script 04. Rerun scripts 03 and 04 (and then 05, 07).
- *models could not be loaded*: run scripts 05 and 07 first.
- `FileNotFoundError` for `cleaned.parquet`: run script 03.
- Memory: the script loads the cleaned data once (as script 04 does); close other programs if it fails.