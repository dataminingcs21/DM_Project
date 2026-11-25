# Comparison of true labels and predictions

Generated: 2026-11-22T17:44:45  |  sampling seed 2026  |  mode: 300 random flows  |  models: 11 voting

## What this is

- The flows are **real rows of CIC-IDS2017 from the held-out test split** (the 20% the models never trained on). They are not from another network.
- The true label of every flow is the dataset's own label. Nothing was invented or edited.
- The prediction is the majority vote of the models of the final set, the same answer the app gives for the same CSV file.

## Result

- Flows compared: **300**; refused as invalid input: **0**
- Correct: **299** of 300 (accuracy **0.9967**)
- Macro F1 on these flows: **0.9945** (only classes that occur in this sample or in the predictions count)

Accuracy is dominated by BENIGN and the other large classes when the sample is random. Read the per-class table, not only the total.

## Per class

| class | flows | correct | recall | test_rows_in_dataset |
|---|---|---|---|---|
| BENIGN | 263 | 262 | 0.996 | 419297 |
| DDoS | 11 | 11 | 1.000 | 25603 |
| DoS GoldenEye | 1 | 1 | 1.000 | 2057 |
| DoS Hulk | 19 | 19 | 1.000 | 34570 |
| PortScan | 6 | 6 | 1.000 | 18164 |

## Mistakes (true label to predicted label)

| true_label | predicted_label | flows |
|---|---|---|
| BENIGN | DoS Hulk | 1 |

## How to compare yourself

1. Open the app, tab **Upload a CSV file**, and upload `comparison_input.csv`.
2. Download the app's result, or read it on the page.
3. Compare its prediction column with `comparison_answers.csv` line by line (row n of the answers is row n of the input).
   `comparison.csv` has already done this with the script's own run of the same predictor.
