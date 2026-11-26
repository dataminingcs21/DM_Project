# Hermetic model benchmark (step 20)

Sample: 31,143 flows of the cleaned data (at most 3,000 per class, small classes whole), split 80% development / 20% hold-out (stratified, seed 42).
Cross-validation: StratifiedKFold, 5 folds, on the 24,914 development rows. Run time 2.1 min.

**No leakage by construction.** Each candidate is one `Pipeline`: a `ColumnTransformer` (selects the 66 feature columns, drops `Label` / `source_file`, then infinite -> missing, median imputation, signed log1p, constant / correlated column filter, standard scaling) followed by the model. The whole pipeline is cloned and refitted inside every fold, so imputation medians, the kept columns and the scaler are learned on the training folds only.

The sample is class-capped, so macro F1 here is higher than on the real class mix and is not comparable with the full-data scores of step 06; it is the comparison between models that matters.

## Cross-validated comparison (mean over folds)

| display | family | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | accuracy_mean | mcc_mean | detection_rate_mean | false_alarm_rate_mean | fit_seconds_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | ensemble | 0.9170 | 0.0429 | 0.9140 | 0.9754 | 0.9730 | 0.9995 | 0.0050 | 8.2282 |
| XGBoost | ensemble | 0.9044 | 0.0443 | 0.8950 | 0.9750 | 0.9726 | 0.9989 | 0.0079 | 7.3478 |
| Extra Trees | ensemble | 0.9009 | 0.0226 | 0.8858 | 0.9724 | 0.9697 | 0.9984 | 0.0142 | 1.3151 |
| Random Forest | ensemble | 0.8983 | 0.0246 | 0.8884 | 0.9742 | 0.9717 | 0.9980 | 0.0075 | 2.8579 |
| Decision Tree | baseline | 0.8869 | 0.0500 | 0.8943 | 0.9698 | 0.9668 | 0.9984 | 0.0196 | 0.9415 |
| Logistic Regression | baseline | 0.8255 | 0.0132 | 0.8316 | 0.9600 | 0.9565 | 0.9984 | 0.0762 | 3.3266 |
| Dummy (most frequent) | floor | 0.0117 | 0.0000 | 0.0667 | 0.0963 | 0.0000 | 0.0000 | 0.0000 | 0.5395 |

Best model: **LightGBM**. Best simple baseline: **Decision Tree**.

## Paired comparison against the best baseline

Same folds for every model, so the fold-by-fold difference removes the fold difficulty.

| model | display | mean_diff_vs_decision_tree | std_diff | folds_better | folds |
|---|---|---:|---:|---:|---:|
| lightgbm | LightGBM | 0.0301 | 0.0129 | 5 | 5 |
| xgboost | XGBoost | 0.0175 | 0.0075 | 5 | 5 |
| extra_trees | Extra Trees | 0.0140 | 0.0473 | 2 | 5 |
| random_forest | Random Forest | 0.0114 | 0.0573 | 3 | 5 |
| decision_tree | Decision Tree | 0.0000 | 0.0000 | 0 | 5 |
| logistic_regression | Logistic Regression | -0.0614 | 0.0588 | 1 | 5 |
| dummy | Dummy (most frequent) | -0.8752 | 0.0500 | 0 | 5 |

## Error analysis (out-of-fold predictions of the best model)

Most frequent confusions:

| true | predicted | flows | share_of_true_class |
|---|---|---:|---:|
| Web Attack - XSS | Web Attack - Brute Force | 316 | 0.6065 |
| Web Attack - Brute Force | Web Attack - XSS | 217 | 0.1845 |
| DoS Slowhttptest | DoS slowloris | 14 | 0.0058 |
| DoS slowloris | DoS Slowhttptest | 6 | 0.0025 |
| BENIGN | Bot | 3 | 0.0013 |
| Bot | BENIGN | 3 | 0.0019 |
| DoS Hulk | Web Attack - XSS | 3 | 0.0013 |
| DoS Hulk | DoS slowloris | 2 | 0.0008 |

Hardest classes (best F1 reached by any model):

| index | best F1 | support |
|---|---:|---:|
| Web Attack - XSS | 0.4309 | 521 |
| Web Attack - Sql Injection | 0.7879 | 17 |
| Web Attack - Brute Force | 0.8114 | 1176 |
| Infiltration | 0.9825 | 29 |

Reading: the Web Attack Brute Force / XSS pair stays confused by every model, since their flows are nearly identical at the flow-feature level (same server, same port, HTTP requests of similar size). Infiltration, Heartbleed and Sql Injection have only tens of flows, so their F1 moves a lot from fold to fold. The linear baseline fails on the classes that need interactions between features (thresholds on ports and window sizes), which is where tree ensembles gain the most.

## Hold-out (scored once)

LightGBM refitted on the 24,914 development rows and scored on the 6,229 untouched hold-out rows:

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | mcc | detection_rate | false_alarm_rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9734 | 0.9317 | 0.9362 | 0.9727 | 0.9707 | 0.9993 | 0.0083 |

| true | predicted | flows | share_of_true_class |
|---|---|---:|---:|
| Web Attack - XSS | Web Attack - Brute Force | 83 | 0.6336 |
| Web Attack - Brute Force | Web Attack - XSS | 60 | 0.2041 |
| DoS slowloris | DoS Slowhttptest | 4 | 0.0067 |
| DoS Slowhttptest | DoS slowloris | 4 | 0.0067 |
| BENIGN | Bot | 2 | 0.0033 |
| PortScan | SSH-Patator | 2 | 0.0033 |
| DoS GoldenEye | DoS Hulk | 1 | 0.0017 |
| BENIGN | DoS Hulk | 1 | 0.0017 |

Saved: `best_pipeline.joblib` (raw flow columns in, class index out; class names in `best_pipeline.json`) and `holdout_flows.csv` (the hold-out rows with their label, to try batch inference in the app).
