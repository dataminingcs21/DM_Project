# Evaluation report

Generated 2026-11-21T07:58:08.

## Scope

- training rows: 2,017,889
- test rows: 504,473
- classes: 15
- features: 40
- cross-validation: 5-fold on 2,017,889 rows (all training rows)
- tie rule of the vote: prefer_non_benign
- random seed: 42

## Test-set comparison

| Rank | Model | Macro F1 | Macro F1 (classes >= 30 rows) | Balanced acc. | Accuracy | MCC | Detection rate | False alarm rate | ROC-AUC (macro) | CV macro F1 | CV std | Training rows | Subsample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest | 0.9150 | 0.9165 | 0.9013 | 0.9984 | 0.9947 | 99.58% | 0.07% | 0.9996 | 0.8463 | 0.0141 | 2,017,889 | no |
| 2 | Decision Tree | 0.9141 | 0.9149 | 0.9337 | 0.9981 | 0.9938 | 99.51% | 0.08% | 0.9669 | 0.8430 | 0.0317 | 2,017,889 | no |
| 3 | XGBoost | 0.9120 | 0.9127 | 0.8975 | 0.9989 | 0.9963 | 99.88% | 0.07% | 0.9997 | 0.8668 | 0.0062 | 2,017,889 | no |
| 4 | Majority Vote (ensemble) | 0.9095 | 0.9096 | 0.9020 | 0.9986 | 0.9954 | 99.82% | 0.09% | - | 0.8488 | 0.0174 | - | - |
| 5 | Gradient Boosting | 0.7039 | 0.8673 | 0.8430 | 0.9956 | 0.9855 | 99.96% | 0.47% | 0.9602 | 0.7237 | 0.0292 | 100,000 | yes |
| 6 | KNN | 0.6772 | 0.8177 | 0.8581 | 0.9902 | 0.9686 | 99.95% | 1.11% | 0.9825 | 0.6791 | 0.0055 | 100,000 | yes |
| 7 | Logistic Regression | 0.6609 | 0.8261 | 0.6550 | 0.9964 | 0.9882 | 99.03% | 0.14% | 0.9929 | 0.6772 | 0.0257 | 2,017,889 | no |
| 8 | SVC (RBF kernel) | 0.6471 | 0.7330 | 0.8369 | 0.9645 | 0.8978 | 99.91% | 4.19% | 0.9928 | 0.6778 | 0.0234 | 50,000 | yes |
| 9 | SGD Classifier | 0.4864 | 0.6080 | 0.4636 | 0.9896 | 0.9651 | 95.46% | 0.24% | 0.9752 | 0.4867 | 0.0015 | 2,017,889 | no |
| 10 | Gaussian Naive Bayes | 0.4790 | 0.4383 | 0.7804 | 0.4318 | 0.3888 | 99.85% | 64.45% | 0.9869 | 0.4613 | 0.0181 | 2,017,889 | no |
| 11 | Bernoulli Naive Bayes | 0.2880 | 0.3600 | 0.6021 | 0.7525 | 0.5342 | 93.68% | 26.87% | 0.9707 | 0.2894 | 0.0019 | 2,017,889 | no |
| 12 | AdaBoost | 0.1673 | 0.2092 | 0.1589 | 0.9015 | 0.6320 | 58.34% | 1.92% | 0.9433 | 0.1664 | 0.0427 | 2,017,889 | no |

### Notes

1. All rows are scored on the same 504,473 held-out test flows, used once. Ranking metric: macro F1 (every class counts equally; accuracy is dominated by BENIGN).
2. Trained on a class-capped subsample, not on all training rows: Gradient Boosting (100,000 of 2,017,889 rows); KNN (100,000 of 2,017,889 rows); SVC (RBF kernel) (50,000 of 2,017,889 rows). Large classes were capped and small classes kept whole, so the class proportions differ strongly from the real traffic (BENIGN is much less frequent in the subsample). That shifts these models toward predicting attacks, so their false alarm rates and precision are NOT directly comparable with the other models.
3. Classes with fewer than 30 test rows (Heartbleed, Infiltration, Web Attack - Sql Injection) have unreliable scores: one flow changes their F1 a lot. `macro_f1` includes them; `macro_f1_reliable` leaves them out.
4. ROC-AUC of SVC (RBF kernel), SGD Classifier comes from decision scores, not probabilities: valid for ranking, but the scores are not calibrated.
5. The ensemble is a majority vote of the models' predicted labels: one equal vote per model, no probability averaging, deterministic tie-break. It is not a trained model, and it has no ROC-AUC (labels only). Voters (11): KNN, Decision Tree, Random Forest, Gaussian Naive Bayes, SVC (RBF kernel), Bernoulli Naive Bayes, Logistic Regression, SGD Classifier, AdaBoost, Gradient Boosting, XGBoost.
6. No confidence intervals are computed: differences in the third decimal are not evidence that one model is better.
7. Column 'CV macro F1' is the mean over 5 folds on all training rows; the test set was not used for it.

### Observations (computed from the table above)

- Best single model on the test set by macro F1: Random Forest (0.9150).
- The majority vote has macro F1 0.9095: 0.0055 below the best single model.
- 8 single model(s) score more than 0.10 macro F1 below the vote: Gradient Boosting, KNN, Logistic Regression, SVC (RBF kernel), SGD Classifier, Gaussian Naive Bayes, Bernoulli Naive Bayes, AdaBoost.
- Test macro F1 differs from the cross-validation mean by more than 0.05 for: Random Forest, Decision Tree, Majority Vote (ensemble) (each cross-validation model trains on 1 - 1/folds of the training rows and each fold holds only that share of every rare class; these numbers alone do not say which explains the gap).

## Cross-validation (training rows only)

5-fold, stratified. `indistinguishable_from_best` means within one standard deviation of the best mean: the folds cannot separate them.

| Rank | Model | Macro F1 (mean) | Std | Accuracy | Detection rate | False alarm rate | Fit (s) | Within 1 std of best |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | 0.8668 | 0.0062 | 0.9989 | 99.85% | 0.07% | 19.0 | True |
| 2 | Majority Vote (ensemble) | 0.8488 | 0.0174 | 0.9986 | 99.82% | 0.09% | - | False |
| 3 | Random Forest | 0.8463 | 0.0141 | 0.9984 | 99.57% | 0.07% | 86.8 | False |
| 4 | Decision Tree | 0.8430 | 0.0317 | 0.9981 | 99.50% | 0.08% | 66.8 | False |
| 5 | Gradient Boosting | 0.7237 | 0.0292 | 0.9955 | 99.94% | 0.48% | 480.1 | False |
| 6 | KNN | 0.6791 | 0.0055 | 0.9908 | 99.94% | 1.04% | 0.0 | False |
| 7 | SVC (RBF kernel) | 0.6778 | 0.0234 | 0.9679 | 99.91% | 3.78% | 2.7 | False |
| 8 | Logistic Regression | 0.6772 | 0.0257 | 0.9965 | 99.03% | 0.15% | 57.8 | False |
| 9 | SGD Classifier | 0.4867 | 0.0015 | 0.9893 | 95.29% | 0.25% | 7.5 | False |
| 10 | Gaussian Naive Bayes | 0.4613 | 0.0181 | 0.4323 | 99.86% | 64.26% | 0.8 | False |
| 11 | Bernoulli Naive Bayes | 0.2894 | 0.0019 | 0.7511 | 93.59% | 27.05% | 0.9 | False |
| 12 | AdaBoost | 0.1664 | 0.0427 | 0.8879 | 65.46% | 3.31% | 201.1 | False |

## Best single model: Random Forest

### Most frequent confusions (true -> predicted)

- PortScan -> BENIGN: 186 flows (1.02% of that class)
- BENIGN -> PortScan: 176 flows (0.04% of that class)
- Bot -> BENIGN: 78 flows (19.95% of that class)
- Web Attack - XSS -> Web Attack - Brute Force: 73 flows (56.15% of that class)
- Web Attack - Brute Force -> Web Attack - XSS: 61 flows (20.75% of that class)
- DoS Hulk -> BENIGN: 58 flows (0.17% of that class)
- BENIGN -> DoS Hulk: 57 flows (0.01% of that class)
- BENIGN -> Bot: 43 flows (0.01% of that class)

### Classes with unreliable scores (fewer than 30 test rows)

- Heartbleed: 2 test rows, precision 1.000, recall 1.000, F1 1.000
- Infiltration: 7 test rows, precision 1.000, recall 0.571, F1 0.727
- Web Attack - Sql Injection: 4 test rows, precision 1.000, recall 1.000, F1 1.000

## Automatic checks

- [x] all 11 models scored on the test set
- [x] every row of the table is scored on the same test rows ([504473])
- [x] class supports are identical for every model
- [x] no missing or infinite macro F1

## Limits

- The three rarest classes have 2 to 7 test rows: their scores are shown with their support and must not be over-read.
- The train/test split is stratified and random; flows of the same attack campaign can appear on both sides, so the scores can be optimistic compared with a temporal split.
- KNN, SVC and Gradient Boosting were trained on class-capped subsamples; their numbers are not directly comparable (see the notes).
- Test-set results were not used to choose or tune any model.
