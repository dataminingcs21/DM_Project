# Model comparison (test set)

| Rank | Model | Macro F1 | Macro F1 (classes >= 30 rows) | Balanced acc. | Accuracy | MCC | Detection rate | False alarm rate | ROC-AUC (macro) | Training rows | Predict (s) | Subsample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest | 0.9150 | 0.9165 | 0.9013 | 0.9984 | 0.9947 | 99.58% | 0.07% | 0.9996 | 2,017,889 | 1.7 | no |
| 2 | Decision Tree | 0.9141 | 0.9149 | 0.9337 | 0.9981 | 0.9938 | 99.51% | 0.08% | 0.9669 | 2,017,889 | 0.1 | no |
| 3 | XGBoost | 0.9120 | 0.9127 | 0.8975 | 0.9989 | 0.9963 | 99.88% | 0.07% | 0.9997 | 2,017,889 | 1.4 | no |
| 4 | Majority Vote (ensemble) | 0.9095 | 0.9096 | 0.9020 | 0.9986 | 0.9954 | 99.82% | 0.09% | - | - | - | - |
| 5 | Gradient Boosting | 0.7039 | 0.8673 | 0.8430 | 0.9956 | 0.9855 | 99.96% | 0.47% | 0.9602 | 100,000 | 11.8 | yes |
| 6 | KNN | 0.6772 | 0.8177 | 0.8581 | 0.9902 | 0.9686 | 99.95% | 1.11% | 0.9825 | 100,000 | 41.5 | yes |
| 7 | Logistic Regression | 0.6609 | 0.8261 | 0.6550 | 0.9964 | 0.9882 | 99.03% | 0.14% | 0.9929 | 2,017,889 | 0.1 | no |
| 8 | SVC (RBF kernel) | 0.6471 | 0.7330 | 0.8369 | 0.9645 | 0.8978 | 99.91% | 4.19% | 0.9928 | 50,000 | 162.9 | yes |
| 9 | SGD Classifier | 0.4864 | 0.6080 | 0.4636 | 0.9896 | 0.9651 | 95.46% | 0.24% | 0.9752 | 2,017,889 | 0.1 | no |
| 10 | Gaussian Naive Bayes | 0.4790 | 0.4383 | 0.7804 | 0.4318 | 0.3888 | 99.85% | 64.45% | 0.9869 | 2,017,889 | 1.9 | no |
| 11 | Bernoulli Naive Bayes | 0.2880 | 0.3600 | 0.6021 | 0.7525 | 0.5342 | 93.68% | 26.87% | 0.9707 | 2,017,889 | 0.3 | no |
| 12 | AdaBoost | 0.1673 | 0.2092 | 0.1589 | 0.9015 | 0.6320 | 58.34% | 1.92% | 0.9433 | 2,017,889 | 3.3 | no |

## Notes

1. All rows are scored on the same 504,473 held-out test flows, used once. Ranking metric: macro F1 (every class counts equally; accuracy is dominated by BENIGN).
2. Trained on a class-capped subsample, not on all training rows: Gradient Boosting (100,000 of 2,017,889 rows); KNN (100,000 of 2,017,889 rows); SVC (RBF kernel) (50,000 of 2,017,889 rows). Large classes were capped and small classes kept whole, so the class proportions differ strongly from the real traffic (BENIGN is much less frequent in the subsample). That shifts these models toward predicting attacks, so their false alarm rates and precision are NOT directly comparable with the other models.
3. Classes with fewer than 30 test rows (Heartbleed, Infiltration, Web Attack - Sql Injection) have unreliable scores: one flow changes their F1 a lot. `macro_f1` includes them; `macro_f1_reliable` leaves them out.
4. ROC-AUC of SVC (RBF kernel), SGD Classifier comes from decision scores, not probabilities: valid for ranking, but the scores are not calibrated.
5. The ensemble is a majority vote of the models' predicted labels: one equal vote per model, no probability averaging, deterministic tie-break. It is not a trained model, and it has no ROC-AUC (labels only). Voters (11): KNN, Decision Tree, Random Forest, Gaussian Naive Bayes, SVC (RBF kernel), Bernoulli Naive Bayes, Logistic Regression, SGD Classifier, AdaBoost, Gradient Boosting, XGBoost.
6. No confidence intervals are computed: differences in the third decimal are not evidence that one model is better.
7. Column 'CV macro F1' is the mean over 5 folds on all training rows; the test set was not used for it.
