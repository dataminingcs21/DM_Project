# Evaluation report: tuned models

Generated 2026-11-21T22:44:52.

## Scope

- test rows: 504,473
- classes: 15
- tie rule of the vote: prefer_non_benign
- random seed: 42
- cross-validation: not repeated for tuned models (tuning used cross-validation on training rows only)

## How the final set was chosen

The final set (used by the application) follows the Checkpoint 5 evidence rule: a model uses its tuned version when the tuning folds showed an improvement beyond fold noise, otherwise its baseline. The rule used tuning results only and was fixed before any tuned model had a test score. The test results below are reported beside it and do not change it.

| Model | Final set uses | Reason (tuning evidence) |
|---|---|---|
| KNN | tuned | tuning improvement beyond fold noise |
| Decision Tree | tuned | tuning improvement beyond fold noise |
| Random Forest | baseline | tuning improvement within fold noise |
| Gaussian Naive Bayes | baseline | the baseline setting was already the best |
| SVC (RBF kernel) | tuned | tuning improvement beyond fold noise |
| Bernoulli Naive Bayes | tuned | tuning improvement beyond fold noise |
| Logistic Regression | baseline | tuning improvement within fold noise |
| SGD Classifier | tuned | tuning improvement beyond fold noise |
| AdaBoost | tuned | tuning improvement beyond fold noise |
| Gradient Boosting | tuned | tuning improvement beyond fold noise |
| XGBoost | baseline | tuning improvement within fold noise |

## Tuned vs baseline on the test set (each scored once)

| Model | Tuned parameters (changed) | Baseline macro F1 | Tuned macro F1 | Delta | Baseline false alarms | Tuned false alarms | Baseline detection | Tuned detection |
|---|---|---|---|---|---|---|---|---|
| KNN | weights=distance | 0.6772 | 0.6831 | 0.0059 | 1.11% | 1.01% | 99.95% | 99.97% |
| Decision Tree | class_weight=balanced, criterion=entropy | 0.9141 | 0.8979 | -0.0162 | 0.08% | 0.08% | 99.51% | 99.62% |
| Random Forest | max_features=0.5, n_estimators=300 | 0.9150 | 0.9158 | 0.0008 | 0.07% | 0.07% | 99.58% | 99.63% |
| Gaussian Naive Bayes | same as baseline | 0.4790 | 0.4790 | 0.0000 | 64.45% | 64.45% | 99.85% | 99.85% |
| SVC (RBF kernel) | C=10.0, gamma=0.1 | 0.6471 | 0.7078 | 0.0607 | 4.19% | 2.40% | 99.91% | 99.95% |
| Bernoulli Naive Bayes | alpha=0.1, binarize=0.5 | 0.2880 | 0.3869 | 0.0989 | 26.87% | 25.84% | 93.68% | 91.61% |
| Logistic Regression | C=10.0 | 0.6609 | 0.7138 | 0.0529 | 0.14% | 0.14% | 99.03% | 99.04% |
| SGD Classifier | alpha=1e-06, loss=modified_huber | 0.4864 | 0.6523 | 0.1659 | 0.24% | 0.29% | 95.46% | 97.71% |
| AdaBoost | base_max_depth=4, learning_rate=0.5, n_estimators=100 | 0.1673 | 0.7908 | 0.6234 | 1.92% | 0.05% | 58.34% | 97.47% |
| Gradient Boosting | learning_rate=0.2, max_depth=5 | 0.7039 | 0.6101 | -0.0938 | 0.47% | 5.78% | 99.96% | 99.96% |
| XGBoost | max_depth=8, n_estimators=300 | 0.9120 | 0.9183 | 0.0063 | 0.07% | 0.07% | 99.88% | 99.86% |

## Majority votes (one equal vote per model; ties: prefer_non_benign)

| Vote | Voters | Macro F1 | Macro F1 (classes >= 30 rows) | Balanced acc. | Accuracy | MCC | Detection | False alarms | Tied rows | Unanimous rows |
|---|---|---|---|---|---|---|---|---|---|---|
| Vote of the 11 baseline models | 11 | 0.9095 | 0.9096 | 0.9020 | 0.9986 | 0.9954 | 99.82% | 0.09% | 0.02% | 25.96% |
| Vote of the 11 tuned models | 11 | 0.9142 | 0.9155 | 0.9052 | 0.9987 | 0.9958 | 99.88% | 0.09% | 0.02% | 31.91% |
| Vote of the final set (evidence rule) | 11 | 0.9054 | 0.9138 | 0.9042 | 0.9987 | 0.9958 | 99.89% | 0.09% | 0.02% | 31.91% |

### Observations (computed from the tables above)

- Test macro F1, tuned minus baseline: 7 model(s) higher by more than 0.005, 2 lower by more than 0.005, 2 within 0.005.
- Tuned models that scored LOWER than their baseline on the test set: Decision Tree (-0.0162), Gradient Boosting (-0.0938). The tuning gain did not carry over to these rows.
- Of the 7 model(s) the evidence rule takes in tuned form, 5 also scored higher on the test set than their baseline.
- Models kept at baseline by the rule (tuned minus baseline on the test set): Random Forest (+0.0008), Gaussian Naive Bayes (+0.0000), Logistic Regression (+0.0529), XGBoost (+0.0063). The rule was fixed before these scores existed and is not changed by them.
- Macro F1, final-set vote vs baseline vote: 0.9054 vs 0.9095 (-0.0041).
- Macro F1, tuned vote vs baseline vote: 0.9142 vs 0.9095 (+0.0047).
- Best single tuned model on the test set: XGBoost (0.9183); final-set vote 0.9054.

## Final-set vote: most frequent confusions (true -> predicted)

- BENIGN -> PortScan: 183 flows (0.04% of that class)
- Web Attack - XSS -> Web Attack - Brute Force: 90 flows (69.23% of that class)
- BENIGN -> DoS Hulk: 89 flows (0.02% of that class)
- BENIGN -> Bot: 58 flows (0.01% of that class)
- Bot -> BENIGN: 55 flows (14.07% of that class)
- Web Attack - Brute Force -> Web Attack - XSS: 38 flows (12.93% of that class)
- BENIGN -> DoS GoldenEye: 20 flows (0.00% of that class)
- BENIGN -> DoS Slowhttptest: 16 flows (0.00% of that class)

### Classes with unreliable scores (fewer than 30 test rows)

- Heartbleed: 2 test rows, precision 1.000, recall 1.000, F1 1.000
- Infiltration: 7 test rows, precision 1.000, recall 0.571, F1 0.727
- Web Attack - Sql Injection: 4 test rows, precision 0.800, recall 1.000, F1 0.889

## Automatic checks

- [x] all 11 tuned models scored on the test set
- [x] all 11 baseline test results found for the same test rows
- [x] final set equals the set recorded in Checkpoint 5 (decided before any tuned test score) (rule gave: knn, decision_tree, svm, bernoulli_nb, sgd, adaboost, gradient_boosting)
- [x] final-set vote has all 11 voters
- [x] every score is on the same test rows ([504473])
- [x] class supports are identical for every model
- [x] no missing or infinite macro F1

## Limits

- The three rarest classes have 2 to 7 test rows: their scores must not be over-read.
- Parameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000); some winners sit at the edge of their grid.
- KNN, SVC and Gradient Boosting are trained on class-capped subsamples (their class prior is shifted).
- The split is stratified and random; the scores can be optimistic compared with a temporal split.
- Test-set scores were not used to choose the final set.
