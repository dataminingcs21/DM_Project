# Tuning report (Checkpoint 5)

Generated 2026-11-21T16:58:37.

## Scope

- training rows: 2,017,889
- classes: 15
- selection score: prior-corrected macro F1 (plain macro F1 recorded beside it)
- tuning sample: class-capped sample of the training rows (KNN 50,000, Decision Tree 100,000, Random Forest 100,000, Gaussian Naive Bayes 100,000, SVC (RBF kernel) 30,000, Bernoulli Naive Bayes 100,000, Logistic Regression 100,000, SGD Classifier 100,000, AdaBoost 100,000, Gradient Boosting 30,000, XGBoost 100,000)
- random seed: 42
- test set: not read

## Summary

| Model | Grid points | Rows | Folds | Best parameters | Baseline corrected F1 | Best corrected F1 | Improvement | Beyond fold noise | Best plain F1 | Minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| KNN | 8 | 50,000 | 5 | n_neighbors=5, weights=distance | 0.6859 | 0.7142 | 0.0284 | yes | 0.8769 | 0.3 |
| Decision Tree | 16 | 100,000 | 5 | class_weight=balanced, criterion=entropy, min_samples_leaf=1 | 0.7821 | 0.8539 | 0.0717 | yes | 0.9059 | 0.3 |
| Random Forest | 16 | 100,000 | 5 | class_weight=None, max_features=0.5, min_samples_leaf=1, n_estimators=300 | 0.8630 | 0.8842 | 0.0211 | no | 0.9058 | 8.3 |
| Gaussian Naive Bayes | 5 | 100,000 | 5 | var_smoothing=1e-09 | 0.4841 | 0.4841 | 0.0000 | is the baseline | 0.6582 | 0.1 |
| SVC (RBF kernel) | 12 | 30,000 | 5 | C=10.0, class_weight=None, gamma=0.1 | 0.6561 | 0.7590 | 0.1028 | yes | 0.8496 | 0.6 |
| Bernoulli Naive Bayes | 9 | 100,000 | 5 | alpha=0.1, binarize=0.5 | 0.2800 | 0.3687 | 0.0886 | yes | 0.6015 | 0.0 |
| Logistic Regression | 8 | 100,000 | 5 | C=10.0, class_weight=None | 0.6491 | 0.6613 | 0.0122 | no | 0.8099 | 1.8 |
| SGD Classifier | 24 | 100,000 | 5 | alpha=1e-06, class_weight=None, loss=modified_huber | 0.5766 | 0.6486 | 0.0720 | yes | 0.7912 | 4.8 |
| AdaBoost | 12 | 100,000 | 5 | base_max_depth=4, learning_rate=0.5, n_estimators=100 | 0.3098 | 0.8495 | 0.5397 | yes | 0.8876 | 4.2 |
| Gradient Boosting | 4 | 30,000 | 3 | learning_rate=0.2, max_depth=5 | 0.6902 | 0.7387 | 0.0485 | yes | 0.8707 | 3.7 |
| XGBoost | 12 | 100,000 | 5 | learning_rate=0.3, max_depth=8, n_estimators=300 | 0.8710 | 0.8790 | 0.0080 | no | 0.9052 | 6.6 |

Corrected F1 = macro F1 with every held-out row weighted by `full_count / sample_count` of its class, i.e. an estimate for the real class mix. Plain F1 is measured on the balanced sample folds and is shown only for comparison. "Beyond fold noise": the mean improvement over the baseline is larger than the standard deviation of the paired fold differences.

## Observations (computed from the results)

- Improved beyond fold noise (7): KNN, Decision Tree, SVC (RBF kernel), Bernoulli Naive Bayes, SGD Classifier, AdaBoost, Gradient Boosting.
- The baseline setting was already the best (1): Gaussian Naive Bayes.
- A different setting scored higher, but not beyond fold noise (3): Random Forest, Logistic Regression, XGBoost.
- Class weights, Decision Tree: best corrected F1 without class weights 0.8319, with balanced weights 0.8539: balanced is better.
- Class weights, Random Forest: best corrected F1 without class weights 0.8842, with balanced weights 0.8808: no class weighting is better or equal.
- Class weights, SVC (RBF kernel): best corrected F1 without class weights 0.7590, with balanced weights 0.7425: no class weighting is better or equal.
- Class weights, Logistic Regression: best corrected F1 without class weights 0.6613, with balanced weights 0.5930: no class weighting is better or equal.
- Class weights, SGD Classifier: best corrected F1 without class weights 0.6486, with balanced weights 0.5815: no class weighting is better or equal.
- Gaussian Naive Bayes: winner at the edge of the grid (var_smoothing=1e-09 (smallest value tried)); a wider grid might find more.
- Bernoulli Naive Bayes: winner at the edge of the grid (alpha=0.1 (smallest value tried)); a wider grid might find more.
- Logistic Regression: winner at the edge of the grid (C=10.0 (largest value tried)); a wider grid might find more.
- SGD Classifier: winner at the edge of the grid (alpha=1e-06 (smallest value tried)); a wider grid might find more.
- AdaBoost: winner at the edge of the grid (base_max_depth=4 (largest value tried)); a wider grid might find more.

## Grid results per model (top 5 grid points; the baseline is always shown)

### KNN

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | n_neighbors=5, weights=distance | 0.7142 | 0.0412 | 0.8769 | 0.0 | best |
| 2 | n_neighbors=11, weights=distance | 0.7106 | 0.0211 | 0.8641 | 0.0 |  |
| 3 | n_neighbors=3, weights=distance | 0.7071 | 0.0307 | 0.8864 | 0.0 |  |
| 4 | n_neighbors=3, weights=uniform | 0.7034 | 0.0311 | 0.8707 | 0.0 |  |
| 5 | n_neighbors=5, weights=uniform | 0.6859 | 0.0327 | 0.8581 | 0.0 | baseline |

### Decision Tree

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | class_weight=balanced, criterion=entropy, min_samples_leaf=1 | 0.8539 | 0.0129 | 0.9059 | 2.5 | best |
| 2 | class_weight=None, criterion=entropy, min_samples_leaf=5 | 0.8319 | 0.0385 | 0.8982 | 2.3 |  |
| 3 | class_weight=balanced, criterion=entropy, min_samples_leaf=2 | 0.8263 | 0.0203 | 0.9082 | 2.5 |  |
| 4 | class_weight=balanced, criterion=gini, min_samples_leaf=1 | 0.8227 | 0.0128 | 0.8905 | 2.3 |  |
| 5 | class_weight=None, criterion=entropy, min_samples_leaf=2 | 0.8202 | 0.0448 | 0.8985 | 2.3 |  |
| 10 | class_weight=None, criterion=gini, min_samples_leaf=1 | 0.7821 | 0.0466 | 0.9012 | 1.8 | baseline |

### Random Forest

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | class_weight=None, max_features=0.5, min_samples_leaf=1, n_estimators=300 | 0.8842 | 0.0213 | 0.9058 | 11.9 | best |
| 2 | class_weight=balanced, max_features=0.5, min_samples_leaf=1, n_estimators=100 | 0.8808 | 0.0190 | 0.9023 | 5.2 |  |
| 3 | class_weight=None, max_features=0.5, min_samples_leaf=1, n_estimators=100 | 0.8799 | 0.0271 | 0.9001 | 4.1 |  |
| 4 | class_weight=balanced, max_features=0.5, min_samples_leaf=1, n_estimators=300 | 0.8791 | 0.0170 | 0.9028 | 15.1 |  |
| 5 | class_weight=None, max_features=0.5, min_samples_leaf=3, n_estimators=100 | 0.8707 | 0.0226 | 0.8939 | 4.0 |  |
| 10 | class_weight=None, max_features=sqrt, min_samples_leaf=1, n_estimators=100 | 0.8630 | 0.0210 | 0.8815 | 1.5 | baseline |

### Gaussian Naive Bayes

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | var_smoothing=1e-09 | 0.4841 | 0.0125 | 0.6582 | 0.3 | best, baseline |
| 2 | var_smoothing=1e-05 | 0.4627 | 0.0166 | 0.6298 | 0.4 |  |
| 3 | var_smoothing=0.001 | 0.4543 | 0.0125 | 0.6039 | 0.3 |  |
| 4 | var_smoothing=1e-07 | 0.4249 | 0.0156 | 0.6030 | 0.4 |  |
| 5 | var_smoothing=0.1 | 0.2934 | 0.0306 | 0.5299 | 0.4 |  |

### SVC (RBF kernel)

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | C=10.0, class_weight=None, gamma=0.1 | 0.7590 | 0.0342 | 0.8496 | 4.0 | best |
| 2 | C=10.0, class_weight=balanced, gamma=0.1 | 0.7425 | 0.0248 | 0.8365 | 2.9 |  |
| 3 | C=10.0, class_weight=None, gamma=0.01 | 0.7381 | 0.0218 | 0.8352 | 2.4 |  |
| 4 | C=10.0, class_weight=None, gamma=scale | 0.7373 | 0.0192 | 0.8447 | 2.5 |  |
| 5 | C=1.0, class_weight=None, gamma=0.1 | 0.7358 | 0.0195 | 0.8203 | 5.6 |  |
| 9 | C=1.0, class_weight=None, gamma=scale | 0.6561 | 0.0239 | 0.8277 | 4.1 | baseline |

### Bernoulli Naive Bayes

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | alpha=0.1, binarize=0.5 | 0.3687 | 0.0262 | 0.6015 | 0.3 | best |
| 2 | alpha=1.0, binarize=0.5 | 0.3310 | 0.0068 | 0.5682 | 0.3 |  |
| 3 | alpha=10.0, binarize=0.5 | 0.3250 | 0.0051 | 0.5648 | 0.3 |  |
| 4 | alpha=1.0, binarize=1.0 | 0.3158 | 0.0094 | 0.5340 | 0.3 |  |
| 5 | alpha=0.1, binarize=1.0 | 0.3082 | 0.0062 | 0.5425 | 0.3 |  |
| 8 | alpha=1.0, binarize=0.0 | 0.2800 | 0.0068 | 0.4366 | 0.2 | baseline |

### Logistic Regression

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | C=10.0, class_weight=None | 0.6613 | 0.0215 | 0.8099 | 36.3 | best |
| 2 | C=1.0, class_weight=None | 0.6491 | 0.0303 | 0.8131 | 26.2 | baseline |
| 3 | C=10.0, class_weight=balanced | 0.5930 | 0.0229 | 0.8186 | 49.5 |  |
| 4 | C=0.1, class_weight=None | 0.5900 | 0.0433 | 0.7773 | 15.8 |  |
| 5 | C=1.0, class_weight=balanced | 0.5589 | 0.0174 | 0.7923 | 48.0 |  |

### SGD Classifier

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | alpha=1e-06, class_weight=None, loss=modified_huber | 0.6486 | 0.0459 | 0.7912 | 5.2 | best |
| 2 | alpha=1e-06, class_weight=None, loss=hinge | 0.6222 | 0.0468 | 0.7814 | 1.9 |  |
| 3 | alpha=1e-06, class_weight=None, loss=log_loss | 0.6221 | 0.0545 | 0.7844 | 2.7 |  |
| 4 | alpha=1e-05, class_weight=None, loss=log_loss | 0.6089 | 0.0256 | 0.7631 | 1.2 |  |
| 5 | alpha=1e-05, class_weight=None, loss=hinge | 0.6049 | 0.0360 | 0.7621 | 0.8 |  |
| 10 | alpha=0.0001, class_weight=None, loss=hinge | 0.5766 | 0.0188 | 0.7025 | 0.5 | baseline |

### AdaBoost

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | base_max_depth=4, learning_rate=0.5, n_estimators=100 | 0.8495 | 0.0300 | 0.8876 | 105.7 | best |
| 2 | base_max_depth=4, learning_rate=1.0, n_estimators=100 | 0.8289 | 0.0400 | 0.8638 | 84.9 |  |
| 3 | base_max_depth=4, learning_rate=1.0, n_estimators=50 | 0.8087 | 0.0480 | 0.8587 | 55.7 |  |
| 4 | base_max_depth=4, learning_rate=0.5, n_estimators=50 | 0.7993 | 0.0266 | 0.8478 | 57.4 |  |
| 5 | base_max_depth=2, learning_rate=1.0, n_estimators=100 | 0.6251 | 0.0381 | 0.7307 | 62.5 |  |
| 10 | base_max_depth=1, learning_rate=1.0, n_estimators=50 | 0.3098 | 0.0449 | 0.4030 | 18.1 | baseline |

### Gradient Boosting

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | learning_rate=0.2, max_depth=5 | 0.7387 | 0.0812 | 0.8707 | 199.7 | best |
| 2 | learning_rate=0.1, max_depth=5 | 0.7341 | 0.0671 | 0.8740 | 218.5 |  |
| 3 | learning_rate=0.2, max_depth=3 | 0.7029 | 0.0476 | 0.8776 | 148.9 |  |
| 4 | learning_rate=0.1, max_depth=3 | 0.6902 | 0.0750 | 0.8775 | 154.2 | baseline |

### XGBoost

| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |
|---|---|---|---|---|---|---|
| 1 | learning_rate=0.3, max_depth=8, n_estimators=300 | 0.8790 | 0.0178 | 0.9052 | 8.6 | best |
| 2 | learning_rate=0.3, max_depth=8, n_estimators=100 | 0.8781 | 0.0136 | 0.9041 | 3.7 |  |
| 3 | learning_rate=0.3, max_depth=10, n_estimators=300 | 0.8760 | 0.0235 | 0.9003 | 9.3 |  |
| 4 | learning_rate=0.1, max_depth=10, n_estimators=300 | 0.8754 | 0.0263 | 0.9000 | 11.2 |  |
| 5 | learning_rate=0.3, max_depth=10, n_estimators=100 | 0.8746 | 0.0271 | 0.9010 | 3.9 |  |
| 8 | learning_rate=0.3, max_depth=6, n_estimators=100 | 0.8710 | 0.0263 | 0.8982 | 3.0 | baseline |

## Tuned models (refitted on the same rows as their baselines)

| Model | Action | Training rows | Fit s | Baseline fit s | File MB | Note |
|---|---|---|---|---|---|---|
| KNN | refitted | 100000 | 0.0 | 0.0 | 16.02 |  |
| Decision Tree | refitted | 2017889 | 34.7 | 77.6 | 1.4 |  |
| Random Forest | refitted | 2017889 | 1293.0 | 105.0 | 280.16 |  |
| Gaussian Naive Bayes | reused baseline | 2017889 | 1.2 | 1.2 | 0.01 |  |
| SVC (RBF kernel) | refitted | 50000 | 4.0 | 3.1 | 1.83 |  |
| Bernoulli Naive Bayes | refitted | 2017889 | 1.3 | 1.2 | 0.01 |  |
| Logistic Regression | refitted | 2017889 | 87.8 | 75.9 | 0.01 |  |
| SGD Classifier | refitted | 2017889 | 38.5 | 10.2 | 0.0 |  |
| AdaBoost | refitted | 2017889 | 1869.8 | 253.9 | 0.58 |  |
| Gradient Boosting | refitted | 100000 | 803.5 | 490.6 | 5.04 |  |
| XGBoost | refitted | 2017889 | 69.3 | 28.6 | 7.28 |  |

## Automatic checks

- [x] no tuning or refit failed
- [x] KNN: tuned model reloads (checksum and freshness verified)
- [x] KNN: saved parameters equal the winning parameters
- [x] KNN: predicts valid class codes on a training slice
- [x] Decision Tree: tuned model reloads (checksum and freshness verified)
- [x] Decision Tree: saved parameters equal the winning parameters
- [x] Decision Tree: predicts valid class codes on a training slice
- [x] Random Forest: tuned model reloads (checksum and freshness verified)
- [x] Random Forest: saved parameters equal the winning parameters
- [x] Random Forest: predicts valid class codes on a training slice
- [x] Gaussian Naive Bayes: tuned model reloads (checksum and freshness verified)
- [x] Gaussian Naive Bayes: saved parameters equal the winning parameters
- [x] Gaussian Naive Bayes: predicts valid class codes on a training slice
- [x] SVC (RBF kernel): tuned model reloads (checksum and freshness verified)
- [x] SVC (RBF kernel): saved parameters equal the winning parameters
- [x] SVC (RBF kernel): predicts valid class codes on a training slice
- [x] Bernoulli Naive Bayes: tuned model reloads (checksum and freshness verified)
- [x] Bernoulli Naive Bayes: saved parameters equal the winning parameters
- [x] Bernoulli Naive Bayes: predicts valid class codes on a training slice
- [x] Logistic Regression: tuned model reloads (checksum and freshness verified)
- [x] Logistic Regression: saved parameters equal the winning parameters
- [x] Logistic Regression: predicts valid class codes on a training slice
- [x] SGD Classifier: tuned model reloads (checksum and freshness verified)
- [x] SGD Classifier: saved parameters equal the winning parameters
- [x] SGD Classifier: predicts valid class codes on a training slice
- [x] AdaBoost: tuned model reloads (checksum and freshness verified)
- [x] AdaBoost: saved parameters equal the winning parameters
- [x] AdaBoost: predicts valid class codes on a training slice
- [x] Gradient Boosting: tuned model reloads (checksum and freshness verified)
- [x] Gradient Boosting: saved parameters equal the winning parameters
- [x] Gradient Boosting: predicts valid class codes on a training slice
- [x] XGBoost: tuned model reloads (checksum and freshness verified)
- [x] XGBoost: saved parameters equal the winning parameters
- [x] XGBoost: predicts valid class codes on a training slice

## Limits

- Parameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000) and refitted on more rows; they may not be optimal there.
- The grids are small; a winner at the edge of its grid means a wider grid might do better.
- Corrected F1 is an estimate for the real class mix from a modest number of rows of the large classes; it is noisy for the three rarest classes.
- The test set was not read by this step. The tuned models are scored on it once, by `06_evaluate.py --stage tuned`.
