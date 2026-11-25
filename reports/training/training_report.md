# Training report (baseline models)

Generated: 2026-11-20T07:14:45  
Training rows available: 2,017,889; features: 40; classes: 15  
Class weights: `none`; seed 42; threads 14; XGBoost device setting `auto`  
Libraries: Python 3.13.15, scikit-learn 1.8.0, XGBoost 3.1.3, NumPy 2.3.5  
Training time of this session: 17.6 min (1056 s)

## Results

| Model | Status | Fit (s) | Predict 5,000 rows (s) | Rows used | Subsample | Converged | Saved size (MB) | Smoke-test accuracy* |
|---|---|---:|---:|---:|---|---|---:|---:|
| KNN | trained | 0.0 | 3.27 | 100,000 | yes, of 2,017,889 |  | 16.02 | 0.921 |
| Decision Tree | trained | 77.6 | 0.00 | 2,017,889 | no |  | 1.34 | 0.998 |
| Random Forest | trained | 105.0 | 0.09 | 2,017,889 | no |  | 122.26 | 0.999 |
| Gaussian Naive Bayes | trained | 1.2 | 0.04 | 2,017,889 | no |  | 0.01 | 0.574 |
| SVC (RBF kernel) | trained | 3.1 | 3.31 | 50,000 | yes, of 2,017,889 |  | 2.57 | 0.875 |
| Bernoulli Naive Bayes | trained | 1.2 | 0.01 | 2,017,889 | no |  | 0.01 | 0.473 |
| Logistic Regression | trained | 75.9 | 0.00 | 2,017,889 | no | yes | 0.01 | 0.753 |
| SGD Classifier | trained | 10.2 | 0.00 | 2,017,889 | no | yes | 0.0 | 0.493 |
| AdaBoost | trained | 253.9 | 0.04 | 2,017,889 | no |  | 0.05 | 0.198 |
| Gradient Boosting | trained | 490.6 | 0.09 | 100,000 | yes, of 2,017,889 |  | 1.79 | 0.917 |
| XGBoost | trained | 28.6 | 0.03 | 2,017,889 | no |  | 2.27 | 0.942 |

\* Smoke test only: accuracy on up to 200 **training** rows per class, which the models have seen. It checks that a model is not broken; it is **not** a performance measure. Real scores come from the test set in step 6.

## Models trained on a subsample

Class-capped stratified subsample (small classes kept whole, large classes capped): KNN (100,000 of 2,017,889 rows), SVC (RBF kernel) (50,000 of 2,017,889 rows), Gradient Boosting (100,000 of 2,017,889 rows). The subsample is more balanced than the real traffic, so these models learn a different class prior and their test scores are not strictly comparable with the models trained on all rows.

## Automatic checks

- PASS: no model failed
- PASS: KNN: reloads from disk (checksum, freshness)
- PASS: KNN: predictions and scores have the right shape and range
- PASS: KNN: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)
- PASS: Decision Tree: reloads from disk (checksum, freshness)
- PASS: Decision Tree: predictions and scores have the right shape and range
- PASS: Decision Tree: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)
- PASS: Random Forest: reloads from disk (checksum, freshness)
- PASS: Random Forest: predictions and scores have the right shape and range
- PASS: Random Forest: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)
- PASS: Gaussian Naive Bayes: reloads from disk (checksum, freshness)
- PASS: Gaussian Naive Bayes: predictions and scores have the right shape and range
- PASS: Gaussian Naive Bayes: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)
- PASS: SVC (RBF kernel): reloads from disk (checksum, freshness)
- PASS: SVC (RBF kernel): predictions and scores have the right shape and range
- PASS: SVC (RBF kernel): is not a constant predictor (predicts at least 2 classes) (13 distinct classes predicted)
- PASS: Bernoulli Naive Bayes: reloads from disk (checksum, freshness)
- PASS: Bernoulli Naive Bayes: predictions and scores have the right shape and range
- PASS: Bernoulli Naive Bayes: is not a constant predictor (predicts at least 2 classes) (14 distinct classes predicted)
- PASS: Logistic Regression: reloads from disk (checksum, freshness)
- PASS: Logistic Regression: predictions and scores have the right shape and range
- PASS: Logistic Regression: is not a constant predictor (predicts at least 2 classes) (13 distinct classes predicted)
- PASS: SGD Classifier: reloads from disk (checksum, freshness)
- PASS: SGD Classifier: predictions and scores have the right shape and range
- PASS: SGD Classifier: is not a constant predictor (predicts at least 2 classes) (9 distinct classes predicted)
- PASS: AdaBoost: reloads from disk (checksum, freshness)
- PASS: AdaBoost: predictions and scores have the right shape and range
- PASS: AdaBoost: is not a constant predictor (predicts at least 2 classes) (4 distinct classes predicted)
- PASS: Gradient Boosting: reloads from disk (checksum, freshness)
- PASS: Gradient Boosting: predictions and scores have the right shape and range
- PASS: Gradient Boosting: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)
- PASS: XGBoost: reloads from disk (checksum, freshness)
- PASS: XGBoost: predictions and scores have the right shape and range
- PASS: XGBoost: is not a constant predictor (predicts at least 2 classes) (15 distinct classes predicted)

Checks passed: 34/34
