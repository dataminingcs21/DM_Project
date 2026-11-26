# Better ensembles: weighted votes and stacking

Every weight and meta-model below was learnt from the **out-of-fold predictions on the training set** (5-fold cross-validation of the baseline models, File 49). The test set (504,473 flows) was used once, to score the finished combiners. The base models are the 11 saved **baseline** models.

## OOF macro F1 of each model (the weights)

| model | oof_macro_f1 |
|---|---:|
| XGBoost | 0.8711 |
| Random Forest | 0.8497 |
| Decision Tree | 0.8465 |
| Gradient Boosting | 0.7097 |
| Logistic Regression | 0.6823 |
| KNN | 0.6781 |
| SVC (RBF kernel) | 0.6687 |
| SGD Classifier | 0.4867 |
| Gaussian Naive Bayes | 0.4628 |
| Bernoulli Naive Bayes | 0.2893 |
| AdaBoost | 0.1758 |

## Comparison on the test set

`oof_macro_f1` is the same combiner scored on the training OOF predictions (empty for the stackers, which were fitted on them). `macro_f1_reliable` leaves out the classes with fewer than 30 test flows (Heartbleed, Sql Injection, Infiltration).

| combiner | oof_macro_f1 | test_macro_f1 | test_macro_f1_reliable | test_accuracy | test_detection_rate | test_false_alarm_rate |
|---|---:|---:|---:|---:|---:|---:|
| Single model: Random Forest | 0.8497 | 0.9150 | 0.9165 | 0.9984 | 0.9958 | 0.0007 |
| Equal vote, top 3 by OOF F1 (XGBoost, Random Forest, Decision Tree) | 0.8627 | 0.9149 | 0.9163 | 0.9986 | 0.9966 | 0.0007 |
| Weighted vote, OOF macro F1^4 | 0.8590 | 0.9134 | 0.9145 | 0.9988 | 0.9983 | 0.0007 |
| Single model: XGBoost | 0.8711 | 0.9120 | 0.9127 | 0.9989 | 0.9988 | 0.0007 |
| Weighted vote, OOF macro F1^1 | 0.8642 | 0.9113 | 0.9118 | 0.9987 | 0.9986 | 0.0008 |
| Equal vote, 11 models (baseline set) | 0.8528 | 0.9095 | 0.9096 | 0.9986 | 0.9982 | 0.0009 |
| Equal vote, top 5 by OOF F1 (XGBoost, Random Forest, Decision Tree, Gradient Boosting, Logistic Regression) | 0.8519 | 0.9076 | 0.9072 | 0.9988 | 0.9983 | 0.0007 |
| Stacking, XGBoost |  | 0.8941 | 0.9019 | 0.9989 | 0.9988 | 0.0007 |
| Stacking, logistic regression |  | 0.8847 | 0.9064 | 0.9988 | 0.9986 | 0.0007 |
| Weighted vote, per-class OOF F1 | 0.7725 | 0.7724 | 0.8613 | 0.9984 | 0.9955 | 0.0007 |

Best: **Single model: Random Forest**, test macro F1 0.9150 vs 0.9095 for the equal vote of the same 11 models (+0.0055).

## Per-class F1 of the difficult classes

| class | test_support | Equal vote, 11 models (baseline set) | Weighted vote, OOF macro F1^1 | Weighted vote, OOF macro F1^4 | Weighted vote, per-class OOF F1 | Equal vote, top 3 by OOF F1 (XGBoost, Random Forest, Decision Tree) | Equal vote, top 5 by OOF F1 (XGBoost, Random Forest, Decision Tree, Gradient Boosting, Logistic Regression) | Stacking, logistic regression | Stacking, XGBoost | Single model: XGBoost | Single model: Random Forest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bot | 391 | 0.844 | 0.846 | 0.858 | 0.549 | 0.858 | 0.858 | 0.851 | 0.854 | 0.862 | 0.838 |
| Web Attack - Brute Force | 294 | 0.777 | 0.786 | 0.786 | 0.821 | 0.771 | 0.796 | 0.771 | 0.803 | 0.773 | 0.773 |
| Web Attack - XSS | 130 | 0.333 | 0.341 | 0.357 | 0.000 | 0.398 | 0.259 | 0.280 | 0.196 | 0.345 | 0.421 |
| Infiltration | 7 | 0.727 | 0.727 | 0.727 | 0.250 | 0.727 | 0.727 | 0.727 | 0.923 | 0.727 | 0.727 |
| Web Attack - Sql Injection | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.667 | 0.667 | 1.000 | 1.000 |

Full table: `ensembles_per_class_f1.csv`.

## Reading the results

* The equal vote is dragged down by the weak models; weighting by OOF F1 (and more so F1^4, or keeping only the top models) lets the tree ensembles decide.
* Per-class weights trust each model only for the classes it handles well.
* Stacking learns combinations (for example *XGBoost says XSS but Random Forest says Brute Force*), which a vote cannot express. The OOF predictions come from models trained on 80% of the training rows, slightly weaker than the final models the stacker is applied to; this is the usual, small stacking bias.
* Differences of a few thousandths are within the noise of the tiny classes (2-7 test flows).