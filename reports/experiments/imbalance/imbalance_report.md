# Class imbalance: weights, SMOTE and undersampling

Training rows: 2,017,889 (83.1% BENIGN). SMOTE raises every class below 20,000 rows to 20,000 (2,210,534 rows after SMOTE); the undersampling variant first keeps 500,000 random BENIGN rows (1,033,347 rows after SMOTE). Resampling touches training rows only; every variant is scored on the same 504,473 test flows.

## Overall

| model \| variant | train_rows | macro_f1 | macro_f1_reliable | accuracy | detection_rate | false_alarm_rate | fit_seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | none | 2017889 | 0.9141 | 0.9149 | 0.9981 | 0.9951 | 0.0008 |  |
| Decision Tree | balanced weights | 2017889 | 0.9086 | 0.9130 | 0.9983 | 0.9964 | 0.0009 | 65.9210 |
| Decision Tree | SMOTE | 2210534 | 0.8833 | 0.9029 | 0.9979 | 0.9951 | 0.0010 | 94.1980 |
| Decision Tree | undersample + SMOTE | 1033347 | 0.8579 | 0.9015 | 0.9982 | 0.9986 | 0.0014 | 22.3090 |
| Random Forest | none | 2017889 | 0.9150 | 0.9165 | 0.9984 | 0.9958 | 0.0007 |  |
| Random Forest | balanced weights | 2017889 | 0.9019 | 0.9120 | 0.9984 | 0.9959 | 0.0007 | 310.7600 |
| Random Forest | SMOTE | 2210534 | 0.9089 | 0.9147 | 0.9983 | 0.9964 | 0.0009 | 406.6870 |
| Random Forest | undersample + SMOTE | 1033347 | 0.9082 | 0.9137 | 0.9985 | 0.9989 | 0.0011 | 107.2870 |
| XGBoost | none | 2017889 | 0.9120 | 0.9127 | 0.9989 | 0.9988 | 0.0007 |  |
| XGBoost | balanced weights | 2017889 | 0.9082 | 0.9173 | 0.9986 | 0.9999 | 0.0012 | 232.1900 |
| XGBoost | SMOTE | 2210534 | 0.9037 | 0.9129 | 0.9987 | 0.9997 | 0.0010 | 251.7400 |
| XGBoost | undersample + SMOTE | 1033347 | 0.8682 | 0.9116 | 0.9986 | 0.9999 | 0.0012 | 120.9120 |

## F1 of the small classes

Test support: Bot 391, Web Attack - Brute Force 294, Web Attack - XSS 130, Infiltration 7, Web Attack - Sql Injection 4, Heartbleed 2.

| model \| variant | Bot | Web Attack - Brute Force | Web Attack - XSS | Infiltration | Web Attack - Sql Injection | Heartbleed |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree \| none | 0.838 | 0.715 | 0.475 | 0.933 | 0.800 | 1.000 |
| Decision Tree \| balanced weights | 0.842 | 0.733 | 0.431 | 0.923 | 0.750 | 1.000 |
| Decision Tree \| SMOTE | 0.804 | 0.673 | 0.420 | 0.615 | 0.800 | 1.000 |
| Decision Tree \| undersample + SMOTE | 0.770 | 0.682 | 0.421 | 0.588 | 0.462 | 1.000 |
| Random Forest \| none | 0.838 | 0.773 | 0.421 | 0.727 | 1.000 | 1.000 |
| Random Forest \| balanced weights | 0.841 | 0.768 | 0.369 | 0.727 | 0.857 | 1.000 |
| Random Forest \| SMOTE | 0.824 | 0.707 | 0.479 | 0.769 | 0.889 | 1.000 |
| Random Forest \| undersample + SMOTE | 0.809 | 0.696 | 0.494 | 0.769 | 0.889 | 1.000 |
| XGBoost \| none | 0.862 | 0.773 | 0.345 | 0.727 | 1.000 | 1.000 |
| XGBoost \| balanced weights | 0.802 | 0.745 | 0.503 | 1.000 | 0.615 | 1.000 |
| XGBoost \| SMOTE | 0.814 | 0.667 | 0.503 | 1.000 | 0.600 | 1.000 |
| XGBoost \| undersample + SMOTE | 0.802 | 0.677 | 0.497 | 0.750 | 0.667 | 0.667 |

## Reading the results

* Weighting and oversampling move the decision boundary toward the small classes: their recall usually rises, their precision can fall, and BENIGN gets a few more false alarms.
* For tree ensembles that already fit the training data almost perfectly, the effect is small; it is larger for single trees and for classes that overlap (Web Attack Brute Force vs XSS).
* SMOTE interpolates in the scaled log-feature space; synthetic flows are plausible mixes of real ones but are not real traffic.
* Heartbleed, Sql Injection and Infiltration have 2-7 test flows: one flow moves their F1 by 0.15-0.5.