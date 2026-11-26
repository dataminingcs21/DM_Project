# Label granularity: 15 classes, 8 families, binary

## Families

| class | family |
|---|---|
| BENIGN | BENIGN |
| Bot | Bot |
| DDoS | DDoS |
| DoS GoldenEye | DoS |
| DoS Hulk | DoS |
| DoS Slowhttptest | DoS |
| DoS slowloris | DoS |
| FTP-Patator | Brute Force |
| Heartbleed | DoS |
| Infiltration | Infiltration |
| PortScan | PortScan |
| SSH-Patator | Brute Force |
| Web Attack - Brute Force | Web Attack |
| Web Attack - Sql Injection | Web Attack |
| Web Attack - XSS | Web Attack |

## Results on the test set

For the binary task, `macro_f1` averages the F1 of BENIGN and of attack; `detection_rate` and `false_alarm_rate` are the security view (share of attacks caught, share of normal flows flagged).

| model | task | variant | macro_f1 | macro_f1_reliable | accuracy | detection_rate | false_alarm_rate |
|---|---|---|---:|---:|---:|---:|---:|
| Random Forest | 15 classes | trained on 15 classes | 0.9150 | 0.9165 | 0.9984 | 0.9958 | 0.0007 |
| Random Forest | 8 families | trained on families | 0.9424 | 0.9731 | 0.9987 | 0.9959 | 0.0007 |
| Random Forest | 8 families | 15-class model, collapsed | 0.9422 | 0.9730 | 0.9987 | 0.9958 | 0.0007 |
| Random Forest | binary | trained binary | 0.9977 | 0.9977 | 0.9987 | 0.9959 | 0.0007 |
| Random Forest | binary | 15-class model, collapsed | 0.9977 | 0.9977 | 0.9987 | 0.9958 | 0.0007 |
| XGBoost | 15 classes | trained on 15 classes | 0.9120 | 0.9127 | 0.9989 | 0.9988 | 0.0007 |
| XGBoost | 8 families | trained on families | 0.9451 | 0.9762 | 0.9992 | 0.9987 | 0.0007 |
| XGBoost | 8 families | 15-class model, collapsed | 0.9463 | 0.9776 | 0.9992 | 0.9988 | 0.0007 |
| XGBoost | binary | trained binary | 0.9987 | 0.9987 | 0.9992 | 0.9987 | 0.0006 |
| XGBoost | binary | 15-class model, collapsed | 0.9987 | 0.9987 | 0.9992 | 0.9988 | 0.0007 |

## F1 per family

| class | test_support | Random Forest \| trained on families | Random Forest \| 15-class, collapsed | XGBoost \| trained on families | XGBoost \| 15-class, collapsed |
|---|---:|---:|---:|---:|---:|
| BENIGN | 419297 | 0.9992 | 0.9992 | 0.9995 | 0.9995 |
| Bot | 391 | 0.8411 | 0.8380 | 0.8539 | 0.8619 |
| Brute Force | 1831 | 0.9986 | 0.9986 | 0.9995 | 0.9992 |
| DDoS | 25603 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| DoS | 38752 | 0.9981 | 0.9981 | 0.9987 | 0.9988 |
| Infiltration | 7 | 0.7273 | 0.7273 | 0.7273 | 0.7273 |
| PortScan | 18164 | 0.9898 | 0.9897 | 0.9945 | 0.9945 |
| Web Attack | 428 | 0.9848 | 0.9871 | 0.9871 | 0.9895 |

## Reading the results

* Families remove the hardest distinctions (Web Attack Brute Force vs XSS, the DoS variants) and absorb the three classes with 2-7 test flows, so their macro F1 is both higher and far more reliable.
* If the collapsed 15-class model is as good as the model trained on families, nothing is lost by keeping the fine labels: the families can always be derived afterwards.
* For an alerting system the binary view matters most: how many attacks are missed and how many false alarms an analyst must look at.