# Ablation: the models without `Destination Port`

Same baseline parameters, same 2,017,889 preprocessed training rows, 39 features instead of 40. Scored on the same 504,473 test flows. *with* = the saved baseline model.

| model | macro_f1_with | macro_f1_without | delta | reliable_with | reliable_without | detection_with | detection_without | false_alarms_with | false_alarms_without | fit_seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9141 | 0.8788 | -0.0353 | 0.9149 | 0.9041 | 0.9951 | 0.9974 | 0.0008 | 0.0010 | 93.9690 |
| Random Forest | 0.9150 | 0.8719 | -0.0431 | 0.9165 | 0.9126 | 0.9958 | 0.9975 | 0.0007 | 0.0008 | 308.4520 |
| XGBoost | 0.9120 | 0.9065 | -0.0055 | 0.9127 | 0.9059 | 0.9988 | 0.9983 | 0.0007 | 0.0008 | 233.2740 |

## Change of F1 per class (without minus with)

| class | support | Decision Tree | Random Forest | XGBoost |
|---|---:|---:|---:|---:|
| BENIGN | 419297 | 0.0001 | 0.0001 | -0.0001 |
| DoS Hulk | 34570 | -0.0003 | -0.0002 | -0.0004 |
| DDoS | 25603 | -0.0001 | -0.0000 | 0.0000 |
| PortScan | 18164 | 0.0076 | 0.0049 | -0.0000 |
| DoS GoldenEye | 2057 | 0.0002 | -0.0002 | -0.0005 |
| FTP-Patator | 1187 | -0.0029 | -0.0030 | -0.0013 |
| DoS slowloris | 1077 | -0.0019 | 0.0005 | 0.0009 |
| DoS Slowhttptest | 1046 | -0.0416 | -0.0245 | -0.0258 |
| SSH-Patator | 644 | -0.0093 | -0.0031 | -0.0031 |
| Bot | 391 | -0.0192 | -0.0065 | -0.0262 |
| Web Attack - Brute Force | 294 | -0.0158 | 0.0043 | 0.0007 |
| Web Attack - XSS | 130 | -0.0465 | -0.0194 | -0.0262 |
| Infiltration | 7 | -0.2667 | 0.0000 | 0.0000 |
| Web Attack - Sql Injection | 4 | -0.1333 | -0.6000 | 0.0000 |
| Heartbleed | 2 | 0.0000 | 0.0000 | 0.0000 |

Classes that lose the most on average: Web Attack - Sql Injection (-0.244), Infiltration (-0.089), Web Attack - XSS (-0.031), DoS Slowhttptest (-0.031).

## Reading the results

* A small drop means the other flow statistics carry nearly the same information: the models did not depend on the port, and the main results do not rest on a lab artefact.
* A large drop for a class means its recognition leans on the port its attack used in this capture; on another network that part of the score would not carry over.
* Without the port, the scores are the more honest estimate of what the flow statistics alone can do.