# Split by capture day: detecting attacks never seen in training

Binary detector (BENIGN vs attack), model **xgboost**. Each fold holds out one capture file; the detector (and the preprocessing pipeline) is trained on the other files, with at most 400,000 BENIGN rows and every attack row. Every attack class sits in exactly one file, so the held-out attacks are **unseen types**.

## Folds

| held_out | unseen_attacks | attack_rows | detection_rate | false_alarm_rate | f1_attack | roc_auc | train_rows | seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Friday-Afternoon-DDos | DDoS | 128016 | 0.6313 | 0.0023 | 0.7732 | 0.9921 | 697862 | 50.7000 |
| Friday-Afternoon-PortScan | PortScan | 90819 | 0.0061 | 0.0054 | 0.0120 | 0.8459 | 735059 | 55.4000 |
| Friday-Morning | Bot | 1953 | 0.0000 | 0.0003 | 0.0000 | 0.9023 | 823925 | 52.0000 |
| Thursday-Afternoon-Infilteration | Infiltration | 36 | 0.0000 | 0.1528 | 0.0000 | 0.6002 | 825842 | 58.7000 |
| Thursday-Morning-WebAttacks | Web Attack - Brute Force, Web Attack - Sql Injection, Web Attack - XSS | 2143 | 0.8665 | 0.0003 | 0.9173 | 0.9996 | 823735 | 59.7000 |
| Tuesday | FTP-Patator, SSH-Patator | 9152 | 0.0022 | 0.0006 | 0.0043 | 0.9311 | 816726 | 56.6000 |
| Wednesday | DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, Heartbleed | 193759 | 0.3021 | 0.0004 | 0.4637 | 0.9968 | 632119 | 52.0000 |

## Same detector, random split (the optimistic setting of the main pipeline)

Detection rate 0.9998, false-alarm rate 0.0010, F1 0.9993, ROC-AUC 1.0000.

Leave-one-day-out, all attack flows pooled: detection rate **0.3329**.

## Per attack class: unseen vs seen

`detected_unseen` = share flagged when the class was never in training; `detected_random_split` = share flagged with the random split, where the class was in training.

| class | held_out_day | rows | detected_unseen | random_split_test_rows | detected_random_split | gap |
|---|---|---:|---:|---:|---:|---:|
| Heartbleed | Wednesday | 11 | 0.000 | 2 | 1.000 | -1.000 |
| FTP-Patator | Tuesday | 5933 | 0.001 | 1187 | 1.000 | -0.999 |
| DoS Slowhttptest | Wednesday | 5228 | 0.002 | 1046 | 1.000 | -0.998 |
| SSH-Patator | Tuesday | 3219 | 0.005 | 644 | 1.000 | -0.995 |
| PortScan | Friday-Afternoon-PortScan | 90819 | 0.006 | 18164 | 1.000 | -0.994 |
| DoS slowloris | Wednesday | 5385 | 0.008 | 1077 | 0.997 | -0.990 |
| Bot | Friday-Morning | 1953 | 0.000 | 391 | 0.977 | -0.977 |
| DoS GoldenEye | Wednesday | 10286 | 0.183 | 2057 | 1.000 | -0.817 |
| Infiltration | Thursday-Afternoon-Infilteration | 36 | 0.000 | 7 | 0.714 | -0.714 |
| DoS Hulk | Wednesday | 172849 | 0.327 | 34570 | 1.000 | -0.673 |
| DDoS | Friday-Afternoon-DDos | 128016 | 0.631 | 25603 | 1.000 | -0.369 |
| Web Attack - Sql Injection | Thursday-Morning-WebAttacks | 21 | 0.667 | 4 | 1.000 | -0.333 |
| Web Attack - Brute Force | Thursday-Morning-WebAttacks | 1470 | 0.850 | 294 | 1.000 | -0.150 |
| Web Attack - XSS | Thursday-Morning-WebAttacks | 652 | 0.910 | 130 | 0.992 | -0.083 |

## Reading the results

* The random split shows how well known attacks are recognised; this experiment shows how well a detector generalises to **new** attack types. The gap is what the random split hides.
* An unseen attack is still caught when it resembles a seen one (e.g. a DoS variant after training on DDoS); attacks with their own pattern (web attacks, infiltration, Bot) are mostly missed.
* The false-alarm rate also changes, because BENIGN traffic of the held-out day was never seen either.
* Compare with the one-class detectors of step 10, which need no attack labels at all.