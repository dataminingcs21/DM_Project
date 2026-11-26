# Association rules (FP-Growth)

Transactions: 60,000 class-capped flows of the **training** part (same stratified split and seed as the main pipeline), 88 items. Numeric features are cut into 3 quantile bins learnt on these flows (`=0` is its own item), flags become `<flag>=1`, destination ports are named by service.

## 1. General rules between flow properties

FP-Growth: 96,388 frequent itemsets at minimum support 0.002, 121,822 rules with confidence >= 0.8 and no class item. Top 20 by lift:

| antecedents | consequents | support | confidence | lift | leverage | conviction |
|---|---|---:|---:|---:|---:|---:|
| {Fwd Packet Length Mean=0, Total Length of Bwd Packets=mid} | {Bwd Packet Length Mean=mid, Total Length of Fwd Packets=0} | 0.005 | 1.000 | 215.054 | 0.005 |  |
| {Total Length of Bwd Packets=mid, Total Length of Fwd Packets=0} | {Bwd Packet Length Mean=mid, Fwd Packet Length Mean=0} | 0.005 | 1.000 | 215.054 | 0.005 |  |
| {Bwd Packet Length Mean=mid, Fwd Packet Length Mean=0} | {Total Length of Bwd Packets=mid, Total Length of Fwd Packets=0} | 0.005 | 0.978 | 215.054 | 0.005 | 46.288 |
| {Bwd Packet Length Mean=mid, Total Length of Fwd Packets=0} | {Fwd Packet Length Mean=0, Total Length of Bwd Packets=mid} | 0.005 | 0.978 | 215.054 | 0.005 | 46.288 |
| {Fwd Packet Length Mean=0, Total Length of Bwd Packets=mid} | {Bwd Packet Length Mean=mid, URG Flag=1} | 0.004 | 0.989 | 175.564 | 0.004 | 90.487 |
| {Total Length of Bwd Packets=mid, Total Length of Fwd Packets=0} | {Bwd Packet Length Mean=mid, URG Flag=1} | 0.004 | 0.989 | 175.564 | 0.004 | 90.487 |
| {Bwd Packet Length Mean=mid, Fwd Packet Length Mean=0} | {Total Length of Bwd Packets=mid, URG Flag=1} | 0.004 | 0.968 | 171.789 | 0.004 | 30.825 |
| {Bwd Packet Length Mean=mid, Total Length of Fwd Packets=0} | {Total Length of Bwd Packets=mid, URG Flag=1} | 0.004 | 0.968 | 171.789 | 0.004 | 30.825 |
| {Total Length of Fwd Packets=high, URG Flag=1} | {Fwd Packet Length Mean=high, Total Backward Packets=0} | 0.004 | 0.845 | 163.054 | 0.004 | 6.425 |
| {Fwd Packet Length Mean=high, Total Backward Packets=0} | {Total Length of Fwd Packets=high, URG Flag=1} | 0.004 | 0.842 | 163.054 | 0.004 | 6.314 |
| {Init_Win_bytes_forward=0, Total Length of Bwd Packets=0} | {Bwd Packet Length Mean=0, FIN Flag=1} | 0.006 | 1.000 | 162.602 | 0.006 |  |
| {Bwd Packet Length Mean=0, Init_Win_bytes_forward=0} | {FIN Flag=1, Total Length of Bwd Packets=0} | 0.006 | 1.000 | 162.602 | 0.006 |  |
| {FIN Flag=1, Total Length of Bwd Packets=0} | {Bwd Packet Length Mean=0, Init_Win_bytes_forward=0} | 0.006 | 0.997 | 162.602 | 0.006 | 366.737 |
| {Bwd Packet Length Mean=0, FIN Flag=1} | {Init_Win_bytes_forward=0, Total Length of Bwd Packets=0} | 0.006 | 0.997 | 162.602 | 0.006 | 366.737 |
| {Total Backward Packets=0, Total Length of Fwd Packets=high} | {Flow IAT Mean=high, URG Flag=1} | 0.004 | 0.867 | 157.223 | 0.004 | 7.497 |
| {Total Backward Packets=0, Total Length of Fwd Packets=high} | {Flow Packets/s=low, URG Flag=1} | 0.004 | 0.867 | 155.811 | 0.004 | 7.496 |
| {Fwd Packet Length Mean=high, Total Backward Packets=0} | {Flow IAT Mean=high, URG Flag=1} | 0.004 | 0.820 | 148.629 | 0.004 | 5.523 |
| {Fwd Packet Length Mean=high, Total Backward Packets=0} | {Flow Packets/s=low, URG Flag=1} | 0.004 | 0.820 | 147.294 | 0.004 | 5.523 |
| {Init_Win_bytes_forward=0, Total Backward Packets=low} | {Bwd Packet Length Mean=0, FIN Flag=1} | 0.004 | 0.876 | 142.519 | 0.004 | 8.047 |
| {Init_Win_bytes_forward=0, Total Backward Packets=low} | {FIN Flag=1, Total Length of Bwd Packets=0} | 0.004 | 0.876 | 142.519 | 0.004 | 8.047 |

The highest-lift rules mostly restate one quantity through two features (a total length of 0 implies a mean length of 0); they confirm the redundancy the correlation filter of step 4 removes, rather than reveal new behaviour. The class rules below are the informative part.

## 2. Rules that point to a class

A global minimum support hides small classes, so the itemsets were also mined **inside each class** (at least 30% of that class) and kept when their confidence on all flows is >= 0.8. `class_coverage` = share of the class the rule covers (recall); `confidence` = share of covered flows that really are that class (precision); `lift` = confidence / overall share of the class.

| class | rule | class_coverage | confidence | lift |
|---|---|---:|---:|---:|
| BENIGN | {Bwd Packet Length Mean=mid, Init_Win_bytes_backward=low, Init_Win_bytes_forward=low} -> BENIGN | 0.428 | 1.000 | 7.339 |
| BENIGN | {Init_Win_bytes_forward=low, port=DNS} -> BENIGN | 0.411 | 1.000 | 7.339 |
| Bot | {Init_Win_bytes_backward=high, port=HTTP-alt} -> Bot | 0.380 | 1.000 | 38.412 |
| Bot | {Flow IAT Mean=low, Init_Win_bytes_backward=high, port=HTTP-alt} -> Bot | 0.378 | 1.000 | 38.412 |
| DDoS | {Init_Win_bytes_backward=mid, Init_Win_bytes_forward=mid, Total Length of Bwd Packets=high} -> DDoS | 0.450 | 1.000 | 7.339 |
| DDoS | {Init_Win_bytes_forward=mid, Total Length of Bwd Packets=high, Total Length of Fwd Packets=low} -> DDoS | 0.450 | 1.000 | 7.339 |
| DoS GoldenEye | {Bwd Packet Length Mean=high, Down/Up Ratio=0, Init_Win_bytes_forward=high} -> DoS GoldenEye | 0.484 | 0.991 | 7.278 |
| DoS GoldenEye | {Bwd Packet Length Mean=high, Down/Up Ratio=0, PSH Flag=1} -> DoS GoldenEye | 0.487 | 0.991 | 7.275 |
| DoS Hulk | {Bwd Packet Length Mean=high, Flow Bytes/s=low, Init_Win_bytes_backward=high} -> DoS Hulk | 0.509 | 1.000 | 7.339 |
| DoS Hulk | {Init_Win_bytes_backward=high, Total Fwd Packets=mid, Total Length of Bwd Packets=high} -> DoS Hulk | 0.400 | 1.000 | 7.339 |
| DoS Slowhttptest | {Flow Duration=high, PSH Flag=1, Total Backward Packets=0} -> DoS Slowhttptest | 0.539 | 0.998 | 14.322 |
| DoS Slowhttptest | {PSH Flag=1, Total Backward Packets=0, Total Fwd Packets=high} -> DoS Slowhttptest | 0.539 | 0.998 | 14.322 |
| DoS slowloris | {Flow IAT Mean=mid, PSH Flag=1, Total Backward Packets=0} -> DoS slowloris | 0.305 | 0.988 | 13.760 |
| DoS slowloris | {Flow Packets/s=mid, PSH Flag=1, Total Backward Packets=0} -> DoS slowloris | 0.305 | 0.988 | 13.760 |
| FTP-Patator | {Flow Duration=mid, Init_Win_bytes_forward=high, port=FTP} -> FTP-Patator | 0.668 | 1.000 | 12.642 |
| FTP-Patator | {Flow Packets/s=mid, Init_Win_bytes_forward=high, port=FTP} -> FTP-Patator | 0.668 | 1.000 | 12.642 |
| Heartbleed | {Bwd Packet Length Mean=high, port=444} -> Heartbleed | 1.000 | 1.000 | 6666.667 |
| Heartbleed | {Bwd Packet Length Mean=high, Flow Packets/s=high, port=444} -> Heartbleed | 1.000 | 1.000 | 6666.667 |
| Infiltration | {Fwd Packet Length Mean=high, port=444} -> Infiltration | 0.828 | 1.000 | 2068.966 |
| Infiltration | {Fwd Packet Length Mean=high, Total Length of Fwd Packets=high, port=444} -> Infiltration | 0.828 | 1.000 | 2068.966 |
| PortScan | {Flow Bytes/s=high, Fwd Packet Length Mean=0, PSH Flag=1} -> PortScan | 0.495 | 1.000 | 7.338 |
| PortScan | {Flow Bytes/s=high, Fwd Packet Length Mean=0, Init_Win_bytes_forward=high} -> PortScan | 0.495 | 1.000 | 7.338 |
| SSH-Patator | {Flow Packets/s=mid, port=SSH} -> SSH-Patator | 0.919 | 1.000 | 23.301 |
| SSH-Patator | {Flow IAT Mean=mid, port=SSH} -> SSH-Patator | 0.919 | 1.000 | 23.301 |

Classes with no rule at this confidence: Web Attack - Brute Force, Web Attack - Sql Injection, Web Attack - XSS. Their flows share their typical values with other classes, so no short rule separates them.

Rules found by plain FP-Growth with a class on the right: 6,359 (file `fpgrowth_class_rules.csv`).

## 3. The rules as a classifier

60 class rules, ordered by confidence then coverage; first matching rule wins; no match -> BENIGN. Scored on 30,000 class-capped **test** flows (never used for mining).

Macro F1: **0.583** (XGBoost on all features: about 0.92 on the full test set). Per class:

| class | support | precision | recall | f1 |
|---|---:|---:|---:|---:|
| BENIGN | 5791 | 0.354 | 0.988 | 0.522 |
| Bot | 391 | 1.000 | 0.409 | 0.581 |
| DDoS | 5790 | 1.000 | 0.453 | 0.623 |
| DoS GoldenEye | 2057 | 0.949 | 0.469 | 0.628 |
| DoS Hulk | 5790 | 1.000 | 0.836 | 0.911 |
| DoS Slowhttptest | 1046 | 0.991 | 0.508 | 0.671 |
| DoS slowloris | 1077 | 0.960 | 0.292 | 0.447 |
| FTP-Patator | 1187 | 1.000 | 0.677 | 0.808 |
| Heartbleed | 2 | 1.000 | 1.000 | 1.000 |
| Infiltration | 7 | 1.000 | 0.857 | 0.923 |
| PortScan | 5790 | 0.999 | 0.506 | 0.672 |
| SSH-Patator | 644 | 1.000 | 0.919 | 0.958 |
| Web Attack - Brute Force | 294 | 0.000 | 0.000 | 0.000 |
| Web Attack - Sql Injection | 4 | 0.000 | 0.000 | 0.000 |
| Web Attack - XSS | 130 | 0.000 | 0.000 | 0.000 |

## Limits

* Rules depend on the binning; three quantile bins are coarse by design (readable rules).
* The sample is class-capped: support and lift are measured on it, not on real traffic proportions.
* Port items make many rules easy (FTP-Patator uses port 21); this is the same lab-setup effect as the Destination Port feature (see `reports/experiments/port_ablation/`).