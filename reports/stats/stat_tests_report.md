# Statistical validation (step 19)

Sample: **22,143 flows** of the cleaned data (2,522,362 rows), at most 2,000 per class, every small class kept whole; seed 42. 66 features tested. Significance level 0.05, Holm-Bonferroni correction across the features of each test.

With thousands of rows almost every p-value is tiny, so the conclusions below rest on the **effect sizes**.

## Summary

| test | title | features_tested | reject_h0_holm | large_effect | top_feature |
|---|---|---:|---:|---:|---|
| chi2 | Chi-square test of independence (binned feature x class) | 58 | 57 | 56 | Destination Port |
| anova | One-way ANOVA (class means) | 63 | 62 | 51 | Subflow Bwd Bytes |
| kruskal | Kruskal-Wallis H test (class distributions) | 63 | 62 | 62 | Destination Port |
| shapiro | Shapiro-Wilk normality test | 62 | 62 | 0 | Total Length of Bwd Packets |
| mannwhitney | Mann-Whitney U test (BENIGN vs attack) | 63 | 56 | 2 | Init_Win_bytes_forward |

## 1. Shapiro-Wilk: are the features normal?

H0 (normal) is rejected for **62 of 62** testable features (on a random subsample of at most 5000 rows). Median W = 0.347 on the raw values and 0.782 after signed log1p. Consequences: (a) the parametric ANOVA is only indicative, so the Kruskal-Wallis test is used as the reference; (b) the log1p step of the preprocessing pipeline is justified; (c) a standard scaler on raw values would be dominated by the extreme flows.

| feature | statistic | statistic_log | skewness | p_holm |
|---|---:|---:|---:|---:|
| Total Length of Bwd Packets | 0.0105 | 0.8249 | 69.6515 | 0.0000 |
| Subflow Bwd Bytes | 0.0105 | 0.8249 | 69.6515 | 0.0000 |
| Subflow Fwd Bytes | 0.0344 | 0.8995 | 50.3400 | 0.0000 |
| Total Length of Fwd Packets | 0.0344 | 0.8995 | 50.3400 | 0.0000 |
| Flow Bytes/s | 0.0425 | 0.9269 | 44.6613 | 0.0000 |
| act_data_pkt_fwd | 0.0470 | 0.8363 | 62.7997 | 0.0000 |
| Total Fwd Packets | 0.0488 | 0.9178 | 44.6914 | 0.0000 |
| Fwd Header Length | 0.0566 | 0.9670 | 48.3482 | 0.0000 |
| Total Backward Packets | 0.0667 | 0.9127 | 44.5215 | 0.0000 |
| Bwd Header Length | 0.0848 | 0.9064 | 44.9928 | 0.0000 |

## 2. Kruskal-Wallis and ANOVA: do the classes differ?

Kruskal-Wallis rejects H0 for **62 of 63** features; 62 have a large effect (epsilon squared >= 0.14). ANOVA rejects H0 for 62. Both tests reach the same decision (H0 rejected) for the same features, so that conclusion does not depend on the normality assumption. Their effect-size rankings agree only moderately (Spearman rho between epsilon squared and eta squared = 0.38): eta squared compares means, which a few extreme flows dominate, while epsilon squared compares ranks. The rank-based ranking is the one to trust here.

| feature | statistic | effect_size | effect | p_holm |
|---|---:|---:|---|---:|
| Destination Port | 18530.2744 | 0.8369 | large | 0.0000 |
| Bwd Packet Length Min | 14878.5492 | 0.6720 | large | 0.0000 |
| Flow IAT Mean | 13998.1913 | 0.6322 | large | 0.0000 |
| Flow Packets/s | 13960.6056 | 0.6307 | large | 0.0000 |
| min_seg_size_forward | 13558.2861 | 0.6123 | large | 0.0000 |
| Bwd Packets/s | 13450.2779 | 0.6075 | large | 0.0000 |
| Fwd Packets/s | 13428.3188 | 0.6065 | large | 0.0000 |
| Flow IAT Max | 13112.9787 | 0.5922 | large | 0.0000 |
| Fwd IAT Mean | 12949.4476 | 0.5848 | large | 0.0000 |
| Flow Duration | 12570.7738 | 0.5677 | large | 0.0000 |
| Fwd IAT Max | 12475.1199 | 0.5634 | large | 0.0000 |
| Idle Mean | 12102.2199 | 0.5466 | large | 0.0000 |
| Idle Max | 12098.1699 | 0.5464 | large | 0.0000 |
| Idle Min | 12033.7821 | 0.5435 | large | 0.0000 |
| Fwd IAT Total | 11914.5490 | 0.5381 | large | 0.0000 |

## 3. Chi-square: is the binned feature independent of the class?

Each feature is cut into 5 quantile bins. Independence is rejected for **57** features; 56 have a large Cramer's V. `Destination Port` is among the strongest: attack types are tied to ports in this capture (see the port ablation, step 14).

| feature | statistic | dof | effect_size | effect | p_holm | note |
|---|---:|---:|---:|---|---:|---|
| Destination Port | 43466.0418 | 42 | 0.8089 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Bwd Packet Length Max | 40104.5919 | 42 | 0.7770 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Bwd Packet Length Std | 26041.5691 | 28 | 0.7668 | large | 0.0000 | 3 bins x 15 classes; 9% of expected counts < 5 |
| Bwd Packet Length Mean | 38628.4164 | 42 | 0.7626 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Avg Bwd Segment Size | 38628.4164 | 42 | 0.7626 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Init_Win_bytes_backward | 48771.0948 | 56 | 0.7420 | large | 0.0000 | 5 bins x 15 classes; 13% of expected counts < 5 |
| Max Packet Length | 35717.2021 | 42 | 0.7333 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Subflow Bwd Bytes | 35713.0554 | 42 | 0.7332 | large | 0.0000 | 4 bins x 15 classes; 10% of expected counts < 5 |
| Total Length of Bwd Packets | 35713.0554 | 42 | 0.7332 | large | 0.0000 | 4 bins x 15 classes; 10% of expected counts < 5 |
| Packet Length Std | 35694.0817 | 42 | 0.7330 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Packet Length Variance | 35692.6659 | 42 | 0.7330 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Packet Length Mean | 34720.5779 | 42 | 0.7230 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Average Packet Size | 34329.1347 | 42 | 0.7189 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| Total Backward Packets | 33466.6144 | 42 | 0.7098 | large | 0.0000 | 4 bins x 15 classes; 12% of expected counts < 5 |
| min_seg_size_forward | 22276.4868 | 28 | 0.7092 | large | 0.0000 | 3 bins x 15 classes; 9% of expected counts < 5 |

## 4. Mann-Whitney U: BENIGN vs attack flows

H0 rejected for **56** features. Positive r: attack flows have larger values. This is the binary question a detector answers; the biggest shifts are on durations, inter-arrival times and window sizes.

| feature | statistic | effect_size | effect | p_holm | note |
|---|---:|---:|---|---:|---|
| Init_Win_bytes_forward | 9177390.0000 | 0.5444 | large | 0.0000 | median BENIGN 118 vs attack 2.92e+04 |
| Min Packet Length | 30386862.5000 | -0.5086 | large | 0.0000 | median BENIGN 6 vs attack 0 |
| Fwd Packet Length Min | 30113304.5000 | -0.4950 | medium | 0.0000 | median BENIGN 6 vs attack 0 |
| Fwd IAT Mean | 10859248.5000 | 0.4609 | medium | 0.0000 | median BENIGN 4 vs attack 7.918e+05 |
| Fwd IAT Max | 10955785.5000 | 0.4561 | medium | 0.0000 | median BENIGN 4 vs attack 3.151e+06 |
| Fwd IAT Std | 10958087.0000 | 0.4560 | medium | 0.0000 | median BENIGN 0 vs attack 9.395e+05 |
| Bwd Packet Length Min | 29174175.0000 | -0.4484 | medium | 0.0000 | median BENIGN 6 vs attack 0 |
| Fwd IAT Total | 11156568.0000 | 0.4461 | medium | 0.0000 | median BENIGN 4 vs attack 5.777e+06 |
| PSH Flag Count | 11289965.5000 | 0.4395 | medium | 0.0000 | median BENIGN 0 vs attack 1 |
| Flow IAT Mean | 11438672.5000 | 0.4321 | medium | 0.0000 | median BENIGN 1.622e+04 vs attack 4.916e+05 |
| Flow Packets/s | 28674886.5000 | -0.4254 | medium | 0.0000 | median BENIGN 84.88 vs attack 2.357 |
| Fwd Packets/s | 28641415.5000 | -0.4219 | medium | 0.0000 | median BENIGN 42.85 vs attack 0.9989 |
| Flow IAT Max | 11752405.0000 | 0.4166 | medium | 0.0000 | median BENIGN 3.127e+04 vs attack 3.272e+06 |
| Flow IAT Std | 11771746.5000 | 0.4156 | medium | 0.0000 | median BENIGN 129.8 vs attack 9.685e+05 |
| Flow Duration | 11793023.5000 | 0.4145 | medium | 0.0000 | median BENIGN 3.741e+04 vs attack 8.431e+06 |

## Limits

- The sample is class-capped: effect sizes describe a balanced mix of classes, not the real traffic (83% BENIGN).
- Flows of one attack session are not independent observations; p-values are therefore optimistic. Effect sizes are less affected.
- The chi-square uses quantile bins; a few cells have expected counts below 5 (the share is given per feature).

Figures: `figures/effect_sizes.png`, `figures/normality_raw_vs_log.png`.
