# Data mining extensions (scripts 09-17)

**Location:** `src/nids/mining/`, `src/nids/experiments/`, `scripts/09_*.py` to `scripts/17_*.py`, `tests/test_mining.py`, `tests/test_experiments.py`
**Added:** after Checkpoint 5, on top of the finished classification pipeline (scripts 01-08), which is unchanged.

## Why

The main pipeline is supervised classification only. A data mining project also needs the other
classic tasks — **clustering, association rules, anomaly (outlier) detection** — plus an honest look at
how far the classification scores carry. These extensions add both, reusing the cleaned and preprocessed
data; nothing in `data/` or `artifacts/` is rewritten.

## Rules shared by every new script

* **Same data, same split.** Preprocessed arrays come from `nids.training.data_access`; scripts that read the
  cleaned data reuse `nids.preprocessing.splitter.stratified_split` (same seed, same test rows).
* **The test set is used only to score.** Every fitted object (clusters, rule bins, anomaly thresholds,
  ensemble weights, stackers, SMOTE) learns from training rows only.
* **Class-capped samples** (`class_capped_indices`) are used where an algorithm cannot take 2 million rows
  (clustering, rules, projections, importance). Every small class is kept whole, so the samples are *more
  balanced than real traffic*; each report says so.
* **Settings** live in `config/settings.py` ("Data mining extensions" block). None of them is part of
  `preprocessing_config()`, so changing them never makes the preprocessed data or the saved models stale.
* **Outputs** go to `reports/mining/<topic>/` and `reports/experiments/<topic>/`: one `.md` report, its `.json`
  twin, CSV tables and figures. Test-set predictions of the saved models are cached once in
  `reports/experiments/test_predictions/<stage>_<model>.npy`.

## Modules

### `nids.mining.reporting`
`markdown_table`, `save_figure` (Agg backend), `write_report` (Markdown + JSON), `class_names(codec)`.

### `nids.mining.clustering`
| Function | What it does |
|---|---|
| `purity`, `cluster_majority_map`, `apply_majority_map` | label-based judgement of a clustering |
| `cluster_scores` | ARI, NMI, homogeneity, completeness, V-measure, purity; silhouette (random subset) and Davies-Bouldin; noise points excluded from internal scores |
| `holdout_macro_f1` | cluster-majority classifier: map learnt on fit rows, scored on other rows |
| `kmeans_sweep` | k-means for every k of `CLUSTER_K_VALUES` |
| `agglomerative` | Ward hierarchical clustering (O(n²) memory: small sample) |
| `k_distance`, `dbscan_grid` | DBSCAN with `eps` at quantiles of the k-distance curve |
| `cluster_composition` | per cluster: size, majority class and share, second class |

### `nids.mining.anomaly`
One-class detectors with one interface (`fit(X_benign)`, `score(X)`, higher = more anomalous):
`IsolationForestDetector`, `LOFDetector` (novelty mode, brute-force neighbours), `AutoencoderDetector`
(`MLPRegressor` 40-32-16-8-16-32-40, reconstruction MSE). `thresholds_from_calibration` sets the threshold for a
target false-alarm rate as a quantile of held-out BENIGN **training** scores; `evaluate_detector` returns ROC-AUC,
average precision, the operating points and the share of each class flagged.

### `nids.mining.association`
`ItemEncoder` turns cleaned flows (original units) into items: destination port named by service, numeric features
in quantile bins learnt on the mining rows (`=0` its own item; features with ≤ 4 distinct values keep their exact
values), flags as `<flag>=1`, and the class. `mine_rules` (FP-Growth, mlxtend), `rules_with_class_consequent`,
`class_rules` (itemsets mined **inside each class**, then confidence and lift measured on all flows — CBA idea, so
the tiny classes get rules too) and `RuleClassifier` (first matching rule by confidence wins, default BENIGN).

### `nids.mining.projection`
`pca_analysis` (2-D coordinates, scree curve, top loadings), `tsne_map` (PCA to 20 dimensions, then t-SNE).

### `nids.mining.importance`
`builtin_importance` (impurity / gain), `permutation_table` (macro-F1 drop, n repeats), `rank_agreement`
(shared top-k and Spearman correlation of two rankings).

### `nids.experiments.common`
`cached_test_predictions`, `predict_batched`, `fit_and_predict` (new model with baseline parameters),
`score` (macro F1 on all classes and on classes with ≥ 30 test rows, detection and false-alarm rates), `per_class_f1`.

### `nids.experiments.ensembles`
`oof_model_weights`, `oof_class_weights` (from the 5-fold out-of-fold predictions of `06_evaluate.py`),
`weighted_vote` (scalar or per-class weights, optional power; same tie rule as File 53, and identical to the
project's majority vote when weights are equal — tested), `best_subset`, `one_hot_votes`, `fit_stacker`
(logistic regression or XGBoost meta-model).

### `nids.experiments.day_split`
`leave_one_file_out` (binary detector trained on seven capture files, tested on the eighth; the preprocessing
pipeline is refitted inside each fold), `random_split_reference` (same detector, random stratified split),
`cap_benign`, `evaluate_binary`, `short_name`.

### `nids.capture.pcap_flows` and `nids.capture.demo`
`extract_flows(source, max_packets, flag_mode)` reads a `.pcap` / `.pcapng` (path, bytes or file object; Ethernet,
Linux cooked, loopback or raw IP; IPv4 and IPv6; TCP and UDP only) and returns `CaptureFlows`: the 40 model features
per flow, the flow metadata (endpoints, protocol, start time, size) and counters. Flows follow CICFlowMeter:
bidirectional, forward = direction of the first packet, end on FIN/RST or after 120 s, activity timeout 5 s, payload
lengths including the Ethernet padding of minimum-size frames (CICFlowMeter counted it: a bare RST shows 6 bytes in
CIC-IDS2017), times in whole microseconds, sample standard deviations, undefined rates of zero-length flows left
missing. `flag_mode="cic"` (default) reproduces the pattern of the dataset's faulty flag columns (Engelen et al.,
2021): `PSH Flag Count` = the flow starts with a SYN, `ACK Flag Count` = its complement, the other flag columns 0;
`flag_mode="real"` gives the true flags. `build_demo_pcap()` writes a deterministic synthetic capture (web and DNS
traffic, one SYN scan of 300 ports) used by the app's demo button. Tests: `tests/test_pcap_flows.py` (hand-worked
session, UDP, padding, flow boundaries, idle periods, pcapng, invalid files, packet limit, flag modes, demo).

## Scripts

| Script | Task | Main outputs | Typical time |
|---|---|---|---|
| `09_cluster.py` | k-means sweep, Ward, DBSCAN | `reports/mining/clustering/` | 2 min |
| `10_anomaly.py` | Isolation Forest, LOF, autoencoder on BENIGN only | `reports/mining/anomaly/` | 5 min |
| `11_association_rules.py` | FP-Growth general and class rules, rule classifier | `reports/mining/association/` | 2 min |
| `12_explain.py` | PCA, t-SNE, built-in and permutation importance | `reports/mining/explain/` | 10 min |
| `13_ensembles.py` | weighted votes, top-n votes, stacking | `reports/experiments/ensembles/` | 25 min (first run predicts with KNN and SVC) |
| `14_port_ablation.py` | retrain without `Destination Port` | `reports/experiments/port_ablation/` | 10 min |
| `15_day_split.py` | leave-one-capture-file-out binary detection | `reports/experiments/day_split/` | 10 min |
| `16_imbalance.py` | balanced weights, SMOTE, undersampling + SMOTE | `reports/experiments/imbalance/` | 30 min |
| `17_granularity.py` | binary and 8 families vs 15 classes | `reports/experiments/granularity/` | 10 min |

Every script prints its main table and the path of its report; `--help` lists the quick-run options.
Exit codes: 0 done; 1 input missing or stale.

## Dependencies added

`mlxtend==0.25.0` (FP-Growth, association rules) and `imbalanced-learn==0.14.2` (SMOTE), pinned in `requirements.txt`.

## Tests

`tests/test_mining.py` and `tests/test_experiments.py` (36 tests, synthetic data only): hand-worked purity,
perfect-clustering scores, noise handling, calibration quantiles, detectors on planted outliers, item encoding
(zero items, exact values, one bin per feature), class rules and rule classifier, FP-Growth class filter,
permutation importance finding the one useful feature, weighted vote = project majority vote with equal weights,
weights and powers, per-class weights, tie rule, stacking input, scoring of reliable classes, benign capping and
leave-one-file-out on synthetic capture days.

## Limits

* Clustering, rules, projections and importance work on class-capped samples, not on real class proportions.
* Rule items depend on the binning (three quantile bins, chosen for readability).
* One-class thresholds come from BENIGN training rows of the same capture; another network needs recalibration.
* The day split is binary: a held-out day's attack classes do not exist in its training data.
