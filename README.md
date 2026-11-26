# Network intrusion detection on CIC-IDS2017

A modular Python system that classifies network flows into **BENIGN or one of 14 attack types** with **eleven supervised models**, combines them by **majority vote**, and serves the result in a **Streamlit app** where you can analyse a **packet capture (.pcap / .pcapng)**, type the features of a flow, load a real example, or upload a CSV file.

On top of the classification pipeline, the **data mining extensions** (scripts 09-17) add clustering, unsupervised anomaly detection, association rules, projections and feature importance, and five experiments that test how far the classification scores carry. The whole process is written up in a French PDF report (`reports\report\rapport_data_mining.pdf`, built by script 18).

- **Course:** Data Mining · **Team:** CS21 · **Domain:** network security
- **Tasks covered:** classification, clustering, association rules, anomaly detection
- **Classes:** 15 (BENIGN + 14 attack types) · **Classifiers:** 11 · **Ensemble:** equal-weight majority vote

Every number in this file comes from a report the scripts wrote on the real data; where a figure is not stated here, the named report has it.

## Read this first: what the results do and do not mean

1. **The random split is optimistic.** The 80/20 split is stratified but random. Flows of the same attack session (and near-identical flows) can land on both sides, so the main scores say how well the models recognise flows like those they trained on. The harder test, holding out one capture day at a time, is step 15 (see point 7).
2. **Three classes are too small to judge.** After cleaning, the test set holds **2 Heartbleed, 4 Web Attack - Sql Injection and 7 Infiltration** flows. One flow more or less moves their scores by tens of percent. Scores for these classes, and the all-class macro F1 that includes them, are indicative only. The app and the reports show the test support next to every per-class score.
3. **Three models were trained on a class-capped subsample** (KNN 100,000 rows, SVC 50,000, Gradient Boosting 100,000 of 2,017,889). Their class mix differs from real traffic, so their predictions are shifted toward the small classes.
4. **Tuning used even smaller samples.** Hyperparameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000) and the winners were refitted on the sizes above. Several winners sit at the edge of their grid (Gaussian NB `var_smoothing=1e-9`, Bernoulli NB `alpha=0.1`, Logistic Regression `C=10`, SGD `alpha=1e-6`, AdaBoost depth 4), so a wider grid might do better. The grids were not widened, because the selection rule was fixed beforehand.
5. **The majority vote is not clearly better than its best members.** See "Results". Random Forest alone (0.9150) is as good as any combination tried in step 13.
6. **`Destination Port` is a model feature, and the most important one.** Attack classes tie to specific ports in this lab capture. Without it, macro F1 drops by 0.035 (Decision Tree), 0.043 (Random Forest) and 0.006 (XGBoost); see "Data mining extensions".
7. **Attacks of a type never seen in training are mostly missed.** Holding out one capture day at a time, a binary detector catches 33% of the unseen attack flows (pooled), against 99.98% with the random split.
8. **A prediction is a vote, not proof.** The input check catches impossible values (text, negatives where impossible, a port above 65535) and warns about values outside the training range. It cannot tell whether a combination of values is a realistic flow.

## Data

**Dataset:** CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick ([official page](https://www.unb.ca/cic/datasets/ids-2017.html)). It is **not included** in this repository: download the eight CSV files of network-flow features (78 feature columns plus `Label`) and put them in `data\raw\`. Respect the dataset's terms of use and cite it in any report based on it. The raw data is needed to reproduce the pipeline, **not** to run the already-trained application (see "Open the app from a clone").

| Step | Result |
|---|---|
| Raw dataset | 2,830,743 rows, 78 feature columns, 15 classes, BENIGN 80.3% |
| Smallest raw classes | Heartbleed (11), Web Attack - Sql Injection (21), Infiltration (36) |
| Cleaning (`03_clean_data.py`) | 308,381 exact duplicates (10.89%) and 12 identical copy columns removed; the three web-attack labels with a damaged character fixed; impossible negatives (2,926 rows, 93% BENIGN) and infinite values (2,867 rows, all with a flow duration of 0) set to missing |
| Cleaned dataset | 2,522,362 rows, 66 feature columns |
| Split (`04_preprocess.py`) | stratified 80/20, seed 42: 2,017,889 train / 504,473 test |
| Model features | 40 of 66 (1 constant and 25 correlated columns removed, on training rows only) |

698 feature vectors carry two different labels (1,396 rows, 0.055%); they were kept and are an unavoidable error floor.

## Methodology

```text
 CIC-IDS2017 CSV files
        |
 01 inspection -> 02 exploratory analysis -> 03 cleaning -> 04 preprocessing (split + pipeline fitted on training rows)
                                                                   |
        +----------------------+---------------------+-------------+-------------+-----------------------+
        |                      |                     |                           |                       |
 05 train, 06 evaluate,   09 clustering        10 anomaly detection      11 association rules     12 projections and
 07 tune, final set,      (k-means, Ward,      (BENIGN only: IF, LOF,    (FP-Growth)              feature importance
 majority vote            DBSCAN)              autoencoder)
        |
 08 comparison files, 13-17 critical experiments (ensembles, port ablation, day split, imbalance, granularity)
        |
 Streamlit application (flows, CSV, .pcap)  ·  18 PDF report
```

The preprocessing pipeline (infinite -> missing, median imputation, signed log1p, removal of constant and correlated columns at 0.95, standard scaling) is fitted on the training rows only and saved, with the feature schema, so that the application transforms incoming traffic exactly like the training data. Every later step that learns something (models, clusters, rule bins, anomaly thresholds, ensemble weights, SMOTE) also learns from training rows only; the test set only scores.

## The eleven models

K-Nearest Neighbours, Decision Tree, Random Forest, Gaussian Naive Bayes, SVC (RBF), Bernoulli Naive Bayes, Logistic Regression, SGD Classifier, AdaBoost, Gradient Boosting and XGBoost. Each exists in a **baseline** version (default-style settings, unweighted) and a **tuned** version: GridSearchCV with stratified cross-validation (5 folds, 3 for Gradient Boosting), score = prior-corrected macro F1, class weighting chosen by cross-validation where the model supports it. The test set is never used to tune or to choose a version.

**Which version the final system uses** was fixed by a rule, before any tuned model was scored on the test set, and never changed afterwards: *use the tuned model if its mean paired improvement over the baseline across the tuning folds is larger than the standard deviation of the paired fold differences; otherwise use the baseline.* This is a pragmatic threshold, not a significance test.

| Final set uses | Models |
|---|---|
| Tuned (7) | KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting |
| Baseline (4) | Random Forest, XGBoost, Logistic Regression, Gaussian NB |

**Majority vote:** one equal vote per model, no weights. On a tie the class that is not BENIGN wins (`prefer_non_benign`). Weighted votes and stacking were tried later (step 13).

## Results (held-out test set, 504,473 flows)

Macro F1 is the main metric: BENIGN is 83% of the rows, so accuracy is above 99.8% for almost every model and says little.

| Majority vote | Macro F1 (all classes) | Macro F1 (reliable classes) |
|---|---:|---:|
| Baseline models only | 0.9095 | 0.9096 |
| Tuned models only | **0.9142** | **0.9155** |
| Final set (evidence rule) | 0.9054 | 0.9138 |

The final set detects 99.89% of attack flows with 0.092% false alarms. It has the lowest all-class macro F1; the differences are small and moved by the three tiny classes, so none of the three votes can be called clearly better. The final set was kept because its rule was fixed beforehand, not because it won.

Test macro F1 of each **tuned** version (for the four models where the final set uses the baseline, this number is shown for reporting only): KNN 0.6831, Decision Tree 0.8979, Random Forest 0.9158, Gaussian NB 0.4790, SVC 0.7078, Bernoulli NB 0.3869, Logistic Regression 0.7138, SGD 0.6523, AdaBoost 0.7908, Gradient Boosting 0.6101, XGBoost 0.9183. The tree ensembles are far ahead; the weak models drag the equal-weight vote down.

Main mistakes of the final vote:

- Web Attack - XSS is labelled Web Attack - Brute Force: 90 flows (69.23% of XSS). The reverse, Brute Force labelled XSS: 38 flows (12.93%). These two classes are close in the feature space.
- Bot labelled BENIGN: 55 flows (14.07% of Bot), that is, missed attacks.
- BENIGN labelled PortScan 183, DoS Hulk 89, Bot 58 (false alarms; BENIGN is the large class, so the shares are small).

Full tables, per-class scores, confusion matrices and figures are in `reports\evaluation\` (`evaluation_report_tuned.md`, `model_comparison_tuned.md`) and `reports\optimization\tuning_report.md`.

## Data mining extensions (scripts 09-17)

Everything here reuses the cleaned and preprocessed data and never rewrites `data\` or `artifacts\`. Clustering, rules, projections and importance use **class-capped samples** (small classes kept whole), so they over-represent the rare attacks. Design and module reference: `docs\project\data_mining_extensions.md`. Full reports: `reports\mining\` and `reports\experiments\`.

### Unsupervised mining

| Task | Script | Main result |
|---|---|---|
| **Clustering** (k-means k = 2-50, Ward, DBSCAN) | `09_cluster.py` | Without labels, clusters recover much of the class structure: purity 0.77 at k = 15 and 0.92 at k = 50 (cluster-majority macro F1 on other rows 0.63). Silhouette is best at k = 40 (0.57). On the same 10,000 rows, Ward beats k-means (ARI 0.45 vs 0.40). DBSCAN reaches purity 0.94 with 8% noise points. |
| **Anomaly detection** (trained on BENIGN only) | `10_anomaly.py` | Autoencoder ROC-AUC **0.964**, LOF 0.918, Isolation Forest 0.860 on the full test set. At a 1% false-alarm rate (threshold set on BENIGN training rows; 0.99% measured on the test set) the autoencoder catches 28% of attack flows (51% of DoS Hulk), the two others almost none. Brute force (FTP/SSH-Patator), web attacks and Bot look normal to all three (0-1% flagged). Supervised XGBoost: 99.9% at 0.07%. |
| **Association rules** (FP-Growth on cleaned flows, original units) | `11_association_rules.py` | 96,388 frequent itemsets, 121,822 rules (confidence >= 0.8). Class rules mined inside each class, e.g. `{port=SSH, Flow IAT Mean=mid} -> SSH-Patator` (coverage 0.92, confidence 1.0), `{port=444, Bwd Packet Length Mean=high} -> Heartbleed`. **No rule** reaches 0.8 confidence for the three web attacks. As a classifier, the 60 class rules give macro F1 0.58 on held-out test flows. |
| **Projections** (PCA, t-SNE) | `12_explain.py` | PC1 + PC2 hold 54% of the variance; 11 components are needed for 90%. On the t-SNE map most attacks form their own islands; Web Attack Brute Force and XSS share one. |
| **Feature importance** (built-in and permutation) | `12_explain.py` | `Destination Port` ranks **#1** by permutation importance for both Random Forest and XGBoost, followed by the initial TCP window sizes. The two models share 7 of their top 10 features (Spearman 0.69). |

### How far the classification scores carry

| Experiment | Script | Main result |
|---|---|---|
| **Better ensembles** (weights and stackers learnt from the 5-fold out-of-fold predictions) | `13_ensembles.py` | The equal vote of the 11 baseline models reproduces the evaluation exactly (macro F1 0.9095). Weighting by OOF F1 (0.9113; F1^4: 0.9134) or voting with the top 3 models (0.9149) helps, but **Random Forest alone (0.9150) is as good as any combination**. Stacking (XGBoost 0.894, logistic 0.885) and per-class weights (0.772: every model's BENIGN votes outweigh its small-class votes) do worse. |
| **Without `Destination Port`** | `14_port_ablation.py` | Macro F1: Decision Tree 0.9141 -> 0.8788, Random Forest 0.9150 -> 0.8719, XGBoost 0.9120 -> **0.9065**. The large classes barely move; the losses are in Sql Injection, Infiltration, XSS and DoS Slowhttptest. XGBoost finds the information in the other features. |
| **Split by capture day** (binary XGBoost, leave one file out) | `15_day_split.py` | Random split: 99.98% of attacks detected. Unseen attack types: **33%** pooled. Web attacks (85-91%) and DDoS (63%) are still caught, as they resemble other HTTP floods; PortScan, FTP/SSH-Patator, Bot, Slowhttptest, slowloris and Infiltration **0-1%**. Holding out the Infiltration day raises false alarms to 15%. |
| **Class imbalance** (balanced weights, SMOTE, undersampling + SMOTE) | `16_imbalance.py` | No variant improves the all-class macro F1 of Decision Tree, Random Forest or XGBoost (changes from -0.004 to -0.056); they trade small-class precision for recall and add false alarms. XGBoost with balanced weights gives the best reliable-class macro F1 (0.9173 vs 0.9127). |
| **Label granularity** (binary, 8 families, 15 classes) | `17_granularity.py` | Families: macro F1 0.942 (Random Forest) and 0.945 (XGBoost), 0.973-0.976 on reliable classes; binary: 0.998-0.999. A 15-class model **collapsed** to families or binary is as good as one trained on them (differences <= 0.0012), so keeping the fine labels costs nothing. |

**What it adds up to:** on the random split the models look near perfect, but part of that rests on the lab setup (ports, one capture per attack type, CICFlowMeter's flag bug, see below). Known attack types on this network are recognised very well; new attack types and other networks are a different, much harder problem, where a one-class detector (the autoencoder) is the only method here that needs no attack labels.

### Packet captures in the app

`src\nids\capture\pcap_flows.py` turns a `.pcap` / `.pcapng` into flows and computes the 40 model features following CICFlowMeter (bidirectional flows, FIN/RST termination, 120 s flow timeout, 5 s activity timeout, payload lengths including Ethernet padding, times in whole microseconds). The CICFlowMeter version used for CIC-IDS2017 filled the TCP flag columns incorrectly (Engelen et al., IEEE S&P Workshops 2021); in the cleaned data `PSH Flag Count` and `ACK Flag Count` are almost mutually exclusive. The default **CIC-compatible** mode reproduces that pattern: on the synthetic demo capture it lets the models recognise 282 of 300 port-scan flows (0 of 300 with the real flags), with no false alarm on the normal traffic. It is an empirical approximation, and the app says so.

## Install

Windows 11, PowerShell, Python 3.13 (the pinned versions in `requirements.txt` were tested on it).

```powershell
git clone https://github.com/dataminingcs21/DM_Project.git
cd DM_Project
python -m venv .venv
.venv\Scripts\Activate.ps1          # if PowerShell blocks scripts, adjust the execution policy for the current user
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

XGBoost uses a GPU when one is usable and the CPU otherwise.

| Library | Version | Used for |
|---|---|---|
| Python | 3.13 | |
| pandas / NumPy / PyArrow | 2.3.3 / 2.3.5 / 22.0.0 | data handling, Parquet |
| scikit-learn | 1.8.0 | preprocessing, 10 of the 11 classifiers, clustering, anomaly detection, metrics |
| XGBoost | 3.1.3 | 11th classifier (GPU capable) |
| mlxtend | 0.25.0 | FP-Growth and association rules |
| imbalanced-learn | 0.14.2 | SMOTE |
| dpkt | 1.9.8 | reading packet captures |
| Matplotlib / seaborn | 3.10.8 / 0.13.2 | figures |
| Streamlit | 1.52.2 | the application |
| ReportLab | 4.5.1 | the PDF report (5.x needs Pillow 13, which Streamlit 1.52 refuses) |
| pytest | 9.0.2 | tests |

## Run the pipeline

From the project folder with the environment active. Each script prints what it did and checks its own output; each has a document in `docs\scripts\`.

```powershell
python scripts\01_inspect_dataset.py      # facts about the raw files
python scripts\02_run_eda.py              # exploratory analysis
python scripts\03_clean_data.py           # cleaning -> data\cleaned\
python scripts\04_preprocess.py           # split + pipeline -> data\preprocessed\
python scripts\05_train.py                # the 11 baseline models (about 18 minutes in the reference run)
python scripts\06_evaluate.py             # baseline evaluation (cross-validation is long; see --cv-sample-size, --skip-cv)
python scripts\07_optimize.py             # tuning and refit (hours; --dry-run shows the plan first)
python scripts\06_evaluate.py --stage tuned   # scores the tuned models once, three votes, the final set
```

Data mining extensions (each writes `reports\mining\<topic>\` or `reports\experiments\<topic>\`; `--help` lists the quick-run options):

```powershell
python scripts\09_cluster.py               # k-means sweep, Ward, DBSCAN (2 min)
python scripts\10_anomaly.py               # Isolation Forest, LOF, autoencoder on BENIGN only (5 min)
python scripts\11_association_rules.py     # FP-Growth rules and the rule classifier (2 min)
python scripts\12_explain.py               # PCA, t-SNE, feature importance (10 min)
python scripts\13_ensembles.py             # weighted votes and stacking (25 min the first time)
python scripts\14_port_ablation.py         # retrain without Destination Port (10 min)
python scripts\15_day_split.py             # leave one capture day out (10 min)
python scripts\16_imbalance.py             # class weights, SMOTE, undersampling (30 min)
python scripts\17_granularity.py           # binary and attack families (10 min)
python scripts\18_build_report_pdf.py      # the full written report (French) -> reports\report\rapport_data_mining.pdf (1 min)
```

Outputs go to `data\`, `artifacts\models\` (trained and tuned), `reports\` and `logs\`. `data\` and `artifacts\` are not in the repository (`.gitignore`) and are rebuilt by the scripts.

## Use the app

```powershell
streamlit run app\streamlit_app.py
```

Streamlit prints a local address, normally <http://localhost:8501>. The page has four tabs:

- **Analyze a capture:** upload a `.pcap` / `.pcapng` (Wireshark, tcpdump), or press **Try the demo capture** (synthetic web and DNS traffic plus one SYN port scan). The page builds the flows, lets the models vote on each, and shows a verdict, the predicted classes, a timeline, the hosts involved in flagged flows and every flow with the agreement of the models; everything can be downloaded as CSV. *Options*: real TCP flags instead of the CIC-compatible ones, packet limit.
- **Single flow:** type the 40 feature values (grouped by the first word of the name), or load a real example of any class saved from the test split, then press **Predict**. The page shows the winning class, how many models agree, every model's vote, the tally and what the test set says about that class.
- **Upload a CSV file:** one flow per row, feature names as headers (a `Label` column is ignored; at most 50,000 rows). Rows with an invalid value are refused with the reason; the others are predicted; the result can be downloaded.
- **About the models:** which version of each model is used and why, the test scores, the known mistakes and the limits.

A missing value is refused by default; the sidebar can switch to "fill with the training median", and the page then says so. The app loads every model it can; a model that cannot be loaded is named on the page and the vote continues with the others.

### Open the app from a clone (no dataset, no training)

The trained models and fitted preprocessing artifacts are too large for the repository; they are distributed separately. **You do not need CIC-IDS2017, the preprocessing run or any training to run the application.**

1. Download `DM_Project_Inference_Bundle.zip` from <https://drive.google.com/file/d/1nQbKxl84xnVSq0zpluSfuUBq-jJJ3HQ7/view?usp=sharing> (about 20 MB; 33 files, about 150 MB once extracted).
2. Extract it **into the root of the cloned repository, keeping its folder structure**. It restores `artifacts\models\trained\`, `artifacts\models\tuned\`, `data\preprocessed\` and `reports\evaluation\`: the 11 selected model binaries (`.joblib`) and their metadata (`.json`), the fitted preprocessing pipeline, the label encoder, the feature schema, the preprocessing metadata, and optional sample flows and evaluation files.
3. Install (see "Install") and run `streamlit run app\streamlit_app.py`.

The training arrays (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`) are not needed for inference. To reproduce training and evaluation from scratch, download the dataset and run the pipeline above.

## Compare predictions with true labels

```powershell
python scripts\08_make_comparison_files.py                 # 300 random held-out flows
python scripts\08_make_comparison_files.py --per-class 20  # every class gets flows
```

It writes `reports\comparison\comparison_input.csv` (upload it in the app), `comparison_answers.csv` (the dataset's true labels) and `comparison.csv` (true and predicted side by side). The flows are real rows of the held-out test split of CIC-IDS2017, so they are unseen by the models but **not** from another network.

## Tests

```powershell
python -m pytest -q
```

Reference run: **719 passed** (665 for the pipeline and the app, 36 for the mining and experiment modules, 18 for packet captures and the capture tab). The GPU test runs only on a machine with a usable GPU and is skipped otherwise. Tests use synthetic data only, do not need the dataset, and never touch the real `data\`, `artifacts\` or `logs\` folders. The app is tested headlessly with Streamlit's `AppTest`.

## Project layout

```
config\       settings.py (paths, seed, split, rules), model_params.py (grids, display names)
src\nids\     data, cleaning, eda, preprocessing, models, training, evaluation, optimization,
              voting, persistence, prediction, utils,
              mining (clustering, anomaly, association, projection, importance, reporting),
              experiments (common, ensembles, day_split), capture (pcap_flows, demo)
scripts\      01 ... 08 (the pipeline and the comparison files), 09 ... 17 (data mining extensions),
              18 (PDF report)
app\          streamlit_app.py, ui_components.py, capture_view.py, style.py
.streamlit\   config.toml (theme, upload size)
tests\        pytest files
docs\         one .md per file (mirrors the layout) and docs\project\ for decisions
reports\      inspection, eda, training, evaluation, optimization, comparison, mining, experiments, report
```

The decisions made after each checkpoint, with their reasons, are in `docs\project\`.

## Not done

- A wider search around the edge-of-grid winners.
- A multiclass split by attack campaign (the day split is binary, because a held-out day's classes do not exist in its training data).
- Any test on real, labelled traffic from another network. The app can read such captures, but no labelled capture was available to score it.
- An exact reproduction of CICFlowMeter's flag bug (the capture reader uses an empirical approximation).

Done since the first version: the comparison without `Destination Port` (step 14), a split by source file (step 15) and a weighted vote (step 13).

## Objective

The goal is not only a high classification score, but a reproducible data mining system whose behaviour is understood: prepare network traffic data, compare eleven classifiers with metrics suited to an imbalanced multiclass problem, study what unsupervised methods reveal, combine models, make the system usable through an application, and state clearly what the results demonstrate and what they do not.
