<<<<<<< HEAD
# Network intrusion detection on CIC-IDS2017
=======
﻿# Network Attack Detection Using Machine Learning
>>>>>>> origin/main

A modular Python system that classifies network flows into **BENIGN or one of 14 attack types** with **eleven supervised models**, combines them by **majority vote**, and serves the result in a **Streamlit app** where you can analyse a **packet capture (.pcap / .pcapng)**, type the features of a flow, load a real example, or upload a CSV file.

<<<<<<< HEAD
On top of the classification pipeline, the **data mining extensions** (scripts 09-17) add clustering, unsupervised anomaly detection, association rules, projections and feature importance, and five experiments that test how far the classification scores carry (see "Data mining extensions").

Every number in this file comes from a report the scripts wrote on the real data; where a figure is not stated here, the named report has it.

## Read this first: what the results do and do not mean
=======
This project develops a machine learning system that analyzes network traffic flows and classifies them as **BENIGN (normal traffic)** or one of **14 attack types**, resulting in a 15-class classification problem.

Eleven supervised machine learning classifiers are trained, evaluated, and tuned using the CIC-IDS2017 dataset. Their predictions are combined through an **equal-weight majority voting system** to produce the final classification.

A **Streamlit web application** provides an interactive interface where users can:
- Enter the 40 required network-flow features manually.
- Load a real example from the held-out test dataset.
- Upload a CSV file containing multiple network flows.
- View the final prediction and the individual votes of all 11 classifiers.
- Explore model information, evaluation results, and known limitations.

The project was developed as part of the **Data Mining** course.
>>>>>>> origin/main

1. **The split is random, so the scores are optimistic.** The 80/20 split is stratified but random. Flows of the same attack session (and near-identical flows) can land on both sides. The scores say how well the models recognise flows like those they trained on, not how they would do on another network or a new capture. Every attack class sits in one source file of the dataset, so a split by file would test attacks the models have never seen; that harder test was not run.
2. **Three classes are too small to judge.** After cleaning, the test set holds **2 Heartbleed, 4 Web Attack - Sql Injection and 7 Infiltration** flows. One flow more or less moves their scores by tens of percent. Scores for these classes, and the all-class macro F1 that includes them, are indicative only. The app and the reports show the test support next to every per-class score.
3. **Three models were trained on a class-capped subsample** (KNN 100,000 rows, SVC 50,000, Gradient Boosting 100,000 of 2,017,889). Their class mix differs from real traffic, so their predictions are shifted toward the small classes.
4. **Tuning used even smaller samples.** Hyperparameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000) and the winners were refitted on the sizes above. Several winners sit at the edge of their grid (Gaussian NB `var_smoothing=1e-9`, Bernoulli NB `alpha=0.1`, Logistic Regression `C=10`, SGD `alpha=1e-6`, AdaBoost depth 4), so a wider grid might do better. The grids were not widened, because the selection rule was fixed beforehand.
5. **The majority vote is not clearly better than its best members.** See "Results". The vote of the final model set has the lowest all-class macro F1 of the three votes that were computed.
6. **`Destination Port` is a model feature, and the most important one.** Attack classes tie to specific ports in this lab capture. Without it, macro F1 drops by 0.035 (Decision Tree), 0.043 (Random Forest) and 0.006 (XGBoost); see "Data mining extensions".
7. **Attacks of a type never seen in training are mostly missed.** Holding out one capture day at a time, a binary detector catches 33% of the unseen attack flows (pooled), against 99.98% with the random split.
8. **A prediction is a vote, not proof.** The input check catches impossible values (text, negatives where impossible, a port above 65535) and warns about values outside the training range. It cannot tell whether a combination of values is a realistic flow.

<<<<<<< HEAD
## Data
=======
- **Project Title:** Network Attack Detection Using Machine Learning
- **Approach:** Supervised multiclass classification.
- **Number of Classes:** 15 (BENIGN + 14 attack types).
- **Number of Classifiers:** 11 supervised machine learning models.
- **Ensemble Strategy:** Equal-weight majority voting.
- **Domain:** Cybersecurity / Network Security.
- **Course:** Data Mining.
- **Methods Not Used:** Clustering and association-rule mining are outside the scope of this project.
>>>>>>> origin/main

- **Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick). It is **not included**. Put the eight CSV files in `data\raw\`.
- **As found:** 2,830,743 rows, 78 feature columns, 15 classes, BENIGN 80.3%. The smallest raw classes are Heartbleed (11), Web Attack - Sql Injection (21) and Infiltration (36).
- **Cleaning** (`03_clean_data.py`): removed 308,381 exact duplicates (10.89%) and 12 identical copy columns; fixed the three web-attack labels that contain a damaged character; set impossible negatives (2,926 rows, 93% BENIGN) to missing; infinite values (2,867 rows, all with a flow duration of 0) are treated as missing. Result: **2,522,362 rows, 66 feature columns**. 698 feature vectors carry two different labels (1,396 rows, 0.055%); they were kept and are an unavoidable error floor.
- **Preprocessing** (`04_preprocess.py`): stratified split, seed 42, **2,017,889 train / 504,473 test**. The pipeline (median imputation, signed log1p, removal of constant and correlated columns at 0.95, standard scaling) is fitted on the training rows only and keeps **40 of 66 features**.

<<<<<<< HEAD
## The eleven models

K-Nearest Neighbours, Decision Tree, Random Forest, Gaussian Naive Bayes, SVC (RBF), Bernoulli Naive Bayes, Logistic Regression, SGD Classifier, AdaBoost, Gradient Boosting and XGBoost. Each exists in a **baseline** version (default-style settings, unweighted) and a **tuned** version (GridSearchCV, 5-fold, score = prior-corrected macro F1, class weighting chosen by cross-validation where the model supports it).

**Which version the final system uses** was fixed by a rule, before any tuned model was scored on the test set, and never changed afterwards: *use the tuned model if its mean paired improvement over the baseline across the tuning folds is larger than the standard deviation of the paired fold differences; otherwise use the baseline.* This is a pragmatic threshold, not a significance test.
=======
**Dataset:** CIC-IDS2017 — Canadian Institute for Cybersecurity, University of New Brunswick.

**Official dataset page:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

The original dataset is **not included in this repository**. To reproduce the complete pipeline, download the eight CSV files containing the network-flow features (78 feature columns plus the `Label` column) from the official page.

Read and respect the dataset's terms of use, and cite the dataset in any report or publication based on it.

Place the downloaded files in:

```text
data/raw/
```

The raw dataset is required to reproduce the data preparation and model-training workflow. It is **not required to run the already-trained application**. See [Open the App From a Clone](#open-the-app-from-a-clone-no-dataset-no-training).

### Dataset Characteristics

The following figures were obtained from the project's dataset inspection and cleaning stages.

| Step | Result |
|---|---|
| Raw dataset | 2,830,743 rows, 78 feature columns, 15 classes |
| BENIGN proportion | Approximately 80.3% |
| Smallest raw classes | Heartbleed (11), Web Attack - Sql Injection (21), Infiltration (36) |
| Exact duplicates removed | 308,381 |
| Identical copy columns removed | 12 |
| Other cleaning operations | Damaged web-attack labels corrected; impossible negative and infinite values treated as missing |
| Cleaned dataset | 2,522,362 rows, 66 feature columns |
| Training split | 2,017,889 rows |
| Test split | 504,473 rows |
| Split strategy | Stratified 80/20 split, random seed 42 |
| Final model features | 40 features |

Feature selection was performed using training data only. One constant feature and 25 highly correlated features were removed during the feature-selection process.

### Dataset Usage

The dataset is used to:

- Inspect and understand network traffic characteristics.
- Clean and prepare the raw network-flow data.
- Explore class distributions and feature relationships.
- Select relevant features for classification.
- Train supervised machine learning classifiers.
- Evaluate individual models and ensemble predictions.
- Analyze the ability of the system to distinguish benign traffic from malicious traffic.

## 🧠 Methodology

The project follows a sequential machine learning pipeline, from raw dataset inspection to the final interactive application.

```text
                  CIC-IDS2017 CSV Files
                           |
                           v
             01. Dataset Inspection
                           |
                           v
              02. Exploratory Data Analysis
                           |
                           v
                   03. Data Cleaning
                           |
                           v
                 04. Data Preprocessing
                           |
             +-------------+-------------+
             |                           |
        Stratified Split           Training Data
             |                           |
             v                           v
        Test Dataset          Fit Preprocessing Pipeline
                                         |
                                         v
                              05. Baseline Model Training
                                         |
                                         v
                                06. Model Evaluation
                                         |
                                         v
                                07. Hyperparameter
                                      Optimization
                                         |
                                         v
                             Tuned Model Evaluation
                                         |
                                         v
                              Final Model Selection
                                         |
                                         v
                              Equal-Weight Majority
                                      Voting
                                         |
                                         v
                              08. Comparison Files
                                         |
                                         v
                              Streamlit Application
```

### Preprocessing Workflow

The preprocessing pipeline is fitted on the training data only to avoid information leakage from the test set.

The main operations are:

1. Stratified train/test splitting.
2. Median imputation for missing values.
3. Signed logarithmic transformation.
4. Removal of constant and highly correlated features.
5. Feature standardization.
6. Storage of the fitted preprocessing pipeline and feature schema for inference.

The same fitted preprocessing artifacts are reused by the application so that incoming traffic is transformed consistently with the data used to train the models.

## 🤖 Machine Learning

The project implements eleven supervised classification algorithms. Each classifier has a baseline version and a tuned version.

### Implemented Models

1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree**
3. **Random Forest**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVC, RBF kernel)**
6. **Bernoulli Naive Bayes**
7. **Logistic Regression**
8. **Stochastic Gradient Descent (SGD) Classifier**
9. **AdaBoost**
10. **Gradient Boosting**
11. **XGBoost**

### Hyperparameter Optimization
>>>>>>> origin/main

Hyperparameter tuning is performed using **GridSearchCV** with stratified cross-validation.

- Five-fold stratified cross-validation is used in general.
- Three folds are used for Gradient Boosting.
- The scoring criterion is a prior-corrected macro F1-score.
- Tuning is performed on class-capped samples of the training data.
- The held-out test set is not used to tune parameters or select the final model versions.

### Final Model Selection

The final system does not automatically use the tuned version of every classifier.

Instead, a selection rule was established before evaluating tuned models on the test set.

For each classifier, the tuned version is selected only when its mean paired improvement over the baseline across the tuning folds exceeds the standard deviation of the paired fold differences. Otherwise, the baseline version is retained.

The resulting final configuration is:

| Version Used | Models |
|---|---|
| **Tuned (7)** | KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting |
| **Baseline (4)** | Random Forest, XGBoost, Logistic Regression, Gaussian NB |

<<<<<<< HEAD
**Majority vote:** one equal vote per model, no weights. On a tie the class that is not BENIGN wins (`prefer_non_benign`). A weighted vote (weights from training-data cross-validation only) is a possible later variant; it was not built.
=======
This approach retains the version supported by the tuning-fold evidence rather than assuming that hyperparameter optimization always improves generalization.

### Majority Voting

The final prediction is generated using an **equal-weight majority vote**.

Each of the eleven classifiers contributes exactly one vote. No classifier receives a manually assigned weight.

The predicted class is the class receiving the largest number of votes.

**Tie-breaking rule:** When a tie occurs between classes, the system uses the `prefer_non_benign` policy, preferring an attack class over BENIGN.
>>>>>>> origin/main

## Results (held-out test set, 504,473 flows)

<<<<<<< HEAD
Macro F1 of the three votes (all classes, and the classes with enough test flows to be reliable):
=======
The classifiers are evaluated using the held-out test set containing **504,473 network flows**. Baseline models are also evaluated using stratified cross-validation on the training data.

The evaluation considers several complementary metrics:

- **Accuracy:** Overall proportion of correctly classified flows.
- **Precision:** Proportion of predicted instances of a class that are correct.
- **Recall:** Proportion of actual instances of a class correctly detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro F1-score:** Average F1-score across all classes, giving each class equal importance.
- **Detection Rate:** Ability to identify malicious traffic.
- **False Alarm Rate:** Proportion of benign traffic incorrectly classified as an attack.
- **Confusion Matrix:** Distribution of correct predictions and classification errors.
- **ROC and Precision-Recall Curves:** Class-wise evaluation of classification performance.
- **Support:** Number of actual test samples belonging to each class.

Because the dataset is highly imbalanced, **Macro F1-score** is used as the main comparison metric. Accuracy alone can hide poor performance on minority attack classes.

For cybersecurity applications, particular attention is also given to **Recall**, since failing to detect an actual attack can have important security consequences.

### Results

The following table compares the three majority-voting configurations on the held-out test set.
>>>>>>> origin/main

| Majority Vote | Macro F1 (All Classes) | Macro F1 (Reliable Classes) |
|---|---:|---:|
| Baseline models only | 0.9095 | 0.9096 |
| Tuned models only | **0.9142** | **0.9155** |
| Final set (evidence rule) | 0.9054 | 0.9138 |

<<<<<<< HEAD
The final set has the lowest all-class macro F1. The differences are small and the all-class figure is moved by the three tiny classes, so none of the three votes can be called clearly better. The final set was kept because its rule was fixed beforehand, not because it won.

Test macro F1 of each **tuned** version (for the four models where the final set uses the baseline, this number is shown for reporting only and is not what the final set uses): KNN 0.6831, Decision Tree 0.8979, Random Forest 0.9158, Gaussian NB 0.4790, SVC 0.7078, Bernoulli NB 0.3869, Logistic Regression 0.7138, SGD 0.6523, AdaBoost 0.7908, Gradient Boosting 0.6101, XGBoost 0.9183. The tree ensembles are far ahead; several single models are well below the vote and drag the equal-weight vote down, which is the reason the weighted vote is worth trying later.

Main mistakes of the final vote:

- Web Attack - XSS is labelled Web Attack - Brute Force: 90 flows (69.23% of XSS). The reverse, Brute Force labelled XSS: 38 flows (12.93%). These two classes are close in the feature space.
- Bot labelled BENIGN: 55 flows (14.07% of Bot), that is, missed attacks.
- BENIGN labelled PortScan 183, DoS Hulk 89, Bot 58 (false alarms; BENIGN is the large class, so the shares are small).

Full tables, per-class scores, confusion matrices and figures are in `reports\evaluation\` (`evaluation_report_tuned.md`, `model_comparison_tuned.md`) and `reports\optimization\tuning_report.md`.

## Data mining extensions (scripts 09-17)
=======
The tuned-only vote achieves the highest reported Macro F1-score for both comparisons. However, the final system uses the evidence-based selection rule established before test evaluation, rather than choosing the configuration that achieved the best test score.

The differences between the three configurations are relatively small. These results should therefore be interpreted together with the dataset limitations discussed below.

### Individual Tuned Model Results

The following are the test Macro F1-scores for the tuned versions of all eleven classifiers.

For Random Forest, XGBoost, Logistic Regression, and Gaussian Naive Bayes, the tuned results are reported for comparison only: the final system uses their baseline versions.

| Model | Tuned Test Macro F1 |
|---|---:|
| KNN | 0.6831 |
| Decision Tree | 0.8979 |
| Random Forest | 0.9158 |
| Gaussian Naive Bayes | 0.4790 |
| SVC | 0.7078 |
| Bernoulli Naive Bayes | 0.3869 |
| Logistic Regression | 0.7138 |
| SGD Classifier | 0.6523 |
| AdaBoost | 0.7908 |
| Gradient Boosting | 0.6101 |
| XGBoost | 0.9183 |

### Main Classification Errors

The final majority vote still makes some classification errors. Examples from the held-out test set include:

- **Web Attack - XSS → Brute Force:** 90 flows (69.23% of XSS test flows).
- **Brute Force → XSS:** 38 flows (12.93% of Brute Force test flows).
- **Bot → BENIGN:** 55 flows (14.07% of Bot test flows), representing missed malicious traffic.
- **BENIGN → PortScan:** 183 flows.
- **BENIGN → DoS Hulk:** 89 flows.
- **BENIGN → Bot:** 58 flows.

These errors highlight the difficulty of distinguishing certain attack categories and the importance of examining per-class metrics rather than relying only on an overall score.

Detailed evaluation tables, class-wise metrics, confusion matrices, and figures are available in:

```text
reports/evaluation/
reports/optimization/tuning_report.md
```

These reports are generated by the project pipeline.

### ⚠️ Limits of These Results

The evaluation results must be interpreted in the context of the following limitations.

1. **Random train/test splitting may produce optimistic results.**

   The dataset is split randomly and stratified. Flows belonging to the same attack session may therefore appear in both the training and test sets. Consequently, the reported scores do not necessarily represent performance on an independent network or a completely unseen attack campaign. A split by source file was not performed.

2. **Three attack classes have very limited representation.**

   The test set contains only:
   - 2 Heartbleed flows.
   - 4 Web Attack - Sql Injection flows.
   - 7 Infiltration flows.

   Performance estimates for these classes are highly uncertain. Their small support also affects the interpretation of the all-class Macro F1-score.

3. **Some classifiers use class-capped training samples.**

   KNN, SVC, and Gradient Boosting were trained using subsamples of the original training data:
   - KNN: 100,000 rows.
   - SVC: 50,000 rows.
   - Gradient Boosting: 100,000 rows.

   These subsamples may have a different class distribution from the complete training dataset.

4. **Hyperparameter tuning used smaller training samples.**

   Tuning used at most 100,000 training rows, with smaller samples for some models:
   - KNN: 50,000 rows.
   - SVC: 30,000 rows.
   - Gradient Boosting: 30,000 rows.

   Several selected parameters reached the boundaries of their search grids, including Gaussian NB, Bernoulli NB, Logistic Regression, SGD, and AdaBoost. A wider search could produce different results.

5. **Destination Port may introduce dataset-specific patterns.**

   `Destination Port` is included as a model feature. Some attack categories are associated with specific ports in the CIC-IDS2017 capture. The classifiers may therefore learn patterns specific to this laboratory dataset. An evaluation without this feature was not performed.

6. **A prediction is not proof of malicious activity.**

   The application validates input values and checks for impossible values, but it cannot guarantee that every combination of network-flow features represents a realistic network connection. Predictions should be treated as model outputs, not as definitive security verdicts.
>>>>>>> origin/main

Everything here reuses the cleaned and preprocessed data and never rewrites `data\` or `artifacts\`. Every fitted object (clusters, rule bins, anomaly thresholds, ensemble weights, SMOTE) learns from **training rows only**; the test set only scores. Clustering, rules, projections and importance use **class-capped samples** (small classes kept whole), so they over-represent the rare attacks. Design and module reference: `docs\project\data_mining_extensions.md`. Full reports: `reports\mining\` and `reports\experiments\`.

<<<<<<< HEAD
### Unsupervised mining

| Task | Script | Main result |
|---|---|---|
| **Clustering** (k-means k = 2-50, Ward, DBSCAN) | `09_cluster.py` | Without labels, clusters recover much of the class structure: purity 0.77 at k = 15 and 0.92 at k = 50 (cluster-majority macro F1 on other rows 0.63). Silhouette is best at k = 40 (0.57). On the same 10,000 rows, Ward beats k-means (ARI 0.45 vs 0.40). DBSCAN reaches purity 0.94 with 8% noise points. |
| **Anomaly detection** (trained on BENIGN only) | `10_anomaly.py` | Autoencoder ROC-AUC **0.964**, LOF 0.918, Isolation Forest 0.860 on the full test set. At a 1% false-alarm rate (threshold set on BENIGN training rows; 0.99% measured on the test set) the autoencoder catches 28% of attack flows (51% of DoS Hulk), the two others almost none. Brute force (FTP/SSH-Patator), web attacks and Bot look normal to all three (0-1% flagged). Supervised XGBoost: 99.9% at 0.07%. |
| **Association rules** (FP-Growth on cleaned flows, original units) | `11_association_rules.py` | 96,388 frequent itemsets, 121,822 rules (confidence >= 0.8). Class rules mined inside each class, e.g. `{port=SSH, Flow IAT Mean=mid} -> SSH-Patator` (coverage 0.92, confidence 1.0), `{port=444, Bwd Packet Length Mean=high} -> Heartbleed`. **No rule** reaches 0.8 confidence for the three web attacks. As a classifier, the 60 class rules give macro F1 0.58 on held-out test flows. |
| **Projections** (PCA, t-SNE) | `12_explain.py` | PC1 + PC2 hold 54% of the variance; 11 components are needed for 90%. On the t-SNE map most attacks form their own islands; Web Attack Brute Force and XSS share one. |
| **Feature importance** (built-in and permutation) | `12_explain.py` | `Destination Port` ranks **#1** by permutation importance for both Random Forest and XGBoost, followed by the initial TCP window sizes. The two models share 7 of their top 10 features (Spearman 0.69). |
=======
The project uses the following technologies and libraries:

| Technology | Version / Purpose |
|---|---|
| Python | 3.13 |
| Pandas | 2.3.3 — data manipulation |
| NumPy | 2.3.5 — numerical operations |
| PyArrow | Parquet file support |
| Scikit-learn | 1.8.0 — preprocessing, 10 classifiers, evaluation, and GridSearchCV |
| XGBoost | 3.1.3 — gradient-boosted classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | 1.52.2 — interactive application |
| Pytest | Automated testing |

GPU acceleration is used by XGBoost when the available environment and configuration support it.

All pinned dependencies are specified in:

```text
requirements.txt
```

The project is implemented through Python modules and scripts rather than Jupyter notebooks.
>>>>>>> origin/main

### How far the classification scores carry

<<<<<<< HEAD
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
=======
The repository is organized into separate modules for data preparation, model training, evaluation, optimization, prediction, and application development.

```text
DM_Project/
│
├── README.md
├── requirements.txt              # Pinned project dependencies
├── pyproject.toml                # Python packaging configuration
├── .gitignore                    # Excludes datasets, large arrays, logs, and local files
├── .gitattributes                # Git LFS tracking configuration
│
├── config/
│   ├── settings.py               # Paths, random seed, split and preprocessing settings
│   └── model_params.py           # Model definitions, default parameters and tuning grids
│
├── src/nids/
│   ├── data/                     # Raw data loading and inspection
│   ├── cleaning/                 # Dataset cleaning rules
│   ├── eda/                      # Exploratory data analysis
│   ├── preprocessing/            # Splitting, transformations and feature selection
│   ├── models/                   # Eleven classifiers and model registry
│   ├── persistence/              # Model storage, loading and metadata validation
│   ├── training/                 # Baseline model training
│   ├── evaluation/               # Metrics, cross-validation, curves and reports
│   ├── optimization/             # Hyperparameter tuning
│   ├── voting/                   # Majority voting implementation
│   ├── prediction/               # Input validation and prediction engine
│   └── utils/                    # Logging and file utilities
│
├── scripts/
│   ├── 01_inspect_dataset.py     # Dataset inspection
│   ├── 02_run_eda.py             # Exploratory data analysis
│   ├── 03_clean_data.py          # Dataset cleaning
│   ├── 04_preprocess.py          # Preprocessing and data splitting
│   ├── 05_train.py               # Baseline model training
│   ├── 06_evaluate.py            # Model evaluation
│   ├── 07_optimize.py            # Hyperparameter optimization
│   └── 08_make_comparison_files.py # Generate prediction comparison files
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── ui_components.py          # Application interface components
│
├── tests/                        # Automated pytest suite
│
├── docs/
│   ├── scripts/                  # Script documentation
│   └── project/                  # Project decisions and checkpoints
│
├── data/
│   ├── raw/                      # Original CIC-IDS2017 CSV files (not distributed)
│   ├── cleaned/                  # Cleaned dataset (not distributed)
│   └── preprocessed/             # Fitted inference preprocessing artifacts
│
├── artifacts/
│   └── models/
│       ├── trained/              # Baseline model artifacts
│       └── tuned/                # Tuned model artifacts
│
└── reports/
    ├── inspection/               # Dataset inspection results
    ├── eda/                      # Exploratory analysis reports
    ├── training/                 # Training results
    ├── evaluation/               # Evaluation metrics and figures
    ├── optimization/             # Hyperparameter tuning reports
    └── comparison/               # True-label and prediction comparisons
```

**Note:** The repository contains the source code and project documentation. Large inference artifacts are distributed separately through the Google Drive bundle described below. Raw data and training-only arrays are not included in that bundle.

## ⚙️ Installation

The following instructions describe how to prepare the development environment on **Windows 11 using PowerShell and Python 3.13**.

### 1. Clone the Repository
>>>>>>> origin/main

```powershell
git clone https://github.com/dataminingcs21/DM_Project.git
cd DM_Project
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you may need to adjust the execution policy for the current user or use an appropriate activation method for your environment.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the Project

```powershell
pip install -e .
```

<<<<<<< HEAD
XGBoost uses a GPU when one is usable and the CPU otherwise.

## Run the pipeline
=======
The environment is now prepared. To run the full training pipeline, the original dataset must also be downloaded and placed in the expected data directory.

## ▶️ Run the Pipeline
>>>>>>> origin/main

The following scripts reproduce the main stages of the machine learning workflow.

Run them from the repository root with the virtual environment activated.

Each script performs a specific task and has corresponding documentation under `docs/scripts/`.

```powershell
<<<<<<< HEAD
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

Outputs go to `data\`, `artifacts\models\` (trained and tuned), `reports\` and `logs\`; these folders are not part of the source and are rebuilt by the scripts.

## Use the app
=======
# 01 - Inspect the raw dataset
python scripts\01_inspect_dataset.py

# 02 - Perform exploratory data analysis
python scripts\02_run_eda.py

# 03 - Clean the dataset
python scripts\03_clean_data.py

# 04 - Split and preprocess the data
python scripts\04_preprocess.py

# 05 - Train the eleven baseline models
python scripts\05_train.py

# 06 - Evaluate baseline models
python scripts\06_evaluate.py

# 07 - Perform hyperparameter optimization
python scripts\07_optimize.py

# 08 - Evaluate tuned models and the three voting configurations
python scripts\06_evaluate.py --stage tuned
```

### Execution Notes

- The baseline training stage took approximately 18 minutes in the reference run; actual execution time depends on the hardware.
- Baseline evaluation supports `--cv-sample-size` and `--skip-cv` for shorter runs.
- Hyperparameter optimization can take several hours depending on the models and hardware.
- Use `--dry-run` with the optimization script to inspect the tuning plan before running the full optimization.
- The tuned evaluation stage compares baseline-only, tuned-only, and final evidence-selected voting configurations.

## 🖥️ The Application
>>>>>>> origin/main

The project provides an interactive Streamlit interface for classifying individual network flows or batches of flows.

### Launch the Application

From the project root, with the virtual environment activated and the inference artifacts installed:

```powershell
streamlit run app\streamlit_app.py
```

<<<<<<< HEAD
- **Analyze a capture:** upload a `.pcap` / `.pcapng` (Wireshark, tcpdump), or press **Try the demo capture** (synthetic web and DNS traffic plus one SYN port scan; also saved as `data\samples\demo_capture.pcap`). The page builds the flows, lets the models vote on each, and shows a verdict, the predicted classes, a timeline, the hosts involved in flagged flows and every flow with the agreement of the models; everything can be downloaded as CSV. *Options*: real TCP flags instead of the CIC-compatible ones, packet limit.
- **Single flow:** type the 40 feature values (grouped by the first word of the name), or load a real example of any class saved from the test split, then press **Predict**. The page shows the winning class, how many models agree, every model's vote, the tally and what the test set says about that class.
- **Upload a CSV file:** one flow per row, feature names as headers (a `Label` column is ignored; at most 50,000 rows). Rows with an invalid value are refused with the reason; the others are predicted; the result can be downloaded.
- **About the models:** which version of each model is used and why, the test scores, the known mistakes and the limits.
- A missing value is refused by default; the sidebar can switch to "fill with the training median", and the page then says so.

The app loads every model it can; a model that cannot be loaded is named on the page and the vote continues with the others.

## Compare predictions with true labels
=======
Streamlit will start a local web server and provide a URL, normally:

```text
http://localhost:8501
```

Open that address in your browser to access the application.

### Application Features

**1. Single-Flow Prediction**

- Enter the 40 required network-flow features manually.
- Alternatively, load a real example from the held-out test split.
- Submit the input using the Predict button.
- View the predicted class and the number of models agreeing with the result.
- Inspect the individual vote of each classifier.
- Review relevant information about the predicted class from the evaluation results.

**2. CSV Batch Prediction**

- Upload a CSV file containing multiple network flows.
- Each row represents one flow.
- Column names must correspond to the expected feature names.
- An optional `Label` column is ignored.
- The application accepts up to 50,000 rows.
- Invalid rows are rejected with an explanation; valid rows can still be processed.
- Download the resulting predictions.

**3. Model Information and Evaluation**

The application also presents information about:
- The model versions selected for the final ensemble.
- The reasons behind the baseline/tuned selection.
- Evaluation scores and known classification errors.
- The limitations of the experimental results.

### Open the App From a Clone (No Dataset, No Training)

The repository contains the application source code, but the trained inference models and fitted preprocessing artifacts are distributed separately through Google Drive because of their size.

**You do not need to download CIC-IDS2017, rebuild the preprocessing pipeline, or retrain the models to run the application.**

### 1. Download the Inference Bundle

<https://drive.google.com/file/d/1nQbKxl84xnVSq0zpluSfuUBq-jJJ3HQ7/view?usp=sharing>

Download:

```text
DM_Project_Inference_Bundle.zip
```

The archive contains **33 files**, approximately 149.58 MB after extraction (the ZIP itself is approximately 20.03 MB).

### 2. Extract the Bundle

Extract the archive directly into the root directory of the cloned repository.

For example:

```text
DM_Project/
├── artifacts/
├── data/
├── reports/
├── app/
├── src/
├── scripts/
└── README.md
```

The extracted files restore the following directories:

```text
artifacts/models/trained/
artifacts/models/tuned/
data/preprocessed/
reports/evaluation/
```

The bundle contains:

- The 11 selected model binaries (`.joblib`).
- Required model metadata (`.json`).
- The fitted preprocessing pipeline.
- The label encoder.
- The feature schema.
- Preprocessing metadata.
- Optional sample-flow and evaluation files.

**Important:** Preserve the original directory structure when extracting. Do not place all the files together in a single directory.

### 3. Install Dependencies and Launch

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

streamlit run app\streamlit_app.py
```

The application will load the selected models and preprocessing artifacts automatically.

### Important Notes

- The inference bundle is separate from the GitHub repository.
- The application requires the 11 selected model binaries and their corresponding metadata.
- The fitted preprocessing pipeline and feature schema must also be present.
- The original CIC-IDS2017 CSV files are not required for inference.
- Training arrays such as `X_train.npy`, `X_test.npy`, `y_train.npy`, and `y_test.npy` are not required to run the application.
- To reproduce training and evaluation from scratch, download the original dataset and execute the pipeline described above.

## 📊 Compare Predictions With True Labels

The project includes a script for generating comparison files using real network flows from the held-out test dataset.

Run:
>>>>>>> origin/main

```powershell
# Generate 300 randomly selected test flows
python scripts\08_make_comparison_files.py

# Generate 20 samples per class
python scripts\08_make_comparison_files.py --per-class 20
```

<<<<<<< HEAD
It writes `reports\comparison\comparison_input.csv` (upload it in the app), `comparison_answers.csv` (the dataset's true labels) and `comparison.csv` (true and predicted side by side). The flows are real rows of the held-out test split of CIC-IDS2017, so they are unseen by the models but **not** from another network.
=======
The script generates:

| File | Description |
|---|---|
| `comparison_input.csv` | Network-flow features to upload to the application |
| `comparison_answers.csv` | Corresponding true class labels |
| `comparison.csv` | Combined true and predicted labels for comparison |

Output directory:

```text
reports/comparison/
```

These flows come from the held-out CIC-IDS2017 test split. They were not used to train the models, but they originate from the same dataset and capture environment.

**Important:** This comparison evaluates predictions on unseen rows from the same dataset. It does not demonstrate generalization to traffic from an independent network.
>>>>>>> origin/main

## Tests

The project includes an automated test suite implemented using **Pytest**.

Run all tests from the repository root:

```powershell
python -m pytest -q
```

<<<<<<< HEAD
Reference run: **719 passed** (665 for the pipeline and the app, 36 for the mining and experiment modules, 18 for packet captures and the capture tab). The GPU test runs only on a machine with a usable GPU and is skipped otherwise. Tests use synthetic data only and never touch the real `data\`, `artifacts\` or `logs\` folders. The app is tested headlessly with Streamlit's `AppTest`.
=======
### Reference Test Results

The reference execution produced:

| Result | Count |
|---|---:|
| Passed | 664 |
| Skipped | 1 |
| Expected convergence warnings | 2 |

The skipped test is intended for environments without a usable GPU.

### Testing Notes

- Tests primarily use synthetic data.
- The test suite does not require the original CIC-IDS2017 dataset.
- Tests do not modify the real `data/`, `artifacts/`, or `logs/` directories.
- The Streamlit interface is tested headlessly using Streamlit's `AppTest`.

The results above describe the reference test run; execution results may vary depending on the environment.
>>>>>>> origin/main

## Project layout

<<<<<<< HEAD
```
config\       settings.py (paths, seed, split, rules), model_params.py (grids, display names)
src\nids\     data, cleaning, eda, preprocessing, models, training, evaluation, optimization,
              voting, persistence, prediction, utils,
              mining (clustering, anomaly, association, projection, importance, reporting),
              experiments (common, ensembles, day_split), capture (pcap_flows, demo)
scripts\      01 ... 08 (the pipeline and the comparison files), 09 ... 17 (data mining extensions)
app\          streamlit_app.py, ui_components.py, capture_view.py, style.py
.streamlit\   config.toml (theme, upload size)
tests\        pytest files
docs\         one .md per file (mirrors the layout) and docs\project\ for decisions
```
=======
- **Team:** CS21
- **Course:** Data Mining
>>>>>>> origin/main

The decisions made after each checkpoint, with their reasons, are in `docs\project\checkpoint_1_decisions.md` to `checkpoint_5_decisions.md`.

<<<<<<< HEAD
## Not done

- A wider search around the edge-of-grid winners.
- A multiclass split by attack campaign (the day split is binary, because a held-out day's classes do not exist in its training data).
- Any test on real, labelled traffic from another network. The app can read such captures, but no labelled capture was available to score it.
- An exact reproduction of CICFlowMeter's flag bug (the capture reader uses an empirical approximation).

Done since the first version: the comparison without `Destination Port` (step 14), a split by source file (step 15) and a weighted vote (step 13).
=======
<badge>Completed</badge>

The main project components have been implemented:

- Dataset inspection and exploratory data analysis.
- Data cleaning and preprocessing.
- Training of eleven baseline classifiers.
- Model evaluation and comparison.
- Hyperparameter optimization.
- Evidence-based final model selection.
- Equal-weight majority voting.
- Streamlit application for individual and batch predictions.
- Automated test suite.
- Technical documentation and evaluation reports.

Project decisions and the reasoning behind them are documented in:

```text
docs/project/
```

### Known Uncompleted Experiments

The following experiments were not performed and remain possible future improvements:

- Evaluation without the `Destination Port` feature.
- Train/test splitting by source file or attack campaign.
- Wider hyperparameter searches around parameters at the edges of their grids.
- Weighted majority voting.
- Evaluation using traffic collected from an independent network.

These limitations are documented to distinguish completed project work from possible future extensions.

## 📚 Objective

The main objective of this project is to design, implement, and evaluate a supervised machine learning solution for network attack detection using network-flow data.

More specifically, the project aims to:

1. Understand and prepare network traffic data for machine learning.
2. Implement and compare eleven supervised classification algorithms.
3. Evaluate the models using appropriate metrics for an imbalanced multiclass dataset.
4. Investigate the effect of hyperparameter optimization.
5. Combine model predictions using an equal-weight majority voting strategy.
6. Develop an interactive application that makes the trained system accessible for individual and batch predictions.
7. Analyze the strengths, weaknesses, and limitations of the experimental results.

The ultimate goal is not simply to obtain a high classification score, but to build a reproducible machine learning system, understand its behavior, and clearly communicate what its results demonstrate—and what they do not.
>>>>>>> origin/main
