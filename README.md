# Network Attack Detection Using Machine Learning

## 📌 Project Overview

This project develops a machine learning system that analyzes network traffic data and decides whether a network flow is **normal (BENIGN)** or belongs to one of **14 attack types**.

Eleven supervised classifiers are trained, evaluated and tuned on the CIC-IDS2017 dataset, then combined by a **majority vote**. A **Streamlit application** lets a user type the features of one flow, load a real example, or upload a CSV file and see the prediction and how each model voted.

The project is developed as part of the **Data Mining** course.

## 🎯 Project Information

* **Project Title:** Network Attack Detection Using Machine Learning
* **Chosen Approach:** Supervised multiclass classification (15 classes: BENIGN + 14 attacks), 11 models, equal-weight majority vote
* **Domain:** Cybersecurity / Network Security
* **Course:** Data Mining
* **Not used:** clustering and association rules (not part of this project)

## 📊 Dataset

**Dataset:** CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

The dataset is **not included** in this repository. Download the eight CSV files with the flow features (78 feature columns plus `Label`) from the page above, read the terms of use there, cite the dataset in any report, and put the files in `data\raw\`. The dataset is only needed to rebuild everything from scratch; to just look at the results or use the app, see "Open the App From a Clone".

What the real data looks like (from the inspection and cleaning reports):

| Step | Result |
|---|---|
| Raw | 2,830,743 rows, 78 feature columns, 15 classes, BENIGN 80.3% |
| Smallest raw classes | Heartbleed (11), Web Attack - Sql Injection (21), Infiltration (36) |
| Cleaning | 308,381 exact duplicates removed, 12 identical copy columns dropped, damaged web-attack labels fixed, impossible negatives and infinite values treated as missing |
| After cleaning | 2,522,362 rows, 66 feature columns |
| Split (stratified, seed 42) | 2,017,889 training rows, 504,473 test rows |
| Used by the models | 40 features (1 constant and 25 highly correlated columns dropped, on training data only) |

The dataset is used to:

* Analyze and understand network traffic.
* Preprocess and clean the data.
* Select relevant features.
* Train supervised machine learning models.
* Evaluate the performance of the models.
* Classify network traffic as normal or malicious.

## 🧠 Methodology

```text
CIC-IDS2017 CSV files
   ↓
Data Inspection (scripts/01)  ·  Data Exploration (scripts/02)
   ↓
Data Cleaning (scripts/03)
   ↓
Preprocessing (scripts/04): stratified train/test split, then, fitted on training rows only:
   median imputation → signed log transform → constant/correlated feature removal → standardisation
   ↓
Supervised Classification (scripts/05): 11 baseline models
   ↓
Model Evaluation (scripts/06): test set, 5-fold cross-validation, majority vote
   ↓
Hyperparameter Tuning (scripts/07): GridSearchCV → evidence rule → final model set
   ↓
Evaluation of the tuned models and the three votes (scripts/06 --stage tuned)
   ↓
Network Attack Detection: Streamlit app (app/streamlit_app.py)
```

## 🤖 Machine Learning

All eleven required supervised models are trained, each in a **baseline** and a **tuned** version:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gaussian Naive Bayes
* Support Vector Machine (SVC, RBF kernel)
* Bernoulli Naive Bayes
* Logistic Regression
* SGD Classifier
* AdaBoost
* Gradient Boosting
* XGBoost

**Tuning:** GridSearchCV with stratified 5-fold cross-validation (3 folds for Gradient Boosting), scored by a prior-corrected macro F1, on class-capped samples of the training rows. The test set is never used for tuning or for choosing the final set.

**Which version the final system uses** was fixed by a rule before any tuned model was scored on the test set: use the tuned model if its mean paired improvement over the baseline across the tuning folds is larger than the standard deviation of the paired fold differences; otherwise use the baseline.

| Final set uses | Models |
|---|---|
| Tuned (7) | KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting |
| Baseline (4) | Random Forest, XGBoost, Logistic Regression, Gaussian NB |

**Majority vote:** one equal vote per model, no weights. On a tie the attack class wins over BENIGN (`prefer_non_benign`).

## 📈 Evaluation

The models are evaluated on the held-out test set (504,473 flows) and, for the baselines, with stratified cross-validation on the training rows. Metrics:

* Accuracy
* Precision, Recall and F1-score per class
* **Macro F1** (all classes count equally; the main comparison metric, because BENIGN is 80% of the data)
* Detection rate and false-alarm rate (attack versus benign)
* Confusion matrix and the most frequent confusions
* ROC and precision-recall curves per class
* Support (number of test flows) next to every per-class score

For network attack detection, particular attention is given to **Recall**, since failing to detect an actual attack has security implications.

### Results

Macro F1 of the three majority votes on the test set:

| Vote | Macro F1, all classes | Macro F1, reliable classes |
|---|---:|---:|
| Baseline models only | 0.9095 | 0.9096 |
| Tuned models only | **0.9142** | **0.9155** |
| Final set (evidence rule) | 0.9054 | 0.9138 |

The final set has the lowest all-class macro F1 of the three; the differences are small and none of the votes can be called clearly better. The final set was kept because its rule was fixed beforehand, not because it won.

Test macro F1 of each **tuned** version (for the four models where the final set uses the baseline, this number is for reporting only): KNN 0.6831, Decision Tree 0.8979, Random Forest 0.9158, Gaussian NB 0.4790, SVC 0.7078, Bernoulli NB 0.3869, Logistic Regression 0.7138, SGD 0.6523, AdaBoost 0.7908, Gradient Boosting 0.6101, XGBoost 0.9183.

Main mistakes of the final vote: Web Attack - XSS labelled Brute Force (90 flows, 69.23% of XSS); Brute Force labelled XSS (38 flows, 12.93%); Bot labelled BENIGN (55 flows, 14.07%, missed attacks); BENIGN labelled PortScan (183), DoS Hulk (89) and Bot (58).

Full tables, per-class scores, confusion matrices and figures are in `reports\evaluation\` and `reports\optimization\tuning_report.md` after running the pipeline.

### ⚠️ Limits of these results

1. **The split is random, so the scores are optimistic.** Flows of the same attack session can land on both sides. The scores do not show how the models would do on another network. A split by source file (every attack class sits in one file) was not run.
2. **Three classes are too small to judge:** the test set has only **2 Heartbleed, 4 Web Attack - Sql Injection and 7 Infiltration** flows. Their scores, and the all-class macro F1, are indicative only.
3. **KNN, SVC and Gradient Boosting were trained on class-capped subsamples** (100,000, 50,000 and 100,000 of 2,017,889 rows), so their class mix differs from real traffic.
4. **Tuning used smaller samples** (at most 100,000 training rows; KNN 50,000; SVC and Gradient Boosting 30,000). Several winners sit at the edge of their grid (Gaussian NB `var_smoothing=1e-9`, Bernoulli NB `alpha=0.1`, Logistic Regression `C=10`, SGD `alpha=1e-6`, AdaBoost depth 4); a wider grid might do better.
5. **`Destination Port` is a model feature.** Attack classes tie to specific ports in this lab capture, so the models may partly learn the lab setup. The comparison without it was **not run**.
6. A prediction is a vote, not proof. The input check catches impossible values but cannot tell whether a combination of values is a realistic flow.

## 🛠️ Technologies

* **Python 3.13**
* **Pandas** 2.3.3, **NumPy** 2.3.5, **PyArrow** (Parquet files)
* **Scikit-learn** 1.8.0 (10 of the 11 models, preprocessing, metrics, GridSearchCV)
* **XGBoost** 3.1.3 (GPU used when available)
* **Matplotlib**, **Seaborn** (figures)
* **Streamlit** 1.52.2 (application)
* **Pytest** (tests); no notebooks: the work is done by scripts

Pinned versions are in `requirements.txt`.

## 📁 Project Structure

```text
DM_Project/
│
├── README.md
├── requirements.txt          pinned dependencies
├── pyproject.toml            packaging (editable install)
├── .gitignore                keeps the raw data, big arrays, logs and editor files out of Git
├── .gitattributes            stores *.joblib model files with Git LFS
│
├── config/
│   ├── settings.py           paths, seed, split, cleaning and preprocessing rules
│   └── model_params.py       model keys, default parameters, tuning grids
│
├── src/nids/
│   ├── data/                 loading and inspection of the raw files
│   ├── cleaning/             cleaning rules
│   ├── eda/                  exploratory statistics and plots
│   ├── preprocessing/        split, pipeline, feature filter, feature schema
│   ├── models/               the 11 classifiers and the registry
│   ├── persistence/          saving and loading models with metadata
│   ├── training/             baseline training
│   ├── evaluation/           metrics, curves, cross-validation, comparison, plots
│   ├── optimization/         GridSearchCV tuning
│   ├── voting/               majority vote
│   ├── prediction/           input validator and predictor
│   └── utils/                logging and file helpers
│
├── scripts/                  the pipeline, one step per script
│   ├── 01_inspect_dataset.py     02_run_eda.py        03_clean_data.py
│   ├── 04_preprocess.py          05_train.py          06_evaluate.py
│   ├── 07_optimize.py            08_make_comparison_files.py
│   └── 09_package_app_assets.py  (optional: zips the files the app needs)
│
├── app/
│   ├── streamlit_app.py      the application
│   └── ui_components.py      form, upload and result views
│
├── tests/                    pytest suite (synthetic data only)
├── docs/                     one .md per code file, plus docs/project/ decision records
│
├── data/preprocessed/        fitted pipeline, label encoder, feature schema, example flows (raw/cleaned data and big arrays are not in Git)
├── artifacts/models/         trained and tuned models (*.joblib via Git LFS) with their metadata
└── reports/                  inspection, EDA, training, evaluation, tuning and comparison reports and figures
```

## ⚙️ Installation

Windows 11, PowerShell, Python 3.13.

```powershell
python -m venv C:\venvs\nids
C:\venvs\nids\Scripts\Activate.ps1
cd <project folder>
pip install -r requirements.txt
pip install -e .
```

## ▶️ Run the Pipeline

From the project folder with the environment active. Each script prints what it did and checks its own output; each has a document in `docs\scripts\`.

```powershell
python scripts\01_inspect_dataset.py      # facts about the raw files
python scripts\02_run_eda.py              # exploratory analysis
python scripts\03_clean_data.py           # cleaning -> data\cleaned\
python scripts\04_preprocess.py           # split + pipeline -> data\preprocessed\
python scripts\05_train.py                # the 11 baseline models (about 18 minutes in the reference run)
python scripts\06_evaluate.py             # baseline evaluation (see --cv-sample-size, --skip-cv for a shorter run)
python scripts\07_optimize.py             # tuning and refit (hours; --dry-run shows the plan first)
python scripts\06_evaluate.py --stage tuned   # scores the tuned models once, three votes, the final set
```

## 🖥️ The Application

```powershell
streamlit run app\streamlit_app.py
```

* **Single flow:** type the 40 feature values, or load a real example of any class from the test split, then press **Predict**. The page shows the winning class, how many models agree, every model's vote and what the test set says about that class.
* **Upload a CSV file:** one flow per row, feature names as headers (a `Label` column is ignored; at most 50,000 rows). Invalid rows are refused with the reason; the others are predicted; the result can be downloaded.
* **About the models:** which version of each model is used and why, the test scores, the known mistakes and the limits.

### Open the App From a Clone (no dataset, no training)

This repository contains the **results too**: the trained models (`artifacts\models\`), the fitted preprocessing files (`data\preprocessed\`, without the large arrays), all reports and figures (`reports\`) and the documents (`docs\`). The model files (`*.joblib`) are stored with **Git LFS**, because GitHub refuses normal files above 100 MB.

```powershell
git lfs install          # once per computer (Git for Windows includes Git LFS)
git clone https://github.com/dataminingcs21/DM_Project.git
cd DM_Project
git lfs pull             # only needed if the model files look like tiny text files
python -m venv C:\venvs\nids
C:\venvs\nids\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app\streamlit_app.py
```

If the app says the models could not be loaded and a `.joblib` file is only a few lines of text (`version https://git-lfs...`), Git LFS was not active: run `git lfs install` and `git lfs pull`. The raw dataset, the cleaned data and the training arrays are **not** in the repository (size); the dataset link is above, and `python scripts\04_preprocess.py` rebuilds the arrays from the cleaned data.

### Compare Predictions With True Labels

```powershell
python scripts\08_make_comparison_files.py                 # 300 random held-out flows
python scripts\08_make_comparison_files.py --per-class 20  # every class gets flows
```

It writes `reports\comparison\comparison_input.csv` (upload it in the app), `comparison_answers.csv` (the true labels) and `comparison.csv` (true and predicted side by side). The flows are real rows of the held-out test split of CIC-IDS2017: unseen by the models, but **not** from another network.

## 🧪 Tests

```powershell
python -m pytest -q
```

Reference run: **664 passed, 1 skipped** (the one skipped test only runs on a computer without a usable GPU) and two expected convergence warnings. Tests use synthetic data only and never touch the real `data\`, `artifacts\` or `logs\` folders. The app is tested headlessly with Streamlit's `AppTest`.

## 👥 Team

* **Team:** CS21

## 🚧 Project Status

🟢 **Completed**

The pipeline (inspection, cleaning, preprocessing, training, evaluation, tuning), the majority vote, the Streamlit application and the test suite are finished. Decisions made after each stage, with their reasons, are in `docs\project\checkpoint_1_decisions.md` to `checkpoint_5_decisions.md`.

**Not done:** the comparison without `Destination Port`; a split by source file or attack campaign; a wider search around the edge-of-grid winners; a weighted vote; any test on traffic from another network (no such data was available).

## 📚 Objective:

The objective is to build and evaluate a supervised machine learning solution that detects network attacks from network traffic data, to compare eleven classification approaches and their majority vote, and to state clearly what the measured performance does and does not show.