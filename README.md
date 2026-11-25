# Network Attack Detection Using Machine Learning

## 📌 Project Overview

This project develops a machine learning system that analyzes network traffic flows and classifies them as **BENIGN (normal traffic)** or one of **14 attack types**, resulting in a 15-class classification problem.

Eleven supervised machine learning classifiers are trained, evaluated, and tuned using the CIC-IDS2017 dataset. Their predictions are combined through an **equal-weight majority voting system** to produce the final classification.

A **Streamlit web application** provides an interactive interface where users can:
- Enter the 40 required network-flow features manually.
- Load a real example from the held-out test dataset.
- Upload a CSV file containing multiple network flows.
- View the final prediction and the individual votes of all 11 classifiers.
- Explore model information, evaluation results, and known limitations.

The project was developed as part of the **Data Mining** course.

## 🎯 Project Information

- **Project Title:** Network Attack Detection Using Machine Learning
- **Approach:** Supervised multiclass classification.
- **Number of Classes:** 15 (BENIGN + 14 attack types).
- **Number of Classifiers:** 11 supervised machine learning models.
- **Ensemble Strategy:** Equal-weight majority voting.
- **Domain:** Cybersecurity / Network Security.
- **Course:** Data Mining.
- **Methods Not Used:** Clustering and association-rule mining are outside the scope of this project.

## 📊 Dataset

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

This approach retains the version supported by the tuning-fold evidence rather than assuming that hyperparameter optimization always improves generalization.

### Majority Voting

The final prediction is generated using an **equal-weight majority vote**.

Each of the eleven classifiers contributes exactly one vote. No classifier receives a manually assigned weight.

The predicted class is the class receiving the largest number of votes.

**Tie-breaking rule:** When a tie occurs between classes, the system uses the `prefer_non_benign` policy, preferring an attack class over BENIGN.

## 📈 Evaluation

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

| Majority Vote | Macro F1 (All Classes) | Macro F1 (Reliable Classes) |
|---|---:|---:|
| Baseline models only | 0.9095 | 0.9096 |
| Tuned models only | **0.9142** | **0.9155** |
| Final set (evidence rule) | 0.9054 | 0.9138 |

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

## 🛠️ Technologies

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

## 📁 Project Structure

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

The environment is now prepared. To run the full training pipeline, the original dataset must also be downloaded and placed in the expected data directory.

## ▶️ Run the Pipeline

The following scripts reproduce the main stages of the machine learning workflow.

Run them from the repository root with the virtual environment activated.

Each script performs a specific task and has corresponding documentation under `docs/scripts/`.

```powershell
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

The project provides an interactive Streamlit interface for classifying individual network flows or batches of flows.

### Launch the Application

From the project root, with the virtual environment activated and the inference artifacts installed:

```powershell
streamlit run app\streamlit_app.py
```

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

```powershell
# Generate 300 randomly selected test flows
python scripts\08_make_comparison_files.py

# Generate 20 samples per class
python scripts\08_make_comparison_files.py --per-class 20
```

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

## 🧪 Tests

The project includes an automated test suite implemented using **Pytest**.

Run all tests from the repository root:

```powershell
python -m pytest -q
```

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

## 👥 Team

- **Team:** CS21
- **Course:** Data Mining

## 🚧 Project Status

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