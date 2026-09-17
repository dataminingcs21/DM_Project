# Network Attack Detection Using Machine Learning

## 📌 Project Overview

This project focuses on **Network Attack Detection Using Machine Learning**.

The main objective is to develop a machine learning model capable of analyzing network traffic data and detecting whether a network activity corresponds to a normal behavior or a potential attack.

The project is developed as part of the **Data Mining** course.

## 🎯 Project Information

* **Project Title:** Network Attack Detection Using Machine Learning
* **Chosen Approach:** Supervised Classification
* **Domain:** Cybersecurity / Network Security
* **Course:** Data Mining

## 📊 Dataset

The project uses a network traffic dataset containing features describing network connections and their corresponding attack or normal traffic labels.

**Dataset:** `https://unb.ca/cic/datasets/ids-2017.html`

The dataset will be used to:

* Analyze and understand network traffic.
* Preprocess and clean the data.
* Select relevant features.
* Train supervised machine learning models.
* Evaluate the performance of the models.
* Classify network traffic as normal or malicious.

> **Note:** The exact dataset name and source will be added once the dataset is confirmed.

## 🧠 Methodology

The project will follow the following data mining pipeline:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Selection / Engineering
   ↓
Train / Test Split
   ↓
Supervised Classification
   ↓
Model Evaluation
   ↓
Network Attack Detection
```

## 🤖 Machine Learning

Several supervised classification algorithms may be investigated and compared, such as:

* Decision Tree
* Random Forest
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes

The final selection of models will depend on the characteristics of the dataset and the experimental results.

## 📈 Evaluation

The classification models will be evaluated using appropriate metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

For network attack detection, particular attention will be given to **Recall**, since failing to detect an actual attack can have significant security implications.

## 🛠️ Technologies

The project may use the following technologies:

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

## 📁 Project Structure

```text
DM_Project/
│
├── README.md
├── dataset/
│   └── dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── training.py
│   └── evaluation.py
│
└── results/
    └── figures/
```

## 👥 Team

* **Team:** CS24

## 🚧 Project Status

🟡 **In Progress**

The project is currently under development. Data preprocessing, exploratory analysis, model training, and evaluation will be added progressively.

## 📚 Objective

The final objective is to build and evaluate a supervised machine learning solution capable of detecting network attacks from network traffic data and to analyze the performance of different classification approaches.

