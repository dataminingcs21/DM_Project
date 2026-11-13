# Network Attack Detection Using Machine Learning

## 📌 Project Overview

This project develops machine-learning classifiers for **Network Intrusion Detection** on the [CIC-IDS-2017](https://unb.ca/cic/datasets/ids-2017.html) dataset (2.8 M raw flow records, 15 traffic classes).

Two tasks are addressed:
1. **Binary classification** — BENIGN (0) vs ATTACK (1)
2. **Multiclass classification** — 15 → 12 classes (see below)

Developed for the **Data Mining** course — Team CS24.

## 📊 Dataset

**CIC-IDS-2017** — 8 CSV files (Mon–Fri), ~2.83 M rows × 79 raw columns.

After cleaning (inf→NaN, constant columns, NaN rows, deduplication):
**2,498,185 rows × 66 features** (80/20 stratified train/test split).

### Encoding note
The Thursday WebAttacks CSV contains an encoding issue in its label strings.
Using `encoding='latin-1'` when reading, followed by label normalisation that
strips non-alphabetic characters, resolves the issue without data loss.

### Deduplication
329,691 exact duplicate rows (identical feature values + identical label) were
removed — 11.7% of the cleaned dataset.  Duplicates are rows with the same
feature vector and the same label; the CSVs do not contain IP/port/timestamp
columns.

### Conflicting-label rows
After deduplication, a small number of feature-identical rows carry different
labels.  These are kept in the dataset; their label pairs are reported in
`results/tables/conflicting_label_pairs.csv`.

## 🧠 Pipeline

```text
8 CSV files  → load (latin-1, float32 downcast per file)
  ↓ clean: inf→NaN, constant cols, NaN rows, dedup
  ↓ encode labels → (X, y_multi, y_bin)
  ↓ 80/20 stratified split on y_multi (shared by both tasks)
  ↓ near-zero column filter (computed on X_train only)
  ↓ StandardScaler (fitted on X_train only, saved to cache/)
  ↓ cache/splits.npz  (reused by notebooks 02 & 03)
```

## 🤖 Models

### Task 1 — Binary (BENIGN vs ATTACK)

| Model | Training data | Notes |
|---|---|---|
| Logistic Regression | full (2M rows) | C tuned by 3-fold CV on 50k subsample |
| Decision Tree | full | max_depth tuned by 3-fold CV on 50k subsample |
| Random Forest | full | bounded config (100 trees, max_depth=20, n_jobs=2) |
| Naive Bayes (Gaussian) | 100k subsample | no tuning |
| KNN | 100k subsample | k tuned by 3-fold CV on 20k subsample |
| Linear SVM | 100k subsample | C tuned by 3-fold CV on 50k subsample |

### Task 2 — Multiclass (planned)

**Rare-class strategy:**
- The 3 Web Attack sub-types are merged into a single **"Web Attack"** class.
- **Heartbleed** (11 samples) and **Infiltration** (36 samples) are excluded
  from the main experiment — documented as a limitation.
- An optional second run keeps all classes with `class_weight='balanced'` and
  flags the per-class scores of Heartbleed and Infiltration as unreliable.
- No SMOTE — the same saved split is reused (filter/merge on existing indices).

## 📈 Evaluation

**Binary task:** recall, precision, F1 of the ATTACK class, confusion matrix,
ROC-AUC, PR-AUC, training time.  Also: per-attack-type recall using original
multiclass labels (shows which attacks the binary classifier misses).

**Priority metric:** Recall on ATTACK — a missed attack has worse security
consequences than a false alarm.

## 📁 Project Structure

```text
DM_Project/
├── README.md
├── requirements.txt
├── download_dataset.py
├── run_binary.py           ← Phase 2 runner
│
├── dataset/                ← 8 CIC-IDS-2017 CSVs
├── notebooks/
│   ├── 01_exploration_cleaning.ipynb
│   ├── 02_binary_classification.ipynb       (planned)
│   └── 03_multiclass_classification.ipynb   (planned)
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
│   ├── tables/             ← CSV result files
│   └── figures/            ← PNG plots
│
├── models/                 ← fitted model .joblib files
├── cache/                  ← splits.npz, scaler, indices
├── documentation/          ← internal reference docs
└── ids_project/            ← Django web dashboard (future)
```

## 🛠️ Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, psutil.

## 🚧 Status

🟡 **Phase 2 in progress** — binary classification models trained and evaluated.
Phase 3 (multiclass) next.
