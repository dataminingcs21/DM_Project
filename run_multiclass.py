"""
run_multiclass.py
=================
Phase 3 — Multiclass classification on CIC-IDS-2017 (13 classes).

Models trained:
Full training set (1,998,548 rows):
  1. Decision Tree        — max_depth=15, min_samples_leaf=5, class_weight='balanced'
  2. Random Forest        — bounded config: 50 trees, max_depth=12, min_samples_leaf=5, n_jobs=4, class_weight='balanced'
  3. Logistic Regression  — solver='lbfgs', max_iter=1000, class_weight='balanced', n_jobs=-1

Stratified subsample of train (100,000 rows):
  4. Gaussian Naive Bayes — fast baseline on 100k rows
  5. Linear SVM           — One-vs-Rest LinearSVC(class_weight='balanced', max_iter=3000)
  6. KNN (k=3)            — trained on 100k rows, evaluated on full test (499,637 rows) in 25k chunks

Outputs:
  results/tables/multiclass_results.csv
  results/tables/multiclass_confusion_matrix_<model>.csv (all 6 models)
  results/tables/multiclass_rf_feature_importance.csv
  results/figures/multiclass_confusion_matrices.png
  results/figures/multiclass_macro_f1_comparison.png
  results/figures/multiclass_per_class_recall.png
  results/figures/multiclass_rf_feature_importance.png
  models/multiclass_<model>.joblib
"""

from __future__ import annotations
import os, sys, time, gc, tempfile, warnings
from pathlib import Path

# Silence loky and matplotlib warnings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from src import config as C
from src import multiclass_utils as M

try:
    import psutil
    def get_ram_info():
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / 1e9
        sys_pct = psutil.virtual_memory().percent
        return rss_gb, sys_pct
except ImportError:
    def get_ram_info():
        return float('nan'), float('nan')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
# Directories & Constants
# ---------------------------------------------------------------------------
CACHE      = C.CACHE_DIR
TABLES     = C.TABLES_DIR
FIGURES    = C.RESULTS_DIR / 'figures'
MODELS_DIR = C.MODELS_DIR
for d in [TABLES, FIGURES, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED        = 42
SUBSAMPLE_N = 100_000   # train subsample for NB, SVM, KNN
SEP         = '=' * 74

def log(msg=""):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Load Multiclass Data (13 classes)
# ---------------------------------------------------------------------------
log(SEP)
log('PHASE 3 — LOADING 13-CLASS DATASET FROM CACHE')
log(SEP)
t0 = time.perf_counter()
data = M.load_multiclass_data(CACHE)
X_train       = data['X_train']       # (1,998,548, 66) float32
X_test        = data['X_test']        # (499,637, 66) float32
y_train       = data['y_train']       # (1,998,548,) int32
y_test        = data['y_test']        # (499,637,) int32
feature_names = data['feature_names']
class_names   = data['class_names']   # 13 class names

ram_gb, ram_pct = get_ram_info()
log(f'  X_train: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)')
log(f'  X_test:  {X_test.shape}   ({X_test.nbytes / 1e6:.1f} MB)')
log(f'  Classes: {len(class_names)} classes: {class_names}')
log(f'  Process RAM: {ram_gb:.2f} GB  |  System RAM used: {ram_pct:.1f}%')
log(f'  Loaded in {time.perf_counter()-t0:.1f}s\n')

# ---------------------------------------------------------------------------
# Stratified Subsample for slow models
# ---------------------------------------------------------------------------
log(SEP)
log(f'PREPARING STRATIFIED SUBSAMPLE ({SUBSAMPLE_N:,} rows, seed={SEED})')
log(SEP)
rng = np.random.default_rng(SEED)
classes, counts = np.unique(y_train, return_counts=True)
idx_parts = []
for c, cnt in zip(classes, counts):
    c_idx = np.where(y_train == c)[0]
    k = max(2, int(round(SUBSAMPLE_N * cnt / len(y_train))))
    k = min(k, cnt)
    chosen = rng.choice(c_idx, size=k, replace=False)
    idx_parts.append(chosen)
sub_idx = np.concatenate(idx_parts)
rng.shuffle(sub_idx)

X_tr_sub = X_train[sub_idx]
y_tr_sub = y_train[sub_idx]
log(f'  Subsample shape: {X_tr_sub.shape}')
log(f'  Class representation in subsample: {dict(zip(classes, np.bincount(y_tr_sub)))}\n')

all_results = []
models_trained = {}

# ===========================================================================
# 1. Decision Tree (full train)
# ===========================================================================
log(SEP)
log('MODEL 1 — DECISION TREE (full train: 1,998,548 rows)')
log('  Config: max_depth=15, min_samples_leaf=5, class_weight="balanced"')
log(SEP)
_dt_path = MODELS_DIR / 'multiclass_dt.joblib'
if _dt_path.exists():
    log('  [CACHE HIT] Loading Decision Tree from cache (fit_time=206.4s, ram_delta=+0.00 GB)')
    dt = joblib.load(_dt_path)
    fit_time_dt = 206.4
    ram_delta_dt = 0.00
    r_after, _ = get_ram_info()
else:
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=SEED
    )
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()
    dt.fit(X_train, y_train)
    fit_time_dt = time.perf_counter() - t_fit
    r_after, _ = get_ram_info()
    ram_delta_dt = r_after - r_before
    joblib.dump(dt, _dt_path)

log(f'  Fit completed in {fit_time_dt:.1f}s | RAM delta: {ram_delta_dt:+.2f} GB | RAM now: {r_after:.2f} GB')
log(f'  Predicting on full test set ({len(X_test):,} rows) ...')
t_pred = time.perf_counter()
y_pred_dt = dt.predict(X_test)
pred_time_dt = time.perf_counter() - t_pred

res_dt = M.compute_multiclass_metrics(
    y_test, y_pred_dt, name='Decision Tree', fit_time=fit_time_dt,
    n_train=len(X_train), n_test=len(X_test), class_names=class_names
)
res_dt['ram_delta_gb'] = round(ram_delta_dt, 2)
res_dt['predict_seconds'] = round(pred_time_dt, 2)
log(f"    Accuracy:     {res_dt['accuracy']:.4f}")
log(f"    Macro-F1:     {res_dt['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_dt['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_dt['macro_recall']:.4f}\n")
all_results.append(res_dt)
models_trained['Decision Tree'] = dt
gc.collect()

# ===========================================================================
# 2. Random Forest (full train, bounded config)
# ===========================================================================
log(SEP)
log('MODEL 2 — RANDOM FOREST (full train: 1,998,548 rows)')
log('  Config: n_estimators=50, max_depth=12, min_samples_leaf=5, max_features="sqrt", n_jobs=4, class_weight="balanced"')
log(SEP)
_rf_path = MODELS_DIR / 'multiclass_rf.joblib'
if _rf_path.exists():
    log('  [CACHE HIT] Loading Random Forest from cache (fit_time=110.8s, ram_delta=+0.02 GB)')
    rf = joblib.load(_rf_path)
    fit_time_rf = 110.8
    ram_delta_rf = 0.02
    r_after, _ = get_ram_info()
else:
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=12, min_samples_leaf=5,
        max_features='sqrt', n_jobs=4, class_weight='balanced',
        random_state=SEED
    )
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()
    rf.fit(X_train, y_train)
    fit_time_rf = time.perf_counter() - t_fit
    r_after, _ = get_ram_info()
    ram_delta_rf = r_after - r_before
    joblib.dump(rf, _rf_path)

log(f'  Fit completed in {fit_time_rf:.1f}s | RAM delta: {ram_delta_rf:+.2f} GB | RAM now: {r_after:.2f} GB')
log(f'  Predicting on full test set ({len(X_test):,} rows) ...')
t_pred = time.perf_counter()
y_pred_rf = rf.predict(X_test)
pred_time_rf = time.perf_counter() - t_pred

res_rf = M.compute_multiclass_metrics(
    y_test, y_pred_rf, name='Random Forest', fit_time=fit_time_rf,
    n_train=len(X_train), n_test=len(X_test), class_names=class_names
)
res_rf['ram_delta_gb'] = round(ram_delta_rf, 2)
res_rf['predict_seconds'] = round(pred_time_rf, 2)
log(f"    Accuracy:     {res_rf['accuracy']:.4f}")
log(f"    Macro-F1:     {res_rf['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_rf['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_rf['macro_recall']:.4f}\n")
all_results.append(res_rf)
models_trained['Random Forest'] = rf

# Feature importances
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
importances.to_csv(TABLES / 'multiclass_rf_feature_importance.csv', header=['importance'])
fig, ax = plt.subplots(figsize=(10, 6))
importances.head(15).sort_values().plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='white')
ax.set_title('Random Forest — Top 15 Feature Importances (Multiclass 13)', fontsize=12)
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig(FIGURES / 'multiclass_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
gc.collect()

# ===========================================================================
# 3. Logistic Regression (Multinomial, full train)
# ===========================================================================
log(SEP)
log('MODEL 3 — LOGISTIC REGRESSION (full train: 1,998,548 rows)')
log('  Config: solver="lbfgs", max_iter=1000, class_weight="balanced", n_jobs=-1')
log(SEP)
_lr_path = MODELS_DIR / 'multiclass_lr.joblib'
if _lr_path.exists():
    log('  [CACHE HIT] Loading Logistic Regression from cache (fit_time=3105.0s, ram_delta=+0.62 GB)')
    lr = joblib.load(_lr_path)
    fit_time_lr = 3105.0
    ram_delta_lr = 0.62
    r_after, _ = get_ram_info()
else:
    lr = LogisticRegression(
        solver='lbfgs', max_iter=1000,
        class_weight='balanced', n_jobs=-1, random_state=SEED
    )
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(X_train, y_train)
        fit_time_lr = time.perf_counter() - t_fit
        r_after, _ = get_ram_info()
        ram_delta_lr = r_after - r_before
        joblib.dump(lr, _lr_path)

    conv_warn = [w for w in caught_warnings if issubclass(w.category, ConvergenceWarning)]
    if conv_warn:
        log(f'  [WARNING] LogisticRegression issued ConvergenceWarning: {conv_warn[0].message}')
    else:
        log(f'  LogisticRegression converged successfully in {lr.n_iter_} iterations!')

log(f'  Fit completed in {fit_time_lr:.1f}s | RAM delta: {ram_delta_lr:+.2f} GB | RAM now: {r_after:.2f} GB')
log(f'  Predicting on full test set ({len(X_test):,} rows) ...')
t_pred = time.perf_counter()
y_pred_lr = lr.predict(X_test)
pred_time_lr = time.perf_counter() - t_pred

res_lr = M.compute_multiclass_metrics(
    y_test, y_pred_lr, name='Logistic Regression', fit_time=fit_time_lr,
    n_train=len(X_train), n_test=len(X_test), class_names=class_names
)
res_lr['ram_delta_gb'] = round(ram_delta_lr, 2)
res_lr['predict_seconds'] = round(pred_time_lr, 2)
log(f"    Accuracy:     {res_lr['accuracy']:.4f}")
log(f"    Macro-F1:     {res_lr['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_lr['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_lr['macro_recall']:.4f}\n")
all_results.append(res_lr)
models_trained['Logistic Regression'] = lr
gc.collect()

# ===========================================================================
# 4. Gaussian Naive Bayes (subsample 100k)
# ===========================================================================
log(SEP)
log(f'MODEL 4 — GAUSSIAN NAIVE BAYES (subsample: {SUBSAMPLE_N:,} train rows)')
log(SEP)
_nb_path = MODELS_DIR / 'multiclass_nb.joblib'
if _nb_path.exists():
    log('  [CACHE HIT] Loading Naive Bayes from cache (fit_time=0.06s, ram_delta=+0.00 GB)')
    nb = joblib.load(_nb_path)
    fit_time_nb = 0.06
    ram_delta_nb = 0.00
    r_after, _ = get_ram_info()
else:
    nb = GaussianNB()
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()
    nb.fit(X_tr_sub, y_tr_sub)
    fit_time_nb = time.perf_counter() - t_fit
    r_after, _ = get_ram_info()
    ram_delta_nb = r_after - r_before
    joblib.dump(nb, _nb_path)

log(f'  Fit completed in {fit_time_nb:.2f}s | RAM delta: {ram_delta_nb:+.2f} GB | RAM now: {r_after:.2f} GB')
log(f'  Predicting on full test set ({len(X_test):,} rows) ...')
t_pred = time.perf_counter()
y_pred_nb = nb.predict(X_test)
pred_time_nb = time.perf_counter() - t_pred

res_nb = M.compute_multiclass_metrics(
    y_test, y_pred_nb, name='Naive Bayes', fit_time=fit_time_nb,
    n_train=len(X_tr_sub), n_test=len(X_test), class_names=class_names
)
res_nb['ram_delta_gb'] = round(ram_delta_nb, 2)
res_nb['predict_seconds'] = round(pred_time_nb, 2)
log(f"    Accuracy:     {res_nb['accuracy']:.4f}")
log(f"    Macro-F1:     {res_nb['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_nb['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_nb['macro_recall']:.4f}\n")
all_results.append(res_nb)
models_trained['Naive Bayes'] = nb
gc.collect()

# ===========================================================================
# 5. Linear SVM (One-vs-Rest, subsample 100k)
# ===========================================================================
log(SEP)
log(f'MODEL 5 — LINEAR SVM (subsample: {SUBSAMPLE_N:,} train rows)')
log('  Config: LinearSVC(class_weight="balanced", max_iter=3000, multi_class="ovr")')
log(SEP)
_svm_path = MODELS_DIR / 'multiclass_svm.joblib'
if _svm_path.exists():
    log('  [CACHE HIT] Loading Linear SVM from cache (fit_time=537.5s, ram_delta=-0.00 GB)')
    svm = joblib.load(_svm_path)
    fit_time_svm = 537.5
    ram_delta_svm = 0.00
    r_after, _ = get_ram_info()
else:
    svm = LinearSVC(
        class_weight='balanced', max_iter=3000, random_state=SEED
    )
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()
    svm.fit(X_tr_sub, y_tr_sub)
    fit_time_svm = time.perf_counter() - t_fit
    r_after, _ = get_ram_info()
    ram_delta_svm = r_after - r_before
    joblib.dump(svm, _svm_path)

log(f'  Fit completed in {fit_time_svm:.1f}s | RAM delta: {ram_delta_svm:+.2f} GB | RAM now: {r_after:.2f} GB')
log(f'  Predicting on full test set ({len(X_test):,} rows) ...')
t_pred = time.perf_counter()
y_pred_svm = svm.predict(X_test)
pred_time_svm = time.perf_counter() - t_pred

res_svm = M.compute_multiclass_metrics(
    y_test, y_pred_svm, name='Linear SVM', fit_time=fit_time_svm,
    n_train=len(X_tr_sub), n_test=len(X_test), class_names=class_names
)
res_svm['ram_delta_gb'] = round(ram_delta_svm, 2)
res_svm['predict_seconds'] = round(pred_time_svm, 2)
log(f"    Accuracy:     {res_svm['accuracy']:.4f}")
log(f"    Macro-F1:     {res_svm['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_svm['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_svm['macro_recall']:.4f}\n")
all_results.append(res_svm)
models_trained['Linear SVM'] = svm
gc.collect()

# ===========================================================================
# 6. KNN (k=3, subsample 100k train, batch evaluation on 499k test)
# ===========================================================================
log(SEP)
log(f'MODEL 6 — KNN (k=3, subsample: {SUBSAMPLE_N:,} train rows, full test: {len(X_test):,} rows, n_jobs=4)')
log(SEP)
_knn_path = MODELS_DIR / 'multiclass_knn.joblib'
if _knn_path.exists():
    log('  [CACHE HIT] Loading KNN from cache (fit_time=0.02s, ram_delta=+0.00 GB)')
    knn = joblib.load(_knn_path)
    fit_time_knn = 0.02
    ram_delta_knn = 0.00
else:
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
    r_before, _ = get_ram_info()
    t_fit = time.perf_counter()
    knn.fit(X_tr_sub, y_tr_sub)
    fit_time_knn = time.perf_counter() - t_fit
    r_after, _ = get_ram_info()
    ram_delta_knn = r_after - r_before
    joblib.dump(knn, _knn_path)

log(f'  Fit completed in {fit_time_knn:.2f}s | RAM delta: {ram_delta_knn:+.2f} GB')
log(f'  Evaluating KNN in batches of 25,000 queries on 499,637 test flows ...')
t_pred_start = time.perf_counter()
batch_size = 25000
n_samples = len(X_test)
y_preds_knn = []

for i in range(0, n_samples, batch_size):
    end = min(i + batch_size, n_samples)
    y_preds_knn.append(knn.predict(X_test[i:end]))
    elapsed = time.perf_counter() - t_pred_start
    pct = end / n_samples * 100
    if (i // batch_size) % 4 == 0 or end == n_samples:
        rss_gb, _ = get_ram_info()
        log(f'    Progress: {end:>7,}/{n_samples:,} ({pct:5.1f}%) | Elapsed: {elapsed:5.1f}s | RAM: {rss_gb:.2f} GB')

y_pred_knn = np.concatenate(y_preds_knn)
pred_time_knn = time.perf_counter() - t_pred_start

res_knn = M.compute_multiclass_metrics(
    y_test, y_pred_knn, name='KNN', fit_time=fit_time_knn,
    n_train=len(X_tr_sub), n_test=len(X_test), class_names=class_names
)
res_knn['ram_delta_gb'] = round(ram_delta_knn, 2)
res_knn['predict_seconds'] = round(pred_time_knn, 2)
log(f"    Accuracy:     {res_knn['accuracy']:.4f}")
log(f"    Macro-F1:     {res_knn['macro_f1']:.4f}")
log(f"    Weighted-F1:  {res_knn['weighted_f1']:.4f}")
log(f"    Macro-Recall: {res_knn['macro_recall']:.4f}\n")
all_results.append(res_knn)
models_trained['KNN'] = knn
gc.collect()

# ===========================================================================
# Save Tables
# ===========================================================================
log(SEP)
log('SAVING MULTICLASS RESULTS TABLES')
log(SEP)
# 1. multiclass_results.csv
df_summary = pd.DataFrame([{
    'model': r['model'],
    'n_train': r['n_train'],
    'n_test_evaluated': r['n_test_evaluated'],
    'fit_time': r['fit_time'],
    'accuracy': r['accuracy'],
    'macro_f1': r['macro_f1'],
    'weighted_f1': r['weighted_f1'],
    'macro_recall': r['macro_recall'],
    'per_class_recall': r['per_class_recall'],
} for r in all_results])
out_summary = TABLES / 'multiclass_results.csv'
df_summary.to_csv(out_summary, index=False)
log(f'Saved -> {out_summary}')

# 2. Confusion matrices 13x13 for EACH model
for r in all_results:
    m_name = r['model'].lower().replace(' ', '_')
    cm_df = pd.DataFrame(r['_cm'], index=class_names, columns=class_names)
    cm_path = TABLES / f'multiclass_confusion_matrix_{m_name}.csv'
    cm_df.to_csv(cm_path)
    log(f'Saved 13x13 CM -> {cm_path}')

# ===========================================================================
# Save Figures
# ===========================================================================
log(SEP)
log('SAVING MULTICLASS EVALUATION FIGURES')
log(SEP)
M.plot_multiclass_confusion_matrices(all_results, class_names, FIGURES / 'multiclass_confusion_matrices.png')
M.plot_multiclass_macro_f1(all_results, FIGURES / 'multiclass_macro_f1_comparison.png')
M.plot_multiclass_per_class_recall_heatmap(all_results, class_names, FIGURES / 'multiclass_per_class_recall.png')

# ===========================================================================
# Final Summary Table
# ===========================================================================
log(SEP)
log('PHASE 3 MULTICLASS EXPERIMENT COMPLETE — SUMMARY TABLE')
log(SEP)
disp_cols = ['model', 'n_train', 'n_test_evaluated', 'fit_time', 'accuracy', 'macro_f1', 'weighted_f1', 'macro_recall']
log(df_summary[disp_cols].sort_values('macro_f1', ascending=False).to_string(index=False))

log('\nPer-Class Recall Summary Table (13 classes):')
pcr_rows = []
for r in all_results:
    row = {'model': r['model']}
    row.update(r['_per_class_dict'])
    pcr_rows.append(row)
df_pcr = pd.DataFrame(pcr_rows).set_index('model')
log(df_pcr.to_string())
log(SEP)
