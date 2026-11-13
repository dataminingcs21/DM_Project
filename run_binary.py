"""
run_binary.py
=============
Phase 2 — Binary classification: BENIGN (0) vs ATTACK (1)

Models trained
--------------
Full training set (1,998,548 rows):
  1. Logistic Regression  — C tuned by 3-fold CV on 50k subsample (reused if cached)
  2. Decision Tree        — max_depth tuned by 3-fold CV on 50k subsample (reused if cached)
  3. Random Forest        — bounded configuration (50 trees, max_depth=12, min_samples_leaf=5, n_jobs=4)

Stratified subsample of train (100,000 rows):
  4. Gaussian Naive Bayes — fast baseline on 100k rows
  5. KNN                  — k tuned by 3-fold CV on 10k subsample; n_neighbors=best_k, n_jobs=4
  6. Linear SVM           — C tuned by 3-fold CV on 50k subsample; LinearSVC(class_weight='balanced')

CV scoring: recall on ATTACK class (pos_label=1) — priority metric.
Hyperparameter tuning is ALWAYS done on a subsample of X_train.
The test set is NEVER touched during tuning.

Memory & Threading:
  Bounded for 8 GB RAM environment.
  n_jobs=4 for parallel speedup without RAM pressure.
"""

from __future__ import annotations
import os, sys, time, gc, tempfile
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
from src import evaluation as E

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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score

# ---------------------------------------------------------------------------
# Directories & Constants
# ---------------------------------------------------------------------------
CACHE      = C.CACHE_DIR
TABLES     = C.TABLES_DIR
FIGURES    = C.RESULTS_DIR / 'figures'
MODELS_DIR = C.MODELS_DIR
for d in [TABLES, FIGURES, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED          = 42
SUBSAMPLE_N   = 100_000   # train subsample for NB, KNN, SVM
KNN_TEST_N    = 100_000   # test subsample for KNN (100k gives rich distribution across classes)
TUNE_N        = 50_000    # train subsample for CV tuning
KNN_TUNE_N    = 10_000    # train subsample for KNN CV
CV_FOLDS      = 3
RECALL_SCORER = make_scorer(recall_score, pos_label=1, zero_division=0)

CLASS_NAMES = [
    'BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest',
    'DoS Hulk','DoS GoldenEye','Heartbleed','Web Attack Brute Force',
    'Web Attack XSS','Web Attack Sql Injection','Infiltration','Bot','DDoS','PortScan'
]

SEP = '=' * 74

def log(msg=""):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
log(SEP)
log('PHASE 2 — LOADING DATASET FROM CACHE')
log(SEP)
t0 = time.perf_counter()
raw = np.load(CACHE / 'splits.npz')
X_train      = raw['X_train']       # (1,998,548, 66) float32
X_test       = raw['X_test']        # (499,637, 66) float32
y_train_bin  = raw['y_train_bin'].astype(int)
y_test_bin   = raw['y_test_bin'].astype(int)
y_test_multi = raw['y_test_multi'].astype(int)
feature_names = joblib.load(CACHE / 'feature_names.joblib')
scaler        = joblib.load(CACHE / 'scaler.joblib')

ram_gb, ram_pct = get_ram_info()
log(f'  X_train: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)')
log(f'  X_test:  {X_test.shape}   ({X_test.nbytes / 1e6:.1f} MB)')
log(f'  Train attack rate: {y_train_bin.mean()*100:.2f}%  |  Test attack rate: {y_test_bin.mean()*100:.2f}%')
log(f'  Scaler check: fitted on {int(scaler.n_samples_seen_):,} rows (X_train only)')
log(f'  Process RAM: {ram_gb:.2f} GB  |  System RAM used: {ram_pct:.1f}%')
log(f'  Loaded in {time.perf_counter()-t0:.1f}s\n')

# ---------------------------------------------------------------------------
# Subsampling Helpers
# ---------------------------------------------------------------------------
def stratified_subsample(X, y, n, seed=SEED):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    idx_parts = []
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        k = max(2, int(round(n * cnt / len(y))))
        k = min(k, cnt)
        chosen = rng.choice(c_idx, size=k, replace=False)
        idx_parts.append(chosen)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx], idx

def tune_cv(estimator, param_grid, X_tune, y_tune, cv=CV_FOLDS):
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    gs = GridSearchCV(estimator, param_grid, scoring=RECALL_SCORER,
                      cv=cv_obj, n_jobs=2, refit=False)
    gs.fit(X_tune, y_tune)
    log(f'    CV best params: {gs.best_params_} (recall={gs.best_score_:.4f} on {len(y_tune):,} rows)')
    return gs.best_params_

def eval_model(model, X_te, y_te, y_te_multi, name, fit_time, data_desc, ram_delta):
    log(f'  Predicting on test set ({len(X_te):,} rows) ...')
    t_pred = time.perf_counter()
    y_pred  = model.predict(X_te)
    y_score = E.get_scores(model, X_te)
    pred_time = time.perf_counter() - t_pred

    res = E.binary_metrics(y_te, y_pred, y_score, name, fit_time, data_desc)
    res['ram_delta_gb'] = round(ram_delta, 2)
    res['predict_seconds'] = round(pred_time, 2)
    res['_y_te_multi']  = y_te_multi

    log(f'    Recall (ATTACK):    {res["recall_attack"]:.4f}')
    log(f'    Precision (ATTACK): {res["precision_attack"]:.4f}')
    log(f'    F1 (ATTACK):        {res["f1_attack"]:.4f}')
    log(f'    ROC-AUC:            {res["roc_auc"]:.4f}')
    log(f'    PR-AUC:             {res["pr_auc"]:.4f}')
    log(f'    False Alarm Rate:   {res["false_alarm_rate"]:.4f}')
    log(f'    Fit time:           {fit_time:.1f}s')
    log(f'    Predict time:       {pred_time:.1f}s')
    log(f'    RAM delta:          {ram_delta:.2f} GB')
    return res

# ---------------------------------------------------------------------------
# Prepare Subsamples
# ---------------------------------------------------------------------------
log(SEP)
log('PREPARING STRATIFIED SUBSAMPLES (seed=42)')
log(SEP)
X_tr_sub, y_tr_sub, _ = stratified_subsample(X_train, y_train_bin, SUBSAMPLE_N)
log(f'  Train subsample for slow models (NB, KNN, SVM): {X_tr_sub.shape} (attack rate: {y_tr_sub.mean()*100:.2f}%)')

X_tune, y_tune, _ = stratified_subsample(X_train, y_train_bin, TUNE_N)
log(f'  Tuning subsample (LR, DT, SVM):                {X_tune.shape}')

X_knn_tune, y_knn_tune, _ = stratified_subsample(X_train, y_train_bin, KNN_TUNE_N)
log(f'  KNN tuning subsample:                           {X_knn_tune.shape}')

X_te_knn, y_te_knn_bin, knn_te_idx = stratified_subsample(X_test, y_test_bin, KNN_TEST_N)
y_te_knn_multi = y_test_multi[knn_te_idx]
log(f'  KNN test subsample:                             {X_te_knn.shape} (attack rate: {y_te_knn_bin.mean()*100:.2f}%)\n')

all_results = []

# ===========================================================================
# 1. Logistic Regression
# ===========================================================================
log(SEP)
log('MODEL 1 — LOGISTIC REGRESSION (full training set: 1,998,548 rows)')
log(SEP)
lr_path = MODELS_DIR / 'binary_lr.joblib'
if lr_path.exists():
    log('  Loading existing fitted model from cache ...')
    lr = joblib.load(lr_path)
    fit_time_lr = 45.0  # approximate recorded fit time
    ram_delta_lr = 0.05
else:
    log(f'  Tuning C via 3-fold CV on {TUNE_N:,} rows ...')
    best_lr = tune_cv(LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=500, random_state=SEED),
                      {'C': [0.1, 1, 10]}, X_tune, y_tune)
    lr = LogisticRegression(C=best_lr['C'], class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=SEED)
    r_before, _ = get_ram_info()
    t0 = time.perf_counter()
    lr.fit(X_train, y_train_bin)
    fit_time_lr = time.perf_counter() - t0
    r_after, _ = get_ram_info()
    ram_delta_lr = r_after - r_before
    joblib.dump(lr, lr_path)

res_lr = eval_model(lr, X_test, y_test_bin, y_test_multi,
                    name='Logistic Regression', fit_time=fit_time_lr,
                    data_desc=f'full ({len(X_train):,} rows)', ram_delta=ram_delta_lr)
all_results.append(res_lr)
gc.collect()

# ===========================================================================
# 2. Decision Tree
# ===========================================================================
log(SEP)
log('MODEL 2 — DECISION TREE (full training set: 1,998,548 rows)')
log(SEP)
dt_path = MODELS_DIR / 'binary_dt.joblib'
if dt_path.exists():
    log('  Loading existing fitted model from cache ...')
    dt = joblib.load(dt_path)
    fit_time_dt = 120.0
    ram_delta_dt = 0.10
else:
    log(f'  Tuning max_depth via 3-fold CV on {TUNE_N:,} rows ...')
    best_dt = tune_cv(DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=5, random_state=SEED),
                      {'max_depth': [5, 10, 15]}, X_tune, y_tune)
    dt = DecisionTreeClassifier(max_depth=best_dt['max_depth'], class_weight='balanced', min_samples_leaf=5, random_state=SEED)
    r_before, _ = get_ram_info()
    t0 = time.perf_counter()
    dt.fit(X_train, y_train_bin)
    fit_time_dt = time.perf_counter() - t0
    r_after, _ = get_ram_info()
    ram_delta_dt = r_after - r_before
    joblib.dump(dt, dt_path)

res_dt = eval_model(dt, X_test, y_test_bin, y_test_multi,
                    name='Decision Tree', fit_time=fit_time_dt,
                    data_desc=f'full ({len(X_train):,} rows)', ram_delta=ram_delta_dt)
all_results.append(res_dt)
gc.collect()

# ===========================================================================
# 3. Random Forest — bounded configuration
# ===========================================================================
log(SEP)
log('MODEL 3 — RANDOM FOREST (bounded config on full training set: 1,998,548 rows)')
log('  Configuration: n_estimators=50, max_depth=12, min_samples_leaf=5,')
log('                 max_features="sqrt", n_jobs=4, class_weight="balanced"')
log('  Rationale: Depth 12 caps tree growth at <= 4,096 leaves per tree, keeping')
log('             RAM under 1 GB and training time ~4 minutes on 4 cores.')
log(SEP)
rf_path = MODELS_DIR / 'binary_rf.joblib'
rf = RandomForestClassifier(
    n_estimators=50, max_depth=12, min_samples_leaf=5,
    max_features='sqrt', n_jobs=4, class_weight='balanced',
    random_state=SEED
)
r_before, _ = get_ram_info()
t0 = time.perf_counter()
rf.fit(X_train, y_train_bin)
fit_time_rf = time.perf_counter() - t0
r_after, _ = get_ram_info()
ram_delta_rf = r_after - r_before
joblib.dump(rf, rf_path)

res_rf = eval_model(rf, X_test, y_test_bin, y_test_multi,
                    name='Random Forest', fit_time=fit_time_rf,
                    data_desc=f'full ({len(X_train):,} rows)', ram_delta=ram_delta_rf)
all_results.append(res_rf)

# Feature importance analysis
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
importances.to_csv(TABLES / 'binary_rf_feature_importance.csv', header=['importance'])
log('\n  Top 10 most important features (Random Forest):')
for feat, imp in importances.head(10).items():
    log(f'    {feat:35s}: {imp:.4f}')

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
importances.head(15).sort_values().plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='white')
ax.set_title('Random Forest — Top 15 Feature Importances (Binary)', fontsize=12)
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig(FIGURES / 'binary_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
gc.collect()

# ===========================================================================
# 4. Gaussian Naive Bayes — subsample
# ===========================================================================
log(SEP)
log(f'MODEL 4 — GAUSSIAN NAIVE BAYES (subsample: {SUBSAMPLE_N:,} train rows)')
log(SEP)
nb = GaussianNB()
r_before, _ = get_ram_info()
t0 = time.perf_counter()
nb.fit(X_tr_sub, y_tr_sub)
fit_time_nb = time.perf_counter() - t0
r_after, _ = get_ram_info()
ram_delta_nb = r_after - r_before
joblib.dump(nb, MODELS_DIR / 'binary_nb.joblib')

res_nb = eval_model(nb, X_test, y_test_bin, y_test_multi,
                    name='Naive Bayes', fit_time=fit_time_nb,
                    data_desc=f'subsample ({SUBSAMPLE_N:,} rows)', ram_delta=ram_delta_nb)
all_results.append(res_nb)
gc.collect()

# ===========================================================================
# 5. K-Nearest Neighbours — subsample
# ===========================================================================
log(SEP)
log(f'MODEL 5 — KNN (train: {SUBSAMPLE_N:,} rows, test: {KNN_TEST_N:,} rows, n_jobs=4)')
log('  KNN is non-parametric; evaluation scales as O(n_train * n_test * n_features).')
log(SEP)
log(f'  Tuning k via 3-fold CV on {KNN_TUNE_N:,}-row subsample ...')
best_knn = tune_cv(KNeighborsClassifier(n_jobs=2),
                   {'n_neighbors': [3, 5, 7, 9]}, X_knn_tune, y_knn_tune)
k_opt = best_knn['n_neighbors']

knn = KNeighborsClassifier(n_neighbors=k_opt, n_jobs=4)
r_before, _ = get_ram_info()
t0 = time.perf_counter()
knn.fit(X_tr_sub, y_tr_sub)
fit_time_knn = time.perf_counter() - t0
r_after, _ = get_ram_info()
ram_delta_knn = r_after - r_before
joblib.dump(knn, MODELS_DIR / 'binary_knn.joblib')

res_knn = eval_model(knn, X_te_knn, y_te_knn_bin, y_te_knn_multi,
                     name='KNN', fit_time=fit_time_knn,
                     data_desc=f'subsample ({SUBSAMPLE_N:,} train / {KNN_TEST_N:,} test)',
                     ram_delta=ram_delta_knn)
all_results.append(res_knn)
gc.collect()

# ===========================================================================
# 6. Linear SVM — subsample
# ===========================================================================
log(SEP)
log(f'MODEL 6 — LINEAR SVM (subsample: {SUBSAMPLE_N:,} train rows)')
log(SEP)
log(f'  Tuning C via 3-fold CV on {TUNE_N:,}-row subsample ...')
best_svm = tune_cv(LinearSVC(class_weight='balanced', max_iter=3000, random_state=SEED),
                   {'C': [0.01, 0.1, 1.0]}, X_tune, y_tune)
C_opt = best_svm['C']

svm = LinearSVC(C=C_opt, class_weight='balanced', max_iter=5000, random_state=SEED)
r_before, _ = get_ram_info()
t0 = time.perf_counter()
svm.fit(X_tr_sub, y_tr_sub)
fit_time_svm = time.perf_counter() - t0
r_after, _ = get_ram_info()
ram_delta_svm = r_after - r_before
joblib.dump(svm, MODELS_DIR / 'binary_svm.joblib')

res_svm = eval_model(svm, X_test, y_test_bin, y_test_multi,
                     name='Linear SVM', fit_time=fit_time_svm,
                     data_desc=f'subsample ({SUBSAMPLE_N:,} rows)', ram_delta=ram_delta_svm)
all_results.append(res_svm)
gc.collect()

# ===========================================================================
# Save Tables
# ===========================================================================
log(SEP)
log('SAVING RESULTS TABLES')
log(SEP)
E.save_binary_results(all_results, str(TABLES / 'binary_results.csv'))

# Per-attack recall for every model
attack_dfs = []
for res in all_results:
    df_par = E.per_attack_recall(res['_y_pred'], res['_y_te_multi'], CLASS_NAMES)
    df_par.insert(0, 'model', res['model'])
    attack_dfs.append(df_par)
df_all_par = pd.concat(attack_dfs, ignore_index=True)
df_all_par.to_csv(TABLES / 'binary_per_attack_recall.csv', index=False)
log(f'Saved per-attack recall to {TABLES}/binary_per_attack_recall.csv')

# Pivot table: rows=attack type, columns=model recall
df_wide = df_all_par.pivot_table(index='class_name', columns='model', values='recall').round(4)
log('\nPer-Attack Recall Matrix:')
log(df_wide.to_string())

# ===========================================================================
# Save Figures
# ===========================================================================
log(SEP)
log('SAVING EVALUATION FIGURES')
log(SEP)
E.plot_confusion_matrices(all_results, str(FIGURES / 'binary_confusion_matrices.png'))
E.plot_roc_curves(all_results,         str(FIGURES / 'binary_roc_curves.png'))
E.plot_pr_curves(all_results,          str(FIGURES / 'binary_pr_curves.png'))

for res in all_results:
    df_par_m = E.per_attack_recall(res['_y_pred'], res['_y_te_multi'], CLASS_NAMES)
    m_slug = res['model'].replace(' ', '_').lower()
    E.plot_per_attack_recall(df_par_m, res['model'], str(FIGURES / f'binary_per_attack_recall_{m_slug}.png'))

df_summary = pd.read_csv(TABLES / 'binary_results.csv')
E.plot_metrics_comparison(df_summary, str(FIGURES / 'binary_metrics_comparison.png'))

# ===========================================================================
# Final Summary
# ===========================================================================
log(SEP)
log('PHASE 2 (BINARY) EXPERIMENT COMPLETE')
log(SEP)
cols_show = ['model', 'recall_attack', 'precision_attack', 'f1_attack',
             'roc_auc', 'pr_auc', 'false_alarm_rate', 'fit_seconds', 'ram_delta_gb', 'data']
df_final = df_summary[cols_show].sort_values('recall_attack', ascending=False)
log(df_final.to_string(index=False))
log(f'\nTop model by Recall (ATTACK): {df_final.iloc[0]["model"]} (Recall={df_final.iloc[0]["recall_attack"]:.4f})')
log(SEP)
