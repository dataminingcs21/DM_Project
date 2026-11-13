"""
run_binary.py  (temporary — deleted after run)
===============================================
Phase 2 — Binary classification: BENIGN (0) vs ATTACK (1)

Models trained
--------------
Full training set (1,998,548 rows):
  1. Logistic Regression  — baseline, C tuned by 3-fold CV on 50k subsample
  2. Decision Tree        — max_depth tuned by 3-fold CV on 50k subsample
  3. Random Forest        — bounded config (see below), no extra CV

Stratified subsample of train (100,000 rows):
  4. Gaussian Naive Bayes — no hyperparameter tuning
  5. KNN                  — k tuned by 3-fold CV on 20k sub-subsample
  6. Linear SVM           — C tuned by 3-fold CV on 50k subsample

CV scoring: recall on ATTACK class (pos_label=1) — priority metric.
Hyperparameter tuning is ALWAYS done on a subsample of X_train.
The test set is NEVER touched during tuning.

Random Forest configuration rationale:
  n_estimators=50   — capped to control wall time on 2M rows
  max_depth=20      — meaningful bound; with min_samples_leaf=5 on 2M rows
                      actual depth is bounded by ~log2(2M/5)≈18
  min_samples_leaf=5 — reduces tree size, helps generalisation
  max_features='sqrt' — sqrt(66)≈8, speeds up each split
  n_jobs=2          — parallel building; >2 risks RAM pressure on 8 GB
  class_weight='balanced' — compensates 83/17 BENIGN/ATTACK imbalance

Outputs
-------
  results/tables/binary_results.csv
  results/tables/binary_per_attack_recall.csv
  results/figures/binary_confusion_matrices.png
  results/figures/binary_roc_curves.png
  results/figures/binary_pr_curves.png
  results/figures/binary_per_attack_recall_<model>.png   (best model)
  results/figures/binary_metrics_comparison.png
  models/  <model>.joblib  (all fitted models)
"""

from __future__ import annotations
import os, sys, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src import config as C
from src import evaluation as E

try:
    import psutil
    def _ram_gb():
        return psutil.Process().memory_info().rss / 1e9
except ImportError:
    def _ram_gb():
        return float('nan')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE      = C.CACHE_DIR
TABLES     = C.TABLES_DIR
FIGURES    = C.RESULTS_DIR / 'figures'
MODELS_DIR = C.MODELS_DIR
for d in [TABLES, FIGURES, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED          = 42
SUBSAMPLE_N   = 100_000   # train subsample for NB, KNN, SVM
KNN_TEST_N    = 50_000    # test subsample for KNN (prediction is O(n_train×n_test))
TUNE_N        = 50_000    # train subsample used for CV hyperparameter search
KNN_TUNE_N    = 20_000    # smaller subsample for KNN CV (k-NN CV is slow)
CV_FOLDS      = 3
RECALL_SCORER = make_scorer(recall_score, pos_label=1, zero_division=0)

CLASS_NAMES = [
    'BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest',
    'DoS Hulk','DoS GoldenEye','Heartbleed','Web Attack Brute Force',
    'Web Attack XSS','Web Attack Sql Injection','Infiltration','Bot','DDoS','PortScan'
]

SEP = '=' * 72

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print(SEP)
print('LOADING DATA FROM CACHE')
print(SEP)
t0 = time.perf_counter()
raw = np.load(CACHE / 'splits.npz')
X_train      = raw['X_train']       # (1,998,548, 66) float32
X_test       = raw['X_test']        # (499,637, 66) float32
y_train_bin  = raw['y_train_bin'].astype(int)
y_test_bin   = raw['y_test_bin'].astype(int)
y_test_multi = raw['y_test_multi'].astype(int)
feature_names = joblib.load(CACHE / 'feature_names.joblib')
print(f'  X_train: {X_train.shape}  X_test: {X_test.shape}')
print(f'  Train attack rate: {y_train_bin.mean()*100:.2f}%  '
      f'Test attack rate: {y_test_bin.mean()*100:.2f}%')
print(f'  RAM after load: {_ram_gb():.2f} GB')
print(f'  [{time.perf_counter()-t0:.1f}s]\n')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stratified_subsample(X, y, n, seed=SEED):
    """Return a stratified subsample of n rows from (X, y)."""
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
    """Grid search on subsample. Returns best params dict.
    n_jobs=1: Windows multiprocessing via joblib/loky has high pickling
    overhead that makes parallel CV slower than serial on small subsamples.
    """
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    gs = GridSearchCV(estimator, param_grid, scoring=RECALL_SCORER,
                      cv=cv_obj, n_jobs=1, refit=False)
    gs.fit(X_tune, y_tune)
    print(f'    CV best params: {gs.best_params_}  '
          f'(recall={gs.best_score_:.4f} on {len(y_tune):,} rows, {cv}-fold)')
    return gs.best_params_


def fit_and_eval(model, X_tr, y_tr, X_te, y_te, y_te_multi,
                 name, data_desc, subsample_desc=''):
    """Train, predict, compute metrics. Returns result dict."""
    print(f'  Fitting {name} on {data_desc} ...')
    ram_before = _ram_gb()
    t_fit = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t_fit
    ram_after = _ram_gb()

    print(f'  Predicting ...')
    y_pred  = model.predict(X_te)
    y_score = E.get_scores(model, X_te)

    res = E.binary_metrics(y_te, y_pred, y_score, name, fit_time, data_desc)
    res['ram_delta_gb'] = round(ram_after - ram_before, 2)
    res['_y_te_multi']  = y_te_multi   # keep for per-attack recall

    print(f'    recall(ATTACK)={res["recall_attack"]:.4f}  '
          f'precision={res["precision_attack"]:.4f}  '
          f'f1={res["f1_attack"]:.4f}  '
          f'ROC-AUC={res["roc_auc"]:.4f}  '
          f'PR-AUC={res["pr_auc"]:.4f}')
    print(f'    fit={fit_time:.1f}s  RAM delta={res["ram_delta_gb"]:.2f} GB  '
          f'RAM now={ram_after:.2f} GB')
    return res


# ---------------------------------------------------------------------------
# Subsamples
# ---------------------------------------------------------------------------
print(SEP)
print(f'PREPARING SUBSAMPLES  (seed={SEED})')
print(SEP)

# 100k train subsample — for NB, KNN, SVM
X_tr_sub, y_tr_sub, _ = stratified_subsample(X_train, y_train_bin, SUBSAMPLE_N)
print(f'  Train subsample for slow models: {X_tr_sub.shape}  '
      f'attack rate={y_tr_sub.mean()*100:.2f}%')

# 50k subsample — for CV tuning of LR, DT, SVM
X_tune, y_tune, _ = stratified_subsample(X_train, y_train_bin, TUNE_N)
print(f'  Tune subsample (for CV):         {X_tune.shape}')

# 20k subsample — for KNN CV (CV is O(n^2) for KNN)
X_knn_tune, y_knn_tune, _ = stratified_subsample(X_train, y_train_bin, KNN_TUNE_N)
print(f'  KNN tune subsample:              {X_knn_tune.shape}')

# 50k test subsample — for KNN prediction (O(n_test × n_train))
X_te_knn, y_te_knn_bin, knn_te_idx = stratified_subsample(X_test, y_test_bin, KNN_TEST_N)
y_te_knn_multi = y_test_multi[knn_te_idx]
print(f'  KNN test subsample:              {X_te_knn.shape}  '
      f'attack rate={y_te_knn_bin.mean()*100:.2f}%\n')

all_results = []

# ===========================================================================
# 1. Logistic Regression — baseline
# ===========================================================================
print(SEP)
print('MODEL 1 — LOGISTIC REGRESSION (baseline, full training set)')
print(SEP)

print(f'  Tuning C via {CV_FOLDS}-fold CV on {TUNE_N:,}-row subsample ...')
lr_params = tune_cv(
    LogisticRegression(class_weight='balanced', solver='lbfgs',
                       max_iter=500, random_state=SEED),
    {'C': [0.01, 0.1, 1, 10]},
    X_tune, y_tune,
)
best_C_lr = lr_params['C']

lr = LogisticRegression(
    C=best_C_lr, class_weight='balanced',
    solver='lbfgs', max_iter=1000, random_state=SEED,
)
res_lr = fit_and_eval(
    lr, X_train, y_train_bin, X_test, y_test_bin, y_test_multi,
    name='Logistic Regression',
    data_desc=f'full ({len(X_train):,} rows)',
)
all_results.append(res_lr)
joblib.dump(lr, MODELS_DIR / 'binary_lr.joblib')
gc.collect()

# ===========================================================================
# 2. Decision Tree
# ===========================================================================
print(SEP)
print('MODEL 2 — DECISION TREE (full training set)')
print(SEP)

print(f'  Tuning max_depth via {CV_FOLDS}-fold CV on {TUNE_N:,}-row subsample ...')
dt_params = tune_cv(
    DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=5,
                           random_state=SEED),
    {'max_depth': [5, 10, 15, 20]},
    X_tune, y_tune,
)
best_depth_dt = dt_params['max_depth']

dt = DecisionTreeClassifier(
    max_depth=best_depth_dt, class_weight='balanced',
    min_samples_leaf=5, random_state=SEED,
)
res_dt = fit_and_eval(
    dt, X_train, y_train_bin, X_test, y_test_bin, y_test_multi,
    name='Decision Tree',
    data_desc=f'full ({len(X_train):,} rows)',
)
all_results.append(res_dt)
joblib.dump(dt, MODELS_DIR / 'binary_dt.joblib')
gc.collect()

# ===========================================================================
# 3. Random Forest — bounded configuration
# ===========================================================================
print(SEP)
print('MODEL 3 — RANDOM FOREST (bounded config, full training set)')
print(f'  Config: n_estimators=50, max_depth=20, min_samples_leaf=5,')
print(f'          max_features=sqrt, n_jobs=2, class_weight=balanced')
print(f'  Rationale: bounded to control RAM and training time on 2M rows.')
print(f'             With min_samples_leaf=5, effective depth ≈ log2(2M/5) ≈ 18.')
print(SEP)

rf = RandomForestClassifier(
    n_estimators=50, max_depth=20, min_samples_leaf=5,
    max_features='sqrt', n_jobs=2, class_weight='balanced',
    random_state=SEED,
)
res_rf = fit_and_eval(
    rf, X_train, y_train_bin, X_test, y_test_bin, y_test_multi,
    name='Random Forest',
    data_desc=f'full ({len(X_train):,} rows)',
)
# Feature importance — top 20
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print('\n  Top 20 feature importances (Random Forest):')
print(importances.head(20).to_string())
importances.to_csv(TABLES / 'binary_rf_feature_importance.csv', header=['importance'])
all_results.append(res_rf)
joblib.dump(rf, MODELS_DIR / 'binary_rf.joblib')
gc.collect()

# Feature importance figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
importances.head(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Random Forest — Top 20 Feature Importances (Binary Task)', fontsize=12)
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig(FIGURES / 'binary_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ===========================================================================
# 4. Gaussian Naive Bayes — subsample
# ===========================================================================
print(SEP)
print(f'MODEL 4 — NAIVE BAYES (subsample: {SUBSAMPLE_N:,} train rows)')
print(SEP)

nb = GaussianNB()
res_nb = fit_and_eval(
    nb, X_tr_sub, y_tr_sub, X_test, y_test_bin, y_test_multi,
    name='Naive Bayes',
    data_desc=f'subsample ({SUBSAMPLE_N:,} rows)',
)
all_results.append(res_nb)
joblib.dump(nb, MODELS_DIR / 'binary_nb.joblib')
gc.collect()

# ===========================================================================
# 5. K-Nearest Neighbours — subsample (train AND test)
# ===========================================================================
print(SEP)
print(f'MODEL 5 — KNN (train: {SUBSAMPLE_N:,} rows, test: {KNN_TEST_N:,} rows)')
print(f'  KNN prediction is O(n_train × n_test × n_features), so both')
print(f'  train and test sets are capped to stay within time/memory limits.')
print(SEP)

print(f'  Tuning k via {CV_FOLDS}-fold CV on {KNN_TUNE_N:,}-row subsample ...')
knn_params = tune_cv(
    KNeighborsClassifier(n_jobs=1, algorithm='ball_tree'),
    {'n_neighbors': [3, 5, 7, 11]},
    X_knn_tune, y_knn_tune,
)
best_k = knn_params['n_neighbors']

knn = KNeighborsClassifier(
    n_neighbors=best_k, n_jobs=1, algorithm='ball_tree',
)
res_knn = fit_and_eval(
    knn, X_tr_sub, y_tr_sub, X_te_knn, y_te_knn_bin, y_te_knn_multi,
    name='KNN',
    data_desc=f'subsample ({SUBSAMPLE_N:,} train / {KNN_TEST_N:,} test)',
)
all_results.append(res_knn)
joblib.dump(knn, MODELS_DIR / 'binary_knn.joblib')
gc.collect()

# ===========================================================================
# 6. Linear SVM — subsample
# ===========================================================================
print(SEP)
print(f'MODEL 6 — LINEAR SVM (subsample: {SUBSAMPLE_N:,} train rows)')
print(SEP)

print(f'  Tuning C via {CV_FOLDS}-fold CV on {TUNE_N:,}-row subsample ...')
svm_params = tune_cv(
    LinearSVC(class_weight='balanced', max_iter=5000, random_state=SEED),
    {'C': [0.01, 0.1, 1]},
    X_tune, y_tune,
)
best_C_svm = svm_params['C']

svm = LinearSVC(
    C=best_C_svm, class_weight='balanced',
    max_iter=5000, random_state=SEED,
)
res_svm = fit_and_eval(
    svm, X_tr_sub, y_tr_sub, X_test, y_test_bin, y_test_multi,
    name='Linear SVM',
    data_desc=f'subsample ({SUBSAMPLE_N:,} rows)',
)
all_results.append(res_svm)
joblib.dump(svm, MODELS_DIR / 'binary_svm.joblib')
gc.collect()

# ===========================================================================
# Save tables
# ===========================================================================
print(SEP)
print('SAVING TABLES')
print(SEP)
E.save_binary_results(all_results, str(TABLES / 'binary_results.csv'))

# Per-attack recall for every model
attack_dfs = []
for res in all_results:
    df_par = E.per_attack_recall(
        res['_y_pred'], res['_y_te_multi'], CLASS_NAMES
    )
    df_par.insert(0, 'model', res['model'])
    attack_dfs.append(df_par)
df_all_par = pd.concat(attack_dfs, ignore_index=True)
df_all_par.to_csv(TABLES / 'binary_per_attack_recall.csv', index=False)
print(f'Saved → {TABLES}/binary_per_attack_recall.csv')

# Per-attack recall table (wide format: models as columns)
df_wide = df_all_par.pivot_table(
    index='class_name', columns='model', values='recall'
).round(4)
print('\nPer-attack recall matrix (rows=attack class, cols=model):')
print(df_wide.to_string())

# ===========================================================================
# Save figures
# ===========================================================================
print(SEP)
print('SAVING FIGURES')
print(SEP)
E.plot_confusion_matrices(all_results, str(FIGURES / 'binary_confusion_matrices.png'))
E.plot_roc_curves(all_results,         str(FIGURES / 'binary_roc_curves.png'))
E.plot_pr_curves(all_results,          str(FIGURES / 'binary_pr_curves.png'))

# Per-attack recall for the best model (by recall_attack)
best_res = max(all_results, key=lambda r: r['recall_attack'])
df_best_par = E.per_attack_recall(
    best_res['_y_pred'], best_res['_y_te_multi'], CLASS_NAMES
)
E.plot_per_attack_recall(
    df_best_par, best_res['model'],
    str(FIGURES / f"binary_per_attack_recall_{best_res['model'].replace(' ','_')}.png")
)
# Also for all models side by side
for res in all_results:
    df_par_m = E.per_attack_recall(res['_y_pred'], res['_y_te_multi'], CLASS_NAMES)
    E.plot_per_attack_recall(
        df_par_m, res['model'],
        str(FIGURES / f"binary_per_attack_recall_{res['model'].replace(' ','_')}.png")
    )

df_summary = pd.read_csv(TABLES / 'binary_results.csv')
E.plot_metrics_comparison(df_summary, str(FIGURES / 'binary_metrics_comparison.png'))

# ===========================================================================
# Final summary
# ===========================================================================
print(SEP)
print('FINAL SUMMARY — BINARY CLASSIFICATION')
print(SEP)
df_disp = df_summary[['model','recall_attack','precision_attack','f1_attack',
                       'roc_auc','pr_auc','fit_seconds','data']].copy()
df_disp = df_disp.sort_values('recall_attack', ascending=False)
print(df_disp.to_string(index=False))
print(f'\nBest recall on ATTACK class: {df_disp.iloc[0]["model"]}  '
      f'({df_disp.iloc[0]["recall_attack"]:.4f})')
print(SEP)
print('Phase 2 complete. Results in results/tables/ and results/figures/')
print(SEP)
