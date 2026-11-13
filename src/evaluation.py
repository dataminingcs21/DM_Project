"""
src/evaluation.py
=================
Shared evaluation utilities for binary and multiclass tasks.

Public API
----------
get_scores(model, X)
    → probability / decision scores for ROC / PR curves

binary_metrics(y_true, y_pred, y_score, name, fit_time, data_desc)
    → dict of all binary metrics

per_attack_recall(y_pred_bin, y_true_multi, class_names)
    → pd.DataFrame  per-attack-class recall on a binary classifier

save_binary_results(rows, out_path)
    → appends / writes results/tables/binary_results.csv

plot_confusion_matrices(results, save_path, normalize=True)
plot_roc_curves(results, save_path)
plot_pr_curves(results, save_path)
plot_per_attack_recall(df, save_path)
plot_metrics_comparison(df, save_path)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display on the laptop)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

sns.set_theme(style='whitegrid', palette='muted')

# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def get_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Return a 1-D array of decision scores for the POSITIVE class.
    Tries predict_proba first, falls back to decision_function.
    Used for ROC-AUC and PR-AUC.
    """
    if hasattr(model, 'predict_proba'):
        try:
            p = model.predict_proba(X)
            return p[:, 1] if p.ndim == 2 else p
        except Exception:
            pass
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return model.predict(X).astype(float)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    name: str,
    fit_time: float,
    data_desc: str = '',
) -> dict:
    """
    Compute all binary classification metrics.

    Priority metric: Recall on the ATTACK class (pos_label=1).
    A missed attack is worse than a false alarm, so recall > precision
    is the design goal.

    Parameters
    ----------
    y_true    : ground-truth binary labels (0=BENIGN, 1=ATTACK).
    y_pred    : binary predictions.
    y_score   : continuous scores (probabilities or decision values).
    name      : model name for the results table.
    fit_time  : training time in seconds.
    data_desc : e.g. 'full (1,998,548 rows)' or 'subsample (100,000 rows)'.

    Returns
    -------
    dict with all metrics; also stores y_true, y_pred, y_score for plotting.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float('nan')

    return {
        # identification
        'model':          name,
        'data':           data_desc,
        'fit_seconds':    round(fit_time, 1),
        # primary metric (ATTACK class)
        'recall_attack':  round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'precision_attack': round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'f1_attack':      round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        # overall
        'accuracy':       round(accuracy_score(y_true, y_pred), 4),
        'roc_auc':        round(roc_auc, 4),
        'pr_auc':         round(pr_auc, 4),
        # confusion matrix values
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        # false alarm rate = FP / (FP+TN) = fraction of benign flows flagged
        'false_alarm_rate': round(fp / (fp + tn) if (fp + tn) > 0 else float('nan'), 4),
        # for plotting
        '_y_true':  y_true,
        '_y_pred':  y_pred,
        '_y_score': y_score,
    }


# ---------------------------------------------------------------------------
# Per-attack-type recall
# ---------------------------------------------------------------------------

def per_attack_recall(
    y_pred_bin: np.ndarray,
    y_true_multi: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """
    For each ATTACK class c (c != 0 = BENIGN), compute the fraction of test
    samples belonging to class c that the binary classifier correctly flagged
    as ATTACK (y_pred_bin == 1).

    This reveals which attack types are most / least detectable.

    Parameters
    ----------
    y_pred_bin   : binary predictions (0 or 1).
    y_true_multi : original multiclass labels (0 = BENIGN, 1-14 = attack types).
    class_names  : list of class name strings, indexed 0-14.

    Returns
    -------
    pd.DataFrame with columns: class_id, class_name, n_test, recall.
    """
    rows = []
    for c in sorted(np.unique(y_true_multi)):
        if c == 0:   # skip BENIGN
            continue
        mask = (y_true_multi == c)
        n = int(mask.sum())
        if n == 0:
            continue
        rec = float((y_pred_bin[mask] == 1).sum()) / n
        rows.append({
            'class_id':   int(c),
            'class_name': class_names[c] if c < len(class_names) else str(c),
            'n_test':     n,
            'recall':     round(rec, 4),
        })
    return pd.DataFrame(rows).sort_values('recall')


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

def save_binary_results(rows: list[dict], out_path: str) -> None:
    """Write / overwrite binary_results.csv (strips internal _keys)."""
    public_keys = [k for k in rows[0].keys() if not k.startswith('_')]
    df = pd.DataFrame([{k: r[k] for k in public_keys} for r in rows])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(df[['model', 'recall_attack', 'precision_attack', 'f1_attack',
              'roc_auc', 'pr_auc', 'fit_seconds', 'data']].to_string(index=False))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_CMAP = plt.cm.Blues
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def plot_confusion_matrices(results: list[dict], save_path: str) -> None:
    """
    2 × 3 grid of normalized confusion matrices for all binary models.
    Rows = Actual, Columns = Predicted.  Normalization = by true class (recall-focused).
    """
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, r in enumerate(results):
        cm = confusion_matrix(r['_y_true'], r['_y_pred'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax = axes[i]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap=_CMAP,
                    xticklabels=['BENIGN', 'ATTACK'],
                    yticklabels=['BENIGN', 'ATTACK'],
                    vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title(f"{r['model']}\nRecall(ATTACK)={r['recall_attack']:.3f}", fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Binary Classification — Normalized Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_roc_curves(results: list[dict], save_path: str) -> None:
    """ROC curves for all binary models on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')

    for i, r in enumerate(results):
        try:
            fpr, tpr, _ = roc_curve(r['_y_true'], r['_y_score'], pos_label=1)
            ax.plot(fpr, tpr, lw=2, color=_COLORS[i % len(_COLORS)],
                    label=f"{r['model']}  (AUC={r['roc_auc']:.4f})")
        except Exception:
            pass

    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Recall on ATTACK)')
    ax.set_title('Binary Classification — ROC Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_pr_curves(results: list[dict], save_path: str) -> None:
    """Precision–Recall curves for all binary models."""
    # Baseline = fraction of positives
    baseline = results[0]['_y_true'].mean() if results else 0.17

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(baseline, color='k', linestyle='--', lw=1,
               label=f'Random (AP={baseline:.3f})')

    for i, r in enumerate(results):
        try:
            prec, rec, _ = precision_recall_curve(r['_y_true'], r['_y_score'], pos_label=1)
            ax.plot(rec, prec, lw=2, color=_COLORS[i % len(_COLORS)],
                    label=f"{r['model']}  (AP={r['pr_auc']:.4f})")
        except Exception:
            pass

    ax.set_xlabel('Recall (ATTACK class)')
    ax.set_ylabel('Precision (ATTACK class)')
    ax.set_title('Binary Classification — Precision–Recall Curves')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_per_attack_recall(df: pd.DataFrame, model_name: str, save_path: str) -> None:
    """
    Horizontal bar chart: per-attack-class recall for one binary model.
    Bars colored green (high recall) to red (low recall).
    """
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.55)))
    colors = ['#d62728' if r < 0.5 else '#ff7f0e' if r < 0.8 else '#2ca02c'
              for r in df['recall']]
    bars = ax.barh(df['class_name'], df['recall'], color=colors, edgecolor='white')
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={int(row['n_test'])}  {row['recall']:.3f}",
                va='center', fontsize=9)
    ax.set_xlabel('Recall (fraction of this attack class detected as ATTACK)')
    ax.set_title(f'Per-Attack-Type Recall — {model_name}', fontsize=12)
    ax.set_xlim([0, 1.18])
    ax.axvline(0.5, color='k', lw=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='k', lw=0.8, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_metrics_comparison(df_results: pd.DataFrame, save_path: str) -> None:
    """
    Grouped bar chart comparing Recall, F1, ROC-AUC, PR-AUC across all models.
    """
    metrics = ['recall_attack', 'f1_attack', 'roc_auc', 'pr_auc']
    labels  = ['Recall (ATTACK)', 'F1 (ATTACK)', 'ROC-AUC', 'PR-AUC']
    n_models  = len(df_results)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(df_results.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2,
                      vals, width, label=row['model'],
                      color=_COLORS[i % len(_COLORS)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel('Score')
    ax.set_title('Binary Classification — Model Comparison', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.axhline(0.9, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")
