"""
src/multiclass_utils.py
=======================
Utilities for Phase 3 Multiclass Classification on CIC-IDS-2017.

Handles:
- Remapping the original 15 classes (0..14) to 13 classes (0..12) by merging
  the 3 Web Attack variants (Brute Force, XSS, Sql Injection) into a single "Web Attack" class.
- Reusing existing train_idx / test_idx splits without re-splitting.
- Multiclass metric computation (Accuracy, Macro-F1, Weighted-F1, Macro-Recall, Per-Class Recall).
- Confusion matrix extraction (13x13) and plotting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Class Definitions (13 classes after Web Attacks merge)
# ---------------------------------------------------------------------------
CLASS_NAMES_13 = [
    'BENIGN',                   # 0
    'FTP-Patator',              # 1
    'SSH-Patator',              # 2
    'DoS slowloris',            # 3
    'DoS Slowhttptest',         # 4
    'DoS Hulk',                 # 5
    'DoS GoldenEye',            # 6
    'Heartbleed',               # 7 (rare: 9 train / 2 test)
    'Web Attack',               # 8 (merged from BF:8, XSS:9, SQLi:10)
    'Infiltration',             # 9 (rare: 29 train / 7 test, was 11)
    'Bot',                      # 10 (was 12)
    'DDoS',                     # 11 (was 13)
    'PortScan',                 # 12 (was 14)
]

ORIGINAL_NAMES_15 = [
    'BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
    'DoS Hulk', 'DoS GoldenEye', 'Heartbleed', 'Web Attack Brute Force',
    'Web Attack XSS', 'Web Attack Sql Injection', 'Infiltration', 'Bot', 'DDoS', 'PortScan'
]


def remap_to_13_classes(y_multi: np.ndarray) -> np.ndarray:
    """
    Map original 15-class target vector (0..14) to 13 classes (0..12).

    Rules:
      - 0..7 stay 0..7 (BENIGN through Heartbleed)
      - 8, 9, 10 map to 8 (Web Attack Brute Force, XSS, SQLi -> Web Attack)
      - 11 -> 9  (Infiltration)
      - 12 -> 10 (Bot)
      - 13 -> 11 (DDoS)
      - 14 -> 12 (PortScan)
    """
    y_arr = np.asarray(y_multi)
    y_new = y_arr.copy()
    y_new[(y_arr == 8) | (y_arr == 9) | (y_arr == 10)] = 8
    y_new[y_arr == 11] = 9
    y_new[y_arr == 12] = 10
    y_new[y_arr == 13] = 11
    y_new[y_arr == 14] = 12
    return y_new.astype(int)


def load_multiclass_data(cache_dir: str | Path) -> dict[str, Any]:
    """
    Loads preprocessed splits from cache, remapping labels to 13 classes.
    Strictly reuses the same train/test split without re-splitting.
    """
    cache_path = Path(cache_dir)
    raw = np.load(cache_path / 'splits.npz')
    feature_names = joblib.load(cache_path / 'feature_names.joblib')
    scaler = joblib.load(cache_path / 'scaler.joblib')

    y_train_13 = remap_to_13_classes(raw['y_train_multi'])
    y_test_13  = remap_to_13_classes(raw['y_test_multi'])

    return {
        'X_train': raw['X_train'],
        'X_test': raw['X_test'],
        'y_train': y_train_13,
        'y_test': y_test_13,
        'feature_names': feature_names,
        'scaler': scaler,
        'class_names': CLASS_NAMES_13,
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    fit_time: float,
    n_train: int,
    n_test: int,
    class_names: list[str] = CLASS_NAMES_13,
) -> dict[str, Any]:
    """
    Computes all standard multiclass evaluation metrics and per-class recall.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class recall dict
    per_class_rec = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    for i, c_name in enumerate(class_names):
        total_c = cm[i, :].sum()
        rec_c = cm[i, i] / total_c if total_c > 0 else 0.0
        per_class_rec[c_name] = round(float(rec_c), 4)

    return {
        'model': name,
        'n_train': int(n_train),
        'n_test_evaluated': int(n_test),
        'fit_time': round(float(fit_time), 1),
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'macro_recall': round(float(macro_rec), 4),
        'per_class_recall': json.dumps(per_class_rec),
        '_per_class_dict': per_class_rec,
        '_cm': cm,
        '_y_true': y_true,
        '_y_pred': y_pred,
    }


def plot_multiclass_confusion_matrices(
    results: list[dict],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """
    Plots a 2x3 grid of normalized confusion matrices for all 6 multiclass models.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).reshape(-1)

    # Shorten class names for readability on 13x13 axes
    short_names = [
        'BENIGN', 'FTP-Pat', 'SSH-Pat', 'slowloris', 'Slowhttp', 'Hulk',
        'GoldenEye', 'Heartbleed', 'WebAttack', 'Infiltr', 'Bot', 'DDoS', 'PortScan'
    ]

    for i, r in enumerate(results):
        cm = r['_cm']
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

        ax = axes[i]
        sns.heatmap(
            cm_norm, annot=False, cmap='Blues', vmin=0, vmax=1,
            xticklabels=short_names, yticklabels=short_names, ax=ax, cbar=False
        )
        ax.set_title(f"{r['model']}\nMacro-F1={r['macro_f1']:.4f} | Acc={r['accuracy']:.4f}", fontsize=11)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        ax.tick_params(axis='both', labelsize=8, rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Multiclass Classification (13 classes) — Normalized Confusion Matrices', fontsize=14, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")


def plot_multiclass_per_class_recall_heatmap(
    results: list[dict],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """
    Plots a heatmap showing per-class recall across all models (rows=models, cols=classes).
    """
    rows = []
    models = []
    for r in results:
        models.append(r['model'])
        rows.append([r['_per_class_dict'].get(c, 0.0) for c in class_names])

    df_heatmap = pd.DataFrame(rows, index=models, columns=class_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df_heatmap, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1,
        linewidths=0.5, cbar_kws={'label': 'Recall'}, ax=ax
    )
    ax.set_title('Multiclass Classification (13 classes) — Per-Class Recall Heatmap', fontsize=13)
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")


def plot_multiclass_macro_f1(results: list[dict], save_path: str | Path) -> None:
    """
    Bar chart comparing Macro-F1 and Accuracy across multiclass models.
    """
    models = [r['model'] for r in results]
    macro_f1s = [r['macro_f1'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width/2, macro_f1s, width, label='Macro-F1', color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='#ff7f0e', alpha=0.85)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel('Score')
    ax.set_title('Multiclass Classification — Macro-F1 vs Accuracy', fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")
