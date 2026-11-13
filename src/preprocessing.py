"""
src/preprocessing.py
====================
Shared pipeline for the CIC-IDS-2017 dataset.  Call in this order:

    1. load_and_merge_datasets(dataset_dir)
         → raw DataFrame (8 CSVs merged, float32 downcast per file, B2/memory fix)
    2. clean_data(df)
         → cleaned DataFrame (inf→NaN, constant cols, NaN rows, dedup)  [B3/B6-fix]
    3. encode_labels(df)
         → (X, y_multi, y_bin)   [B1-fix: y_bin from raw string label]
    4. make_split(n_samples, y_multi, test_size, random_state, cache_dir)
         → (train_idx, test_idx) saved to cache  [B4-fix: stratify on multiclass]
    5. drop_near_zero_cols(X_train, X_test, threshold)
         → (X_train, X_test, dropped_cols)  [B6-fix: computed on TRAIN only]
    6. scale_features(X_train, X_test, cache_dir)
         → (X_train_sc, X_test_sc, scaler)  [B5-fix: saves scaler + feature names]

Also exposes:
    load_split(cache_dir) → (train_idx, test_idx)
    split_and_scale(...)  → backward-compat wrapper for src/training.py
"""

from __future__ import annotations

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# Memory helpers
# ===========================================================================

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 → float32 and int64 → int32 immediately after reading
    each CSV file, BEFORE concatenation.

    Why: CIC-IDS-2017 flow metrics have at most 6-7 significant digits;
    float32 (7 decimal digits) is sufficient precision.  This halves the
    per-row memory footprint (~680 bytes → ~340 bytes) and prevents
    out-of-memory errors on an 8 GB laptop when all 8 CSVs are merged.

    The Label column (dtype=object/str) is intentionally left untouched.
    """
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include='int64').columns:
        df[col] = df[col].astype('int32')
    return df


# ===========================================================================
# 1. Load and merge
# ===========================================================================

def load_and_merge_datasets(dataset_dir: str = './dataset/') -> pd.DataFrame:
    """
    Read all 8 CIC-IDS-2017 CSV files and concatenate into one DataFrame.

    Memory fix: each file is downcast to float32/int32 immediately after
    reading and before being appended to the list, so at most two files
    reside in full float64 memory simultaneously.

    B2-fix: uses encoding='latin-1' to avoid the UnicodeDecodeError caused
    by the Windows-1252 en-dash (0x96) in the Thursday WebAttacks CSV labels
    (e.g. 'Web Attack \x96 Brute Force').  latin-1 is a 1-to-1 byte mapping
    that never raises a decode error; the norm() function in encode_labels()
    strips the special character anyway.

    After merging, prints all unique raw label strings and their row counts
    so you can confirm:
      (a) all 15 expected classes loaded correctly, and
      (b) no UnicodeDecodeError occurred on the WebAttacks file.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the 8 ISCX CSV files.

    Returns
    -------
    pd.DataFrame
        Merged raw dataset, columns whitespace-stripped, dtypes float32/int32.
    """
    files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    ]

    dfs: list[pd.DataFrame] = []
    missing: list[str] = []

    for file in files:
        path = os.path.join(dataset_dir, file)
        if os.path.exists(path):
            print(f"  Reading {file} ...", end=' ', flush=True)
            df_tmp = pd.read_csv(path, encoding='latin-1', low_memory=False)
            df_tmp.columns = df_tmp.columns.str.strip()
            df_tmp = _downcast(df_tmp)          # memory fix: downcast in-place
            mb = df_tmp.memory_usage(deep=True).sum() / 1e6
            print(f"{len(df_tmp):>9,} rows  |  {mb:.0f} MB")
            dfs.append(df_tmp)
        else:
            missing.append(file)
            print(f"  WARNING: not found: {path}")

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir!r}. "
            "Run download_dataset.py first."
        )
    if missing:
        print(f"\n  WARNING: {len(missing)}/8 files missing – "
              "results will be incomplete.\n")

    df = pd.concat(dfs, ignore_index=True)
    del dfs                                     # release intermediate list
    df.columns = df.columns.str.strip()         # safety strip after concat

    gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"\nMerged shape: {df.shape[0]:,} rows × {df.shape[1]} columns  "
          f"({gb:.2f} GB in RAM)\n")

    # B2-fix: show raw label distribution so we can verify encoding worked
    print("Raw label distribution (verify all 15 classes present, no crashes):")
    counts = df['Label'].value_counts().sort_values(ascending=False)
    for lbl, n in counts.items():
        print(f"  {str(lbl)!r:45s}  {n:>9,}")
    print()

    return df


# ===========================================================================
# 2. Clean
# ===========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural cleaning steps that do NOT use train/test split information.

    Steps
    -----
    1. Replace +/-inf → NaN  (division-by-zero in flow rate features).
    2. Drop truly constant columns using nunique()<=1  (B6-fix: catches
       non-zero constants; excludes Label from the check).
    3. Drop rows with NaN.
    4. Drop exact duplicate rows (B3-fix: same flow cannot appear in both
       train and test after the split).

    B6 note: near-zero column filtering is intentionally NOT done here.
    It is applied after the split, computed on X_TRAIN only, by
    drop_near_zero_cols().  Using the full dataset for that decision would
    leak test-set statistics into feature selection.

    B3 note: also reports per-class counts before and after dedup, and
    counts feature-identical rows that carry different labels (using
    pd.util.hash_pandas_object for speed on 2.8M rows).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_merge_datasets().

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with reset integer index.
    """
    print(f"Before cleaning: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("─" * 64)

    # ── Step 1: inf → NaN ────────────────────────────────────────────────
    df_c = df.replace([np.inf, -np.inf], np.nan)
    n_now_nan = df_c.isna().any(axis=1).sum()
    n_was_nan = df.isna().any(axis=1).sum()
    print(f"  Rows newly containing NaN (from inf→NaN): {n_now_nan - n_was_nan:,}")

    # ── Step 2: drop truly constant columns (B6-fix) ─────────────────────
    feature_cols = [c for c in df_c.columns if c != 'Label']
    const_cols = [c for c in feature_cols if df_c[c].nunique() <= 1]
    if const_cols:
        df_c = df_c.drop(columns=const_cols)
        print(f"  Constant columns dropped ({len(const_cols)}): {const_cols}")
    else:
        print("  Constant columns dropped: 0")

    # ── Step 3: drop NaN rows ─────────────────────────────────────────────
    r_before_nan = len(df_c)
    df_c = df_c.dropna()
    r_nan = r_before_nan - len(df_c)
    print(f"  Rows dropped (NaN/inf): {r_nan:,}  "
          f"({r_nan / r_before_nan * 100:.2f}%)")

    # ── B3: per-class counts before deduplication ─────────────────────────
    before_counts = df_c['Label'].value_counts().sort_values(ascending=False)
    print(f"\n  Per-class counts BEFORE deduplication  (total {len(df_c):,}):")
    for lbl, n in before_counts.items():
        print(f"    {str(lbl)!r:45s}  {n:>9,}")

    # ── B3: detect feature-identical rows with conflicting labels ─────────
    feat_cols = [c for c in df_c.columns if c != 'Label']
    print("\n  Checking for feature-identical rows with conflicting labels ...")
    row_hash = pd.util.hash_pandas_object(df_c[feat_cols], index=False)
    hash_lbl = pd.DataFrame(
        {'hash': row_hash.values, 'Label': df_c['Label'].values}
    )
    unique_per_hash = hash_lbl.groupby('hash')['Label'].nunique()
    n_conflict_groups = int((unique_per_hash > 1).sum())
    n_conflict_rows   = int(
        hash_lbl[hash_lbl['hash'].isin(
            unique_per_hash[unique_per_hash > 1].index
        )].shape[0]
    )
    del hash_lbl, row_hash, unique_per_hash   # free memory
    print(f"    Feature groups with conflicting labels: {n_conflict_groups:,}")
    print(f"    Rows involved in conflicts:             {n_conflict_rows:,}")

    # ── Step 4: drop duplicate rows (B3-fix) ─────────────────────────────
    r_before_dup = len(df_c)
    df_c = df_c.drop_duplicates()
    r_dup = r_before_dup - len(df_c)
    print(f"\n  Exact duplicate rows dropped: {r_dup:,}  "
          f"({r_dup / r_before_dup * 100:.2f}%)")

    # ── B3: per-class counts after deduplication ──────────────────────────
    after_counts = df_c['Label'].value_counts().sort_values(ascending=False)
    print(f"\n  Per-class counts AFTER deduplication  (total {len(df_c):,}):")
    for lbl, n in after_counts.items():
        removed = int(before_counts.get(lbl, 0)) - n
        print(f"    {str(lbl)!r:45s}  {n:>9,}  (−{removed:,})")

    print("─" * 64)
    print(f"After cleaning:  {len(df_c):,} rows, {df_c.shape[1]} columns\n")

    return df_c.reset_index(drop=True)


# ===========================================================================
# 3. Encode labels
# ===========================================================================

def encode_labels(df: pd.DataFrame):
    """
    Encode the text 'Label' column and derive both target vectors.

    B1-fix: y_bin is built from the RAW string label
        (df['Label'].str.strip() != 'BENIGN')
    NOT from the numeric encoding index.  This means y_bin stays correct
    even if the order of label_names is ever changed.

    Parameters
    ----------
    df : pd.DataFrame with a 'Label' column containing text labels.
         Must be the output of clean_data() (no NaN, no inf, no duplicates).

    Returns
    -------
    X       : pd.DataFrame, float32  — all feature columns (Label removed).
    y_multi : pd.Series, int32       — class indices 0-14  (multiclass task).
    y_bin   : pd.Series, int32       — ATTACK=1, BENIGN=0  (binary task).
    """
    def norm(s: str) -> str:
        """Strip encoding artifacts: keep only letters, hyphens, spaces."""
        return ' '.join(re.sub(r'[^A-Za-z\- ]', ' ', str(s)).split())

    label_names = [
        'BENIGN',                   # 0
        'FTP-Patator',              # 1
        'SSH-Patator',              # 2
        'DoS slowloris',            # 3
        'DoS Slowhttptest',         # 4
        'DoS Hulk',                 # 5
        'DoS GoldenEye',            # 6
        'Heartbleed',               # 7
        'Web Attack Brute Force',   # 8
        'Web Attack XSS',           # 9
        'Web Attack Sql Injection', # 10
        'Infiltration',             # 11
        'Bot',                      # 12
        'DDoS',                     # 13
        'PortScan',                 # 14
    ]
    label_map = {norm(name): i for i, name in enumerate(label_names)}

    encoded = df['Label'].map(lambda s: label_map.get(norm(s)))

    unknown = encoded.isna()
    if unknown.any():
        raise ValueError(
            f"Unknown label(s) found: {df.loc[unknown, 'Label'].unique()}. "
            "Add them to label_names in encode_labels()."
        )

    y_multi = encoded.astype('int32')

    # B1-fix: build binary target from RAW string — independent of label_map order
    y_bin = (df['Label'].str.strip() != 'BENIGN').astype('int32')

    # Cast features to float32 (they should already be from _downcast, but be safe)
    X = df.drop(columns=['Label']).astype('float32')

    return X, y_multi, y_bin


# ===========================================================================
# 4. Train / test split  (B4-fix)
# ===========================================================================

def make_split(
    n_samples: int,
    y_multi: 'pd.Series | np.ndarray',
    test_size: float = 0.2,
    random_state: int = 42,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create ONE stratified train/test split on the MULTICLASS label and save
    the row indices to cache_dir.

    B4-fix rationale:
      • Stratify on y_multi (15 classes), NOT on y_bin (2 classes).
        A finer stratification ensures that even the rarest attack classes
        (Heartbleed: ~11 samples) appear in both splits.
      • Save integer row indices so that BOTH Task 1 and Task 2 can reload
        the identical split without re-running the cleaning pipeline.
      • The binary target y_bin is derived from these same indices:
            y_train_bin = y_bin.iloc[train_idx]
        This guarantees all three targets (X, y_multi, y_bin) share one split.

    Parameters
    ----------
    n_samples    : int   Total number of rows in the cleaned dataset.
    y_multi      : array-like  Multiclass integer labels (used for stratify=).
    test_size    : float Fraction for the test set (default 0.2 = 80/20).
    random_state : int   Reproducibility seed (default 42).
    cache_dir    : str   If given, saves train_idx.npy and test_idx.npy here.

    Returns
    -------
    train_idx, test_idx : np.ndarray of integer row positions.
    """
    idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multi,
    )

    print(f"Split: {len(train_idx):,} train  /  {len(test_idx):,} test  "
          f"(actual test %: {len(test_idx)/n_samples*100:.2f}%)")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, 'train_idx.npy'), train_idx)
        np.save(os.path.join(cache_dir, 'test_idx.npy'),  test_idx)
        print(f"  → Indices saved: {cache_dir}/train_idx.npy, test_idx.npy")

    return train_idx, test_idx


def load_split(cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reload previously saved train/test indices from cache_dir.
    Use this in notebook 02 and 03 to guarantee identical split.
    """
    train_idx = np.load(os.path.join(cache_dir, 'train_idx.npy'))
    test_idx  = np.load(os.path.join(cache_dir, 'test_idx.npy'))
    print(f"Loaded split from cache: {len(train_idx):,} train / "
          f"{len(test_idx):,} test")
    return train_idx, test_idx


# ===========================================================================
# 5. Drop near-zero columns on TRAIN set only  (B6-fix)
# ===========================================================================

def drop_near_zero_cols(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 99.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Drop feature columns where >= threshold% of TRAINING values are zero.

    B6-fix: computed exclusively on X_TRAIN.  Using the full dataset (or the
    test set) for column selection would constitute data leakage because the
    decision of which columns to keep would depend on test-set statistics.

    After computing dropped_cols on X_train, the SAME columns are removed
    from X_test so both sets remain aligned.

    Parameters
    ----------
    X_train   : pd.DataFrame  Training features (after split).
    X_test    : pd.DataFrame  Test features (after split).
    threshold : float         % of zeros in X_train above which a column is
                              dropped (default 99.0 = ≥99% zeros).

    Returns
    -------
    X_train_filtered, X_test_filtered : DataFrames with dropped_cols removed.
    dropped_cols : list[str]  Column names that were removed.
    """
    n_train = len(X_train)
    zero_pct_train = (X_train == 0).sum() / n_train * 100

    dropped_cols = zero_pct_train[zero_pct_train >= threshold].index.tolist()

    if dropped_cols:
        X_train = X_train.drop(columns=dropped_cols)
        X_test  = X_test.drop(columns=dropped_cols)
        print(f"Near-zero cols dropped (>={threshold}% zeros in train): "
              f"{len(dropped_cols)}")
        for c in dropped_cols:
            print(f"    {c}")
    else:
        print(f"Near-zero cols dropped (>={threshold}% zeros in train): 0")

    return X_train, X_test, dropped_cols


# ===========================================================================
# 6. Scale features  (B5-fix)
# ===========================================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on X_TRAIN only, then transform both splits.

    B5-fix: saves the fitted scaler and the ordered feature name list to
    cache_dir so they can be reloaded for inference on new network flows
    without re-running the entire pipeline.

    Why only fit on train?
    ----------------------
    The StandardScaler learns the mean μ and standard deviation σ of each
    feature.  If we computed these on the full dataset, the scaler would
    'know' something about the test set, inflating performance estimates.

    Parameters
    ----------
    X_train   : pd.DataFrame  Training features.
    X_test    : pd.DataFrame  Test features.
    cache_dir : str           If given, saves scaler.joblib and
                              feature_names.joblib here.

    Returns
    -------
    X_train_sc, X_test_sc : float32 DataFrames.
    scaler                : the fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include='number').columns

    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]).astype('float32'),
        columns=num_cols,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test[num_cols]).astype('float32'),
        columns=num_cols,
        index=X_test.index,
    )

    print(f"Scaled features: {X_train_sc.shape} train, {X_test_sc.shape} test")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(scaler,            os.path.join(cache_dir, 'scaler.joblib'))
        joblib.dump(list(num_cols),    os.path.join(cache_dir, 'feature_names.joblib'))
        print(f"  → scaler.joblib + feature_names.joblib saved to {cache_dir}/")

    return X_train_sc, X_test_sc, scaler


# ===========================================================================
# Legacy wrapper (backward compat for src/training.py)
# ===========================================================================

def split_and_scale(
    X: pd.DataFrame,
    y: 'pd.Series | np.ndarray',
    test_size: float = 0.2,
    random_state: int = 42,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    """
    Backward-compatible wrapper used by src/training.py.

    Calls make_split → drop_near_zero_cols → scale_features in sequence.
    Returns 5 values: (X_train, X_test, y_train, y_test, scaler).
    B5-fix: returns the fitted scaler as the 5th value.
    """
    y_arr = np.asarray(y)
    train_idx, test_idx = make_split(len(X), y_arr, test_size, random_state, cache_dir)

    X_train_raw = X.iloc[train_idx].copy()
    X_test_raw  = X.iloc[test_idx].copy()
    y_train     = y_arr[train_idx]
    y_test      = y_arr[test_idx]

    X_train_f, X_test_f, _ = drop_near_zero_cols(X_train_raw, X_test_raw)
    X_train_sc, X_test_sc, scaler = scale_features(X_train_f, X_test_f, cache_dir)

    return X_train_sc, X_test_sc, y_train, y_test, scaler
