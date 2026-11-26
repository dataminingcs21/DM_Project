"""Step 16: class imbalance - do class weights or SMOTE help the small attack classes?

Usage (from the project root, with the virtual environment active):

    python scripts/16_imbalance.py                                # Decision Tree, Random Forest, XGBoost
    python scripts/16_imbalance.py --models xgboost --smote-target 10000

BENIGN is 83% of the training rows; Heartbleed has 9, Sql Injection 17, Infiltration 29.  Variants, all with
baseline parameters and scored on the same test set:

* **none** - the saved baseline model (unweighted);
* **balanced weights** - every class gets the same total weight (``class_weight='balanced'`` or the
  equivalent sample weights);
* **SMOTE** - synthetic rows for every class below ``--smote-target`` rows, interpolated between a row and one of
  its k nearest neighbours of the same class (training rows only, never the test set);
* **undersample + SMOTE** - BENIGN reduced to ``--benign-target`` random rows, then SMOTE.

Outputs (reports/experiments/imbalance/): imbalance_report.md / .json, per_class_f1.csv

Exit codes: 0 done; 1 data missing.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from config import model_params, settings
from nids.experiments.common import cached_test_predictions, fit_and_predict, per_class_f1, score
from nids.mining.reporting import class_names, markdown_table, write_report
from nids.training.data_access import DataNotReadyError, load_test_data, load_training_data
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.imbalance")
OUT = settings.EXPERIMENTS_REPORT_DIR / "imbalance"
SMALL = ("Bot", "Web Attack - Brute Force", "Web Attack - XSS", "Infiltration", "Web Attack - Sql Injection", "Heartbleed")


def smote(X: np.ndarray, y: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y)
    strategy = {c: target for c, n in enumerate(counts) if 0 < n < target}
    k = int(min(5, counts[list(strategy)].min() - 1)) if strategy else 5
    started = time.time()
    Xr, yr = SMOTE(sampling_strategy=strategy, k_neighbors=max(k, 1), random_state=settings.RANDOM_SEED).fit_resample(X, y)
    logger.info("SMOTE: %d -> %d rows (%d classes raised to %d, k=%d) in %.0f s", len(y), len(yr), len(strategy), target, k, time.time() - started)
    return np.asarray(Xr, dtype="float32"), np.asarray(yr, dtype="int64")


def undersample(y: np.ndarray, benign_index: int, target: int) -> np.ndarray:
    rng = np.random.default_rng(settings.RANDOM_SEED)
    benign = np.flatnonzero(y == benign_index)
    keep = np.concatenate([np.flatnonzero(y != benign_index), rng.choice(benign, size=min(target, benign.size), replace=False)])
    return np.sort(keep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", choices=("decision_tree", "random_forest", "xgboost"), default=["decision_tree", "random_forest", "xgboost"])
    parser.add_argument("--smote-target", type=int, default=20_000)
    parser.add_argument("--benign-target", type=int, default=500_000)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        train, test = load_training_data(), load_test_data()
    except DataNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 1
    n, benign, names = train.n_classes, train.codec.benign_index, class_names(train.codec)
    y_train = np.asarray(train.y, dtype="int64")
    baseline = cached_test_predictions(test.X, args.models, "trained")

    X_sm, y_sm = smote(train.X, y_train, args.smote_target)
    keep = undersample(y_train, benign, args.benign_target)
    X_us, y_us = smote(train.X[keep], y_train[keep], args.smote_target)
    del keep

    rows, per_class = [], {}
    for key in args.models:
        name = model_params.DISPLAY_NAMES[key]
        variants = {"none": (baseline[key].astype("int64"), None, len(y_train))}
        print(f"{name}: balanced weights ...")
        pred, sec = fit_and_predict(key, train.X, y_train, test.X, class_weight_mode="balanced")
        variants["balanced weights"] = (pred, sec, len(y_train))
        print(f"{name}: SMOTE ...")
        pred, sec = fit_and_predict(key, X_sm, y_sm, test.X)
        variants["SMOTE"] = (pred, sec, len(y_sm))
        print(f"{name}: undersample + SMOTE ...")
        pred, sec = fit_and_predict(key, X_us, y_us, test.X)
        variants["undersample + SMOTE"] = (pred, sec, len(y_us))
        for variant, (pred, sec, n_rows) in variants.items():
            s = score(test.y, pred, n, benign)
            rows.append({"model": name, "variant": variant, "train_rows": n_rows, **s, "fit_seconds": sec})
            per_class[(name, variant)] = per_class_f1(test.y, pred, names).set_index("class")["f1"]
    table = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class)
    pc.columns = [f"{m} | {v}" for m, v in pc.columns]
    pc.insert(0, "test_support", np.bincount(test.y, minlength=n))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "imbalance_comparison.csv", index=False)
    pc.to_csv(OUT / "per_class_f1.csv")

    small = pc.loc[[c for c in SMALL if c in pc.index]].T
    small_view = small.drop(index="test_support").reset_index().rename(columns={"index": "model | variant"})
    md = [
        "# Class imbalance: weights, SMOTE and undersampling",
        "",
        f"Training rows: {len(y_train):,} ({np.mean(y_train == benign):.1%} BENIGN). SMOTE raises every class below "
        f"{args.smote_target:,} rows to {args.smote_target:,} ({len(y_sm):,} rows after SMOTE); the undersampling variant first keeps "
        f"{args.benign_target:,} random BENIGN rows ({len(y_us):,} rows after SMOTE). Resampling touches training rows only; "
        f"every variant is scored on the same {test.y.size:,} test flows.",
        "",
        "## Overall",
        "",
        markdown_table(table),
        "",
        "## F1 of the small classes",
        "",
        "Test support: " + ", ".join(f"{c} {int(small.loc['test_support', c])}" for c in small.columns) + ".",
        "",
        markdown_table(small_view, digits=3),
        "",
        "## Reading the results",
        "",
        "* Weighting and oversampling move the decision boundary toward the small classes: their recall usually "
        "rises, their precision can fall, and BENIGN gets a few more false alarms.",
        "* For tree ensembles that already fit the training data almost perfectly, the effect is small; it is "
        "larger for single trees and for classes that overlap (Web Attack Brute Force vs XSS).",
        "* SMOTE interpolates in the scaled log-feature space; synthetic flows are plausible mixes of real ones but "
        "are not real traffic.",
        "* Heartbleed, Sql Injection and Infiltration have 2-7 test flows: one flow moves their F1 by 0.15-0.5.",
    ]
    md_path, _ = write_report(OUT, "imbalance_report", "\n".join(md), {"comparison": table.to_dict(orient="records")})
    print(markdown_table(table[["model", "variant", "macro_f1", "macro_f1_reliable", "false_alarm_rate"]]))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
