"""Step 10: unsupervised anomaly detection - find attacks after learning normal traffic only.

Usage (from the project root, with the virtual environment active):

    python scripts/10_anomaly.py                    # Isolation Forest, LOF, autoencoder
    python scripts/10_anomaly.py --skip-lof         # LOF is the slowest detector
    python scripts/10_anomaly.py --benign-rows 50000 --test-rows 100000   # quick run

Data: every detector is fitted on BENIGN rows of the preprocessed TRAINING set only (no attack is
ever shown to it).  Alarm thresholds are set on other BENIGN training rows (calibration) for the
target false-alarm rates in ``settings.ANOMALY_FALSE_ALARM_TARGETS``.  The full held-out test set
(504,473 flows) is then scored once.  For reference, the supervised XGBoost (which did see labelled
attacks) is shown in the same terms.

Outputs (reports/mining/anomaly/): anomaly_report.md / .json, per_class_detection.csv, figures/*.png

Exit codes: 0 done; 1 data missing or stale.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from config import settings
from nids.experiments.common import cached_test_predictions
from nids.mining import anomaly as A
from nids.mining.reporting import class_names, markdown_table, plt, save_figure, write_report
from nids.training.data_access import DataNotReadyError, load_test_data, load_training_data
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.anomaly")
OUT = settings.MINING_REPORT_DIR / "anomaly"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benign-rows", type=int, default=settings.ANOMALY_BENIGN_TRAIN_SIZE, help="BENIGN training rows to fit on")
    parser.add_argument("--calibration-rows", type=int, default=100_000, help="other BENIGN training rows to set the thresholds")
    parser.add_argument("--lof-rows", type=int, default=50_000, help="LOF fit rows (its scoring cost grows with this)")
    parser.add_argument("--test-rows", type=int, default=None, help="score a random subset of the test set (default: all)")
    parser.add_argument("--skip-lof", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()

    try:
        train, test = load_training_data(mmap=True), load_test_data(mmap=True)
    except DataNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 1
    names = np.asarray(class_names(train.codec), dtype=object)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    benign_pos = rng.permutation(np.flatnonzero(train.y == train.codec.benign_index))
    fit_pos = np.sort(benign_pos[: args.benign_rows])
    cal_pos = np.sort(benign_pos[args.benign_rows : args.benign_rows + args.calibration_rows])
    X_fit, X_cal = np.asarray(train.X[fit_pos]), np.asarray(train.X[cal_pos])

    test_pos = np.arange(test.y.size) if args.test_rows is None else np.sort(rng.choice(test.y.size, size=args.test_rows, replace=False))
    X_test, y_test = np.asarray(test.X[test_pos]), test.y[test_pos]
    y_names = names[y_test]
    is_attack = y_test != test.codec.benign_index
    print(f"Fit: {len(fit_pos):,} BENIGN training rows; calibration: {len(cal_pos):,} other BENIGN training rows; test: {len(y_test):,} rows ({is_attack.mean():.1%} attacks)")

    detectors: list[tuple[A.AnomalyDetector, np.ndarray]] = [(A.IsolationForestDetector(), X_fit), (A.AutoencoderDetector(), X_fit)]
    if not args.skip_lof:
        detectors.insert(1, (A.LOFDetector(), X_fit[: args.lof_rows]))

    results, scores = [], {}
    for detector, X_train in detectors:
        result, s = A.evaluate_detector(detector, X_train, X_cal, X_test, y_names)
        results.append(result)
        scores[result.name] = s

    # supervised reference: XGBoost (baseline) on the same rows, attack = any non-BENIGN class
    xgb = cached_test_predictions(np.asarray(test.X), ["xgboost"])["xgboost"][test_pos]
    xgb_alarm = xgb != test.codec.benign_index
    supervised = {
        "detection_rate": float((xgb_alarm & is_attack).sum() / is_attack.sum()),
        "false_alarm_rate": float((xgb_alarm & ~is_attack).sum() / (~is_attack).sum()),
    }

    # tables
    overview = pd.DataFrame(
        [
            {
                "detector": r.name,
                "fit_rows": r.fit_rows,
                "roc_auc": r.roc_auc,
                "average_precision": r.average_precision,
                **{f"DR@FAR {op['target_false_alarm_rate']:g}": op["detection_rate"] for op in r.operating_points},
                **{f"real FAR@{op['target_false_alarm_rate']:g}": op["false_alarm_rate"] for op in r.operating_points},
                "fit_s": r.fit_seconds,
                "score_s": r.score_seconds,
            }
            for r in results
        ]
    )
    mid = settings.ANOMALY_FALSE_ALARM_TARGETS[len(settings.ANOMALY_FALSE_ALARM_TARGETS) // 2]
    col = f"flagged_at_far_{mid:g}"
    per_class = pd.concat([r.per_class.set_index("class")[[col]].rename(columns={col: r.name}) for r in results], axis=1)
    per_class.insert(0, "test_rows", results[0].per_class.set_index("class")["test_rows"])
    per_class["XGBoost (supervised)"] = pd.Series(xgb_alarm).groupby(y_names).mean()
    per_class = per_class.sort_values("test_rows", ascending=False).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(OUT / "per_class_detection.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(is_attack, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {next(r.roc_auc for r in results if r.name == name):.3f})")
    ax.scatter([supervised["false_alarm_rate"]], [supervised["detection_rate"]], marker="*", s=160, color="black", label="XGBoost (supervised)", zorder=5)
    ax.set(xscale="log", xlim=(1e-4, 1), xlabel="false-alarm rate (log)", ylabel="detection rate", title="One-class detectors on the test set")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, OUT / "figures" / "roc_curves.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_frame = per_class[per_class["class"] != settings.BENIGN_LABEL].set_index("class").drop(columns="test_rows")
    plot_frame.plot.barh(ax=ax, width=0.8)
    ax.set(xlabel=f"share of the class flagged (threshold for {mid:.0%} false alarms on BENIGN calibration rows)", xlim=(0, 1), title="Which attacks look abnormal?")
    ax.invert_yaxis()
    save_figure(fig, OUT / "figures" / "per_class_detection.png")

    fig, axes = plt.subplots(1, len(scores), figsize=(5 * len(scores), 3.8))
    for ax, (name, s) in zip(np.atleast_1d(axes), scores.items()):
        bins = np.histogram_bin_edges(s, bins=80)
        ax.hist(s[~is_attack], bins=bins, alpha=0.6, density=True, label="BENIGN")
        ax.hist(s[is_attack], bins=bins, alpha=0.6, density=True, label="attacks")
        ax.set(title=name, xlabel="anomaly score", yscale="log")
        ax.legend()
    save_figure(fig, OUT / "figures" / "score_distributions.png")

    best = max(results, key=lambda r: r.roc_auc)
    hard = per_class[(per_class["class"] != settings.BENIGN_LABEL)].nsmallest(4, best.name)
    md = [
        "# Unsupervised anomaly detection",
        "",
        f"Each detector learnt **only BENIGN traffic** ({len(fit_pos):,} BENIGN training rows; LOF {min(args.lof_rows, len(fit_pos)):,}). "
        f"Thresholds were set on {len(cal_pos):,} *other* BENIGN training rows for target false-alarm rates of "
        f"{', '.join(f'{a:.1%}' for a in settings.ANOMALY_FALSE_ALARM_TARGETS)}. The test set ({len(y_test):,} flows, "
        f"{is_attack.mean():.1%} attacks) was scored once. No attack label was used to fit or to choose a threshold.",
        "",
        "## Overview",
        "",
        "DR = detection rate (share of attack flows flagged); FAR = false-alarm rate (share of BENIGN flows flagged). "
        "`real FAR` is the false-alarm rate actually measured on the test set at the threshold set for that target.",
        "",
        markdown_table(overview),
        "",
        f"Supervised reference, XGBoost trained **with** labelled attacks: detection rate {supervised['detection_rate']:.4f} "
        f"at a false-alarm rate of {supervised['false_alarm_rate']:.4f}.",
        "",
        "![ROC](figures/roc_curves.png)",
        "",
        f"## Detection per attack class (threshold for {mid:.0%} false alarms)",
        "",
        markdown_table(per_class),
        "",
        f"Hardest classes for the best detector ({best.name}): {', '.join(hard['class'])}. "
        "These attacks look like normal flows in these features; a detector that only knows normal traffic cannot separate them.",
        "",
        "![per class](figures/per_class_detection.png)",
        "",
        "![scores](figures/score_distributions.png)",
        "",
        "## Reading the results",
        "",
        "* A one-class detector needs no attack labels and can in principle flag attacks never seen before; the price is "
        "many more false alarms (or missed attacks) than a supervised model that was shown every attack type.",
        "* Volume attacks (DoS, DDoS, PortScan) change flow statistics strongly; slow or application-level attacks "
        "(web attacks, some brute force, Bot) look like ordinary sessions.",
        "* The thresholds come from BENIGN *training* rows; the real false-alarm rate on the test set stays close to the "
        "target because both are from the same capture. On another network, it would have to be recalibrated.",
    ]
    summary = {"detectors": [r.summary() for r in results], "supervised_reference": supervised, "fit_rows": len(fit_pos), "calibration_rows": len(cal_pos), "test_rows": int(len(y_test))}
    md_path, _ = write_report(OUT, "anomaly_report", "\n".join(md), summary)
    print(markdown_table(overview, digits=4))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
