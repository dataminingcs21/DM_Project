"""Step 20: hermetic benchmark - preprocessing + model in one Pipeline, stratified k-fold CV, then a hold-out.

Usage (from the project root, with the virtual environment active):

    python scripts/20_pipeline_benchmark.py                        # 3,000 rows per class, 5 folds, all candidates
    python scripts/20_pipeline_benchmark.py --cap-per-class 800 --folds 3   # quick run

Why: in steps 04-07 the preprocessing is fitted once on the whole training split, so the cross-validation of
step 06 re-uses medians / kept columns / scaler that have seen every fold.  Here every candidate is a single
``Pipeline(ColumnTransformer(preprocessing) -> model)`` and ``StratifiedKFold`` refits the WHOLE pipeline in each
fold: no statistic of a validation fold ever reaches the model (see src/nids/evaluation/pipeline_benchmark.py).

Data: a class-capped sample of the cleaned data, split 80/20 (stratified).  The 80% is cross-validated; the
best model (by mean CV macro F1) is refitted on it and scored once on the untouched 20%.

Candidates: Dummy (floor), Logistic Regression and Decision Tree (baselines), Random Forest, Extra Trees,
XGBoost and LightGBM (ensembles; the last two only if installed).

Outputs (reports/benchmark/):
    benchmark_report.md / .json      readable report and machine-readable numbers
    cv_folds.csv, cv_summary.csv     every fold, and mean / std per model
    paired_vs_baseline.csv           paired fold differences against the best baseline
    per_class_f1_oof.csv             out-of-fold F1 per class and model
    confusion_<best>_oof.csv / holdout_confusion.csv
    figures/*.png
    best_pipeline.joblib             the refitted best pipeline (raw flow columns in, class index out)
    best_pipeline.json               its classes, features and hold-out scores
    holdout_flows.csv                the untouched hold-out rows WITH their Label (to try batch inference in the app)

Exit codes: 0 done; 1 cleaned data missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config import settings
from nids.eda import eda_stats
from nids.evaluation import pipeline_benchmark as B
from nids.mining.reporting import markdown_table, plt, save_figure, write_report
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.pipeline_benchmark")
OUT = settings.REPORTS_DIR / "benchmark"
COLORS = {"floor": "#9aa0a6", "baseline": "#e0a030", "ensemble": "#3b6ea8"}


def plot_cv(summary: pd.DataFrame, folds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    s = summary.iloc[::-1]
    ax = axes[0]
    ax.barh(s["display"], s["macro_f1_mean"], xerr=s["macro_f1_std"], color=[COLORS[f] for f in s["family"]], capsize=3)
    ax.set(xlabel="macro F1 (mean of the folds, bar = std)", xlim=(0, 1), title="Stratified CV, whole pipeline refitted per fold")
    ax = axes[1]
    order = list(summary["model"])
    data = [folds.loc[folds["model"] == k, "macro_f1"].to_numpy() for k in order if k != "dummy"]
    labels = [B.DISPLAY[k] for k in order if k != "dummy"]
    ax.boxplot(data, vert=False, tick_labels=labels)
    ax.invert_yaxis()
    ax.set(xlabel="macro F1 per fold", title="Fold-to-fold spread (Dummy omitted)")
    save_figure(fig, OUT / "figures" / "cv_macro_f1.png")


def plot_per_class(per_class: pd.DataFrame) -> None:
    table = per_class.drop(columns=["support"])
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 4, 7))
    im = ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), [f"{c} ({n})" for c, n in zip(table.index, per_class["support"])])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Out-of-fold F1 per class (support in brackets)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save_figure(fig, OUT / "figures" / "per_class_f1_oof.png")


def plot_confusion(cm: pd.DataFrame, title: str, name: str) -> None:
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm)), cm.index, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm.iat[i, j]:
                ax.text(j, i, str(cm.iat[i, j]), ha="center", va="center", fontsize=6.5, color="white" if norm.iat[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    save_figure(fig, OUT / "figures" / name)


def top_confusions(cm: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    rows = []
    for t in cm.index:
        total = cm.loc[t].sum()
        for p in cm.columns:
            if t != p and cm.at[t, p]:
                rows.append({"true": t, "predicted": p, "flows": int(cm.at[t, p]), "share_of_true_class": cm.at[t, p] / total})
    return pd.DataFrame(rows).sort_values("flows", ascending=False).head(n).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cap-per-class", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=settings.CV_FOLDS)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--models", nargs="*", default=None, help=f"subset of {tuple(B.DISPLAY)}")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    args = parser.parse_args(argv)
    configure_logging(level="WARNING", force=True)

    if not settings.CLEANED_DATA_FILE.is_file():
        logger.error("Cleaned data not found: %s (run scripts/03_clean_data.py first)", settings.CLEANED_DATA_FILE)
        return 1
    df = pd.read_parquet(settings.CLEANED_DATA_FILE)
    sample = eda_stats.class_capped_sample(df, args.cap_per_class, seed=args.seed).reset_index(drop=True)
    del df
    dev, hold = train_test_split(sample, test_size=args.holdout, stratify=sample[settings.LABEL_COLUMN], random_state=args.seed)
    dev, hold = dev.reset_index(drop=True), hold.reset_index(drop=True)
    models = args.models or B.available_models()
    print(f"Sample {len(sample):,} rows -> CV on {len(dev):,}, hold-out {len(hold):,}; models: {', '.join(models)}")

    start = time.perf_counter()
    result = B.cross_validate_models(dev, models=models, n_splits=args.folds, seed=args.seed, log=print)
    cv_seconds = time.perf_counter() - start
    OUT.mkdir(parents=True, exist_ok=True)
    result.folds.to_csv(OUT / "cv_folds.csv", index=False)
    result.summary.to_csv(OUT / "cv_summary.csv", index=False)
    per_class = result.per_class_f1()
    per_class.to_csv(OUT / "per_class_f1_oof.csv", index_label="class")

    baselines = result.summary[result.summary["family"] == "baseline"]
    best_baseline = baselines.iloc[0]["model"] if len(baselines) else result.summary.iloc[-1]["model"]
    paired = B.paired_comparison(result.folds, best_baseline)
    paired.to_csv(OUT / "paired_vs_baseline.csv", index=False)

    best = result.summary.iloc[0]["model"]
    cm_oof = result.confusion(best)
    cm_oof.to_csv(OUT / f"confusion_{best}_oof.csv")
    plot_cv(result.summary, result.folds)
    plot_per_class(per_class)
    plot_confusion(cm_oof, f"{B.DISPLAY[best]}: out-of-fold confusion matrix (counts; colour = row share)", f"confusion_{best}_oof.png")

    # refit the best pipeline on the whole development part, score it ONCE on the hold-out
    features = B.feature_columns(dev)
    pipe = B.build_model_pipeline(best, features, seed=args.seed)
    y_dev = np.array([result.classes.index(v) for v in dev[settings.LABEL_COLUMN].astype(str)])
    pipe.fit(dev, y_dev)
    hold_scores = B.holdout_scores(pipe, hold, result.classes)
    y_hold = np.array([result.classes.index(v) for v in hold[settings.LABEL_COLUMN].astype(str)])
    hold_pred = np.asarray(pipe.predict(hold)).astype(int).ravel()
    cm_hold = pd.DataFrame(confusion_matrix(y_hold, hold_pred, labels=range(len(result.classes))), index=result.classes, columns=result.classes)
    cm_hold.to_csv(OUT / "holdout_confusion.csv")
    plot_confusion(cm_hold, f"{B.DISPLAY[best]}: hold-out confusion matrix", "holdout_confusion.png")
    joblib.dump(pipe, OUT / "best_pipeline.joblib", compress=3)
    meta = {"model": best, "display": B.DISPLAY[best], "classes": result.classes, "features": features,
            "trained_rows": int(len(dev)), "holdout_scores": hold_scores, "cap_per_class": args.cap_per_class, "seed": args.seed}
    (OUT / "best_pipeline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hold.drop(columns=[settings.SOURCE_FILE_COLUMN], errors="ignore").to_csv(OUT / "holdout_flows.csv", index=False)

    conf_oof = top_confusions(cm_oof)
    conf_hold = top_confusions(cm_hold)
    s = result.summary
    show = s[["display", "family", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "accuracy_mean", "mcc_mean", "detection_rate_mean", "false_alarm_rate_mean", "fit_seconds_mean"]]
    weak = per_class[per_class.columns[1:]].max(axis=1).sort_values().head(4)
    lines = [
        "# Hermetic model benchmark (step 20)",
        "",
        f"Sample: {len(sample):,} flows of the cleaned data (at most {args.cap_per_class:,} per class, small classes whole), "
        f"split {1 - args.holdout:.0%} development / {args.holdout:.0%} hold-out (stratified, seed {args.seed}).",
        f"Cross-validation: StratifiedKFold, {args.folds} folds, on the {len(dev):,} development rows. Run time {cv_seconds / 60:.1f} min.",
        "",
        "**No leakage by construction.** Each candidate is one `Pipeline`: a `ColumnTransformer` (selects the "
        f"{len(features)} feature columns, drops `Label` / `source_file`, then infinite -> missing, median imputation, signed log1p, "
        "constant / correlated column filter, standard scaling) followed by the model. The whole pipeline is cloned and refitted "
        "inside every fold, so imputation medians, the kept columns and the scaler are learned on the training folds only.",
        "",
        "The sample is class-capped, so macro F1 here is higher than on the real class mix and is not comparable with the "
        "full-data scores of step 06; it is the comparison between models that matters.",
        "",
        "## Cross-validated comparison (mean over folds)",
        "",
        markdown_table(show),
        "",
        f"Best model: **{B.DISPLAY[best]}**. Best simple baseline: **{B.DISPLAY[best_baseline]}**.",
        "",
        "## Paired comparison against the best baseline",
        "",
        "Same folds for every model, so the fold-by-fold difference removes the fold difficulty.",
        "",
        markdown_table(paired),
        "",
        "## Error analysis (out-of-fold predictions of the best model)",
        "",
        "Most frequent confusions:",
        "",
        markdown_table(conf_oof),
        "",
        "Hardest classes (best F1 reached by any model):",
        "",
        markdown_table(weak.rename("best F1").to_frame().join(per_class["support"]), index=True),
        "",
        "Reading: the Web Attack Brute Force / XSS pair stays confused by every model, since their flows are nearly identical at "
        "the flow-feature level (same server, same port, HTTP requests of similar size). Infiltration, Heartbleed and "
        "Sql Injection have only tens of flows, so their F1 moves a lot from fold to fold. "
        "The linear baseline fails on the classes that need interactions between features (thresholds on ports and window sizes), "
        "which is where tree ensembles gain the most.",
        "",
        "## Hold-out (scored once)",
        "",
        f"{B.DISPLAY[best]} refitted on the {len(dev):,} development rows and scored on the {len(hold):,} untouched hold-out rows:",
        "",
        markdown_table(pd.DataFrame([hold_scores])),
        "",
        markdown_table(conf_hold),
        "",
        "Saved: `best_pipeline.joblib` (raw flow columns in, class index out; class names in `best_pipeline.json`) and "
        "`holdout_flows.csv` (the hold-out rows with their label, to try batch inference in the app).",
    ]
    report = {
        "sample_rows": int(len(sample)), "dev_rows": int(len(dev)), "holdout_rows": int(len(hold)),
        "folds": args.folds, "cap_per_class": args.cap_per_class, "seed": args.seed, "cv_seconds": cv_seconds,
        "models": models, "best_model": best, "best_baseline": best_baseline,
        "summary": s.to_dict(orient="records"), "holdout": hold_scores,
    }
    write_report(OUT, "benchmark_report", "\n".join(lines) + "\n", report)
    print(show.to_string(index=False))
    print(f"\nHold-out ({B.DISPLAY[best]}): " + ", ".join(f"{k} {v:.4f}" for k, v in hold_scores.items()))
    print(f"Report: {OUT / 'benchmark_report.md'}  pipeline: {(OUT / 'best_pipeline.joblib').stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
