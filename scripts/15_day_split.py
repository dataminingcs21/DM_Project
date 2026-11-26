"""Step 15: split by capture day - does a detector catch attack types it has never seen?

Usage (from the project root, with the virtual environment active):

    python scripts/15_day_split.py                     # XGBoost, all folds + random-split reference
    python scripts/15_day_split.py --model random_forest
    python scripts/15_day_split.py --benign-cap 100000  # quicker

See ``nids.experiments.day_split`` for the method.  The task is binary (BENIGN vs attack), because a held-out
day's attack classes do not exist in its training data.  Reads the CLEANED data and refits the preprocessing
pipeline inside every fold.

Outputs (reports/experiments/day_split/): day_split_report.md / .json, folds.csv, per_class.csv

Exit codes: 0 done; 1 data missing.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import settings
from nids.experiments import day_split as D
from nids.mining.reporting import markdown_table, write_report
from nids.utils.io_utils import read_parquet
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.day_split")
OUT = settings.EXPERIMENTS_REPORT_DIR / "day_split"


def make_factory(name: str):
    if name == "xgboost":
        from xgboost import XGBClassifier

        from nids.models.xgboost_model import resolve_device

        return lambda: XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", device=resolve_device(), random_state=settings.RANDOM_SEED, n_jobs=settings.N_JOBS)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return lambda: RandomForestClassifier(n_estimators=100, n_jobs=settings.N_JOBS, random_state=settings.RANDOM_SEED)
    raise ValueError(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("xgboost", "random_forest"), default="xgboost")
    parser.add_argument("--benign-cap", type=int, default=400_000, help="BENIGN training rows per fold")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        df = read_parquet(settings.CLEANED_DATA_FILE, hint="run scripts/03_clean_data.py")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    factory = make_factory(args.model)

    folds = D.leave_one_file_out(df, factory, benign_cap=args.benign_cap)
    ref_metrics, ref_per_class = D.random_split_reference(df, factory, benign_cap=args.benign_cap)

    fold_table = pd.DataFrame([{"held_out": f.held_out, "unseen_attacks": ", ".join(c for c in f.unseen_classes if c != settings.BENIGN_LABEL), **f.metrics} for f in folds])
    per_class_rows = []
    for f in folds:
        for r in f.per_class.itertuples(index=False):
            if r[0] != settings.BENIGN_LABEL:
                per_class_rows.append({"class": r[0], "held_out_day": f.held_out, "rows": r.rows, "detected_unseen": r.flagged_share})
    per_class = pd.DataFrame(per_class_rows)
    ref = ref_per_class.rename(columns={"flagged_share": "detected_random_split", "rows": "random_split_test_rows"})
    per_class = per_class.merge(ref, on="class", how="left")
    per_class["gap"] = per_class["detected_unseen"] - per_class["detected_random_split"]
    OUT.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(OUT / "folds.csv", index=False)
    per_class.to_csv(OUT / "per_class.csv", index=False)

    attack_rows = fold_table["attack_rows"].sum()
    pooled_dr = float((fold_table["detection_rate"] * fold_table["attack_rows"]).sum() / attack_rows)
    md = [
        "# Split by capture day: detecting attacks never seen in training",
        "",
        f"Binary detector (BENIGN vs attack), model **{args.model}**. Each fold holds out one capture file; the detector "
        f"(and the preprocessing pipeline) is trained on the other files, with at most {args.benign_cap:,} BENIGN rows and "
        "every attack row. Every attack class sits in exactly one file, so the held-out attacks are **unseen types**.",
        "",
        "## Folds",
        "",
        markdown_table(fold_table[["held_out", "unseen_attacks", "attack_rows", "detection_rate", "false_alarm_rate", "f1_attack", "roc_auc", "train_rows", "seconds"]]),
        "",
        "## Same detector, random split (the optimistic setting of the main pipeline)",
        "",
        f"Detection rate {ref_metrics['detection_rate']:.4f}, false-alarm rate {ref_metrics['false_alarm_rate']:.4f}, "
        f"F1 {ref_metrics['f1_attack']:.4f}, ROC-AUC {ref_metrics['roc_auc']:.4f}.",
        "",
        f"Leave-one-day-out, all attack flows pooled: detection rate **{pooled_dr:.4f}**.",
        "",
        "## Per attack class: unseen vs seen",
        "",
        "`detected_unseen` = share flagged when the class was never in training; `detected_random_split` = share flagged "
        "with the random split, where the class was in training.",
        "",
        markdown_table(per_class.sort_values("gap"), digits=3),
        "",
        "## Reading the results",
        "",
        "* The random split shows how well known attacks are recognised; this experiment shows how well a detector "
        "generalises to **new** attack types. The gap is what the random split hides.",
        "* An unseen attack is still caught when it resembles a seen one (e.g. a DoS variant after training on DDoS); "
        "attacks with their own pattern (web attacks, infiltration, Bot) are mostly missed.",
        "* The false-alarm rate also changes, because BENIGN traffic of the held-out day was never seen either.",
        "* Compare with the one-class detectors of step 10, which need no attack labels at all.",
    ]
    summary = {"model": args.model, "benign_cap": args.benign_cap, "folds": fold_table.to_dict(orient="records"), "random_split": ref_metrics, "pooled_detection_unseen": pooled_dr}
    md_path, _ = write_report(OUT, "day_split_report", "\n".join(md), summary)
    print(markdown_table(fold_table[["held_out", "detection_rate", "false_alarm_rate", "roc_auc"]]))
    print(f"Random split reference: detection {ref_metrics['detection_rate']:.4f}, false alarms {ref_metrics['false_alarm_rate']:.4f}")
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
