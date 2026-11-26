"""Step 14: ablation - how much do the models depend on ``Destination Port``?

Usage (from the project root, with the virtual environment active):

    python scripts/14_port_ablation.py                         # Decision Tree, Random Forest, XGBoost
    python scripts/14_port_ablation.py --models xgboost

In this lab capture every attack targets one port (FTP-Patator -> 21, SSH-Patator -> 22, Bot -> 8080, most
DoS and web attacks -> 80), and step 12 shows that Destination Port is the most important feature of both
Random Forest and XGBoost.  A model can thus learn "port 21 = attack" - true in this capture, useless on a
network where the same attack hits another port.  Here the same models (baseline parameters, same
preprocessed training rows) are retrained without that one column and both versions are scored on the
same test set.  Models that use a subsample in the main pipeline are not offered (their scores are noisier).

Outputs (reports/experiments/port_ablation/): port_ablation_report.md / .json, per_class_f1.csv

Exit codes: 0 done; 1 data missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.experiments.common import cached_test_predictions, fit_and_predict, per_class_f1, score
from nids.mining.reporting import class_names, markdown_table, write_report
from nids.training.data_access import DataNotReadyError, load_test_data, load_training_data
from nids.utils.io_utils import read_json
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.port_ablation")
OUT = settings.EXPERIMENTS_REPORT_DIR / "port_ablation"
FEATURE = "Destination Port"
ALLOWED = ("decision_tree", "random_forest", "xgboost", "logistic_regression", "gaussian_nb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", choices=ALLOWED, default=["decision_tree", "random_forest", "xgboost"])
    args = parser.parse_args(argv)
    configure_logging()
    try:
        train, test = load_training_data(), load_test_data()
    except DataNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 1
    features = read_json(settings.FEATURE_NAMES_FILE)["model_feature_names"]
    keep = np.array([f != FEATURE for f in features])
    if keep.all():
        print(f"ERROR: '{FEATURE}' is not a model feature")
        return 1
    n, benign, names = train.n_classes, train.codec.benign_index, class_names(train.codec)
    with_port = cached_test_predictions(test.X, args.models, "trained")
    X_train, X_test = np.ascontiguousarray(train.X[:, keep]), np.ascontiguousarray(test.X[:, keep])

    rows, deltas = [], {}
    for key in args.models:
        print(f"Retraining {model_params.DISPLAY_NAMES[key]} without {FEATURE} on {X_train.shape[0]:,} rows ...")
        without, seconds = fit_and_predict(key, X_train, train.y, X_test)
        a, b = score(test.y, with_port[key], n, benign), score(test.y, without, n, benign)
        rows.append(
            {
                "model": model_params.DISPLAY_NAMES[key],
                "macro_f1_with": a["macro_f1"],
                "macro_f1_without": b["macro_f1"],
                "delta": b["macro_f1"] - a["macro_f1"],
                "reliable_with": a["macro_f1_reliable"],
                "reliable_without": b["macro_f1_reliable"],
                "detection_with": a["detection_rate"],
                "detection_without": b["detection_rate"],
                "false_alarms_with": a["false_alarm_rate"],
                "false_alarms_without": b["false_alarm_rate"],
                "fit_seconds": seconds,
            }
        )
        f_with = per_class_f1(test.y, with_port[key], names).set_index("class")
        f_without = per_class_f1(test.y, without, names).set_index("class")
        deltas[model_params.DISPLAY_NAMES[key]] = f_without["f1"] - f_with["f1"]
        if "support" not in deltas:
            deltas["support"] = f_with["support"]
    table = pd.DataFrame(rows)
    per_class = pd.DataFrame(deltas)
    per_class = per_class[["support", *[c for c in per_class.columns if c != "support"]]].sort_values("support", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(OUT / "per_class_f1_delta.csv")

    model_cols = [c for c in per_class.columns if c != "support"]
    worst = per_class[model_cols].mean(axis=1).nsmallest(4)
    md = [
        f"# Ablation: the models without `{FEATURE}`",
        "",
        f"Same baseline parameters, same {X_train.shape[0]:,} preprocessed training rows, {int(keep.sum())} features instead of "
        f"{keep.size}. Scored on the same {test.y.size:,} test flows. *with* = the saved baseline model.",
        "",
        markdown_table(table),
        "",
        "## Change of F1 per class (without minus with)",
        "",
        markdown_table(per_class.reset_index(), digits=4),
        "",
        "Classes that lose the most on average: " + ", ".join(f"{c} ({v:+.3f})" for c, v in worst.items()) + ".",
        "",
        "## Reading the results",
        "",
        "* A small drop means the other flow statistics carry nearly the same information: the models did not depend "
        "on the port, and the main results do not rest on a lab artefact.",
        "* A large drop for a class means its recognition leans on the port its attack used in this capture; on "
        "another network that part of the score would not carry over.",
        "* Without the port, the scores are the more honest estimate of what the flow statistics alone can do.",
    ]
    md_path, _ = write_report(OUT, "port_ablation_report", "\n".join(md), {"models": table.to_dict(orient="records"), "per_class_delta": per_class.reset_index().to_dict(orient="records")})
    print(markdown_table(table[["model", "macro_f1_with", "macro_f1_without", "delta"]]))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
