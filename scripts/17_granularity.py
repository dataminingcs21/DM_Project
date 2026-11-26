"""Step 17: label granularity - binary detector and attack families vs the 15-class model.

Usage (from the project root, with the virtual environment active):

    python scripts/17_granularity.py
    python scripts/17_granularity.py --models xgboost

Three ways to frame the task, same preprocessed rows, baseline parameters:

* **15 classes** - the saved baseline model (main pipeline);
* **8 families** - BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Bot, Infiltration
  (``settings.ATTACK_FAMILIES``); the three tiny classes are absorbed into larger families;
* **binary** - BENIGN vs attack.

Each coarse model is compared with the 15-class model's predictions *collapsed* to the same labels, which
answers: does training on coarse labels help, or is it enough to collapse the fine predictions?

Outputs (reports/experiments/granularity/): granularity_report.md / .json, family_per_class_f1.csv

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
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.granularity")
OUT = settings.EXPERIMENTS_REPORT_DIR / "granularity"


def family_mapping(names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Array mapping 15-class codes -> family codes (BENIGN first), and the family names."""
    missing = sorted(set(names) - set(settings.ATTACK_FAMILIES))
    if missing:
        raise KeyError(f"classes without a family in settings.ATTACK_FAMILIES: {missing}")
    families = [settings.BENIGN_LABEL] + sorted({settings.ATTACK_FAMILIES[c] for c in names} - {settings.BENIGN_LABEL})
    lookup = {f: i for i, f in enumerate(families)}
    return np.array([lookup[settings.ATTACK_FAMILIES[c]] for c in names], dtype="int64"), families


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", choices=("decision_tree", "random_forest", "xgboost"), default=["random_forest", "xgboost"])
    args = parser.parse_args(argv)
    configure_logging()
    try:
        train, test = load_training_data(), load_test_data()
    except DataNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 1
    names = class_names(train.codec)
    to_family, families = family_mapping(names)
    to_binary = (np.arange(len(names)) != train.codec.benign_index).astype("int64")
    y_tr, y_te = np.asarray(train.y, dtype="int64"), np.asarray(test.y, dtype="int64")
    fine = cached_test_predictions(test.X, args.models, "trained")

    rows, family_f1 = [], {}
    for key in args.models:
        name = model_params.DISPLAY_NAMES[key]
        fine_pred = fine[key].astype("int64")
        print(f"{name}: training on 8 families ...")
        fam_pred, fam_s = fit_and_predict(key, train.X, to_family[y_tr], test.X)
        print(f"{name}: training binary ...")
        bin_pred, bin_s = fit_and_predict(key, train.X, to_binary[y_tr], test.X)
        for task, truth, k, benign, variants in (
            ("15 classes", y_te, len(names), train.codec.benign_index, {"trained on 15 classes": fine_pred}),
            ("8 families", to_family[y_te], len(families), 0, {"trained on families": fam_pred, "15-class model, collapsed": to_family[fine_pred]}),
            ("binary", to_binary[y_te], 2, 0, {"trained binary": bin_pred, "15-class model, collapsed": to_binary[fine_pred]}),
        ):
            for variant, pred in variants.items():
                rows.append({"model": name, "task": task, "variant": variant, **score(truth, pred, k, benign)})
        family_f1[(name, "trained on families")] = per_class_f1(to_family[y_te], fam_pred, families).set_index("class")["f1"]
        family_f1[(name, "15-class, collapsed")] = per_class_f1(to_family[y_te], to_family[fine_pred], families).set_index("class")["f1"]
    table = pd.DataFrame(rows)
    fam = pd.DataFrame(family_f1)
    fam.columns = [f"{m} | {v}" for m, v in fam.columns]
    fam.insert(0, "test_support", np.bincount(to_family[y_te], minlength=len(families)))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "granularity_comparison.csv", index=False)
    fam.to_csv(OUT / "family_per_class_f1.csv")

    mapping = pd.DataFrame({"class": names, "family": [families[i] for i in to_family]})
    md = [
        "# Label granularity: 15 classes, 8 families, binary",
        "",
        "## Families",
        "",
        markdown_table(mapping),
        "",
        "## Results on the test set",
        "",
        "For the binary task, `macro_f1` averages the F1 of BENIGN and of attack; `detection_rate` and "
        "`false_alarm_rate` are the security view (share of attacks caught, share of normal flows flagged).",
        "",
        markdown_table(table),
        "",
        "## F1 per family",
        "",
        markdown_table(fam.reset_index(), digits=4),
        "",
        "## Reading the results",
        "",
        "* Families remove the hardest distinctions (Web Attack Brute Force vs XSS, the DoS variants) and absorb the "
        "three classes with 2-7 test flows, so their macro F1 is both higher and far more reliable.",
        "* If the collapsed 15-class model is as good as the model trained on families, nothing is lost by keeping "
        "the fine labels: the families can always be derived afterwards.",
        "* For an alerting system the binary view matters most: how many attacks are missed and how many false alarms "
        "an analyst must look at.",
    ]
    md_path, _ = write_report(OUT, "granularity_report", "\n".join(md), {"comparison": table.to_dict(orient="records"), "families": dict(zip(names, (families[i] for i in to_family)))})
    print(markdown_table(table[["model", "task", "variant", "macro_f1", "detection_rate", "false_alarm_rate"]]))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
