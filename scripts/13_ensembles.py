"""Step 13: better ensembles - weighted votes and stacking, learnt from out-of-fold predictions.

Usage (from the project root, with the virtual environment active):

    python scripts/13_ensembles.py
    python scripts/13_ensembles.py --stack-rows 300000     # smaller logistic stacker

Inputs: the 11 saved **baseline** models (artifacts/models/trained/) and their 5-fold out-of-fold
predictions on all 2,017,889 training rows (reports/evaluation/cross_validation/cv_<model>_oof.npy,
written by 06_evaluate.py).  Weights and stackers are learnt from the OOF predictions only; the
test set is used once, at the end, to score every combiner.  Test predictions of the base models
are cached in reports/experiments/test_predictions/.

Outputs (reports/experiments/ensembles/): ensembles_report.md / .json, ensembles_comparison.csv,
ensembles_per_class_f1.csv

Exit codes: 0 done; 1 data, models or OOF predictions missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.evaluation.cross_validation import load_oof
from nids.experiments import ensembles as E
from nids.experiments.common import cached_test_predictions, per_class_f1, score
from nids.mining.reporting import class_names, markdown_table, write_report
from nids.training.data_access import DataNotReadyError, load_test_data, load_training_data
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.ensembles")
OUT = settings.EXPERIMENTS_REPORT_DIR / "ensembles"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stack-rows", type=int, default=600_000, help="training rows for the logistic stacker (XGBoost uses all)")
    args = parser.parse_args(argv)
    configure_logging()

    keys = list(model_params.MODEL_KEYS)
    try:
        train, test = load_training_data(mmap=True), load_test_data(mmap=True)
        oof = {k: np.asarray(load_oof(k), dtype="int64") for k in keys}
    except (DataNotReadyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if any(len(p) != train.y.size for p in oof.values()):
        print("ERROR: the out-of-fold predictions do not cover the training set; rerun 06_evaluate.py without --cv-sample-size")
        return 1
    n, benign = train.n_classes, train.codec.benign_index
    names = class_names(train.codec)
    y_train, y_test = np.asarray(train.y), np.asarray(test.y)
    test_pred = {k: v.astype("int64") for k, v in cached_test_predictions(np.asarray(test.X), keys, "trained").items()}

    # weights from training-data OOF predictions only
    w_model = E.oof_model_weights(oof, y_train, n)
    w_class = E.oof_class_weights(oof, y_train, n)
    weights_table = pd.DataFrame({"model": [model_params.DISPLAY_NAMES[k] for k in keys], "oof_macro_f1": [w_model[k] for k in keys]}).sort_values("oof_macro_f1", ascending=False)

    combiners: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}  # name -> (test prediction, OOF prediction or None)
    combiners["Equal vote, 11 models (baseline set)"] = (E.weighted_vote(test_pred, n, None, benign), E.weighted_vote(oof, n, None, benign))
    for power in (1, 4):
        label = f"Weighted vote, OOF macro F1^{power}"
        ws = {k: w_model[k] for k in keys}
        combiners[label] = (E.weighted_vote(test_pred, n, ws, benign, power), E.weighted_vote(oof, n, ws, benign, power))
    combiners["Weighted vote, per-class OOF F1"] = (E.weighted_vote(test_pred, n, w_class, benign), E.weighted_vote(oof, n, w_class, benign))
    for top in (3, 5):
        subset = E.best_subset(w_model, top)
        combiners[f"Equal vote, top {top} by OOF F1 ({', '.join(model_params.DISPLAY_NAMES[k] for k in subset)})"] = (
            E.weighted_vote({k: test_pred[k] for k in subset}, n, None, benign),
            E.weighted_vote({k: oof[k] for k in subset}, n, None, benign),
        )

    # stacking: meta-model on one-hot OOF votes
    Z_train, Z_test = E.one_hot_votes(oof, keys, n), E.one_hot_votes(test_pred, keys, n)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    sub = np.sort(rng.choice(y_train.size, size=min(args.stack_rows, y_train.size), replace=False))
    lr = E.fit_stacker("logistic", Z_train[sub], y_train[sub], n)
    combiners["Stacking, logistic regression"] = (lr.predict(Z_test).astype("int64"), None)
    xgb = E.fit_stacker("xgboost", Z_train, y_train, n)
    combiners["Stacking, XGBoost"] = (np.asarray(xgb.predict(Z_test)).astype("int64"), None)

    # reference single models
    for k in ("xgboost", "random_forest"):
        combiners[f"Single model: {model_params.DISPLAY_NAMES[k]}"] = (test_pred[k], oof[k])

    rows, per_class = [], {}
    for name, (pred, oof_pred) in combiners.items():
        s = score(y_test, pred, n, benign)
        rows.append({"combiner": name, "oof_macro_f1": None if oof_pred is None else score(y_train, oof_pred, n, benign)["macro_f1"], **{f"test_{k}": v for k, v in s.items()}})
        per_class[name] = per_class_f1(y_test, pred, names).set_index("class")["f1"]
    table = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    pc = pd.DataFrame(per_class)
    pc.insert(0, "test_support", np.bincount(y_test, minlength=n))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ensembles_comparison.csv", index=False)
    pc.to_csv(OUT / "ensembles_per_class_f1.csv")

    equal = table.loc[table["combiner"].str.startswith("Equal vote, 11"), "test_macro_f1"].iloc[0]
    best = table.iloc[0]
    focus = pc.loc[[c for c in ("Bot", "Web Attack - Brute Force", "Web Attack - XSS", "Infiltration", "Web Attack - Sql Injection") if c in pc.index]]
    md = [
        "# Better ensembles: weighted votes and stacking",
        "",
        "Every weight and meta-model below was learnt from the **out-of-fold predictions on the training set** "
        "(5-fold cross-validation of the baseline models, File 49). The test set (504,473 flows) was used once, to "
        "score the finished combiners. The base models are the 11 saved **baseline** models.",
        "",
        "## OOF macro F1 of each model (the weights)",
        "",
        markdown_table(weights_table),
        "",
        "## Comparison on the test set",
        "",
        "`oof_macro_f1` is the same combiner scored on the training OOF predictions (empty for the stackers, which "
        "were fitted on them). `macro_f1_reliable` leaves out the classes with fewer than 30 test flows (Heartbleed, "
        "Sql Injection, Infiltration).",
        "",
        markdown_table(table),
        "",
        f"Best: **{best['combiner']}**, test macro F1 {best['test_macro_f1']:.4f} vs {equal:.4f} for the equal vote of the "
        f"same 11 models ({best['test_macro_f1'] - equal:+.4f}).",
        "",
        "## Per-class F1 of the difficult classes",
        "",
        markdown_table(focus.reset_index(), digits=3),
        "",
        "Full table: `ensembles_per_class_f1.csv`.",
        "",
        "## Reading the results",
        "",
        "* The equal vote is dragged down by the weak models; weighting by OOF F1 (and more so F1^4, or keeping only "
        "the top models) lets the tree ensembles decide.",
        "* Per-class weights trust each model only for the classes it handles well.",
        "* Stacking learns combinations (for example *XGBoost says XSS but Random Forest says Brute Force*), which a "
        "vote cannot express. The OOF predictions come from models trained on 80% of the training rows, slightly "
        "weaker than the final models the stacker is applied to; this is the usual, small stacking bias.",
        "* Differences of a few thousandths are within the noise of the tiny classes (2-7 test flows).",
    ]
    md_path, _ = write_report(OUT, "ensembles_report", "\n".join(md), {"weights": w_model, "comparison": table.to_dict(orient="records")})
    print(markdown_table(table[["combiner", "oof_macro_f1", "test_macro_f1", "test_macro_f1_reliable"]]))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
