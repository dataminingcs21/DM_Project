"""Step 8: make comparison files: real held-out flows, their true labels, and the app's predictions.

Usage (from the project root, with the virtual environment active):

    python scripts/08_make_comparison_files.py                  # 300 random held-out flows
    python scripts/08_make_comparison_files.py --n 1000
    python scripts/08_make_comparison_files.py --per-class 20   # 20 flows of every class (fewer if a class has fewer)
    python scripts/08_make_comparison_files.py --seed 7         # another random draw

Where the flows come from: the TEST part of the 80/20 split of CIC-IDS2017 (rows no model was trained on).
The split is recomputed from the cleaned data with the same settings and seed as ``04_preprocess.py`` and
checked against the saved ``y_test.npy``; if it does not match, the script stops. Nothing is invented: every
flow and every true label is a real row of the dataset.

Outputs in ``reports/comparison/`` (or ``--out-dir``):
    comparison_input.csv       the flows, the 40 model features only, NO label  (upload this file in the app)
    comparison_answers.csv     the TRUE label of every row (row = line number in the input file)
    comparison_predictions.csv the app's prediction (majority vote) with the vote counts
    comparison.csv             answers and predictions side by side, with a "correct" column
    comparison_summary.md      accuracy, per-class results, the mistakes
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import settings
from nids.prediction import input_validator as validator
from nids.prediction import predictor as predictor_module
from nids.preprocessing.preprocessor import select_feature_columns
from nids.preprocessing.splitter import stratified_split
from nids.preprocessing.target_builder import build_target, make_codec
from nids.utils.io_utils import load_array, load_joblib, read_parquet
from nids.utils.logger import get_logger

logger = get_logger("scripts.comparison")

DEFAULT_OUT_DIR = settings.REPORTS_DIR / "comparison"
RARE_TEST_ROWS = 10  # classes with fewer test rows than this get a warning in the summary


# ---------------------------------------------------------------------------
# recover the held-out rows
# ---------------------------------------------------------------------------
def load_heldout(codec_saved=None) -> tuple[pd.DataFrame, pd.Series]:
    """The raw test rows (feature columns) and their true class names, in X_test order.

    Raises ValueError when the recomputed split does not equal the saved one.
    """
    df = read_parquet(settings.CLEANED_DATA_FILE, hint="Run scripts/03_clean_data.py first.")
    feature_cols = select_feature_columns(df)
    target, _ = build_target(df[settings.LABEL_COLUMN])
    codec = make_codec(target)
    saved = codec_saved or load_joblib(settings.LABEL_ENCODER_FILE, hint="Run scripts/04_preprocess.py first.")
    if tuple(saved.classes) != tuple(codec.classes):
        raise ValueError("The classes of the cleaned data differ from the saved label encoder. Rerun scripts/04_preprocess.py.")
    _, test_idx = stratified_split(target)
    y_test = load_array(settings.Y_TEST_FILE, hint="Run scripts/04_preprocess.py first.")
    if len(test_idx) != len(y_test) or not np.array_equal(codec.encode(target)[test_idx], y_test):
        raise ValueError(
            "The recomputed test split does not equal the saved y_test.npy (the cleaned data or the split settings changed "
            "after 04_preprocess.py). Rerun scripts/04_preprocess.py."
        )
    raw_test = df[feature_cols].iloc[test_idx].reset_index(drop=True)
    labels = target.iloc[test_idx].astype(str).reset_index(drop=True)
    return raw_test.assign(**{"cleaned_row_position": test_idx}), labels


def draw(labels: pd.Series, n: int | None, per_class: int | None, seed: int) -> np.ndarray:
    """Positions (inside the test part) of the flows to use, in random order."""
    rng = np.random.default_rng(seed)
    if per_class is not None:
        picked = []
        for _, positions in labels.groupby(labels, sort=True).indices.items():
            take = min(per_class, len(positions))
            picked.append(rng.choice(positions, size=take, replace=False))
        chosen = np.concatenate(picked)
        rng.shuffle(chosen)
        return chosen
    n = min(int(n), len(labels))
    return rng.choice(len(labels), size=n, replace=False)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def build_comparison(answers: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"row": np.arange(1, len(answers) + 1), "true_label": answers.to_numpy()})
    out["predicted_label"] = predictions["prediction"].to_numpy()
    out["top_votes"] = predictions["top_votes"].to_numpy()
    out["n_voters"] = predictions["n_voters"].to_numpy()
    out["correct"] = out["true_label"] == out["predicted_label"]
    return out


def per_class_table(comparison: pd.DataFrame, test_counts: pd.Series) -> pd.DataFrame:
    rows = []
    for name, group in comparison.groupby("true_label", sort=True):
        rows.append(
            {"class": name, "flows": len(group), "correct": int(group["correct"].sum()),
             "recall": group["correct"].mean(), "test_rows_in_dataset": int(test_counts.get(name, 0))}
        )
    return pd.DataFrame(rows)


def mistake_table(comparison: pd.DataFrame) -> pd.DataFrame:
    wrong = comparison[~comparison["correct"]]
    if wrong.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "flows"])
    return wrong.groupby(["true_label", "predicted_label"]).size().reset_index(name="flows").sort_values("flows", ascending=False).reset_index(drop=True)


def md_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "|".join("---" for _ in frame.columns) + "|"]
    for _, row in frame.iterrows():
        cells = [formats[c].format(row[c]) if c in formats else str(row[c]) for c in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_summary(comparison: pd.DataFrame, classes: pd.DataFrame, mistakes: pd.DataFrame, info: dict) -> str:
    n = len(comparison)
    correct = int(comparison["correct"].sum())
    macro = f1_score(comparison["true_label"], comparison["predicted_label"], average="macro", zero_division=0)
    rare = classes[classes["test_rows_in_dataset"] < RARE_TEST_ROWS]
    out = [
        "# Comparison of true labels and predictions",
        "",
        f"Generated: {info['generated']}  |  sampling seed {info['seed']}  |  mode: {info['mode']}  |  models: {info['voters']} voting",
        "",
        "## What this is",
        "",
        "- The flows are **real rows of CIC-IDS2017 from the held-out test split** (the 20% the models never trained on). They are not from another network.",
        "- The true label of every flow is the dataset's own label. Nothing was invented or edited.",
        "- The prediction is the majority vote of the models of the final set, the same answer the app gives for the same CSV file.",
        "",
        "## Result",
        "",
        f"- Flows compared: **{n:,}**; refused as invalid input: **{info['refused']}**",
        f"- Correct: **{correct:,}** of {n:,} (accuracy **{correct / n:.4f}**)",
        f"- Macro F1 on these flows: **{macro:.4f}** (only classes that occur in this sample or in the predictions count)",
        "",
        "Accuracy is dominated by BENIGN and the other large classes when the sample is random. Read the per-class table, not only the total.",
        "",
        "## Per class",
        "",
        md_table(classes, {"recall": "{:.3f}"}),
        "",
    ]
    if len(rare):
        names = ", ".join(f"{r['class']} ({r['test_rows_in_dataset']} test rows)" for _, r in rare.iterrows())
        out += [f"**Caution:** these classes have very few test rows in the whole dataset, so their results here are not reliable: {names}.", ""]
    out += ["## Mistakes (true label to predicted label)", ""]
    out += [md_table(mistakes) if len(mistakes) else "No mistakes in this sample.", ""]
    out += [
        "## How to compare yourself",
        "",
        "1. Open the app, tab **Upload a CSV file**, and upload `comparison_input.csv`.",
        "2. Download the app's result, or read it on the page.",
        "3. Compare its prediction column with `comparison_answers.csv` line by line (row n of the answers is row n of the input).",
        "   `comparison.csv` has already done this with the script's own run of the same predictor.",
        "",
    ]
    return "\n".join(out)


def check_roundtrip(input_path: Path, predictor: predictor_module.Predictor, expected: pd.Series) -> int:
    """Read the written CSV back and predict again; returns how many predictions differ from the in-memory ones."""
    again = predictor.predict_frame(pd.read_csv(input_path))
    return int((again.frame["prediction"].to_numpy() != expected.to_numpy()).sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=300, help="number of random held-out flows (default 300)")
    parser.add_argument("--per-class", type=int, default=None, help="instead of --n: this many flows of every class (fewer if the class has fewer)")
    parser.add_argument("--seed", type=int, default=2026, help="seed of the random draw (not the split seed); default 2026")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    if args.n < 1 or (args.per_class is not None and args.per_class < 1):
        logger.error("--n and --per-class must be at least 1")
        return 1

    try:
        predictor = predictor_module.Predictor.load("final")
        held_out, labels = load_heldout()
    except (predictor_module.PredictorError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    chosen = draw(labels, args.n, args.per_class, args.seed)
    features = validator.required_features(predictor.schema)
    sample = held_out.iloc[chosen]
    # the same float32 values the preprocessing used, written with every digit so the file reads back unchanged
    inputs = sample[features].astype(settings.FLOAT_DTYPE).astype("float64").reset_index(drop=True)
    answers = labels.iloc[chosen].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.out_dir / "comparison_input.csv"
    inputs.to_csv(input_path, index=False, encoding="utf-8")

    result = predictor.predict_frame(pd.read_csv(input_path))
    comparison = build_comparison(answers, result.frame)
    refused = int(comparison["predicted_label"].isna().sum())
    evaluated = comparison[comparison["predicted_label"].notna()]
    classes = per_class_table(evaluated, labels.value_counts())
    mistakes = mistake_table(evaluated)

    pd.DataFrame({"row": comparison["row"], "true_label": comparison["true_label"], "cleaned_row_position": sample["cleaned_row_position"].to_numpy()}).to_csv(
        args.out_dir / "comparison_answers.csv", index=False, encoding="utf-8"
    )
    result.frame.reset_index(drop=True).assign(row=comparison["row"]).pipe(lambda f: f[["row"] + [c for c in f.columns if c != "row"]]).to_csv(
        args.out_dir / "comparison_predictions.csv", index=False, encoding="utf-8"
    )
    comparison.to_csv(args.out_dir / "comparison.csv", index=False, encoding="utf-8")
    classes.to_csv(args.out_dir / "comparison_per_class.csv", index=False, encoding="utf-8")

    info = {
        "generated": datetime.now().isoformat(timespec="seconds"), "seed": args.seed, "refused": refused, "voters": len(result.voters),
        "mode": f"{args.per_class} per class" if args.per_class else f"{len(comparison)} random flows",
    }
    (args.out_dir / "comparison_summary.md").write_text(build_summary(evaluated, classes, mistakes, {**info, "refused": refused}), encoding="utf-8")

    # automatic checks
    differ = check_roundtrip(input_path, predictor, result.frame["prediction"])
    checks = [
        ("the flows are held-out test rows (split matches the saved y_test.npy)", True),
        ("input file has only the model features and no label", list(pd.read_csv(input_path, nrows=1).columns) == features),
        ("reading the input file back gives the same predictions", differ == 0),
        ("no flow was refused", refused == 0),
    ]
    accuracy = accuracy_score(evaluated["true_label"], evaluated["predicted_label"]) if len(evaluated) else float("nan")
    print(f"\nFlows: {len(comparison)}   correct: {int(evaluated['correct'].sum())}   accuracy: {accuracy:.4f}")
    print(classes.to_string(index=False, formatters={"recall": "{:.3f}".format}))
    print("\nChecks:")
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAILED'}] {name}")
    print(f"\nFiles written to {args.out_dir}")
    return 0 if all(ok for _, ok in checks) else 2


if __name__ == "__main__":
    sys.exit(main())