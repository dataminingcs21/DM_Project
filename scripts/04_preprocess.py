"""Step 4: preprocess the cleaned data and save it (Checkpoint 3).

Usage (from the project root, with the virtual environment active):

    python scripts/04_preprocess.py
    python scripts/04_preprocess.py --dry-run --sample-per-class 2000   # quick test, writes nothing

Outputs in ``data/preprocessed/``:
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    preprocessing_pipeline.joblib, label_encoder.joblib
    feature_schema.json, feature_names.json, sample_flows.parquet
    preprocessing_metadata.json      (written last: its presence means "complete")
    preprocessing_report.md          readable report (send this one for Checkpoint 3)
    preprocessing_class_counts.csv, preprocessing_features.csv

After saving, the script re-reads what it wrote and runs automatic checks.
``--dry-run`` runs everything in memory and writes nothing, so a quick test can
never overwrite the real preprocessed data.  ``--sample-per-class`` is allowed
only together with ``--dry-run`` for the same reason.
"""

from __future__ import annotations

import argparse
import gc
import sys
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from nids.eda.eda_stats import class_capped_sample
from nids.preprocessing.feature_schema import schema_table
from nids.preprocessing.preprocessor import (
    PreprocessingResult,
    check_preprocessed_fresh,
    preprocess,
    save_preprocessed,
)
from nids.utils.io_utils import config_hash, load_array, load_joblib, read_json, read_parquet
from nids.utils.logger import get_logger

logger = get_logger("scripts.preprocess")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_checks(result: PreprocessingResult, from_disk: bool) -> list[tuple[str, bool, str]]:
    """Automatic checks. With ``from_disk`` the saved files are re-read and compared with memory."""
    checks: list[tuple[str, bool, str]] = []
    split = result.report["split"]

    checks.append(("train and test rows are disjoint and cover all rows", bool(split["train_test_disjoint"] and split["covers_all_rows"]), ""))
    checks.append(("every class is in both train and test", not split["classes_missing_from_train"] and not split["classes_missing_from_test"], ""))
    names = set(result.schema["raw_feature_names"])
    checks.append(("label and source_file are not features", settings.LABEL_COLUMN not in names and settings.SOURCE_FILE_COLUMN not in names, ""))
    checks.append(("no missing or infinite values in X", bool(np.isfinite(result.X_train).all() and np.isfinite(result.X_test).all()), ""))
    centre = float(np.abs(result.X_train.mean(axis=0)).max())
    checks.append(("training data is centred (largest absolute mean < 0.001)", centre < 1e-3, f"largest absolute mean = {centre:.6f}"))
    checks.append(("class codes are in range", int(max(result.y_train.max(), result.y_test.max())) < result.codec.n_classes, ""))

    pipeline = result.pipeline
    if from_disk:
        pipeline = load_joblib(settings.PIPELINE_FILE)
        x_train = load_array(settings.X_TRAIN_FILE, mmap_mode="r")
        x_test = load_array(settings.X_TEST_FILE)
        y_train, y_test = load_array(settings.Y_TRAIN_FILE), load_array(settings.Y_TEST_FILE)
        same = (
            x_train.shape == result.X_train.shape and x_test.shape == result.X_test.shape
            and np.array_equal(x_train[:1000], result.X_train[:1000]) and np.array_equal(x_train[-1000:], result.X_train[-1000:])
            and np.array_equal(x_test, result.X_test) and np.array_equal(y_train, result.y_train) and np.array_equal(y_test, result.y_test)
        )
        checks.append(("saved arrays equal the arrays in memory", bool(same), ""))
        del x_train
        codec = load_joblib(settings.LABEL_ENCODER_FILE)
        checks.append(("saved label encoder equals the one in memory", codec == result.codec, ""))
        fresh, reason = check_preprocessed_fresh()
        checks.append(("saved data is fresh for the current settings", fresh, reason))

    samples = result.sample_flows
    positions = samples["test_row_position"].to_numpy()
    raw = samples[result.schema["raw_feature_names"]].astype(settings.FLOAT_DTYPE)
    reproduced = np.allclose(pipeline.transform(raw).to_numpy(), result.X_test[positions], atol=1e-5)
    label = "the saved pipeline reproduces X_test rows from raw sample flows" if from_disk else "the pipeline reproduces X_test rows from raw sample flows"
    checks.append((label, bool(reproduced), ""))
    return checks


# ---------------------------------------------------------------------------
# Readable report
# ---------------------------------------------------------------------------
def _markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_(none)_"
    shown = df.head(max_rows)
    lines = ["| " + " | ".join(str(c) for c in shown.columns) + " |", "|" + "|".join("---" for _ in shown.columns) + "|"]
    for row in shown.itertuples(index=False):
        cells = [f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in the CSV file._")
    return "\n".join(lines)


def build_markdown_report(result: PreprocessingResult, metadata: dict[str, Any], checks: list[tuple[str, bool, str]], dry_run: bool) -> str:
    report = result.report
    pipe = report["pipeline"]
    feat = schema_table(result.schema)
    counts = pd.DataFrame(report["class_counts"])
    counts["test_percent"] = (counts["test"] / (counts["train"] + counts["test"]) * 100).round(2)
    dropped = feat.loc[~feat["used_by_models"], ["name", "dropped_because"]]
    kept = feat.loc[feat["used_by_models"], ["name", "min", "median", "p99", "max", "all_whole_numbers"]]
    constant = pipe["feature_filter"]["constant_features"]

    lines = [
        "# Preprocessing report",
        "",
        f"Generated {metadata['generated_at']} by `scripts/04_preprocess.py`"
        + (" (**dry run: nothing was saved**)." if dry_run else "."),
        "",
        "## Overview",
        f"- Rows: **{report['n_rows']:,}** = {report['X_train_shape'][0]:,} train + {report['X_test_shape'][0]:,} test "
        f"(test fraction {report['split']['test_fraction']:.4f}, seed {report['split']['random_seed']})",
        f"- Feature columns: **{report['n_raw_features']}** in, **{report['n_model_features']}** used by the models ({report['dtype']})",
        f"- Target mode: `{report['target']['mode']}`, classes: **{report['target']['n_classes']}**, rare-class policy: `{report['target']['rare_class_policy']}`",
        f"- Preprocessing settings hash: `{metadata['config_hash']}`; cleaning rules hash: `{metadata.get('cleaned_data_config_hash')}`",
        f"- Time: {report['seconds']} s",
        "",
        "## Pipeline (fitted on the training rows only)",
        f"- Steps: {' -> '.join(pipe['steps'])}",
        f"- Imputer: `{pipe['imputer_strategy']}`; signed log: `{pipe['log_transform']}`; scaler: `{pipe['scaler_type']}`",
        f"- Correlation threshold: {pipe['feature_filter']['correlation_threshold']} (computed on {pipe['feature_filter']['rows_used_for_correlation']:,} training rows)",
        "",
        "## Rows per class",
        _markdown_table(counts),
        "",
        "Small classes have very few test rows; their scores will be unstable (read the support column everywhere).",
        "",
        "## Feature selection",
        f"- Constant columns: {constant or 'none'}",
        f"- Correlated columns dropped: {len(pipe['feature_filter']['correlated_features'])} (first column in order is kept)",
        "",
        "Dropped columns and why:",
        _markdown_table(dropped),
        "",
        "Columns used by the models (training-data ranges):",
        _markdown_table(kept),
        "",
        "## Automatic checks",
        _markdown_table(pd.DataFrame([(n, "PASS" if ok else "FAIL", d) for n, ok, d in checks], columns=["check", "result", "detail"])),
        "",
        "## Checkpoint 3: what to look at",
        "1. The class table: are the test counts of the rare classes acceptable (Heartbleed, Sql Injection, Infiltration)?",
        "2. The dropped features: does any dropped feature look important to you? (the first of two correlated columns is kept by rule)",
        "3. `Destination Port` is still a model feature (Checkpoint 1, option A); the comparison without it comes later.",
        "4. All checks must read PASS before the training stage.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the cleaned CIC-IDS2017 data.")
    parser.add_argument("--dry-run", action="store_true", help="run everything in memory and write nothing")
    parser.add_argument("--sample-per-class", type=int, default=None, help="use at most N rows per class (needs --dry-run)")
    args = parser.parse_args(argv)

    if args.sample_per_class is not None and not args.dry_run:
        logger.error("--sample-per-class changes the data, so it is only allowed together with --dry-run (it would overwrite the real preprocessed data).")
        return 1
    if args.sample_per_class is not None and args.sample_per_class < 50:
        logger.error("--sample-per-class must be at least 50")
        return 1

    try:
        settings.ensure_directories()
        df = read_parquet(settings.CLEANED_DATA_FILE, hint="Run scripts/03_clean_data.py first.")
        if settings.CLEANING_REPORT_FILE.is_file():
            saved_hash = read_json(settings.CLEANING_REPORT_FILE).get("config_hash")
            if saved_hash != config_hash(settings.cleaning_config()):
                logger.error("The cleaned data was made with other cleaning rules than the current settings. Rerun scripts/03_clean_data.py.")
                return 1
        else:
            logger.warning("No cleaning report found; cannot check that the cleaned data matches the current cleaning rules.")
        if args.sample_per_class is not None:
            df = class_capped_sample(df, args.sample_per_class).reset_index(drop=True)
            logger.info("Dry run on a class-capped sample: %d rows", len(df))
        result = preprocess(df)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    del df
    gc.collect()

    if args.dry_run:
        from datetime import datetime

        metadata = {"generated_at": datetime.now().isoformat(timespec="seconds"), "config_hash": config_hash(settings.preprocessing_config()), "cleaned_data_config_hash": None}
        checks = run_checks(result, from_disk=False)
    else:
        metadata = save_preprocessed(result)
        checks = run_checks(result, from_disk=True)

    report_text = build_markdown_report(result, metadata, checks, args.dry_run)
    if args.dry_run:
        print(report_text)
    else:
        out = settings.PREPROCESSED_DIR
        (out / "preprocessing_report.md").write_text(report_text, encoding="utf-8")
        pd.DataFrame(result.report["class_counts"]).to_csv(out / "preprocessing_class_counts.csv", index=False, encoding="utf-8")
        schema_table(result.schema).to_csv(out / "preprocessing_features.csv", index=False, encoding="utf-8")
        logger.info("Saved to %s", out)

    failed = [name for name, ok, _ in checks if not ok]
    logger.info("Checks passed: %d/%d", len(checks) - len(failed), len(checks))
    if failed:
        logger.error("Failed checks: %s", failed)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())