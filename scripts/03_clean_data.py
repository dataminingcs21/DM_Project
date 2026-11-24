"""Step 3: clean the raw dataset and save it (Checkpoint 2).

Usage (from the project root, with the virtual environment active):

    python scripts/03_clean_data.py
    python scripts/03_clean_data.py --nrows 5000        # quick test on a slice

Outputs in ``data/cleaned/``:
    cleaned.parquet            the cleaned data (features in float64, Label, source_file)
    cleaning_report.json       every number of the cleaning, plus the rules (config) and their hash
    cleaning_report.md         the same, readable
    cleaning_class_counts.csv  rows per class before and after

The raw files are never modified.  Missing values are left in place on purpose:
they are imputed on the training part only, later (File 23).
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from nids.cleaning.cleaner import clean_dataset
from nids.data import schema
from nids.data.loader import load_raw_data
from nids.utils.io_utils import config_hash, write_json, write_parquet
from nids.utils.logger import get_logger

logger = get_logger("scripts.clean_data")

# Diagnostic only (nothing is changed because of it): infinite rates are often the
# result of dividing by a duration of zero.  This checks whether that is the case.
ZERO_DURATION_COLUMN = "Flow Duration"


# ---------------------------------------------------------------------------
# Diagnostic on the raw data (before cleaning)
# ---------------------------------------------------------------------------
def diagnose_nonfinite_rows(df: pd.DataFrame) -> dict[str, Any] | None:
    """Compare rows with a missing or infinite feature value to rows with a zero duration."""
    if ZERO_DURATION_COLUMN not in df.columns:
        return None
    nonfinite = np.zeros(len(df), dtype=bool)
    for col in schema.get_feature_columns(df.columns):
        nonfinite |= ~np.isfinite(df[col].to_numpy(dtype="float64", copy=False))
    zero_duration = df[ZERO_DURATION_COLUMN].to_numpy(dtype="float64", copy=False) == 0
    return {
        "duration_column": ZERO_DURATION_COLUMN,
        "rows_with_missing_or_infinite": int(nonfinite.sum()),
        "of_which_with_zero_duration": int((nonfinite & zero_duration).sum()),
        "rows_with_zero_duration": int(zero_duration.sum()),
        "zero_duration_rows_that_are_missing_or_infinite": int((nonfinite & zero_duration).sum()),
    }


# ---------------------------------------------------------------------------
# Readable report
# ---------------------------------------------------------------------------
def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_(none)_"
    shown = df.head(max_rows)
    lines = [
        "| " + " | ".join(str(c) for c in shown.columns) + " |",
        "|" + "|".join("---" for _ in shown.columns) + "|",
    ]
    for row in shown.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in the CSV file._")
    return "\n".join(lines)


def _dict_table(data: dict, key_name: str, value_name: str) -> str:
    return _markdown_table(pd.DataFrame({key_name: list(data.keys()), value_name: list(data.values())}))


def build_markdown_report(report: dict[str, Any], class_table: pd.DataFrame) -> str:
    step = {s["step"]: s for s in report["steps"]}
    labels, ident = step["apply_label_map"], step["drop_identical_columns"]
    neg, inf = step["handle_negative_values"], step["handle_infinite_values"]
    dup, conf = step["remove_exact_duplicates"], step["handle_conflicting_vectors"]
    missing = report["missing_values_after_cleaning"]

    lines = [
        "# Cleaning report",
        "",
        f"Generated {report['generated_at']} by `scripts/03_clean_data.py`. The raw files were not changed.",
        "",
        "## Overview",
        f"- Rows: **{report['rows_in']:,}** in, **{report['rows_out']:,}** out "
        f"({report['rows_in'] - report['rows_out']:,} removed, {(report['rows_in'] - report['rows_out']) / max(report['rows_in'], 1) * 100:.2f}%)",
        f"- Feature columns: **{report['n_feature_columns_in']}** in, **{report['n_feature_columns_out']}** out",
        f"- Cleaning rules hash: `{report['config_hash']}`",
        "",
        "## Rows removed by step",
        _markdown_table(
            pd.DataFrame(
                [(s["step"], s["rows_before"], s["rows_after"], s["rows_removed"]) for s in report["steps"] if "rows_before" in s],
                columns=["step", "rows_before", "rows_after", "rows_removed"],
            )
        ),
        "",
        "## Rows per class, before and after",
        "`rows_before` counts after the missing-label drop and the label fixes.",
        "",
        _markdown_table(class_table),
        "",
        f"- Classes below the technical minimum of {report['min_class_samples_rule']} rows after cleaning: "
        f"{', '.join(f'`{c}`' for c in report['classes_below_min_class_samples']) or 'none'}",
        "",
        "## Labels",
        f"- Rows relabelled: {labels['rows_changed']:,}",
        _dict_table({k: f"{v['new']} ({v['rows']} rows)" for k, v in labels["applied"].items()}, "old label", "new label"),
        f"- Odd labels still present: {labels['odd_labels_remaining'] or 'none'}",
        "",
        "## Identical columns",
        f"- Groups of identical columns: {ident['identical_groups'] or 'none'}",
        f"- Columns dropped: {ident['columns_dropped'] or 'none'}",
        "",
        f"## Impossible negative values (policy `{neg['policy']}`)",
        f"- Rows affected: {neg['rows_affected']:,}",
        _dict_table(neg["values_per_column"], "column", "negative values"),
        "",
        "Rows affected, per class:",
        _dict_table(neg["rows_affected_per_class"], "class", "rows"),
        f"- Listed columns not present in the data: {neg['columns_not_present'] or 'none'}",
        "",
        "## Infinite values (turned into missing)",
        f"- Rows affected: {inf['rows_affected']:,}",
        _dict_table(inf["values_per_column"], "column", "infinite values"),
        "",
        "Rows affected, per class:",
        _dict_table(inf["rows_affected_per_class"], "class", "rows"),
        "",
        "## Exact duplicates",
        f"- Rows removed: {dup['rows_removed']:,}",
        _dict_table(dup["removed_per_class"], "class", "rows removed"),
        "",
        f"## Feature vectors with more than one label (policy `{conf['policy']}`)",
        f"- Vectors: {conf['n_conflicting_vectors']:,}; rows in them: {conf['rows_in_conflicting_vectors']:,}; rows removed: {conf['rows_removed']:,}",
        _dict_table(conf["label_combinations"], "labels sharing identical features", "vectors"),
        "",
        "Rows per class in such vectors:",
        _dict_table(conf["rows_per_class"], "class", "rows"),
        "",
        "## Missing values left in the cleaned data (imputed later on the training data only)",
        f"- Missing values: {missing['n_missing']:,}; rows with at least one: {missing['rows_with_missing']:,}",
        _dict_table(missing["per_column"], "column", "missing values"),
    ]
    diag = report.get("diagnostics", {}).get("nonfinite_vs_zero_duration")
    if diag:
        lines += [
            "",
            "## Diagnostic (raw data, before cleaning)",
            f"- Rows with a missing or infinite value: {diag['rows_with_missing_or_infinite']:,}",
            f"- Of these, rows with `{diag['duration_column']}` = 0: {diag['of_which_with_zero_duration']:,}",
            f"- All rows with `{diag['duration_column']}` = 0: {diag['rows_with_zero_duration']:,} "
            f"(of which missing or infinite: {diag['zero_duration_rows_that_are_missing_or_infinite']:,})",
        ]
    lines += [
        "",
        "## Checkpoint 2 decisions",
        f"1. `RARE_CLASS_POLICY` (now `{settings.RARE_CLASS_POLICY}`): keep or merge, using the class table above.",
        f"2. `CONFLICT_POLICY` (now `{settings.CONFLICT_POLICY}`): keep or drop the vectors with more than one label.",
        f"3. `NEGATIVE_VALUE_POLICY` (now `{settings.NEGATIVE_VALUE_POLICY}`): confirm, using the per-class numbers above.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean the raw CIC-IDS2017 data.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="folder with the CSV files (default: data/raw)")
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N rows of each file (quick test)")
    args = parser.parse_args(argv)

    try:
        settings.ensure_directories()
        df, load_report = load_raw_data(args.raw_dir, nrows=args.nrows)
        diagnostics = {"nonfinite_vs_zero_duration": diagnose_nonfinite_rows(df)}
        cleaned, report, tables = clean_dataset(df)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    del df  # free the raw frame (about 2 GB) before writing
    gc.collect()

    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["config_hash"] = config_hash(settings.cleaning_config())
    report["load"] = {"n_files": load_report["n_files"], "n_rows_loaded": load_report["n_rows"], "nrows_limit": args.nrows}
    report["diagnostics"] = diagnostics

    out_dir = settings.CLEANED_DATA_DIR
    logger.info("Writing the cleaned data (%d rows x %d columns)", len(cleaned), cleaned.shape[1])
    write_parquet(cleaned, settings.CLEANED_DATA_FILE)
    write_json(report, settings.CLEANING_REPORT_FILE)
    tables["class_counts"].to_csv(out_dir / "cleaning_class_counts.csv", index=False, encoding="utf-8")
    (out_dir / "cleaning_report.md").write_text(build_markdown_report(report, tables["class_counts"]), encoding="utf-8")

    logger.info("Rows: %d in, %d out | feature columns: %d in, %d out", report["rows_in"], report["rows_out"], report["n_feature_columns_in"], report["n_feature_columns_out"])
    logger.info("Saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())