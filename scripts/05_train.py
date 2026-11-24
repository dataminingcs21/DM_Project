"""Step 5: train the 11 baseline models and save them (Checkpoint 4).

Usage (from the project root, with the virtual environment active):

    python scripts/05_train.py                          # all 11 models, saved
    python scripts/05_train.py --models knn svm         # only some models
    python scripts/05_train.py --skip-existing          # resume: keep models that are saved and fresh
    python scripts/05_train.py --dry-run --sample-size 30000   # quick test, saves nothing

Outputs:
    artifacts/models/trained/<model>.joblib and <model>.json   one pair per model (File 43)
    reports/training/training_report.md                        readable report (send this one for Checkpoint 4)
    reports/training/training_summary.json                     the same results, machine readable

Training uses only the training arrays; the test set is not read.  After training,
every saved model is reloaded (checksum and freshness are checked) and run on a
class-balanced slice of the *training* data as a smoke test.  That accuracy is a
sanity check, **not a performance measure** (the models have seen these rows); the
real scores come from the test set in step 6.

``--dry-run`` trains in memory on a class-capped sample and writes nothing, so a
quick test can never overwrite the real models.  Ctrl+C stops the run; models
already saved stay saved, and ``--skip-existing`` resumes.

Exit codes: 0 all good; 1 setup problem (data missing or stale, bad option);
2 a model failed or an automatic check failed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from config import model_params, settings
from nids.persistence import model_store
from nids.training import trainer
from nids.training.data_access import DataNotReadyError, TrainingData, class_capped_indices, load_training_data
from nids.utils.io_utils import write_json
from nids.utils.logger import get_logger

logger = get_logger("scripts.train")

SANITY_ROWS_PER_CLASS = 200


# ---------------------------------------------------------------------------
# Smoke test of the saved models
# ---------------------------------------------------------------------------
def sanity_slice(data: TrainingData) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``SANITY_ROWS_PER_CLASS`` training rows per class (every class present)."""
    idx = np.sort(np.concatenate([np.flatnonzero(data.y == c)[:SANITY_ROWS_PER_CLASS] for c in range(data.n_classes)]))
    return np.asarray(data.X[idx]), data.y[idx]


def check_saved_model(key: str, data: TrainingData, X: np.ndarray, y: np.ndarray) -> list[tuple[str, bool, str]]:
    """Reload one saved model and check it.  Never raises: a problem becomes a failed check."""
    name = model_params.DISPLAY_NAMES[key]
    checks: list[tuple[str, bool, str]] = []
    try:
        model = model_store.load_model(key)  # verifies checksum and freshness
    except Exception as exc:
        return [(f"{name}: reloads from disk (checksum, freshness)", False, f"{type(exc).__name__}: {exc}")]
    checks.append((f"{name}: reloads from disk (checksum, freshness)", True, ""))
    try:
        pred = model.predict(X)
        scores = model.predict_scores(X)
    except Exception as exc:
        checks.append((f"{name}: predicts on training rows", False, f"{type(exc).__name__}: {exc}"))
        return checks
    ok_shape = pred.shape == (len(y),) and scores.shape == (len(y), data.n_classes) and int(pred.min()) >= 0 and int(pred.max()) < data.n_classes
    checks.append((f"{name}: predictions and scores have the right shape and range", bool(ok_shape), ""))
    distinct = int(np.unique(pred).size)
    checks.append((f"{name}: is not a constant predictor (predicts at least 2 classes)", distinct >= 2, f"{distinct} distinct classes predicted"))
    return checks


def fill_skipped_from_metadata(summary: trainer.TrainingSummary) -> None:
    """A model kept from an earlier run (``--skip-existing``) gets its numbers from its saved metadata,
    so the report always describes all models, not only the ones trained in this session."""
    for r in summary.results:
        if r.status != "skipped":
            continue
        try:
            meta = model_store.load_metadata(r.key)
        except Exception:
            continue
        info, extra = meta.get("model_info", {}), meta.get("extra", {})
        sub = extra.get("subsample", {})
        r.fit_seconds = info.get("fit_seconds")
        r.predict_seconds_5000 = extra.get("predict_seconds_5000_train_rows")
        r.n_rows_used = sub.get("n_rows")
        r.n_rows_available = sub.get("n_rows_available")
        r.subsampled = sub.get("subsampled")
        r.converged = info.get("converged")
        r.train_device = info.get("train_device")
        r.file_size_mb = meta.get("file_size_mb")
        r.note = f"kept from the run of {meta.get('saved_at')}"


def sanity_accuracy(key: str, X: np.ndarray, y: np.ndarray) -> float | None:
    try:
        return float((model_store.load_model(key).predict(X) == y).mean())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("scikit-learn", "xgboost", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def build_markdown_report(
    summary: trainer.TrainingSummary,
    data: TrainingData,
    checks: list[tuple[str, bool, str]],
    accuracies: dict[str, float | None],
    sizes_mb: dict[str, float],
    dry_run: bool,
) -> str:
    versions = _versions()
    lines = [
        "# Training report (baseline models)" + (" - DRY RUN, nothing saved" if dry_run else ""),
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Training rows available: {data.n_rows:,}; features: {data.feature_count}; classes: {data.n_classes}  ",
        f"Class weights: `{settings.CLASS_WEIGHT_MODE}`; seed {settings.RANDOM_SEED}; threads {settings.N_JOBS}; XGBoost device setting `{settings.XGBOOST_DEVICE}`  ",
        f"Libraries: Python {versions['python']}, scikit-learn {versions['scikit-learn']}, XGBoost {versions['xgboost']}, NumPy {versions['numpy']}  ",
        f"Training time of this session: {summary.total_seconds / 60:.1f} min ({summary.total_seconds:.0f} s)",
        "",
        "## Results",
        "",
        "| Model | Status | Fit (s) | Predict 5,000 rows (s) | Rows used | Subsample | Converged | Saved size (MB) | Smoke-test accuracy* |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in summary.results:
        acc = accuracies.get(r.key)
        size = r.file_size_mb if r.file_size_mb is not None else sizes_mb.get(r.key)
        lines.append(
            f"| {r.display_name} | {r.status if r.status != 'skipped' else 'kept (earlier run)'} | "
            f"{'' if r.fit_seconds is None else f'{r.fit_seconds:.1f}'} | "
            f"{'' if r.predict_seconds_5000 is None else f'{r.predict_seconds_5000:.2f}'} | "
            f"{'' if r.n_rows_used is None else f'{r.n_rows_used:,}'} | "
            f"{'' if r.subsampled is None else ('yes, of ' + format(r.n_rows_available, ',')) if r.subsampled else 'no'} | "
            f"{'' if r.converged is None else ('yes' if r.converged else '**NO**')} | "
            f"{'' if size is None else size} | "
            f"{'' if acc is None else f'{acc:.3f}'} |"
        )
    lines += [
        "",
        f"\\* Smoke test only: accuracy on up to {SANITY_ROWS_PER_CLASS} **training** rows per class, "
        "which the models have seen. It checks that a model is not broken; it is **not** a performance measure. "
        "Real scores come from the test set in step 6.",
        "",
    ]
    if summary.failed:
        lines += ["## Failed models", ""]
        for r in summary.results:
            if r.status == "failed":
                lines.append(f"- **{r.display_name}**: {r.error}")
        lines.append("")
    if summary.not_converged:
        names = ", ".join(model_params.DISPLAY_NAMES[k] for k in summary.not_converged)
        lines += ["## Not converged", "", f"These models used all their iterations: {names}. Their result is not a true optimum.", ""]
    sub = [r for r in summary.results if r.subsampled]
    if sub:
        lines += [
            "## Models trained on a subsample",
            "",
            "Class-capped stratified subsample (small classes kept whole, large classes capped): "
            + ", ".join(f"{r.display_name} ({r.n_rows_used:,} of {r.n_rows_available:,} rows)" for r in sub)
            + ". The subsample is more balanced than the real traffic, so these models learn a different class prior "
            "and their test scores are not strictly comparable with the models trained on all rows.",
            "",
        ]
    lines += ["## Automatic checks", ""]
    if not checks:
        lines += ["No checks were run (dry run: nothing was saved).", ""]
    for name, ok, detail in checks:
        lines.append(f"- {'PASS' if ok else '**FAIL**'}: {name}" + (f" ({detail})" if detail else ""))
    passed = sum(1 for _, ok, _ in checks if ok)
    lines += ["", f"Checks passed: {passed}/{len(checks)}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline models.")
    parser.add_argument("--models", nargs="+", metavar="KEY", help=f"model keys to train (default: all). Keys: {', '.join(model_params.MODEL_KEYS)}")
    parser.add_argument("--skip-existing", action="store_true", help="do not retrain models that are saved and fresh")
    parser.add_argument("--stop-on-error", action="store_true", help="stop at the first failed model (default: continue)")
    parser.add_argument("--dry-run", action="store_true", help="train in memory and save nothing")
    parser.add_argument("--sample-size", type=int, default=None, help="train on a class-capped sample of N rows (needs --dry-run)")
    args = parser.parse_args(argv)

    if args.sample_size is not None and not args.dry_run:
        logger.error("--sample-size changes the training data, so it is only allowed together with --dry-run.")
        return 1
    if args.sample_size is not None and args.sample_size < 200:
        logger.error("--sample-size must be at least 200")
        return 1
    keys = list(args.models) if args.models else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        logger.error("Unknown model key(s) %s. Known: %s", unknown, list(model_params.MODEL_KEYS))
        return 1
    if args.dry_run and args.skip_existing:
        logger.error("--skip-existing has no meaning with --dry-run (nothing is saved).")
        return 1

    try:
        settings.ensure_directories()
        data = load_training_data()
    except (DataNotReadyError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    if args.sample_size is not None:
        idx = class_capped_indices(data.y, args.sample_size)
        data = TrainingData(X=data.X[idx], y=data.y[idx], codec=data.codec, n_classes=data.n_classes, feature_count=data.feature_count)
        logger.info("Dry run on a class-capped sample of %d rows", data.n_rows)

    print("Plan:")
    for key in keys:
        target = model_params.subsample_size(key)
        rows = data.n_rows if target is None or target >= data.n_rows else target
        print(f"  {model_params.DISPLAY_NAMES[key]:<24} {rows:>10,} rows" + ("  (subsample)" if rows < data.n_rows else ""))
    print()

    started = time.time()
    try:
        summary = trainer.train_models(
            keys=keys,
            data=data,
            save=not args.dry_run,
            skip_existing=args.skip_existing,
            continue_on_error=not args.stop_on_error,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted. Models already saved stay saved; rerun with --skip-existing to continue.")
        return 130

    if not args.dry_run:
        fill_skipped_from_metadata(summary)
    checks: list[tuple[str, bool, str]] = []
    accuracies: dict[str, float | None] = {}
    sizes_mb: dict[str, float] = {}
    checks.append(("no model failed", not summary.failed, ", ".join(summary.failed)))
    if not args.dry_run:
        X, y = sanity_slice(data)
        for r in summary.results:
            if r.status == "failed":
                continue
            checks.extend(check_saved_model(r.key, data, X, y))
            accuracies[r.key] = sanity_accuracy(r.key, X, y)
            try:
                sizes_mb[r.key] = model_store.load_metadata(r.key)["file_size_mb"]
            except Exception:
                pass

    print(trainer.format_summary(summary))
    report = build_markdown_report(summary, data, checks, accuracies, sizes_mb, args.dry_run)
    if args.dry_run:
        print()
        print(report)
    else:
        out = settings.REPORTS_DIR / "training"
        out.mkdir(parents=True, exist_ok=True)
        (out / "training_report.md").write_text(report, encoding="utf-8")
        write_json(
            {**summary.to_dict(), "generated_at": datetime.now().isoformat(timespec="seconds"), "libraries": _versions(),
             "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks], "smoke_test_accuracy": accuracies},
            out / "training_summary.json",
        )
        logger.info("Report written to %s", out / "training_report.md")

    failed_checks = [name for name, ok, _ in checks if not ok]
    logger.info("Checks passed: %d/%d (total %.1f min)", len(checks) - len(failed_checks), len(checks), (time.time() - started) / 60)
    if failed_checks or summary.failed:
        logger.error("Failed: %s", failed_checks or summary.failed)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())