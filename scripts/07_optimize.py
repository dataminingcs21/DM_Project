"""Step 7: tune the 11 models with GridSearchCV and refit the winners (Checkpoint 5).

Usage (from the project root, with the virtual environment active):

    python scripts/07_optimize.py --dry-run             # plan only: grids, sample sizes, what is already saved
    python scripts/07_optimize.py                       # tune all 11 models, refit the winners, write the report
    python scripts/07_optimize.py --models decision_tree random_forest    # only some models
    python scripts/07_optimize.py --skip-refit          # tuning only
    python scripts/07_optimize.py --rerun               # ignore saved tuning results and tune again

What it does, in order (each stage saves as it goes and can be resumed after Ctrl+C):

1. **Grid search** (``nids.optimization.grid_search``): stratified k-fold on a class-capped sample of TRAINING rows,
   scored by prior-corrected macro F1.  Results: ``reports/optimization/tuning_<model>.json``.
2. **Refit**: each winner is trained on the same rows as its baseline (all training rows, or the model's own
   subsample for KNN, SVC and Gradient Boosting) and saved as ``artifacts/models/tuned/<model>.joblib/.json``.
   A winner that IS the baseline setting is not retrained: the baseline model is saved as the tuned model.
3. **Report**: ``reports/optimization/tuning_report.md`` (send this one for Checkpoint 5) and ``tuning_summary.json``.

**This script never reads the test set.**  The tuned models are scored on it once, by ``06_evaluate.py --stage tuned``,
which also puts them next to the baselines.  A baseline-versus-tuned choice is therefore not made on the test set.

Exit codes: 0 all good; 1 setup problem (data or baseline models missing, bad option); 2 a model failed or a check
failed; 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from config import model_params, settings
from nids.optimization import grid_search as gs
from nids.persistence import model_store
from nids.training import trainer
from nids.training.data_access import DataNotReadyError, class_capped_indices, load_training_data
from nids.utils.io_utils import write_json
from nids.utils.logger import get_logger

logger = get_logger("scripts.optimize")

TUNED = "tuned"
SANITY_ROWS_PER_CLASS = 200


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------
def tuned_is_current(key: str, result: dict[str, Any]) -> bool:
    """True when a fresh tuned model for exactly this tuning result is already saved."""
    try:
        meta = model_store.load_metadata(key, TUNED)
    except model_store.ModelStoreError:
        return False
    tuning = (meta.get("extra") or {}).get("tuning") or {}
    return not model_store.check_fresh(meta) and tuning.get("fingerprint") == result.get("fingerprint")


def refit_winner(key: str, result: dict[str, Any], data: Any) -> dict[str, Any]:
    """Train and save the tuned model of ``key``; returns a small record for the report."""
    params = dict(result["best_params"])
    tuning_record = {
        "fingerprint": result["fingerprint"],
        "best_params": params,
        "baseline_params": result["baseline_params"],
        "same_as_baseline": bool(result["best_is_baseline"]),
        "corrected_macro_f1_best": result["best_score"]["corrected_mean"],
        "corrected_macro_f1_baseline": result["baseline_score"]["corrected_mean"],
        "improvement_beyond_noise": bool(result["improvement_beyond_noise"]),
        "tuning_rows": result["tuning_rows"],
    }
    if result["best_is_baseline"]:
        model = model_store.load_model(key, "trained")  # the baseline already is the winner: no need to retrain it
        base_meta = model_store.load_metadata(key, "trained")
        extra = {**(base_meta.get("extra") or {}), "tuning": tuning_record}
        meta = model_store.save_model(model, stage=TUNED, extra=extra)
        info = model.info()
        return {"key": key, "status": "reused baseline", "fit_seconds": info.get("fit_seconds"), "n_rows_used": info.get("n_train_rows"),
                "file_size_mb": meta["file_size_mb"], "params": params}
    r = trainer.train_model(key, data, stage=TUNED, save=True, params=params, extra={"tuning": tuning_record})
    return {"key": key, "status": "refitted", "fit_seconds": r.fit_seconds, "n_rows_used": r.n_rows_used, "file_size_mb": r.file_size_mb,
            "params": params, "converged": r.converged, "note": r.note}


def sanity_check(key: str, result: dict[str, Any], data: Any) -> list[tuple[str, bool, str]]:
    """Reload the saved tuned model (checksum and freshness checked) and run it on a slice of TRAINING rows."""
    checks: list[tuple[str, bool, str]] = []
    name = model_params.DISPLAY_NAMES[key]
    try:
        model = model_store.load_model(key, TUNED)
    except Exception as exc:  # noqa: BLE001
        return [(f"{name}: tuned model reloads", False, f"{type(exc).__name__}: {exc}")]
    checks.append((f"{name}: tuned model reloads (checksum and freshness verified)", True, ""))
    wanted = result["best_params"]
    got = {**gs.baseline_point(key), **model.params}  # a parameter left out of model.params has its default (baseline) value
    differs = {k: (v, got.get(k)) for k, v in wanted.items() if got.get(k) != v}
    checks.append((f"{name}: saved parameters equal the winning parameters", not differs, str(differs) if differs else ""))
    idx = np.sort(np.concatenate([np.flatnonzero(data.y == c)[:SANITY_ROWS_PER_CLASS] for c in range(data.n_classes)]))
    X, y = np.asarray(data.X[idx]), data.y[idx]
    pred = model.predict(X)
    checks.append((f"{name}: predicts valid class codes on a training slice", bool(pred.min() >= 0 and pred.max() < data.n_classes), ""))
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(value: Any, fmt: str = "f4") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if fmt == "f4":
        return f"{value:.4f}"
    if fmt == "min":
        return f"{value / 60:.1f}"
    return str(value)


HARD_MINIMA = {"min_samples_leaf": 1, "base_max_depth": 1, "binarize": 0.0}  # values below these are not allowed: no wider grid exists


def grid_edges(result: dict[str, Any], grid: dict[str, list[Any]]) -> list[str]:
    """Numeric parameters (3 or more values) whose winning value is the smallest or largest tried."""
    edges = []
    for name, values in grid.items():
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) < 3 or len(numeric) != len(values):
            continue
        best = result["best_params"][name]
        if best == min(numeric) and HARD_MINIMA.get(name) == best:
            continue  # already at the smallest allowed value
        if best == min(numeric):
            edges.append(f"{name}={best} (smallest value tried)")
        elif best == max(numeric):
            edges.append(f"{name}={best} (largest value tried)")
    return edges


def class_weight_finding(result: dict[str, Any]) -> str | None:
    """Best corrected score with class_weight=None versus 'balanced' (None when the grid has no class_weight)."""
    rows = [r for r in result["table"] if r["macro_f1_corrected_mean"] is not None and "class_weight" in r["params"]]
    if not rows:
        return None
    best = {}
    for mode in (None, "balanced"):
        scores = [r["macro_f1_corrected_mean"] for r in rows if r["params"]["class_weight"] == mode]
        best[mode] = max(scores) if scores else None
    if best[None] is None or best["balanced"] is None:
        return None
    verdict = "balanced is better" if best["balanced"] > best[None] else "no class weighting is better or equal"
    return f"best corrected F1 without class weights {best[None]:.4f}, with balanced weights {best['balanced']:.4f}: {verdict}"


def build_report(results: dict[str, dict[str, Any]], refits: dict[str, dict[str, Any]], checks: list[tuple[str, bool, str]],
                 scope: dict[str, Any], baseline_fit: dict[str, float | None], failed: dict[str, str]) -> str:
    lines = ["# Tuning report (Checkpoint 5)", "", f"Generated {datetime.now().isoformat(timespec='seconds')}.", "", "## Scope", ""]
    lines += [f"- {k}: {v}" for k, v in scope.items()]
    lines += ["", "## Summary", ""]
    lines += ["| Model | Grid points | Rows | Folds | Best parameters | Baseline corrected F1 | Best corrected F1 | Improvement | Beyond fold noise | Best plain F1 | Minutes |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in results.items():
        params = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        flag = "is the baseline" if r["best_is_baseline"] else ("yes" if r["improvement_beyond_noise"] else "no")
        lines.append(f"| {r['display_name']} | {r['n_combinations']} | {r['tuning_rows']:,} | {r['folds']} | {params} | {_fmt(r['baseline_score']['corrected_mean'])} | "
                     f"{_fmt(r['best_score']['corrected_mean'])} | {_fmt(r['improvement_corrected'])} | {flag} | {_fmt(r['best_score']['plain_mean'])} | {r['seconds'] / 60:.1f} |")
    lines += ["", "Corrected F1 = macro F1 with every held-out row weighted by `full_count / sample_count` of its class, i.e. an estimate for the real class mix. "
              "Plain F1 is measured on the balanced sample folds and is shown only for comparison. "
              "\"Beyond fold noise\": the mean improvement over the baseline is larger than the standard deviation of the paired fold differences."]

    obs = []
    improved = [r["display_name"] for r in results.values() if r["improvement_beyond_noise"]]
    same = [r["display_name"] for r in results.values() if r["best_is_baseline"]]
    unclear = [r["display_name"] for r in results.values() if not r["best_is_baseline"] and not r["improvement_beyond_noise"]]
    obs.append(f"Improved beyond fold noise ({len(improved)}): {', '.join(improved) or 'none'}.")
    obs.append(f"The baseline setting was already the best ({len(same)}): {', '.join(same) or 'none'}.")
    if unclear:
        obs.append(f"A different setting scored higher, but not beyond fold noise ({len(unclear)}): {', '.join(unclear)}.")
    for key, r in results.items():
        finding = class_weight_finding(r)
        if finding:
            obs.append(f"Class weights, {r['display_name']}: {finding}.")
    for key, r in results.items():
        edges = grid_edges(r, model_params.tuning_grid(key))
        if edges:
            obs.append(f"{r['display_name']}: winner at the edge of the grid ({'; '.join(edges)}); a wider grid might find more.")
    lines += ["", "## Observations (computed from the results)", ""] + [f"- {o}" for o in obs]

    lines += ["", "## Grid results per model (top 5 grid points; the baseline is always shown)", ""]
    for key, r in results.items():
        shown = [row for row in r["table"] if row["rank"] <= 5 or row["is_baseline"]]
        lines += [f"### {r['display_name']}", "", "| Rank | Parameters | Corrected F1 (mean) | Std | Plain F1 | Fit s | Note |", "|---|---|---|---|---|---|---|"]
        for row in shown:
            note = ", ".join(n for n, flag in (("best", row["is_best"]), ("baseline", row["is_baseline"])) if flag)
            lines.append(f"| {row['rank']} | {', '.join(f'{k}={v}' for k, v in row['params'].items())} | {_fmt(row['macro_f1_corrected_mean'])} | "
                         f"{_fmt(row['macro_f1_corrected_std'])} | {_fmt(row['macro_f1_plain_mean'])} | {row['fit_seconds_mean']:.1f} | {note} |")
        lines.append("")
        if r["failed_combinations"]:
            lines += [f"{r['failed_combinations']} grid point(s) failed and were left out.", ""]

    if refits:
        lines += ["## Tuned models (refitted on the same rows as their baselines)", "",
                  "| Model | Action | Training rows | Fit s | Baseline fit s | File MB | Note |", "|---|---|---|---|---|---|---|"]
        for key, f in refits.items():
            fit = "-" if f.get("fit_seconds") is None else "%.1f" % f["fit_seconds"]
            base = "-" if baseline_fit.get(key) is None else "%.1f" % baseline_fit[key]
            lines.append(f"| {model_params.DISPLAY_NAMES[key]} | {f['status']} | {_fmt(f.get('n_rows_used'), 'int')} | {fit} | {base} | {f.get('file_size_mb', '-')} | {f.get('note') or ''} |")
        lines.append("")
    if failed:
        lines += ["## Models that failed", ""] + [f"- {model_params.DISPLAY_NAMES.get(k, k)}: {why}" for k, why in failed.items()] + [""]
    lines += ["## Automatic checks", ""] + [f"- [{'x' if ok else ' '}] {name}" + (f" ({detail})" if detail else "") for name, ok, detail in checks]
    lines += ["", "## Limits", "",
              "- Parameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000) and refitted on more rows; they may not be optimal there.",
              "- The grids are small; a winner at the edge of its grid means a wider grid might do better.",
              "- Corrected F1 is an estimate for the real class mix from a modest number of rows of the large classes; it is noisy for the three rarest classes.",
              "- The test set was not read by this step. The tuned models are scored on it once, by `06_evaluate.py --stage tuned`."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the models with GridSearchCV and refit the winners.")
    parser.add_argument("--models", nargs="+", metavar="KEY", help=f"model keys (default: all). Keys: {', '.join(model_params.MODEL_KEYS)}")
    parser.add_argument("--skip-refit", action="store_true", help="only search the grids; do not train the tuned models")
    parser.add_argument("--rerun", action="store_true", help="ignore saved tuning results and search again")
    parser.add_argument("--sample-size", type=int, default=None, help="tuning sample size (default: config.model_params.TUNING_SAMPLE_SIZE; slow models stay capped)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)

    if args.sample_size is not None and args.sample_size < 1000:
        logger.error("--sample-size must be at least 1000")
        return 1
    keys = list(args.models) if args.models else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        logger.error("Unknown model key(s) %s. Known: %s", unknown, list(model_params.MODEL_KEYS))
        return 1

    try:
        settings.ensure_directories()
        data = load_training_data()
    except (DataNotReadyError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    baseline_fit: dict[str, float | None] = {}
    missing_baselines = []
    for key in keys:
        try:
            baseline_fit[key] = (model_store.load_metadata(key, "trained").get("model_info") or {}).get("fit_seconds")
        except model_store.ModelStoreError:
            missing_baselines.append(key)
    if missing_baselines:
        logger.error("No trained baseline for %s: run 05_train.py first (the tuned models are compared with their baselines).", missing_baselines)
        return 1

    print("Plan:")
    print(f"  {'model':<24}{'grid pts':>9}{'folds':>6}{'tuning rows':>13}{'fits':>7}  {'running':<13}saved tuning / tuned model")
    for key in keys:
        grid = model_params.tuning_grid(key)
        folds = model_params.tuning_folds(key)
        rows = min(model_params.tuning_sample_size(key, args.sample_size), data.n_rows)
        saved_t = (gs.tuning_directory() / f"tuning_{key}.json").is_file()
        saved_m = model_store.metadata_file(key, TUNED).is_file()
        mode = "one by one" if key in model_params.TUNING_SEQUENTIAL else "in parallel"
        print(f"  {model_params.DISPLAY_NAMES[key]:<24}{gs.n_combinations(grid):>9}{folds:>6}{rows:>13,}{gs.n_combinations(grid) * folds:>7}  {mode:<13}{'yes' if saved_t else 'no'} / {'yes' if saved_m else 'no'}")
    print("  The test set is not read by this script.\n")
    if args.dry_run:
        print("Dry run: nothing was run.")
        return 0

    started = time.time()
    failed: dict[str, str] = {}
    checks: list[tuple[str, bool, str]] = []
    refits: dict[str, dict[str, Any]] = {}
    try:
        results = gs.run_tuning(keys, data.X, data.y, data.n_classes, sample_size=args.sample_size, skip_existing=not args.rerun)
        ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
        failed = {k: r.get("error", "unknown") for k, r in results.items() if r.get("status") != "ok"}
        if not args.skip_refit:
            for position, (key, result) in enumerate(ok.items(), start=1):
                name = model_params.DISPLAY_NAMES[key]
                if not args.rerun and tuned_is_current(key, result):
                    logger.info("[%d/%d] %s: tuned model already saved for this tuning result", position, len(ok), name)
                    meta = model_store.load_metadata(key, TUNED)
                    refits[key] = {"key": key, "status": "already saved", "fit_seconds": (meta.get("model_info") or {}).get("fit_seconds"),
                                   "n_rows_used": (meta.get("extra") or {}).get("n_train_rows_used"), "file_size_mb": meta.get("file_size_mb"),
                                   "params": result["best_params"]}
                    continue
                logger.info("[%d/%d] refitting %s with %s", position, len(ok), name, result["best_params"])
                try:
                    refits[key] = refit_winner(key, result, data)
                except Exception as exc:  # noqa: BLE001 - one model must not stop the others
                    logger.error("%s refit FAILED: %s: %s", name, type(exc).__name__, exc)
                    failed[key] = f"refit failed: {type(exc).__name__}: {exc}"
            for key in refits:
                checks.extend(sanity_check(key, ok[key], data))
    except KeyboardInterrupt:
        logger.error("Interrupted. Saved tuning results and tuned models stay saved; rerun to continue.")
        return 130

    checks.insert(0, ("no tuning or refit failed", not failed, ", ".join(failed)))
    scope = {
        "training rows": f"{data.n_rows:,}", "classes": data.n_classes, "selection score": "prior-corrected macro F1 (plain macro F1 recorded beside it)",
        "tuning sample": "class-capped sample of the training rows (" + ", ".join(f"{model_params.DISPLAY_NAMES[k]} {model_params.tuning_sample_size(k, args.sample_size):,}" for k in keys) + ")",
        "random seed": settings.RANDOM_SEED, "test set": "not read",
    }
    report = build_report(ok, refits, checks, scope, baseline_fit, failed)
    out = settings.OPTIMIZATION_REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning_report.md").write_text(report, encoding="utf-8")
    table = gs.summary_table(ok)
    write_json({"generated_at": datetime.now().isoformat(timespec="seconds"), "scope": scope, "summary": table.to_dict(orient="records"),
                "refits": refits, "failed": failed, "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks]}, out / "tuning_summary.json")

    print()
    if not table.empty:
        print(table[["model", "best_params", "baseline_corrected_f1", "best_corrected_f1", "improvement", "beyond_noise", "best_is_baseline", "minutes"]].round(4).to_string(index=False))
    logger.info("Report: %s", out / "tuning_report.md")
    failed_checks = [name for name, passed, _ in checks if not passed]
    logger.info("Checks passed: %d/%d (total %.1f min)", len(checks) - len(failed_checks), len(checks), (time.time() - started) / 60)
    if failed_checks or failed:
        logger.error("Incomplete: failed checks %s, failed models %s", failed_checks, list(failed))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())