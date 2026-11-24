"""Step 6: cross-validate the models, score the test set ONCE, vote, compare and report.

Usage (from the project root, with the virtual environment active):

    python scripts/06_evaluate.py --dry-run                  # plan, time estimates, what is already saved; runs nothing
    python scripts/06_evaluate.py                            # everything, full 5-fold cross-validation (long)
    python scripts/06_evaluate.py --cv-sample-size 100000    # cross-validation on a 100,000-row class-capped sample
    python scripts/06_evaluate.py --skip-cv                  # only test-set scoring + report
    python scripts/06_evaluate.py --skip-test                # only cross-validation
    python scripts/06_evaluate.py --models random_forest xgboost   # only some models (the report then lists the others as missing)
    python scripts/06_evaluate.py --rescore                  # score the test set again (normally the saved scores are reused)
    python scripts/06_evaluate.py --stage tuned --dry-run    # plan for the tuned models (after 07_optimize.py)
    python scripts/06_evaluate.py --stage tuned              # score the 11 tuned models ONCE, compare with the baselines, 3 votes

``--stage tuned`` (needs the baseline test results of the default run and the tuned models of 07_optimize.py):

* scores the 11 tuned models on the test set once (``reports/evaluation/test_tuned/``), no cross-validation;
* shows every tuned model beside its baseline (test macro F1, delta, false alarms, tuned parameters);
* takes three majority votes: the 11 baseline models, the 11 tuned models, and the FINAL set;
* the final set is fixed by the Checkpoint 5 evidence rule, read from the tuned models' metadata
  (tuned where the tuning folds showed an improvement beyond fold noise, otherwise the baseline).  It is decided
  from tuning results only, never from test scores, and the script checks it against the set recorded in
  ``docs/project/checkpoint_5_decisions.md``;
* writes ``model_comparison_tuned.*``, ``evaluation_report_tuned.md``, ``evaluation_summary_tuned.json`` and ``figures_tuned/``.

What it does, in order (every stage saves its results and can be resumed after Ctrl+C):

1. **Cross-validation on TRAINING rows only** (stratified, 5 folds by default).  Per model:
   ``reports/evaluation/cross_validation/cv_<model>.json`` and the out-of-fold predictions.
2. **Cross-validation of the majority vote**: the vote of the 11 out-of-fold predictions, scored fold by fold.
3. **Test set, scored once**: every saved model predicts the held-out test rows; predictions, metrics and ROC / precision-recall
   curves are saved in ``reports/evaluation/test/``.  Saved results are reused on a rerun (``--rescore`` forces a new scoring),
   so the test set is not used to tune anything: model choices come from cross-validation only.
4. **Majority vote on the test set**: one equal vote per model, from the saved test predictions.
5. **Report**: ``reports/evaluation/model_comparison.md/.csv/.json``, ``per_class_f1.csv``, ``evaluation_report.md`` and the
   figures in ``reports/evaluation/figures/``.

Exit codes: 0 all good; 1 setup problem (data or models missing, bad option); 2 a model failed or a check failed; 130 interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.evaluation import comparator, cross_validation as cv, curves as curves_mod, metrics as M, plots
from nids.persistence import model_store
from nids.training.data_access import DataNotReadyError, class_capped_indices, load_test_data, load_training_data
from nids.utils.io_utils import load_array, read_json, save_array, write_json
from nids.utils.logger import get_logger
from nids.voting import majority_voting as voting

logger = get_logger("scripts.evaluate")

ENSEMBLE_KEY = comparator.ENSEMBLE_KEY
ENSEMBLE_NAME = comparator.ENSEMBLE_NAME
MAX_OVERLAID_MODELS = 4


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def test_directory(stage: str = "trained") -> Path:
    return settings.EVALUATION_REPORT_DIR / ("test" if stage == "trained" else "test_tuned")


def figure_directory(stage: str = "trained") -> Path:
    return settings.EVALUATION_REPORT_DIR / ("figures" if stage == "trained" else "figures_tuned")


def cv_directory_for(sample_size: int | None) -> Path:
    """Full cross-validation and sample cross-validation never overwrite each other."""
    base = cv.cv_directory()
    return base if sample_size is None else base / f"sample_{sample_size}"


# ---------------------------------------------------------------------------
# Time estimates (from the training metadata of YOUR machine; rough, linear in the number of rows)
# ---------------------------------------------------------------------------
def estimate_seconds(key: str, n_cv_rows: int, n_test_rows: int, folds: int, run_cv: bool, run_test: bool, stage: str = "trained") -> dict[str, float | None]:
    """Estimated seconds for cross-validation and test scoring of one model, or None when unknown."""
    try:
        meta = model_store.load_metadata(key, stage)
    except model_store.ModelStoreError:
        return {"cv": None, "test": None}
    extra = meta.get("extra") or {}
    fit = (meta.get("model_info") or {}).get("fit_seconds")
    used = extra.get("n_train_rows_used")
    per5000 = extra.get("predict_seconds_5000_train_rows")
    out: dict[str, float | None] = {"cv": None, "test": None}
    if per5000 is None:
        return out
    if run_cv and fit is not None and used:
        cap = model_params.subsample_size(key)
        fold_train = int(n_cv_rows * (folds - 1) / folds)
        rows = fold_train if cap is None else min(cap, fold_train)
        out["cv"] = fit * rows / used * folds + per5000 * n_cv_rows / 5000
    if run_test:
        out["test"] = 2 * per5000 * n_test_rows / 5000  # predict + predict_scores
    return out


def format_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


# ---------------------------------------------------------------------------
# Stage 1-2: cross-validation
# ---------------------------------------------------------------------------
def vote_cv_result(oof: dict[str, np.ndarray], y: np.ndarray, codec: Any, folds: list[tuple[np.ndarray, np.ndarray]], expected: list[str]) -> dict[str, Any]:
    """Cross-validation result of the majority vote, in the same format as a single model's result.

    The vote is taken over the out-of-fold predictions (each row was predicted by models that did not train on it),
    then scored fold by fold on the same folds as the models.  No training happens here.
    """
    vote = voting.vote_labels(oof, codec.n_classes, codec.benign_index, expected_keys=expected)
    rows = []
    for number, (_, test_pos) in enumerate(folds, start=1):
        m = M.compute_metrics(y[test_pos], vote.labels[test_pos], codec)
        view = m["attack_vs_normal"] or {}
        rows.append(
            {
                "fold": number,
                **{name: m["overall"][name] for name in cv.FOLD_METRICS},
                "detection_rate": view.get("detection_rate"),
                "false_alarm_rate": view.get("false_alarm_rate"),
                "fit_seconds": None,
                "predict_seconds": None,
                "rows_trained": None,
                "rows_scored": int(test_pos.size),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for name in (*cv.FOLD_METRICS, "detection_rate", "false_alarm_rate", "fit_seconds", "predict_seconds"):
        values = frame[name].dropna().to_numpy(dtype="float64")
        summary[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=1)) if values.size > 1 else None,
        }
    return {
        "key": ENSEMBLE_KEY,
        "display_name": ENSEMBLE_NAME,
        "status": "ok",
        "n_splits": len(folds),
        "n_rows": int(len(y)),
        "subsampled": False,
        "folds": rows,
        "summary": summary,
        "pooled": M.compute_metrics(y, vote.labels, codec),
        "members": list(vote.voters),
        "missing": dict(vote.missing),
        "vote": vote.summary(),
    }


def run_cross_validation(keys: list[str], train: Any, folds: int, sample_size: int | None) -> tuple[dict[str, dict[str, Any]], np.ndarray, Path]:
    """Stage 1 and 2.  Returns (results including the vote, the cross-validated labels, the folder)."""
    directory = cv_directory_for(sample_size)
    results = cv.run_cross_validation(
        keys, train.X, train.y, train.codec, n_splits=folds, sample_size=sample_size, skip_existing=True, directory=directory
    )
    y_cv = train.y if sample_size is None else train.y[class_capped_indices(train.y, sample_size)]
    ok = [k for k, r in results.items() if r.get("status") == "ok"]
    if len(ok) >= 2:
        oof = {k: cv.load_oof(k, directory) for k in ok}
        fold_pairs = cv.make_folds(y_cv, folds)
        results[ENSEMBLE_KEY] = vote_cv_result(oof, y_cv, train.codec, fold_pairs, expected=list(keys))
        write_json(results[ENSEMBLE_KEY], directory / f"cv_{ENSEMBLE_KEY}.json")
    else:
        logger.warning("Fewer than 2 models were cross-validated: no cross-validation of the majority vote.")
    return results, y_cv, directory


# ---------------------------------------------------------------------------
# Stage 3: test set, scored once
# ---------------------------------------------------------------------------
def test_fingerprint(key: str, n_rows: int, y_hash: str, stage: str = "trained") -> dict[str, Any]:
    meta = model_store.load_metadata(key, stage)
    return {"model_sha256": meta.get("sha256"), "model_saved_at": meta.get("saved_at"), "n_test_rows": int(n_rows), "y_test_sha256": y_hash}


def _paths(key: str, stage: str = "trained") -> tuple[Path, Path]:
    d = test_directory(stage)
    return d / f"{key}.json", d / f"{key}_pred.npy"


def score_model(key: str, X: np.ndarray, y: np.ndarray, codec: Any, y_hash: str, rescore: bool, stage: str = "trained") -> tuple[dict[str, Any] | None, str | None]:
    """Score one saved model on the test set (or reuse its saved result).  Returns ``(result, None)`` or ``(None, reason)``."""
    name = model_params.DISPLAY_NAMES[key]
    json_path, pred_path = _paths(key, stage)
    try:
        stamp = test_fingerprint(key, len(y), y_hash, stage)
    except model_store.ModelStoreError:
        return None, f"no saved {stage} model (run {'05_train.py' if stage == 'trained' else '07_optimize.py'})"
    if not rescore and json_path.is_file() and pred_path.is_file():
        saved = read_json(json_path)
        if saved.get("status") == "ok" and saved.get("fingerprint") == stamp:
            logger.info("%s: reusing the saved test-set result (the test set is scored once)", name)
            return saved, None
    try:
        model = model_store.load_model(key, stage)
    except model_store.StaleModelError as exc:
        return None, f"stale: {exc}"
    except model_store.ModelStoreError as exc:
        return None, f"cannot load: {exc}"
    try:
        logger.info("%s: scoring %d test rows", name, len(y))
        started = time.time()
        pred = np.asarray(model.predict(X))
        predict_seconds = time.time() - started
        started = time.time()
        scores = model.predict_scores(X)
        score_seconds = time.time() - started
        metrics = M.compute_metrics(y, pred, codec, scores=scores, score_kind=model.score_kind)
        try:
            curve_data = curves_mod.compute_curves(y, scores, codec, model.score_kind)
        except ValueError as exc:  # curves are optional; the metrics above are what matters
            logger.warning("%s: no curves (%s)", name, exc)
            curve_data = None
    except Exception as exc:  # noqa: BLE001 - one model failing must not stop the others
        logger.error("%s FAILED on the test set: %s: %s", name, type(exc).__name__, exc)
        return None, f"failed while scoring: {type(exc).__name__}: {exc}"
    metrics.update({"model_key": key, "predict_seconds": round(predict_seconds, 3), "score_seconds": round(score_seconds, 3)})
    result = {
        "key": key,
        "display_name": name,
        "status": "ok",
        "fingerprint": stamp,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "curves": curve_data,
    }
    test_directory(stage).mkdir(parents=True, exist_ok=True)
    save_array(pred.astype("int16"), pred_path)
    write_json(result, json_path)  # last: marks the result as complete
    logger.info("%s: macro F1 %.4f, accuracy %.4f (predict %.1f s, scores %.1f s)", name, metrics["overall"]["macro_f1"], metrics["overall"]["accuracy"], predict_seconds, score_seconds)
    return result, None


# ---------------------------------------------------------------------------
# Stage 4: vote on the test set
# ---------------------------------------------------------------------------
def vote_on_test(scored: dict[str, dict[str, Any]], reasons: dict[str, str], y: np.ndarray, codec: Any, expected: list[str]) -> dict[str, Any] | None:
    if len(scored) < 2:
        logger.warning("Fewer than 2 scored models: no majority vote on the test set.")
        return None
    preds = {k: load_array(_paths(k)[1]).astype("int64") for k in scored}
    vote = voting.vote_labels(preds, codec.n_classes, codec.benign_index, expected_keys=expected, missing=reasons)
    metrics = M.compute_metrics(y, vote.labels, codec)
    metrics.update({"model_key": ENSEMBLE_KEY, "members": list(vote.voters), "missing_members": dict(vote.missing), "vote": vote.summary()})
    result = {"key": ENSEMBLE_KEY, "display_name": ENSEMBLE_NAME, "status": "ok", "scored_at": datetime.now().isoformat(timespec="seconds"), "metrics": metrics, "curves": None}
    test_directory().mkdir(parents=True, exist_ok=True)
    save_array(vote.labels.astype("int16"), _paths(ENSEMBLE_KEY)[1])
    write_json(result, _paths(ENSEMBLE_KEY)[0])
    logger.info("Majority vote of %d models: macro F1 %.4f, accuracy %.4f", vote.n_voters, metrics["overall"]["macro_f1"], metrics["overall"]["accuracy"])
    return result


# ---------------------------------------------------------------------------
# Stage 5: report
# ---------------------------------------------------------------------------
def _md_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    """Markdown table from ``(column, heading, format)``; ``format`` is 'f4', 'pct', 'int', 'text' or 'sec'."""
    def cell(value: Any, fmt: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return {"f4": lambda v: f"{v:.4f}", "pct": lambda v: f"{v * 100:.2f}%", "int": lambda v: f"{int(v):,}", "sec": lambda v: f"{v:.1f}"}.get(fmt, str)(value)

    lines = ["| " + " | ".join(h for _, h, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(cell(row[c], f) for c, _, f in columns) + " |")
    return lines


def observations(frame: pd.DataFrame, cv_rank: pd.DataFrame | None, cv_is_sample: bool = False) -> list[str]:
    """Facts computed from the numbers (no interpretation beyond them)."""
    out: list[str] = []
    singles = frame[frame["kind"] == "single"]
    if singles.empty:
        return out
    best = singles.iloc[0]
    out.append(f"Best single model on the test set by macro F1: {best['model']} ({best['macro_f1']:.4f}).")
    vote = frame[frame["kind"] == "ensemble"]
    if not vote.empty:
        v = vote.iloc[0]
        diff = v["macro_f1"] - best["macro_f1"]
        if abs(diff) < 5e-5:
            out.append(f"The majority vote has macro F1 {v['macro_f1']:.4f}: equal to the best single model (to 4 decimals).")
        else:
            out.append(f"The majority vote has macro F1 {v['macro_f1']:.4f}: {abs(diff):.4f} {'above' if diff > 0 else 'below'} the best single model.")
        weaker = singles[singles["macro_f1"] < v["macro_f1"] - 0.10]
        if len(weaker):
            out.append(f"{len(weaker)} single model(s) score more than 0.10 macro F1 below the vote: {', '.join(weaker['model'])}.")
    if cv_rank is not None and not cv_rank.empty and "cv_macro_f1_mean" in frame:
        gap = (frame["macro_f1"] - frame["cv_macro_f1_mean"]).abs()
        far = frame[gap > 0.05]
        if len(far):
            out.append("Test macro F1 differs from the cross-validation mean by more than 0.05 for: " + ", ".join(far["model"]) +
                       (" (the cross-validation sample has a different class mix than the test set; see the notes)." if cv_is_sample else
                        " (each cross-validation model trains on 1 - 1/folds of the training rows and each fold holds only that share of every rare class; these numbers alone do not say which explains the gap)."))
        else:
            out.append("Test macro F1 is within 0.05 of the cross-validation mean for every model.")
    return out


def confusion_lines(metrics: dict[str, Any], n: int = 8) -> list[str]:
    """The most frequent wrong (true -> predicted) pairs, as Markdown list lines."""
    lines = []
    for c in metrics["top_confusions"][:n]:
        share = c.get("share_of_true_class")
        lines.append(f"- {c['true_class']} -> {c['predicted_class']}: {c['rows']:,} flows" + (f" ({share * 100:.2f}% of that class)" if share is not None else ""))
    return lines


def build_report(
    frame: pd.DataFrame,
    notes: list[str],
    results: dict[str, dict[str, Any]],
    cv_results: dict[str, dict[str, Any]] | None,
    cv_rank: pd.DataFrame | None,
    scope: dict[str, Any],
    checks: list[tuple[str, bool, str]],
    missing: dict[str, str],
) -> str:
    lines = ["# Evaluation report", "", f"Generated {datetime.now().isoformat(timespec='seconds')}.", ""]
    lines += ["## Scope", ""]
    lines += [f"- {k}: {v}" for k, v in scope.items()]
    lines += ["", "## Test-set comparison", ""]
    cols = [("rank", "Rank", "int"), ("model", "Model", "text"), ("macro_f1", "Macro F1", "f4"), ("macro_f1_reliable", "Macro F1 (classes >= 30 rows)", "f4"),
            ("balanced_accuracy", "Balanced acc.", "f4"), ("accuracy", "Accuracy", "f4"), ("mcc", "MCC", "f4"),
            ("detection_rate", "Detection rate", "pct"), ("false_alarm_rate", "False alarm rate", "pct"), ("roc_auc_macro", "ROC-AUC (macro)", "f4")]
    if "cv_macro_f1_mean" in frame:
        cols += [("cv_macro_f1_mean", "CV macro F1", "f4"), ("cv_macro_f1_std", "CV std", "f4")]
    cols += [("train_rows", "Training rows", "int"), ("subsample", "Subsample", "text")]
    shown = frame.copy()
    shown["subsample"] = shown["trained_on_subsample"].map({True: "yes", False: "no"}).fillna("-")
    lines += _md_table(shown, cols)
    lines += ["", "### Notes", ""] + [f"{i}. {n}" for i, n in enumerate(notes, 1)]
    obs = observations(frame, cv_rank, cv_is_sample="sample" in str(scope.get("cross-validation", "")))
    if obs:
        lines += ["", "### Observations (computed from the table above)", ""] + [f"- {o}" for o in obs]

    if cv_rank is not None and not cv_rank.empty:
        lines += ["", "## Cross-validation (training rows only)", ""]
        lines += [f"{cv_results[next(k for k, r in cv_results.items() if r.get('status') == 'ok')]['n_splits']}-fold, stratified. "
                  f"`indistinguishable_from_best` means within one standard deviation of the best mean: the folds cannot separate them.", ""]
        lines += _md_table(cv_rank, [("rank", "Rank", "int"), ("model", "Model", "text"), ("macro_f1_mean", "Macro F1 (mean)", "f4"), ("macro_f1_std", "Std", "f4"),
                                     ("accuracy_mean", "Accuracy", "f4"), ("detection_rate_mean", "Detection rate", "pct"), ("false_alarm_rate_mean", "False alarm rate", "pct"),
                                     ("fit_seconds_mean", "Fit (s)", "sec"), ("indistinguishable_from_best", "Within 1 std of best", "text")])
    failed_cv = [k for k, r in (cv_results or {}).items() if r.get("status") == "failed"]
    if failed_cv:
        lines += ["", "Cross-validation failed for: " + ", ".join(f"{k} ({cv_results[k].get('error')})" for k in failed_cv) + "."]

    best_key = frame[frame["kind"] == "single"]["key"].iloc[0] if (frame["kind"] == "single").any() else None
    if best_key:
        m = results[best_key]
        lines += ["", f"## Best single model: {m['display_name']}", "", "### Most frequent confusions (true -> predicted)", ""]
        lines += confusion_lines(m) or ["- none (no wrong predictions)"]
        lines += ["", "### Classes with unreliable scores (fewer than 30 test rows)", ""]
        low = [c for c in m["per_class"] if c["support"] > 0 and c["low_support"]]
        lines += [f"- {c['class']}: {c['support']} test rows, precision {c['precision']:.3f}, recall {c['recall']:.3f}, F1 {c['f1']:.3f}" for c in low] or ["- none"]
    if missing:
        lines += ["", "## Models not in the comparison", ""] + [f"- {model_params.DISPLAY_NAMES.get(k, k)}: {why}" for k, why in missing.items()]
    lines += ["", "## Automatic checks", ""] + [f"- [{'x' if ok else ' '}] {name}" + (f" ({detail})" if detail else "") for name, ok, detail in checks]
    lines += ["", "## Limits", "",
              "- The three rarest classes have 2 to 7 test rows: their scores are shown with their support and must not be over-read.",
              "- The train/test split is stratified and random; flows of the same attack campaign can appear on both sides, so the scores can be optimistic compared with a temporal split.",
              "- KNN, SVC and Gradient Boosting were trained on class-capped subsamples; their numbers are not directly comparable (see the notes).",
              "- Test-set results were not used to choose or tune any model."]
    return "\n".join(lines) + "\n"


def make_figures(frame: pd.DataFrame, results: dict[str, dict[str, Any]], cv_results: dict[str, dict[str, Any]] | None, cv_rank: pd.DataFrame | None) -> list[Path]:
    out_dir = figure_directory()
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def attempt(label: str, fn: Any) -> None:
        try:
            made.append(fn())
        except Exception as exc:  # noqa: BLE001 - a missing picture must not hide the numbers
            logger.warning("Figure %s skipped: %s: %s", label, type(exc).__name__, exc)

    plot_frame = frame.copy()
    plot_frame["trained_on_subsample"] = plot_frame["trained_on_subsample"].map(lambda v: bool(v) if v is not None and v == v else False)
    attempt("test macro F1", lambda: plots.plot_model_comparison(plot_frame, out_dir / "test_macro_f1.png", value="macro_f1", error=None, flag_column=None,
                                                                    xlabel="Macro F1 on the test set", title="Macro F1 on the test set"))
    attempt("detection", lambda: plots.plot_detection_and_false_alarms(plot_frame, out_dir / "test_detection_false_alarms.png", detection="detection_rate", false_alarm="false_alarm_rate",
                                                                         title="Attack detection and false alarms on the test set"))
    if cv_rank is not None and not cv_rank.empty:
        attempt("cv macro F1", lambda: plots.plot_model_comparison(cv_rank, out_dir / "cv_macro_f1.png", title="Macro F1 in cross-validation (training rows)"))
        attempt("cv folds", lambda: plots.plot_fold_spread(cv_results, out_dir / "cv_fold_spread.png"))
    for key, r in results.items():
        m = r["metrics"]
        attempt(f"confusion {key}", lambda m=m, key=key: plots.plot_confusion_matrix(m, out_dir / f"confusion_{key}.png", title=f"{r['display_name']}: test set"))
        attempt(f"per class {key}", lambda m=m, key=key, r=r: plots.plot_per_class_metrics(m, out_dir / f"per_class_{key}.png", title=f"{r['display_name']}: precision and recall per class"))
    with_curves = [(key, r) for key, r in results.items() if r.get("curves")]
    ranked = [row["key"] for _, row in frame.iterrows() if row["key"] in dict(with_curves)]
    if ranked:
        top = ranked[0]
        top_curves = results[top]["curves"]
        attempt("roc per class", lambda: curves_mod.plot_class_curves(top_curves, out_dir / f"curves_roc_{top}.png", "roc", title=f"{results[top]['display_name']}: ROC per class (test set)"))
        attempt("pr per class", lambda: curves_mod.plot_class_curves(top_curves, out_dir / f"curves_pr_{top}.png", "pr", title=f"{results[top]['display_name']}: precision-recall per class (test set)"))
        chosen = {results[k]["display_name"]: results[k]["curves"] for k in ranked[:MAX_OVERLAID_MODELS]}
        attempt("roc macro", lambda: curves_mod.plot_macro_curves(chosen, out_dir / "curves_roc_macro.png", "roc"))
        attempt("pr macro", lambda: curves_mod.plot_macro_curves(chosen, out_dir / "curves_pr_macro.png", "pr"))
    return made


def automatic_checks(results: dict[str, dict[str, Any]], n_test: int, expected: list[str], missing: dict[str, str]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("all 11 models scored on the test set", not missing and all(k in results for k in expected), ", ".join(missing)))
    rows = {k: r["metrics"]["n_rows"] for k, r in results.items()}
    checks.append(("every row of the table is scored on the same test rows", len(set(rows.values())) <= 1 and (not rows or n_test in rows.values()), str(sorted(set(rows.values())))))
    supports = {tuple(c["support"] for c in r["metrics"]["per_class"]) for r in results.values()}
    checks.append(("class supports are identical for every model", len(supports) <= 1, ""))
    bad = [k for k, r in results.items() if not np.isfinite(r["metrics"]["overall"]["macro_f1"])]
    checks.append(("no missing or infinite macro F1", not bad, ", ".join(bad)))
    return checks



# ---------------------------------------------------------------------------
# --stage tuned: the 11 tuned models, scored once, beside the baselines
# ---------------------------------------------------------------------------
BASELINE_VOTE_KEY = "majority_vote_baseline"
TUNED_VOTE_KEY = "majority_vote_tuned"
FINAL_VOTE_KEY = "majority_vote_final"
VOTE_NAMES = {
    BASELINE_VOTE_KEY: "Vote of the 11 baseline models",
    TUNED_VOTE_KEY: "Vote of the 11 tuned models",
    FINAL_VOTE_KEY: "Vote of the final set (evidence rule)",
}
# The final set as recorded in docs/project/checkpoint_5_decisions.md BEFORE any tuned model had a test score.
# The set actually used is read from the tuned models' metadata; the script checks that the two agree.
CHECKPOINT_5_TUNED = ("knn", "decision_tree", "svm", "bernoulli_nb", "sgd", "adaboost", "gradient_boosting")


def final_set(keys: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """The evidence rule: ``(source, reasons)`` where ``source[key]`` is "tuned" or "baseline".

    A model uses its tuned version when the tuning folds showed an improvement beyond fold noise and the tuned
    parameters differ from the baseline.  The rule reads the tuning record stored in the tuned model's metadata;
    it never looks at a test score.
    """
    source: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for key in keys:
        try:
            tuning = (model_store.load_metadata(key, "tuned").get("extra") or {}).get("tuning") or {}
        except model_store.ModelStoreError:
            source[key], reasons[key] = "baseline", "no tuned model saved"
            continue
        if tuning.get("same_as_baseline"):
            source[key], reasons[key] = "baseline", "the baseline setting was already the best"
        elif tuning.get("improvement_beyond_noise"):
            source[key], reasons[key] = "tuned", "tuning improvement beyond fold noise"
        else:
            source[key], reasons[key] = "baseline", "tuning improvement within fold noise"
    return source, reasons


def changed_params(key: str) -> str:
    """Tuned parameters that differ from the baseline setting, as text."""
    try:
        tuning = (model_store.load_metadata(key, "tuned").get("extra") or {}).get("tuning") or {}
    except model_store.ModelStoreError:
        return "-"
    best, base = tuning.get("best_params") or {}, tuning.get("baseline_params") or {}
    diff = {k: v for k, v in best.items() if base.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in diff.items()) or "same as baseline"


def save_vote(vote_key: str, preds: dict[str, np.ndarray], y: np.ndarray, codec: Any, expected: list[str], missing: dict[str, str], sources: dict[str, str]) -> dict[str, Any]:
    """Vote over saved predictions and save the labels and metrics under ``test_tuned/``."""
    vote = voting.vote_labels(preds, codec.n_classes, codec.benign_index, expected_keys=expected, missing=missing)
    metrics = M.compute_metrics(y, vote.labels, codec)
    metrics.update({"model_key": vote_key, "members": list(vote.voters), "member_sources": dict(sources), "missing_members": dict(vote.missing), "vote": vote.summary()})
    result = {"key": vote_key, "display_name": VOTE_NAMES[vote_key], "status": "ok", "scored_at": datetime.now().isoformat(timespec="seconds"), "metrics": metrics, "curves": None}
    test_directory("tuned").mkdir(parents=True, exist_ok=True)
    save_array(vote.labels.astype("int16"), _paths(vote_key, "tuned")[1])
    write_json(result, _paths(vote_key, "tuned")[0])
    logger.info("%s: %d voters, macro F1 %.4f, accuracy %.4f", VOTE_NAMES[vote_key], vote.n_voters, metrics["overall"]["macro_f1"], metrics["overall"]["accuracy"])
    return result


def baseline_test_results(keys: list[str], n_rows: int, y_hash: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """The saved test results of the baseline models, accepted only when they used the same test rows."""
    found: dict[str, dict[str, Any]] = {}
    problems: dict[str, str] = {}
    for key in keys:
        json_path, pred_path = _paths(key, "trained")
        if not (json_path.is_file() and pred_path.is_file()):
            problems[key] = "no baseline test result (run `python scripts/06_evaluate.py` first)"
            continue
        saved = read_json(json_path)
        stamp = saved.get("fingerprint") or {}
        if saved.get("status") != "ok":
            problems[key] = "baseline test result is not complete"
        elif stamp.get("y_test_sha256") != y_hash or stamp.get("n_test_rows") != n_rows:
            problems[key] = "baseline was scored on different test rows"
        else:
            found[key] = saved
    return found, problems


def tuned_table(keys: list[str], base: dict[str, dict[str, Any]], tuned: dict[str, dict[str, Any]], source: dict[str, str], reasons: dict[str, str]) -> pd.DataFrame:
    """One row per model: baseline and tuned test scores side by side (the final-set choice is shown, not made, here)."""
    rows = []
    for key in keys:
        b = base[key]["metrics"] if key in base else None
        t = tuned[key]["metrics"] if key in tuned else None

        def get(m: dict[str, Any] | None, *path: str) -> float | None:
            if m is None:
                return None
            node: Any = m
            for p in path:
                node = (node or {}).get(p)
            return node

        rows.append({
            "key": key,
            "model": model_params.DISPLAY_NAMES[key],
            "final_set_uses": source.get(key),
            "rule_reason": reasons.get(key),
            "tuned_parameters": changed_params(key),
            "baseline_macro_f1": get(b, "overall", "macro_f1"),
            "tuned_macro_f1": get(t, "overall", "macro_f1"),
            "baseline_macro_f1_reliable": comparator._reliable_macro_f1(b) if b else None,
            "tuned_macro_f1_reliable": comparator._reliable_macro_f1(t) if t else None,
            "baseline_accuracy": get(b, "overall", "accuracy"),
            "tuned_accuracy": get(t, "overall", "accuracy"),
            "baseline_detection_rate": get(b, "attack_vs_normal", "detection_rate"),
            "tuned_detection_rate": get(t, "attack_vs_normal", "detection_rate"),
            "baseline_false_alarm_rate": get(b, "attack_vs_normal", "false_alarm_rate"),
            "tuned_false_alarm_rate": get(t, "attack_vs_normal", "false_alarm_rate"),
        })
    frame = pd.DataFrame(rows)
    frame["delta_macro_f1"] = frame["tuned_macro_f1"] - frame["baseline_macro_f1"]
    return frame


def vote_table(votes: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, r in votes.items():
        m = r["metrics"]
        rows.append({
            "key": key, "vote": r["display_name"], "voters": len(m["members"]),
            "macro_f1": m["overall"]["macro_f1"], "macro_f1_reliable": comparator._reliable_macro_f1(m),
            "balanced_accuracy": m["overall"]["balanced_accuracy"], "accuracy": m["overall"]["accuracy"], "mcc": m["overall"]["mcc"],
            "detection_rate": (m.get("attack_vs_normal") or {}).get("detection_rate"),
            "false_alarm_rate": (m.get("attack_vs_normal") or {}).get("false_alarm_rate"),
            "tied_fraction": m["vote"]["tied_fraction"], "unanimous_fraction": m["vote"]["unanimous_fraction"],
        })
    return pd.DataFrame(rows)


def tuned_observations(models: pd.DataFrame, votes: pd.DataFrame) -> list[str]:
    """Facts computed from the tables (no interpretation beyond them)."""
    out: list[str] = []
    ok = models.dropna(subset=["delta_macro_f1"])
    if not ok.empty:
        better = ok[ok["delta_macro_f1"] > 0.005]
        worse = ok[ok["delta_macro_f1"] < -0.005]
        flat = ok[ok["delta_macro_f1"].abs() <= 0.005]
        out.append(f"Test macro F1, tuned minus baseline: {len(better)} model(s) higher by more than 0.005, {len(worse)} lower by more than 0.005, {len(flat)} within 0.005.")
        if len(worse):
            out.append("Tuned models that scored LOWER than their baseline on the test set: " + ", ".join(f"{r.model} ({r.delta_macro_f1:+.4f})" for r in worse.itertuples()) + ". The tuning gain did not carry over to these rows.")
        used_tuned = ok[ok["final_set_uses"] == "tuned"]
        if len(used_tuned):
            out.append(f"Of the {len(used_tuned)} model(s) the evidence rule takes in tuned form, {int((used_tuned['delta_macro_f1'] > 0).sum())} also scored higher on the test set than their baseline.")
        skipped = ok[ok["final_set_uses"] == "baseline"]
        if len(skipped):
            out.append("Models kept at baseline by the rule (tuned minus baseline on the test set): " + ", ".join(f"{r.model} ({r.delta_macro_f1:+.4f})" for r in skipped.itertuples()) + ". The rule was fixed before these scores existed and is not changed by them.")
    if not votes.empty:
        v = votes.set_index("key")
        for a, b, text in ((FINAL_VOTE_KEY, BASELINE_VOTE_KEY, "final-set vote vs baseline vote"), (TUNED_VOTE_KEY, BASELINE_VOTE_KEY, "tuned vote vs baseline vote")):
            if a in v.index and b in v.index:
                d = v.loc[a, "macro_f1"] - v.loc[b, "macro_f1"]
                out.append(f"Macro F1, {text}: {v.loc[a, 'macro_f1']:.4f} vs {v.loc[b, 'macro_f1']:.4f} ({d:+.4f}).")
        if FINAL_VOTE_KEY in v.index and not ok.empty:
            top = ok.loc[ok["tuned_macro_f1"].idxmax()]
            out.append(f"Best single tuned model on the test set: {top['model']} ({top['tuned_macro_f1']:.4f}); final-set vote {v.loc[FINAL_VOTE_KEY, 'macro_f1']:.4f}.")
    return out


def tuned_checks(keys: list[str], tuned: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]], n_test: int, source: dict[str, str], missing: dict[str, str], final_missing: dict[str, str]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("all 11 tuned models scored on the test set", not missing and all(k in tuned for k in keys), ", ".join(missing)))
    checks.append(("all 11 baseline test results found for the same test rows", all(k in base for k in keys), ", ".join(k for k in keys if k not in base)))
    used_tuned = tuple(k for k in keys if source.get(k) == "tuned")
    checks.append(("final set equals the set recorded in Checkpoint 5 (decided before any tuned test score)", set(used_tuned) == set(CHECKPOINT_5_TUNED), "rule gave: " + ", ".join(used_tuned)))
    checks.append(("final-set vote has all 11 voters", not final_missing, ", ".join(final_missing)))
    rows = {r["metrics"]["n_rows"] for r in [*tuned.values(), *base.values()]}
    checks.append(("every score is on the same test rows", rows <= {n_test}, str(sorted(rows))))
    supports = {tuple(c["support"] for c in r["metrics"]["per_class"]) for r in [*tuned.values(), *base.values()]}
    checks.append(("class supports are identical for every model", len(supports) <= 1, ""))
    bad = [k for k, r in tuned.items() if not np.isfinite(r["metrics"]["overall"]["macro_f1"])]
    checks.append(("no missing or infinite macro F1", not bad, ", ".join(bad)))
    return checks


def build_tuned_report(models: pd.DataFrame, votes: pd.DataFrame, obs: list[str], scope: dict[str, Any], checks: list[tuple[str, bool, str]],
                       final_metrics: dict[str, Any] | None, missing: dict[str, str]) -> str:
    lines = ["# Evaluation report: tuned models", "", f"Generated {datetime.now().isoformat(timespec='seconds')}.", ""]
    lines += ["## Scope", ""] + [f"- {k}: {v}" for k, v in scope.items()]
    lines += ["", "## How the final set was chosen", "",
              "The final set (used by the application) follows the Checkpoint 5 evidence rule: a model uses its tuned version when the "
              "tuning folds showed an improvement beyond fold noise, otherwise its baseline. The rule used tuning results only and was "
              "fixed before any tuned model had a test score. The test results below are reported beside it and do not change it.", ""]
    lines += _md_table(models, [("model", "Model", "text"), ("final_set_uses", "Final set uses", "text"), ("rule_reason", "Reason (tuning evidence)", "text")])
    lines += ["", "## Tuned vs baseline on the test set (each scored once)", ""]
    lines += _md_table(models, [("model", "Model", "text"), ("tuned_parameters", "Tuned parameters (changed)", "text"), ("baseline_macro_f1", "Baseline macro F1", "f4"),
                                ("tuned_macro_f1", "Tuned macro F1", "f4"), ("delta_macro_f1", "Delta", "f4"),
                                ("baseline_false_alarm_rate", "Baseline false alarms", "pct"), ("tuned_false_alarm_rate", "Tuned false alarms", "pct"),
                                ("baseline_detection_rate", "Baseline detection", "pct"), ("tuned_detection_rate", "Tuned detection", "pct")])
    lines += ["", "## Majority votes (one equal vote per model; ties: " + settings.TIE_BREAK_RULE + ")", ""]
    lines += _md_table(votes, [("vote", "Vote", "text"), ("voters", "Voters", "int"), ("macro_f1", "Macro F1", "f4"), ("macro_f1_reliable", "Macro F1 (classes >= 30 rows)", "f4"),
                               ("balanced_accuracy", "Balanced acc.", "f4"), ("accuracy", "Accuracy", "f4"), ("mcc", "MCC", "f4"),
                               ("detection_rate", "Detection", "pct"), ("false_alarm_rate", "False alarms", "pct"), ("tied_fraction", "Tied rows", "pct"), ("unanimous_fraction", "Unanimous rows", "pct")])
    if obs:
        lines += ["", "### Observations (computed from the tables above)", ""] + [f"- {o}" for o in obs]
    if final_metrics:
        lines += ["", "## Final-set vote: most frequent confusions (true -> predicted)", ""] + (confusion_lines(final_metrics) or ["- none"])
        lines += ["", "### Classes with unreliable scores (fewer than 30 test rows)", ""]
        low = [c for c in final_metrics["per_class"] if c["support"] > 0 and c["low_support"]]
        lines += [f"- {c['class']}: {c['support']} test rows, precision {c['precision']:.3f}, recall {c['recall']:.3f}, F1 {c['f1']:.3f}" for c in low] or ["- none"]
    if missing:
        lines += ["", "## Tuned models not in the comparison", ""] + [f"- {model_params.DISPLAY_NAMES.get(k, k)}: {why}" for k, why in missing.items()]
    lines += ["", "## Automatic checks", ""] + [f"- [{'x' if ok else ' '}] {name}" + (f" ({detail})" if detail else "") for name, ok, detail in checks]
    lines += ["", "## Limits", "",
              "- The three rarest classes have 2 to 7 test rows: their scores must not be over-read.",
              "- Parameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000); some winners sit at the edge of their grid.",
              "- KNN, SVC and Gradient Boosting are trained on class-capped subsamples (their class prior is shifted).",
              "- The split is stratified and random; the scores can be optimistic compared with a temporal split.",
              "- Test-set scores were not used to choose the final set."]
    return "\n".join(lines) + "\n"


def make_tuned_figures(models: pd.DataFrame, final_result: dict[str, Any] | None) -> list[Path]:
    out_dir = figure_directory("tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    try:
        ok = models.dropna(subset=["baseline_macro_f1", "tuned_macro_f1"]).sort_values("tuned_macro_f1")
        fig, ax = plots._new_figure(9, 0.55 * len(ok) + 1.8)
        pos = np.arange(len(ok))
        ax.barh(pos + 0.19, ok["baseline_macro_f1"], height=0.36, color=plots.ORANGE, label="Baseline")
        ax.barh(pos - 0.19, ok["tuned_macro_f1"], height=0.36, color=plots.BLUE, label="Tuned")
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{m}{' *' if u == 'tuned' else ''}" for m, u in zip(ok["model"], ok["final_set_uses"])], color=plots.INK)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Macro F1 on the test set   (* = tuned version used by the final set)", color=plots.INK_SECONDARY)
        ax.set_title("Baseline and tuned models on the test set", color=plots.INK, loc="left")
        plots._style_axes(ax, "x")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
        made.append(plots._save(fig, out_dir / "tuned_vs_baseline_macro_f1.png"))
    except Exception as exc:  # noqa: BLE001 - a missing picture must not hide the numbers
        logger.warning("Figure tuned_vs_baseline skipped: %s: %s", type(exc).__name__, exc)
    if final_result:
        m = final_result["metrics"]
        for name, fn in (("confusion", plots.plot_confusion_matrix), ("per_class", plots.plot_per_class_metrics)):
            try:
                made.append(fn(m, out_dir / f"{name}_{FINAL_VOTE_KEY}.png", title="Final-set vote: test set"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Figure %s skipped: %s: %s", name, type(exc).__name__, exc)
    return made


def run_tuned_stage(args: argparse.Namespace) -> int:
    keys = list(args.models) if args.models else list(model_params.MODEL_KEYS)
    expected = list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in expected]
    if unknown:
        logger.error("Unknown model key(s) %s. Known: %s", unknown, expected)
        return 1
    if keys != expected:
        logger.error("--stage tuned always scores all 11 models (the final set needs every one of them); do not use --models.")
        return 1
    try:
        settings.ensure_directories()
        y_path = settings.Y_TEST_FILE
        n_test = int(np.load(y_path, mmap_mode="r").shape[0]) if y_path.is_file() else 0
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
    source, reasons = final_set(keys)

    print("Plan (tuned stage; no cross-validation, the test set is scored once per model):")
    print(f"  test set: {n_test:,} rows")
    print(f"  {'model':<24} {'final set uses':<15} {'test (est.)':>12}   saved tuned / baseline test result")
    total = 0.0
    for key in keys:
        est = estimate_seconds(key, 0, n_test, 2, False, True, stage="tuned")
        total += est["test"] or 0.0
        print(f"  {model_params.DISPLAY_NAMES[key]:<24} {source[key]:<15} {format_minutes(est['test']):>12}   "
              f"{'yes' if _paths(key, 'tuned')[0].is_file() else 'no':>3} / {'yes' if _paths(key, 'trained')[0].is_file() else 'no'}")
    print(f"  {'total (rough)':<24} {'':<15} {format_minutes(total):>12}")
    print(f"  final set by the evidence rule: {sum(v == 'tuned' for v in source.values())} tuned + {sum(v == 'baseline' for v in source.values())} baseline")
    print()
    if args.dry_run:
        print("Dry run: nothing was run.")
        return 0

    started = time.time()
    try:
        test = load_test_data(mmap=True)
        X_test, y_test = np.asarray(test.X), test.y
        y_hash = hashlib.sha256(np.ascontiguousarray(y_test).tobytes()).hexdigest()
        base, base_problems = baseline_test_results(keys, len(y_test), y_hash)
        if base_problems:
            logger.error("Baseline test results are needed for the comparison: %s", base_problems)
            return 1

        tuned: dict[str, dict[str, Any]] = {}
        missing: dict[str, str] = {}
        for position, key in enumerate(keys, start=1):
            logger.info("[%d/%d] %s (tuned)", position, len(keys), model_params.DISPLAY_NAMES[key])
            result, reason = score_model(key, X_test, y_test, test.codec, y_hash, args.rescore, stage="tuned")
            if result is None:
                missing[key] = reason or "unknown"
            else:
                tuned[key] = result

        # three votes, all from saved test predictions
        base_preds = {k: load_array(_paths(k, "trained")[1]).astype("int64") for k in keys}
        tuned_preds = {k: load_array(_paths(k, "tuned")[1]).astype("int64") for k in tuned}
        votes: dict[str, dict[str, Any]] = {}
        votes[BASELINE_VOTE_KEY] = save_vote(BASELINE_VOTE_KEY, base_preds, y_test, test.codec, expected, {}, {k: "baseline" for k in keys})
        if len(tuned_preds) >= 2:
            votes[TUNED_VOTE_KEY] = save_vote(TUNED_VOTE_KEY, tuned_preds, y_test, test.codec, expected, missing, {k: "tuned" for k in tuned_preds})
        final_preds, final_missing = {}, {}
        for key in keys:
            if source[key] == "tuned":
                if key in tuned_preds:
                    final_preds[key] = tuned_preds[key]
                else:
                    final_missing[key] = f"tuned model not scored ({missing.get(key, 'unknown')})"
            else:
                final_preds[key] = base_preds[key]
        if final_preds:
            votes[FINAL_VOTE_KEY] = save_vote(FINAL_VOTE_KEY, final_preds, y_test, test.codec, expected, final_missing, {k: source[k] for k in final_preds})

        models = tuned_table(keys, base, tuned, source, reasons)
        vote_frame = vote_table(votes)
        obs = tuned_observations(models, vote_frame)
        checks = tuned_checks(keys, tuned, base, len(y_test), source, missing, final_missing)
        scope = {"test rows": f"{len(y_test):,}", "classes": test.codec.n_classes, "tie rule of the vote": settings.TIE_BREAK_RULE, "random seed": settings.RANDOM_SEED,
                 "cross-validation": "not repeated for tuned models (tuning used cross-validation on training rows only)"}
        directory = settings.EVALUATION_REPORT_DIR
        models.to_csv(directory / "model_comparison_tuned.csv", index=False)
        write_json({"models": models.astype(object).where(models.notna(), None).to_dict(orient="records"),
                    "votes": vote_frame.astype(object).where(vote_frame.notna(), None).to_dict(orient="records")}, directory / "model_comparison_tuned.json")
        final_metrics = votes[FINAL_VOTE_KEY]["metrics"] if FINAL_VOTE_KEY in votes else None
        report = build_tuned_report(models, vote_frame, obs, scope, checks, final_metrics, missing)
        (directory / "evaluation_report_tuned.md").write_text(report, encoding="utf-8")
        (directory / "model_comparison_tuned.md").write_text("\n".join(_md_table(models, [("model", "Model", "text"), ("final_set_uses", "Final set uses", "text"),
                                                                                         ("baseline_macro_f1", "Baseline macro F1", "f4"), ("tuned_macro_f1", "Tuned macro F1", "f4"),
                                                                                         ("delta_macro_f1", "Delta", "f4")])) + "\n", encoding="utf-8")
        write_json({"generated_at": datetime.now().isoformat(timespec="seconds"), "scope": scope, "final_set": source, "rule_reasons": reasons,
                    "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks], "missing": missing, "final_vote_missing": final_missing},
                   directory / "evaluation_summary_tuned.json")
        figures = [] if args.no_figures else make_tuned_figures(models, votes.get(FINAL_VOTE_KEY))
    except KeyboardInterrupt:
        logger.error("Interrupted. Saved test results stay saved; rerun to continue.")
        return 130
    except (DataNotReadyError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    print()
    print(models[["model", "final_set_uses", "baseline_macro_f1", "tuned_macro_f1", "delta_macro_f1", "tuned_false_alarm_rate"]].round(4).to_string(index=False))
    print()
    print(vote_frame[["vote", "voters", "macro_f1", "accuracy", "detection_rate", "false_alarm_rate"]].round(4).to_string(index=False))
    print()
    print(*[f"- {o}" for o in obs], sep="\n")
    logger.info("Report: %s (+ %d figure(s) in %s)", directory / "evaluation_report_tuned.md", len(figures), figure_directory("tuned"))
    failed_checks = [name for name, ok, _ in checks if not ok]
    logger.info("Checks passed: %d/%d (total %.1f min)", len(checks) - len(failed_checks), len(checks), (time.time() - started) / 60)
    if failed_checks or missing:
        logger.error("Incomplete: failed checks %s, missing tuned models %s", failed_checks, list(missing))
        return 2
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validate, score the test set once, vote, compare and report.")
    parser.add_argument("--models", nargs="+", metavar="KEY", help=f"model keys (default: all). Keys: {', '.join(model_params.MODEL_KEYS)}")
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    parser.add_argument("--cv-sample-size", type=int, default=None, help="cross-validate on a class-capped sample of N training rows (default: all rows)")
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation (reuses saved results in the report if they exist)")
    parser.add_argument("--skip-test", action="store_true", help="skip test-set scoring and the report")
    parser.add_argument("--rescore", action="store_true", help="score the test set again instead of reusing saved results")
    parser.add_argument("--no-figures", action="store_true", help="do not draw figures")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and time estimates, run nothing")
    parser.add_argument("--stage", choices=("trained", "tuned"), default="trained",
                        help="trained: the baseline models (default). tuned: score the tuned models once, compare with the baselines and take the three votes")
    args = parser.parse_args(argv)

    if args.stage == "tuned":
        ignored = [flag for flag, used in (("--folds", args.folds != 5), ("--cv-sample-size", args.cv_sample_size is not None), ("--skip-cv", args.skip_cv), ("--skip-test", args.skip_test)) if used]
        if ignored:
            logger.error("%s do not apply to --stage tuned (no cross-validation is run for the tuned models).", ", ".join(ignored))
            return 1
        return run_tuned_stage(args)

    if args.folds < 2:
        logger.error("--folds must be at least 2")
        return 1
    if args.cv_sample_size is not None and args.cv_sample_size < 1000:
        logger.error("--cv-sample-size must be at least 1000")
        return 1
    if args.skip_cv and args.skip_test:
        logger.error("--skip-cv and --skip-test together leave nothing to do")
        return 1
    keys = list(args.models) if args.models else list(model_params.MODEL_KEYS)
    unknown = [k for k in keys if k not in model_params.MODEL_KEYS]
    if unknown:
        logger.error("Unknown model key(s) %s. Known: %s", unknown, list(model_params.MODEL_KEYS))
        return 1
    expected = list(model_params.MODEL_KEYS)  # the table always accounts for all 11 models

    try:
        settings.ensure_directories()
        train = load_training_data(mmap=True)
    except (DataNotReadyError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    n_cv_rows = train.n_rows if args.cv_sample_size is None else min(args.cv_sample_size, train.n_rows)
    n_test = int(np.load(settings.Y_TEST_FILE, mmap_mode="r").shape[0]) if settings.Y_TEST_FILE.is_file() else 0

    # ---- plan -----------------------------------------------------------
    print("Plan:")
    print(f"  cross-validation: {'skipped' if args.skip_cv else f'{args.folds}-fold on {n_cv_rows:,} training rows' + (' (class-capped sample)' if args.cv_sample_size else ' (all rows)')}")
    print(f"  test set: {'skipped' if args.skip_test else f'{n_test:,} rows, scored once (saved results are reused)'}")
    print(f"  {'model':<24} {'CV (est.)':>12} {'test (est.)':>12}   saved CV / test")
    total_cv = total_test = 0.0
    cv_dir = cv_directory_for(args.cv_sample_size)
    for key in keys:
        est = estimate_seconds(key, n_cv_rows, n_test, args.folds, not args.skip_cv, not args.skip_test)
        total_cv += est["cv"] or 0.0
        total_test += est["test"] or 0.0
        saved_cv = (cv_dir / f"cv_{key}.json").is_file()
        saved_test = _paths(key)[0].is_file()
        print(f"  {model_params.DISPLAY_NAMES[key]:<24} {format_minutes(est['cv']):>12} {format_minutes(est['test']):>12}   {'yes' if saved_cv else 'no':>3} / {'yes' if saved_test else 'no'}")
    print(f"  {'total (rough, linear in rows)':<24} {format_minutes(total_cv):>12} {format_minutes(total_test):>12}")
    print()
    if args.dry_run:
        print("Dry run: nothing was run.")
        return 0

    started = time.time()
    cv_results: dict[str, dict[str, Any]] | None = None
    try:
        # ---- 1-2 cross-validation ---------------------------------------
        if not args.skip_cv:
            cv_results, _, cv_dir = run_cross_validation(keys, train, args.folds, args.cv_sample_size)
        else:
            saved = {k: read_json(cv_dir / f"cv_{k}.json") for k in [*expected, ENSEMBLE_KEY] if (cv_dir / f"cv_{k}.json").is_file()}
            cv_results = saved or None
            if cv_results:
                logger.info("Using %d saved cross-validation result(s) from %s", len(cv_results), cv_dir)
        failed_cv = [k for k, r in (cv_results or {}).items() if r.get("status") == "failed"]
        if args.skip_test:
            if cv_results:
                cv_rank = cv.rank_models(cv_results)
                print(cv_rank[["rank", "model", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "detection_rate_mean", "false_alarm_rate_mean", "fit_seconds_mean"]].round(4).to_string(index=False))
            logger.info("Cross-validation finished in %.1f min", (time.time() - started) / 60)
            return 2 if failed_cv else 0

        # ---- 3 test set --------------------------------------------------
        test = load_test_data(mmap=True)
        X_test, y_test = np.asarray(test.X), test.y
        y_hash = hashlib.sha256(np.ascontiguousarray(y_test).tobytes()).hexdigest()
        scored: dict[str, dict[str, Any]] = {}
        missing: dict[str, str] = {k: "not evaluated in this run (--models)" for k in expected if k not in keys}
        for position, key in enumerate(keys, start=1):
            logger.info("[%d/%d] %s", position, len(keys), model_params.DISPLAY_NAMES[key])
            result, reason = score_model(key, X_test, y_test, test.codec, y_hash, args.rescore)
            if result is None:
                missing[key] = reason or "unknown"
            else:
                scored[key] = result
        # models outside --models that already have saved results are still part of the comparison
        for key in expected:
            if key not in keys and key in missing and _paths(key)[0].is_file() and _paths(key)[1].is_file():
                saved = read_json(_paths(key)[0])
                if saved.get("status") == "ok":
                    scored[key] = saved
                    missing.pop(key)

        # ---- 4 vote on the test set --------------------------------------
        results = {k: r for k, r in scored.items()}
        vote = vote_on_test(scored, {k: v for k, v in missing.items()}, y_test, test.codec, expected)
        if vote:
            results[ENSEMBLE_KEY] = vote
        if not results:
            logger.error("No model could be scored: %s", missing)
            return 2

        # ---- 5 report -----------------------------------------------------
        metrics_by_key = {k: {**r["metrics"], "display_name": r["display_name"]} for k, r in results.items()}
        training, _ = comparator.training_info(expected)
        frame = comparator.comparison_table(metrics_by_key, training)
        cv_rank = cv.rank_models(cv_results) if cv_results else None
        if cv_rank is not None and not cv_rank.empty:
            frame = frame.merge(cv_rank[["key", "macro_f1_mean", "macro_f1_std"]].rename(columns={"macro_f1_mean": "cv_macro_f1_mean", "macro_f1_std": "cv_macro_f1_std"}),
                                on="key", how="left")
        per_class = comparator.per_class_comparison(metrics_by_key, "f1", training)
        notes = comparator.table_notes(frame, metrics_by_key, training, missing=list(missing), ensemble_members=[model_params.DISPLAY_NAMES[k] for k in vote["metrics"]["members"]] if vote else None)
        if cv_rank is not None:
            notes.append(f"Column 'CV macro F1' is the mean over {args.folds} folds on "
                         f"{'all training rows' if args.cv_sample_size is None else f'a class-capped sample of {n_cv_rows:,} training rows (balanced classes: not comparable with the test-set class mix)'}; the test set was not used for it.")
        checks = automatic_checks(results, len(y_test), expected, missing)
        paths = comparator.save_comparison(frame, per_class, notes)
        scope = {
            "training rows": f"{train.n_rows:,}", "test rows": f"{len(y_test):,}", "classes": train.n_classes, "features": train.feature_count,
            "cross-validation": "skipped / not available" if not cv_results else f"{args.folds}-fold on {n_cv_rows:,} rows" + (" (class-capped sample)" if args.cv_sample_size else " (all training rows)"),
            "tie rule of the vote": settings.TIE_BREAK_RULE, "random seed": settings.RANDOM_SEED,
        }
        report = build_report(frame, notes, metrics_by_key, cv_results, cv_rank, scope, checks, missing)
        report_path = settings.EVALUATION_REPORT_DIR / "evaluation_report.md"
        report_path.write_text(report, encoding="utf-8")
        write_json({"generated_at": datetime.now().isoformat(timespec="seconds"), "scope": scope, "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
                    "missing": missing, "cv_failed": failed_cv}, settings.EVALUATION_REPORT_DIR / "evaluation_summary.json")
        figures = [] if args.no_figures else make_figures(frame, {k: {**r, "metrics": metrics_by_key[k]} for k, r in results.items()}, cv_results, cv_rank)
    except KeyboardInterrupt:
        logger.error("Interrupted. Saved cross-validation and test results stay saved; rerun to continue.")
        return 130
    except (DataNotReadyError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    print()
    print(frame[["rank", "model", "macro_f1", "macro_f1_reliable", "accuracy", "detection_rate", "false_alarm_rate"]].round(4).to_string(index=False))
    print()
    print(*notes, sep="\n")
    logger.info("Report: %s (+ %d figure(s) in %s); tables: %s", report_path, len(figures), figure_directory(), paths["markdown"])
    failed_checks = [name for name, ok, _ in checks if not ok]
    logger.info("Checks passed: %d/%d (total %.1f min)", len(checks) - len(failed_checks), len(checks), (time.time() - started) / 60)
    if failed_checks or failed_cv or missing:
        logger.error("Incomplete: failed checks %s, failed CV %s, missing models %s", failed_checks, failed_cv, list(missing))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())