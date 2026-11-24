"""Side-by-side comparison of the models on the held-out test set.

Input: for every model a metrics dictionary from ``metrics.compute_metrics`` /
``metrics.evaluate_model`` (all computed on the SAME test rows), optionally the
information about how each model was trained, and optionally the majority-vote
ensemble as one more entry under the key ``"majority_vote"``.  The ensemble is
scored like any other model (``compute_metrics`` on its voted labels, no scores,
so no ROC-AUC); this module does not vote, it only compares.

Output: a ranked table (macro F1 first), a per-class table and a list of notes that
must travel with the table (see ``table_notes``).

Honesty rules built into the table
----------------------------------
* The ranking metric is **macro F1**, because accuracy is dominated by BENIGN.
* ``macro_f1_reliable`` repeats macro F1 over the classes with at least
  ``LOW_SUPPORT_THRESHOLD`` test rows only: Heartbleed, Infiltration and SQL Injection
  have a handful of test rows, so their F1 values swing on one flow.
* Models trained on a **subsample** (KNN, SVC, Gradient Boosting) are flagged in the
  table itself, with the number of rows they saw.  Their class proportions differ
  from the full training data, which shifts what they predict; this is stated in the
  notes, not hidden.
* Models that are missing (not trained, failed) are listed, never silently dropped.
* No confidence intervals are computed here: a difference in the third decimal is not
  evidence that one model is better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.evaluation.metrics import LOW_SUPPORT_THRESHOLD
from nids.persistence import model_store
from nids.utils.io_utils import write_json
from nids.utils.logger import get_logger

logger = get_logger(__name__)

ENSEMBLE_KEY = "majority_vote"
ENSEMBLE_NAME = "Majority Vote (ensemble)"
PRIMARY_METRIC = "macro_f1"
DECISION_SCORE_KIND = "decision"  # what BaseModel.score_kind returns for models without predict_proba


# ---------------------------------------------------------------------------
# Training information
# ---------------------------------------------------------------------------
def training_info(
    keys: list[str] | tuple[str, ...] | None = None,
    stage: str = "trained",
    directory: Path | str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """How each saved model was trained, read from its metadata file.

    Returns ``(info, missing)``: ``info[key]`` has ``display_name``, ``n_train_rows``,
    ``n_rows_available``, ``subsampled``, ``fit_seconds``; ``missing`` lists keys with no saved model.
    """
    info: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key in keys if keys is not None else model_params.MODEL_KEYS:
        try:
            meta = model_store.load_metadata(key, stage, directory)
        except model_store.ModelStoreError:
            missing.append(key)
            continue
        extra = meta.get("extra") or {}
        sub = extra.get("subsample") or {}
        model_info = meta.get("model_info") or {}
        info[key] = {
            "display_name": meta.get("display_name") or model_params.DISPLAY_NAMES.get(key, key),
            "n_train_rows": sub.get("n_rows", extra.get("n_train_rows_used")),
            "n_rows_available": sub.get("n_rows_available"),
            "subsampled": bool(sub.get("subsampled", False)),
            "fit_seconds": model_info.get("fit_seconds"),
        }
    return info, missing


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _name(key: str, training: dict[str, dict[str, Any]]) -> str:
    if key == ENSEMBLE_KEY:
        return ENSEMBLE_NAME
    if key in training and training[key].get("display_name"):
        return training[key]["display_name"]
    return model_params.DISPLAY_NAMES.get(key, key)


def _reliable_macro_f1(metrics: dict[str, Any]) -> float | None:
    """Mean F1 over the classes that are present in the test rows AND have enough support."""
    values = [c["f1"] for c in metrics["per_class"] if c["support"] >= LOW_SUPPORT_THRESHOLD]
    return float(np.mean(values)) if values else None


def comparison_table(
    results: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per model, best macro F1 first (ties keep the input order). Empty input gives an empty frame."""
    training = training or {}
    rows = []
    for key, m in results.items():
        overall = m["overall"]
        binary = m.get("attack_vs_normal") or {}
        t = training.get(key, {})
        is_ensemble = key == ENSEMBLE_KEY
        predict_seconds = m.get("predict_seconds")
        rows.append(
            {
                "model": _name(key, training),
                "key": key,
                "kind": "ensemble" if is_ensemble else "single",
                "macro_f1": overall["macro_f1"],
                "macro_f1_reliable": _reliable_macro_f1(m),
                "balanced_accuracy": overall["balanced_accuracy"],
                "accuracy": overall["accuracy"],
                "mcc": overall["mcc"],
                "weighted_f1": overall["weighted_f1"],
                "detection_rate": binary.get("detection_rate"),
                "false_alarm_rate": binary.get("false_alarm_rate"),
                "roc_auc_macro": overall.get("roc_auc_ovr_macro"),
                "score_kind": m.get("score_kind"),
                "train_rows": None if is_ensemble else t.get("n_train_rows"),
                "trained_on_subsample": None if is_ensemble else (t.get("subsampled") if t else None),
                "predict_seconds": predict_seconds,
                "test_rows": m["n_rows"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(PRIMARY_METRIC, ascending=False, kind="stable").reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def per_class_comparison(
    results: dict[str, dict[str, Any]],
    metric: str = "f1",
    training: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Classes as rows, models as columns (``metric`` = ``f1``, ``precision`` or ``recall``).

    ``support`` and ``low_support`` come first and are always present.
    """
    if metric not in ("f1", "precision", "recall"):
        raise ValueError(f"metric must be 'f1', 'precision' or 'recall', got {metric!r}")
    if not results:
        return pd.DataFrame()
    training = training or {}
    first = next(iter(results.values()))
    frame = pd.DataFrame(
        {
            "class": first["class_names"],
            "support": [c["support"] for c in first["per_class"]],
            "low_support": [c["low_support"] for c in first["per_class"]],
        }
    )
    for key, m in results.items():
        if m["class_names"] != first["class_names"]:
            raise ValueError(f"{key!r} was scored with different classes than the other models")
        if [c["support"] for c in m["per_class"]] != frame["support"].tolist():
            raise ValueError(f"{key!r} was scored on different test rows than the other models")
        frame[_name(key, training)] = [c[metric] for c in m["per_class"]]
    return frame


def best_model_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    """For each class, the model with the highest value (ties: first column) and that value."""
    model_cols = [c for c in per_class.columns if c not in ("class", "support", "low_support")]
    values = per_class[model_cols]
    out = per_class[["class", "support", "low_support"]].copy()
    out["best_model"] = values.idxmax(axis=1)
    out["best_value"] = values.max(axis=1)
    return out


# ---------------------------------------------------------------------------
# Notes that must accompany the table
# ---------------------------------------------------------------------------
def table_notes(
    frame: pd.DataFrame,
    results: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]] | None = None,
    missing: list[str] | None = None,
    ensemble_members: list[str] | None = None,
) -> list[str]:
    """The statements that make the table honest (one string each)."""
    training = training or {}
    notes: list[str] = []
    if frame.empty:
        return ["No model results to compare."]

    n_test = int(frame["test_rows"].iloc[0])
    notes.append(
        f"All rows are scored on the same {n_test:,} held-out test flows, used once. "
        f"Ranking metric: macro F1 (every class counts equally; accuracy is dominated by BENIGN)."
    )

    subsampled = frame[frame["trained_on_subsample"] == True]  # noqa: E712 (None must not count)
    if not subsampled.empty:
        parts = []
        for _, r in subsampled.iterrows():
            t = training.get(r["key"], {})
            avail = t.get("n_rows_available")
            parts.append(f"{r['model']} ({int(r['train_rows']):,}" + (f" of {int(avail):,} rows)" if avail else " rows)"))
        notes.append(
            "Trained on a class-capped subsample, not on all training rows: " + "; ".join(parts) + ". "
            "Large classes were capped and small classes kept whole, so the class proportions differ strongly from the "
            "real traffic (BENIGN is much less frequent in the subsample). That shifts these models toward predicting "
            "attacks, so their false alarm rates and precision are NOT directly comparable with the other models."
        )

    low = sorted({c for m in results.values() for c in m.get("low_support_classes", [])})
    if low:
        notes.append(
            f"Classes with fewer than {LOW_SUPPORT_THRESHOLD} test rows ({', '.join(low)}) have unreliable scores: "
            "one flow changes their F1 a lot. `macro_f1` includes them; `macro_f1_reliable` leaves them out."
        )

    if (frame["score_kind"] == DECISION_SCORE_KIND).any():
        names = ", ".join(frame.loc[frame["score_kind"] == DECISION_SCORE_KIND, "model"])
        notes.append(
            f"ROC-AUC of {names} comes from decision scores, not probabilities: valid for ranking, "
            "but the scores are not calibrated."
        )

    if ENSEMBLE_KEY in results:
        members = ensemble_members or results[ENSEMBLE_KEY].get("members")
        text = (
            "The ensemble is a majority vote of the models' predicted labels: one equal vote per model, no probability "
            "averaging, deterministic tie-break. It is not a trained model, and it has no ROC-AUC (labels only)."
        )
        if members:
            text += f" Voters ({len(members)}): {', '.join(members)}."
        notes.append(text)

    if missing:
        names = ", ".join(_name(k, training) for k in missing)
        notes.append(f"Missing from this table (no saved model or evaluation failed): {names}.")

    notes.append(
        "No confidence intervals are computed: differences in the third decimal are not evidence that one model is better."
    )
    return notes


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
_PERCENT = {"detection_rate", "false_alarm_rate"}
_COLUMN_LABELS = {
    "rank": "Rank",
    "model": "Model",
    "macro_f1": "Macro F1",
    "macro_f1_reliable": "Macro F1 (classes >= 30 rows)",
    "balanced_accuracy": "Balanced acc.",
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "detection_rate": "Detection rate",
    "false_alarm_rate": "False alarm rate",
    "roc_auc_macro": "ROC-AUC (macro)",
    "train_rows": "Training rows",
    "predict_seconds": "Predict (s)",
    "subsample": "Subsample",
}


def _cell(column: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if column in _PERCENT:
        return f"{value * 100:.2f}%"
    if column == "train_rows":
        return f"{int(value):,}"
    if column == "predict_seconds":
        return f"{value:.1f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown(frame: pd.DataFrame, notes: list[str] | None = None, title: str = "Model comparison (test set)") -> str:
    """A readable Markdown table with the notes underneath."""
    shown = frame.copy()
    if "trained_on_subsample" in shown:
        shown["subsample"] = shown["trained_on_subsample"].map({True: "yes", False: "no"}).fillna("-")
    columns = [c for c in _COLUMN_LABELS if c in shown.columns]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(_COLUMN_LABELS[c] for c in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_cell(c, row[c]) for c in columns) + " |")
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"{i}. {n}" for i, n in enumerate(notes, 1)]
    return "\n".join(lines) + "\n"


def save_comparison(
    frame: pd.DataFrame,
    per_class: pd.DataFrame | None,
    notes: list[str],
    directory: Path | str | None = None,
) -> dict[str, Path]:
    """Write ``model_comparison.csv/.md/.json`` (and ``per_class_f1.csv``) and return the paths."""
    directory = Path(directory) if directory is not None else settings.EVALUATION_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": directory / "model_comparison.csv",
        "markdown": directory / "model_comparison.md",
        "json": directory / "model_comparison.json",
    }
    frame.to_csv(paths["csv"], index=False)
    paths["markdown"].write_text(to_markdown(frame, notes), encoding="utf-8")
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    write_json({"ranking_metric": PRIMARY_METRIC, "models": records, "notes": notes}, paths["json"])
    if per_class is not None and not per_class.empty:
        paths["per_class_csv"] = directory / "per_class_f1.csv"
        per_class.to_csv(paths["per_class_csv"], index=False)
    return paths