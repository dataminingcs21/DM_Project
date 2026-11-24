"""Predict the class of one flow, or of a table of flows, with the final set of models.

One path from a typed flow to an answer::

    values -> input_validator -> preprocessing pipeline -> every model's predict -> majority vote

* The input is checked first (File 58).  A flow with an error is **not** predicted; the findings say why.
* The same fitted preprocessing pipeline that produced the training data transforms the flow.
* Every available model gives one label; the vote is the one-model-one-vote majority of File 53
  (no probabilities, no weights, deterministic ties).
* **The final set** follows the Checkpoint 5 evidence rule: each model is loaded in its tuned version
  where tuning showed an improvement beyond fold noise, otherwise as its baseline.  The rule is read from
  the metadata that ``07_optimize.py`` stored in the tuned models (never from test scores).  A model that
  cannot be loaded, or fails while predicting, is NOT replaced by another version: it is listed as not
  voting, with the reason, in every result.

What it does not do: train, tune, evaluate, or draw anything.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import model_params, settings
from nids.persistence import model_store
from nids.prediction import input_validator as validator
from nids.preprocessing import feature_schema as feature_schema_module
from nids.preprocessing import preprocessor
from nids.preprocessing.target_builder import LabelCodec
from nids.utils.io_utils import load_array, load_joblib, read_parquet
from nids.utils.logger import get_logger
from nids.voting import majority_voting as voting

logger = get_logger(__name__)

SELECTIONS = ("final", "baseline", "tuned")
_STAGE = {"baseline": "trained", "tuned": "tuned"}  # version name -> model store stage


class PredictorError(RuntimeError):
    """The predictor cannot be built or cannot predict (missing files, no model available, inconsistent artifacts)."""


# ---------------------------------------------------------------------------
# Which version of each model the final system uses
# ---------------------------------------------------------------------------
def final_set(keys: list[str] | tuple[str, ...] | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """The Checkpoint 5 evidence rule: ``(version, reason)`` per model, version "tuned" or "baseline".

    A model uses its tuned version when the tuning folds showed an improvement beyond fold noise and the tuned
    parameters differ from the baseline.  The decision reads the tuning record stored in the tuned model's metadata.
    """
    version: dict[str, str] = {}
    reason: dict[str, str] = {}
    for key in keys if keys is not None else model_params.MODEL_KEYS:
        try:
            tuning = (model_store.load_metadata(key, "tuned").get("extra") or {}).get("tuning") or {}
        except model_store.ModelStoreError:
            version[key], reason[key] = "baseline", "no tuned model saved"
            continue
        if tuning.get("same_as_baseline"):
            version[key], reason[key] = "baseline", "the baseline setting was already the best in tuning"
        elif tuning.get("improvement_beyond_noise"):
            version[key], reason[key] = "tuned", "tuning improvement beyond fold noise"
        else:
            version[key], reason[key] = "baseline", "tuning improvement within fold noise"
    return version, reason


def _versions_for(selection: str) -> tuple[dict[str, str], dict[str, str]]:
    if selection == "final":
        return final_set()
    if selection in ("baseline", "tuned"):
        keys = model_params.MODEL_KEYS
        return {k: selection for k in keys}, {k: f"all models use their {selection} version (selection={selection!r})" for k in keys}
    raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class FlowPrediction:
    """The answer for one flow.  ``label`` is None when the input was refused (see ``validation``)."""

    validation: validator.ValidationResult
    label: str | None = None
    code: int | None = None
    votes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["model", "version", "vote"]))
    tally: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["class", "votes", "winner"]))
    n_voters: int = 0
    top_votes: int = 0
    tied: bool = False
    missing: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.label is not None

    @property
    def agreement(self) -> str:
        """Plain-language agreement, e.g. ``"9 of 11 models voted for DoS Hulk"``."""
        if not self.ok:
            return "no prediction"
        text = f"{self.top_votes} of {self.n_voters} models voted for {self.label}"
        if self.tied:
            text += f" (tied with another class; the tie rule '{settings.TIE_BREAK_RULE}' decided)"
        return text

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "label": self.label, "code": self.code, "agreement": self.agreement, "n_voters": self.n_voters,
            "top_votes": self.top_votes, "tied": self.tied, "missing_models": dict(self.missing),
            "input": self.validation.summary(), "seconds": round(self.seconds, 4),
        }


@dataclass
class BatchPrediction:
    """The answer for a table of flows: one line per input row, in the input order.

    ``frame`` columns: ``prediction`` (class name, empty for refused rows), ``top_votes``, ``n_voters``, ``tied``,
    then one column per voting model with its own vote.
    """

    frame: pd.DataFrame
    validation: validator.ValidationResult
    voters: list[str] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def n_predicted(self) -> int:
        return int(self.frame["prediction"].notna().sum()) if len(self.frame) else 0

    def summary(self) -> dict[str, Any]:
        counts = self.frame["prediction"].value_counts().to_dict() if self.n_predicted else {}
        return {
            "n_rows": int(len(self.frame)), "n_predicted": self.n_predicted, "n_refused": int(len(self.frame) - self.n_predicted),
            "voters": list(self.voters), "missing_models": dict(self.missing), "class_counts": {str(k): int(v) for k, v in counts.items()},
            "input": self.validation.summary(), "seconds": round(self.seconds, 4),
        }


# ---------------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------------
class Predictor:
    """Holds the fitted pipeline, the label codec, the schema and the loaded models.  Build it with :meth:`load`."""

    def __init__(
        self,
        pipeline: Any,
        codec: LabelCodec,
        schema: Mapping[str, Any],
        models: Mapping[str, Any],
        versions: Mapping[str, str],
        reasons: Mapping[str, str] | None = None,
        not_loaded: Mapping[str, str] | None = None,
        selection: str = "final",
    ) -> None:
        if not models:
            raise PredictorError("no model is available, so nothing can be predicted: " + "; ".join(f"{k}: {v}" for k, v in (not_loaded or {}).items()))
        self.pipeline = pipeline
        self.codec = codec
        self.schema = schema
        self.models = dict(models)
        self.versions = dict(versions)
        self.reasons = dict(reasons or {})
        self.not_loaded = dict(not_loaded or {})
        self.selection = selection

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, selection: str = "final", require_all: bool = False) -> "Predictor":
        """Load the pipeline, the codec, the schema and the models.

        ``selection``: ``"final"`` (the evidence rule; what the application uses), ``"baseline"`` or ``"tuned"`` (all 11 in that version).
        ``require_all=True`` refuses to build the predictor unless all 11 models load.
        """
        versions, reasons = _versions_for(selection)
        fresh, why = preprocessor.check_preprocessed_fresh()
        if not fresh:
            raise PredictorError(why)
        try:
            schema = feature_schema_module.load_feature_schema()
            pipeline = load_joblib(settings.PIPELINE_FILE, hint="run scripts/04_preprocess.py first")
            codec = load_joblib(settings.LABEL_ENCODER_FILE, hint="run scripts/04_preprocess.py first")
        except (FileNotFoundError, ValueError) as error:
            raise PredictorError(str(error)) from error
        if not isinstance(codec, LabelCodec):
            raise PredictorError("label_encoder.joblib does not contain a LabelCodec")
        names = [str(n) for n in pipeline.get_feature_names_out()]
        if names != list(schema["model_feature_names"]):
            raise PredictorError("the saved pipeline and feature_schema.json disagree about the model features; rerun scripts/04_preprocess.py")

        models: dict[str, Any] = {}
        not_loaded: dict[str, str] = {}
        for key in model_params.MODEL_KEYS:
            try:
                models[key] = model_store.load_model(key, _STAGE[versions[key]])
            except Exception as exc:  # noqa: BLE001 - one unloadable model must not stop the others; it is reported
                not_loaded[key] = f"{versions[key]} version not loaded: {type(exc).__name__}: {exc}"
                logger.error("Model %s (%s) not loaded: %s", key, versions[key], not_loaded[key])
        if require_all and not_loaded:
            raise PredictorError("not all models could be loaded: " + "; ".join(f"{k}: {v}" for k, v in not_loaded.items()))
        for key, model in models.items():
            if getattr(model, "n_classes_", codec.n_classes) != codec.n_classes:
                raise PredictorError(f"{key} was trained on {model.n_classes_} classes but the label codec has {codec.n_classes}")
        logger.info("Predictor ready (%s): %d of %d models loaded", selection, len(models), len(model_params.MODEL_KEYS))
        return cls(pipeline, codec, schema, models, versions, reasons, not_loaded, selection)

    # ------------------------------------------------------------------ info
    @property
    def voters(self) -> list[str]:
        return list(self.models)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.codec.classes

    def model_table(self) -> pd.DataFrame:
        """One line per model: version used, why, whether it is loaded, and how it was trained."""
        rows = []
        for key in model_params.MODEL_KEYS:
            version = self.versions.get(key)
            row: dict[str, Any] = {
                "key": key, "model": model_params.DISPLAY_NAMES[key], "version": version, "why": self.reasons.get(key),
                "loaded": key in self.models, "problem": self.not_loaded.get(key),
                "training_rows": None, "trained_on_subsample": None, "tuned_parameters": None, "file_size_mb": None,
            }
            try:
                meta = model_store.load_metadata(key, _STAGE[version]) if version else {}
            except model_store.ModelStoreError:
                meta = {}
            extra = meta.get("extra") or {}
            sub = extra.get("subsample") or {}
            row["training_rows"] = sub.get("n_rows", extra.get("n_train_rows_used"))
            row["trained_on_subsample"] = bool(sub.get("subsampled", False)) if meta else None
            row["file_size_mb"] = meta.get("file_size_mb")
            if version == "tuned":
                best, base = (extra.get("tuning") or {}).get("best_params") or {}, (extra.get("tuning") or {}).get("baseline_params") or {}
                row["tuned_parameters"] = ", ".join(f"{k}={v}" for k, v in best.items() if base.get(k) != v) or "same as baseline"
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ predicting
    def _features(self, validation: validator.ValidationResult) -> np.ndarray:
        """The valid rows through the fitted pipeline, in the dtype the models were trained on."""
        # 04_preprocess.py casts the raw rows to settings.FLOAT_DTYPE (float32) BEFORE the pipeline, so the pipeline
        # computes in that precision.  Doing the same here makes a typed flow go through exactly the same
        # arithmetic as the rows the models were trained and evaluated on (float64 would differ by about 1e-6,
        # enough to move a flow that sits next to a tree's split point to the other side).
        raw = validation.valid_frame.astype(settings.FLOAT_DTYPE)
        out = self.pipeline.transform(raw)
        return np.ascontiguousarray(out.to_numpy(dtype=settings.FLOAT_DTYPE))

    def _vote(self, X: np.ndarray) -> tuple[voting.VoteResult, dict[str, np.ndarray]]:
        predictions, failures = voting.collect_predictions(self.models, X)
        if not predictions:
            raise PredictorError("every model failed while predicting: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
        result = voting.vote_labels(
            predictions, self.codec.n_classes, self.codec.benign_index,
            expected_keys=list(model_params.MODEL_KEYS), missing={**self.not_loaded, **failures},
        )
        return result, predictions

    def predict_flow(self, values: Mapping[str, Any], missing: str = "error") -> FlowPrediction:
        """Predict one flow given as ``{feature name: value}``.  A refused input gives ``label=None`` and the findings."""
        started = time.time()
        validation = validator.validate_flow(values, self.schema, missing=missing)
        if not validation.ok:
            return FlowPrediction(validation=validation, missing=dict(self.not_loaded), seconds=time.time() - started)
        vote, predictions = self._vote(self._features(validation))
        voters, tally = voting.explain_vote(predictions, vote, self.codec, row=0)
        voters.insert(1, "version", [self.versions[k] for k in vote.voters])
        code = int(vote.labels[0])
        return FlowPrediction(
            validation=validation, label=str(self.codec.classes[code]), code=code, votes=voters, tally=tally,
            n_voters=vote.n_voters, top_votes=int(vote.top_votes[0]), tied=bool(vote.tied[0]), missing=dict(vote.missing),
            seconds=time.time() - started,
        )

    def predict_frame(self, data: pd.DataFrame, missing: str = "error") -> BatchPrediction:
        """Predict a table of flows (one row per flow).  Rows with an error are refused, the others are predicted."""
        started = time.time()
        validation = validator.validate_frame(data, self.schema, missing=missing)
        n = validation.n_rows
        index = validation.frame.index
        frame = pd.DataFrame(
            {
                "prediction": pd.Series([None] * n, dtype=object, index=index),
                "top_votes": pd.Series(pd.array([pd.NA] * n, dtype="Int64"), index=index),
                "n_voters": pd.Series(pd.array([pd.NA] * n, dtype="Int64"), index=index),
                "tied": pd.Series(pd.array([pd.NA] * n, dtype="boolean"), index=index),
            }
        )
        if not (validation.column_ok and validation.n_valid > 0):
            return BatchPrediction(frame=frame, validation=validation, missing=dict(self.not_loaded), seconds=time.time() - started)
        vote, predictions = self._vote(self._features(validation))
        where = np.flatnonzero(validation.valid_mask)
        frame.iloc[where, frame.columns.get_loc("prediction")] = self.codec.decode(vote.labels)
        frame.iloc[where, frame.columns.get_loc("top_votes")] = vote.top_votes
        frame.iloc[where, frame.columns.get_loc("n_voters")] = vote.n_voters
        frame.iloc[where, frame.columns.get_loc("tied")] = vote.tied
        for key in vote.voters:
            column = pd.Series([None] * n, dtype=object, index=index)
            column.iloc[where] = self.codec.decode(predictions[key])
            frame[model_params.DISPLAY_NAMES[key]] = column
        return BatchPrediction(frame=frame, validation=validation, voters=list(vote.voters), missing=dict(vote.missing), seconds=time.time() - started)


# ---------------------------------------------------------------------------
# Real example flows (taken from the TEST split by 04_preprocess.py; nothing is invented)
# ---------------------------------------------------------------------------
def load_sample_flows() -> pd.DataFrame:
    """A few real test flows per class: the raw feature columns, the class name (``settings.LABEL_COLUMN``) and ``test_row_position``."""
    return read_parquet(settings.SAMPLE_FLOWS_FILE, hint="run scripts/04_preprocess.py first")


# ---------------------------------------------------------------------------
# Demonstration with consistency checks:  python -m nids.prediction.predictor
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        predictor = Predictor.load("final")
    except (PredictorError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    table = predictor.model_table()
    print("Models of the final set:")
    print(table[["model", "version", "loaded", "training_rows", "trained_on_subsample", "file_size_mb"]].to_string(index=False))
    if predictor.not_loaded:
        print("\nNOT LOADED:", predictor.not_loaded)
    print()

    typical = predict_typical(predictor)
    print(f"Typical flow (per-feature training medians; not a real flow): {typical.label} - {typical.agreement}")
    print()

    failed = 0
    try:
        samples = load_sample_flows()
    except (FileNotFoundError, ValueError) as error:
        print(f"Sample flows not available ({error}); consistency checks skipped.")
        return 0 if not predictor.not_loaded else 2
    truth = samples[settings.LABEL_COLUMN].to_numpy()
    raw = samples.drop(columns=[settings.LABEL_COLUMN, "test_row_position"])
    batch = predictor.predict_frame(raw)
    s = batch.summary()
    print(f"Sample flows: {s['n_rows']} real test flows, {s['n_predicted']} predicted, {s['n_refused']} refused by the input check.")
    predicted = batch.frame["prediction"].to_numpy(dtype=object)
    done = batch.frame["prediction"].notna().to_numpy()
    print(f"  final vote equals the true class on {int((predicted[done] == truth[done]).sum())} of {int(done.sum())} (a handful of flows per class: an illustration, not a metric)")

    # check 1: the typed-flow path reproduces the preprocessed test matrix exactly
    positions = samples["test_row_position"].to_numpy(dtype="int64")
    X_test = load_array(settings.X_TEST_FILE, mmap_mode="r")
    valid = batch.validation.valid_mask
    if valid.any():
        ours = predictor._features(batch.validation)
        theirs = np.asarray(X_test[positions[valid]], dtype="float64")
        ours = ours.astype("float64")
        diff = float(np.max(np.abs(ours - theirs)))
        ok1 = bool(np.array_equal(ours, theirs))  # the same float32 arithmetic as 04_preprocess.py: the values must be identical
        print(f"  check 1, typed-flow path vs saved X_test rows: max difference {diff:.2e}, {int((ours != theirs).sum())} of {theirs.size} values differ -> {'OK (identical)' if ok1 else 'MISMATCH'}")
        failed += 0 if ok1 else 1

    # check 2: the final vote equals the saved evaluation vote on the same test rows
    saved = settings.EVALUATION_REPORT_DIR / "test_tuned" / "majority_vote_final_pred.npy"
    if saved.is_file() and not predictor.not_loaded and valid.any():
        reference = load_array(saved)[positions[valid]]
        mine = predictor.codec.encode(batch.frame["prediction"].to_numpy(dtype=object)[valid])
        same = int((reference == mine).sum())
        ok2 = same == int(valid.sum())
        print(f"  check 2, final vote vs saved evaluation vote (majority_vote_final): {same} of {int(valid.sum())} identical -> {'OK' if ok2 else 'MISMATCH'}")
        failed += 0 if ok2 else 1
    else:
        print("  check 2 skipped (saved evaluation vote not found or not all models loaded)")
    return 2 if failed or predictor.not_loaded else 0


def predict_typical(predictor: Predictor) -> FlowPrediction:
    """Predict the per-feature-median flow (used by the demonstration and by the interface's default form)."""
    return predictor.predict_flow(validator.typical_flow(predictor.schema))


if __name__ == "__main__":
    sys.exit(main())