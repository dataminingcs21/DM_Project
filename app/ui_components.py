"""Building blocks of the Streamlit application (File 62).

Two kinds of code live here, kept apart on purpose:

* **Plain functions** (no Streamlit calls): grouping of the feature fields, default and example values,
  reading an uploaded CSV, turning results into tables and sentences, reading the test-set results that
  ``06_evaluate.py`` saved.  They are unit-tested without a browser.
* **Rendering functions** (``render_*``) and the cached loaders (``get_*``): thin layers that put those
  results on the page.  ``streamlit_app.py`` (File 63) only composes them.

Nothing here predicts or validates by itself: that is ``nids.prediction.predictor`` and
``nids.prediction.input_validator``.  Numbers shown about model quality are read from the reports
``06_evaluate.py`` wrote, never typed in here.
"""

from __future__ import annotations

import io
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from config import model_params, settings
from nids.prediction import input_validator as validator
from nids.prediction import predictor as predictor_module
from nids.utils.io_utils import read_json

MAX_UPLOAD_ROWS = 50_000
MIN_GROUP_SIZE = 3
OTHER_GROUP = "Other"
GROUPING_RULE = (
    "The fields are grouped by the first word of the feature name; a first word shared by fewer than "
    f"{MIN_GROUP_SIZE} features goes to '{OTHER_GROUP}'. The grouping is only for reading; it is not a statement about the features."
)
STATE_PREFIX = "flow"

GENERAL_LIMITS = (
    "A prediction is the majority vote of the available models. It is not proof that a flow is, or is not, an attack.",
    "The models were trained and tested on one dataset (CIC-IDS2017) with a random split; flows of the same attack campaign can appear on both sides, so the scores can be optimistic for traffic from another network.",
    "KNN, SVC and Gradient Boosting were trained on class-capped subsamples, so their class prior differs from the real traffic.",
    "The input check catches impossible values (text, negatives where impossible, a port above 65535) and warns about values outside the training range. It cannot tell whether a combination of values is a realistic flow.",
    "Typing values by hand is for exploring how the models react. Real examples (loaded from your saved test flows) are better starting points than invented values.",
)


# ---------------------------------------------------------------------------
# Plain functions: fields
# ---------------------------------------------------------------------------
def first_word(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def group_features(names: list[str]) -> dict[str, list[str]]:
    """Group feature names by their first word (see ``GROUPING_RULE``); order of first appearance, ``Other`` last."""
    counts: dict[str, int] = {}
    for name in names:
        counts[first_word(name)] = counts.get(first_word(name), 0) + 1
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    for name in names:
        word = first_word(name)
        if counts[word] >= MIN_GROUP_SIZE:
            groups.setdefault(word, []).append(name)
        else:
            other.append(name)
    if other:
        groups[OTHER_GROUP] = other
    return groups


def format_value(value: Any) -> str:
    """A number as text for a text box: whole numbers without a decimal point, others with up to 10 significant digits."""
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if x != x:
        return ""
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    if x == round(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.10g}"


def _fmt(x: Any) -> str:
    return "?" if x is None else f"{x:.6g}"


def field_help(spec: Mapping[str, Any]) -> str:
    """The tooltip of one field: what the training data looked like and which rules apply."""
    parts = [f"Training range {_fmt(spec.get('min'))} to {_fmt(spec.get('max'))}; median {_fmt(spec.get('median'))}."]
    if spec.get("all_whole_numbers"):
        parts.append("Every training value was a whole number.")
    if spec.get("must_be_non_negative"):
        parts.append(f"Must not be negative: {spec.get('non_negative_reason') or 'impossible'}.")
    limit = validator.HARD_LIMITS.get(spec["name"])
    if limit:
        parts.append(f"Allowed {limit[0]:g} to {limit[1]:g}: {limit[2]}.")
    return " ".join(parts)


def field_key(name: str, prefix: str = STATE_PREFIX) -> str:
    return f"{prefix}::{name}"


def median_values(schema: Mapping[str, Any]) -> dict[str, str]:
    """Every model feature at its training median, as text (a starting point, not a real flow)."""
    return {name: format_value(value) for name, value in validator.typical_flow(schema).items()}


def sample_counts(samples: pd.DataFrame) -> dict[str, int]:
    """How many real example flows exist per class (class name order)."""
    return {str(k): int(v) for k, v in samples[settings.LABEL_COLUMN].value_counts().sort_index().items()}


def sample_values(samples: pd.DataFrame, class_name: str, index: int, names: list[str]) -> dict[str, str]:
    """The ``index``-th (0-based) real example flow of a class, as text, for the given feature names."""
    rows = samples[samples[settings.LABEL_COLUMN] == class_name]
    if rows.empty:
        raise ValueError(f"no example flow for class {class_name!r}")
    if not 0 <= index < len(rows):
        raise ValueError(f"class {class_name!r} has {len(rows)} example flow(s); index {index} is outside")
    row = rows.iloc[index]
    return {name: format_value(row[name]) for name in names}


def values_from_state(state: Mapping[str, Any], names: list[str], prefix: str = STATE_PREFIX) -> dict[str, Any]:
    """The typed values (as text) of every field; a field that is not in the state yet is absent from the result."""
    return {name: state[field_key(name, prefix)] for name in names if field_key(name, prefix) in state}


# ---------------------------------------------------------------------------
# Plain functions: uploads and results
# ---------------------------------------------------------------------------
def read_uploaded_csv(file: Any, max_rows: int = MAX_UPLOAD_ROWS) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV into a table; returns ``(table, notes)``.

    Column names are stripped of leading and trailing spaces (the CIC-IDS2017 files have them). If the label column
    is present it is set aside (never used for the prediction). At most ``max_rows`` rows are read.
    Raises ValueError with a plain message when the file cannot be read.
    """
    notes: list[str] = []
    try:
        frame = pd.read_csv(file, nrows=max_rows + 1, skipinitialspace=True)
    except pd.errors.EmptyDataError as error:
        raise ValueError("the file is empty") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"the file could not be read as a CSV table: {error}") from error
    if frame.shape[1] == 0:
        raise ValueError("the file has no columns")
    if len(frame) > max_rows:
        frame = frame.iloc[:max_rows]
        notes.append(f"Only the first {max_rows:,} rows were read.")
    stripped = [str(c).strip() for c in frame.columns]
    if stripped != [str(c) for c in frame.columns]:
        notes.append("Spaces around the column names were removed.")
    frame.columns = stripped
    if settings.LABEL_COLUMN in frame.columns:
        frame = frame.drop(columns=[settings.LABEL_COLUMN])
        notes.append(f"The '{settings.LABEL_COLUMN}' column was ignored (the prediction never uses a label).")
    return frame.reset_index(drop=True), notes


def results_table(batch: predictor_module.BatchPrediction) -> pd.DataFrame:
    """The batch result with 1-based row numbers and the reason a row was refused."""
    frame = batch.frame.copy()
    frame.insert(0, "row", range(1, len(frame) + 1))
    reasons: dict[int, list[str]] = {}
    for issue in batch.validation.errors:
        if issue.row is not None:
            reasons.setdefault(issue.row, []).append(issue.message)
    frame.insert(2, "refused_because", [" | ".join(reasons.get(i, [])) or None for i in frame["row"]])
    return frame


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def agreement_band(top_votes: int, n_voters: int) -> str:
    """Plain-language band for how many models agree (mechanical definition: all / more than half / only a plurality)."""
    if n_voters <= 0:
        return "no prediction"
    if top_votes == n_voters:
        return "all models agree"
    if top_votes * 2 > n_voters:
        return "a majority of the models agree"
    return "no majority: the winning class has only the most votes, not more than half"


def agreement_short(top_votes: int, n_voters: int) -> str:
    """A one-or-two-word form of ``agreement_band`` for a metric tile."""
    if n_voters <= 0:
        return "none"
    if top_votes == n_voters:
        return "All agree"
    return "Majority" if top_votes * 2 > n_voters else "Plurality only"


def is_benign(label: str | None) -> bool:
    return label == settings.BENIGN_LABEL


# ---------------------------------------------------------------------------
# Plain functions: what the evaluation found (read from the saved reports)
# ---------------------------------------------------------------------------
def _eval_dir():
    return settings.EVALUATION_REPORT_DIR


def load_class_reliability() -> dict[str, Any]:
    """Per-class test results of the final vote (precision, recall, support) and its main confusions; ``{}`` if not available."""
    path = _eval_dir() / "test_tuned" / "majority_vote_final.json"
    if not path.is_file():
        return {}
    metrics = (read_json(path).get("metrics")) or {}
    per_class = {str(c["class"]): {k: c.get(k) for k in ("support", "precision", "recall", "f1", "low_support")} for c in metrics.get("per_class", [])}
    return {"per_class": per_class, "confusions": list(metrics.get("top_confusions", [])), "n_rows": metrics.get("n_rows")}


def load_model_scores() -> dict[str, Any]:
    """Test macro F1 of every model (baseline and tuned) and of the three votes, from ``model_comparison_tuned.json``; ``{}`` if not available."""
    path = _eval_dir() / "model_comparison_tuned.json"
    if not path.is_file():
        return {}
    data = read_json(path)
    return {"models": {m["key"]: m for m in data.get("models", [])}, "votes": data.get("votes", [])}


def reliability_lines(label: str, reliability: Mapping[str, Any] | None) -> list[str]:
    """What the test set says about the predicted class (sentences built from the saved numbers)."""
    if not reliability or label not in (reliability.get("per_class") or {}):
        return ["No test-set results are available for this class (run `python scripts\\06_evaluate.py --stage tuned`)."]
    c = reliability["per_class"][label]
    lines = []
    support = c.get("support")
    recall, precision = c.get("recall"), c.get("precision")
    if support is not None and recall is not None:
        lines.append(f"On the test set the final vote found {recall * 100:.1f}% of the {support:,} '{label}' flows (recall).")
    if precision is not None:
        lines.append(f"{precision * 100:.1f}% of the flows it called '{label}' really were '{label}' (precision).")
    if c.get("low_support"):
        lines.append(f"Only {support} test flows belong to this class, so these figures are not reliable.")
    for item in reliability.get("confusions", []):
        if label in (item.get("true_class"), item.get("predicted_class")):
            share = item.get("share_of_true_class")
            lines.append(f"Frequent confusion on the test set: {item['rows']:,} '{item['true_class']}' flow(s) were labelled '{item['predicted_class']}'" + (f" ({share * 100:.1f}% of that class)." if share is not None else "."))
    return lines


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the models ...")
def get_predictor() -> predictor_module.Predictor:
    """The final-set predictor, loaded once per server process."""
    return predictor_module.Predictor.load("final")


@st.cache_data(show_spinner=False)
def get_samples() -> pd.DataFrame:
    return predictor_module.load_sample_flows()


@st.cache_data(show_spinner=False)
def get_reliability() -> dict[str, Any]:
    return load_class_reliability()


@st.cache_data(show_spinner=False)
def get_model_scores() -> dict[str, Any]:
    return load_model_scores()


# ---------------------------------------------------------------------------
# Rendering: form
# ---------------------------------------------------------------------------
def init_form_state(predictor: predictor_module.Predictor, prefix: str = STATE_PREFIX) -> None:
    """Put the training medians into every field that has no value yet (first visit only)."""
    for name, text in median_values(predictor.schema).items():
        st.session_state.setdefault(field_key(name, prefix), text)


def _load_into_state(values: Mapping[str, str], prefix: str) -> None:
    for name, text in values.items():
        st.session_state[field_key(name, prefix)] = text


def _on_load_medians(predictor: predictor_module.Predictor, prefix: str) -> None:
    _load_into_state(median_values(predictor.schema), prefix)
    st.session_state[f"{prefix}::source"] = "the training medians (per-feature medians, not a real flow)"


def _on_load_example(predictor: predictor_module.Predictor, samples: pd.DataFrame, prefix: str) -> None:
    name = st.session_state[f"{prefix}::example_class"]
    index = int(st.session_state[f"{prefix}::example_number"]) - 1
    _load_into_state(sample_values(samples, name, index, validator.required_features(predictor.schema)), prefix)
    st.session_state[f"{prefix}::source"] = f"real example {index + 1} of class '{name}' (a test flow saved by the preprocessing step)"


def render_form(predictor: predictor_module.Predictor, samples: pd.DataFrame | None, prefix: str = STATE_PREFIX) -> tuple[dict[str, Any], bool]:
    """The single-flow form.  Returns ``(typed values, predict button pressed)``.

    The values are text exactly as typed; the input check decides what they mean.
    """
    init_form_state(predictor, prefix)
    names = validator.required_features(predictor.schema)
    specs = {s["name"]: s for s in validator.field_specs(predictor.schema)}

    st.markdown("**Start from**")
    left, right = st.columns([3, 2])
    with left:
        if samples is not None and len(samples):
            counts = sample_counts(samples)
            st.selectbox("Class of the real example", list(counts), key=f"{prefix}::example_class", label_visibility="collapsed")
            chosen = st.session_state.get(f"{prefix}::example_class", next(iter(counts)))
            number_key = f"{prefix}::example_number"
            if st.session_state.get(number_key, 1) > max(counts.get(chosen, 1), 1):
                st.session_state[number_key] = 1  # another class has fewer examples
            st.number_input(f"Which example (1 to {counts.get(chosen, 1)})", min_value=1, max_value=max(counts.get(chosen, 1), 1), value=1, step=1, key=number_key)
            st.button("Load this real example", on_click=_on_load_example, args=(predictor, samples, prefix), key=f"{prefix}::btn_example")
        else:
            st.caption("No real example flows are available (run `scripts\\04_preprocess.py`).")
    with right:
        st.button("Use the training medians", on_click=_on_load_medians, args=(predictor, prefix), key=f"{prefix}::btn_medians")
    st.caption("Currently loaded: " + st.session_state.get(f"{prefix}::source", "the training medians (per-feature medians, not a real flow)"))

    st.markdown("**Feature values** (type any value; the input check explains what it accepts)")
    st.caption(GROUPING_RULE)
    for group, members in group_features(names).items():
        with st.expander(f"{group} ({len(members)})", expanded=False):
            columns = st.columns(2)
            for i, name in enumerate(members):
                columns[i % 2].text_input(name, key=field_key(name, prefix), help=field_help(specs[name]))
    pressed = st.button("Predict", type="primary", key=f"{prefix}::predict")
    return values_from_state(st.session_state, names, prefix), pressed


# ---------------------------------------------------------------------------
# Rendering: findings and results
# ---------------------------------------------------------------------------
def render_findings(validation: validator.ValidationResult, show_rows: bool = False) -> None:
    """The input check's findings: errors, warnings, then information."""
    summary = validation.summary()
    if summary["issues_not_listed"]:
        st.caption(f"{summary['issues_not_listed']:,} more finding(s) are counted but not listed.")
    issues = sorted(validation.issues, key=lambda i: (i.row is not None, i.row or 0)) if show_rows else validation.issues
    for issue in issues:
        text = str(issue) if show_rows else issue.message
        if issue.level == validator.ERROR:
            st.error(text)
        elif issue.level == validator.WARNING:
            st.warning(text)
        else:
            st.info(text)


def render_prediction(prediction: predictor_module.FlowPrediction, reliability: Mapping[str, Any] | None = None) -> None:
    """The answer for one flow: result, agreement, every model's vote, the tally and what the test set says about the class."""
    if not prediction.ok:
        st.error("The flow was not predicted: the input check found errors.")
        render_findings(prediction.validation)
        return
    benign = is_benign(prediction.label)
    headline = f"Predicted class: {prediction.label}"
    (st.success if benign else st.error)(headline + (" (benign traffic)" if benign else " (an attack class)"))
    a, b, c = st.columns(3)
    a.metric("Models voting", f"{prediction.n_voters} of {len(model_params.MODEL_KEYS)}")
    b.metric("Votes for the result", f"{prediction.top_votes} of {prediction.n_voters}")
    c.metric("Agreement", agreement_short(prediction.top_votes, prediction.n_voters))
    st.caption(agreement_band(prediction.top_votes, prediction.n_voters).capitalize() + ".")
    if prediction.tied:
        st.info(f"The vote was tied; the tie rule '{settings.TIE_BREAK_RULE}' decided (an attack class beats BENIGN).")
    if prediction.missing:
        st.warning("Some models did not vote: " + "; ".join(f"{model_params.DISPLAY_NAMES.get(k, k)} ({why})" for k, why in prediction.missing.items()))
    if prediction.validation.warnings:
        with st.expander(f"Input warnings ({len(prediction.validation.warnings)})", expanded=True):
            render_findings(prediction.validation)

    votes = prediction.votes.rename(columns={"model": "Model", "version": "Version", "vote": "Vote"}).copy()
    votes["Agrees with the result"] = votes["Vote"] == prediction.label
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**How each model voted**")
        st.dataframe(votes, hide_index=True, width="stretch")
    with right:
        st.markdown("**Vote tally**")
        tally = prediction.tally.rename(columns={"class": "Class", "votes": "Votes", "winner": "Result"})
        st.dataframe(tally, hide_index=True, width="stretch")
    with st.expander("What the test set says about this class", expanded=False):
        for line in reliability_lines(prediction.label, reliability):
            st.write("- " + line)
        st.caption("These figures describe the final vote on the held-out test flows of the dataset, not this particular flow.")


def render_batch_result(batch: predictor_module.BatchPrediction) -> None:
    """Summary, table and download for a table of flows."""
    summary = batch.summary()
    a, b, c = st.columns(3)
    a.metric("Rows read", f"{summary['n_rows']:,}")
    b.metric("Predicted", f"{summary['n_predicted']:,}")
    c.metric("Refused by the input check", f"{summary['n_refused']:,}")
    if batch.missing:
        st.warning("Some models did not vote: " + "; ".join(f"{model_params.DISPLAY_NAMES.get(k, k)} ({why})" for k, why in batch.missing.items()))
    if summary["class_counts"]:
        st.markdown("**Predicted classes**")
        counts = pd.Series(summary["class_counts"], name="Flows").sort_values(ascending=False)
        st.bar_chart(counts)
    table = results_table(batch)
    st.markdown("**Result per row**")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download the result as CSV", data=to_csv_bytes(table), file_name="predictions.csv", mime="text/csv")
    if batch.validation.issues:
        with st.expander(f"Input findings ({sum(batch.validation.counts.values()):,})", expanded=summary["n_refused"] > 0):
            render_findings(batch.validation, show_rows=True)


# ---------------------------------------------------------------------------
# Rendering: models and limits
# ---------------------------------------------------------------------------
def render_status(predictor: predictor_module.Predictor) -> None:
    """A short status for the sidebar."""
    loaded = len(predictor.models)
    st.metric("Models loaded", f"{loaded} of {len(model_params.MODEL_KEYS)}")
    tuned = sum(1 for k in predictor.models if predictor.versions.get(k) == "tuned")
    st.caption(f"{tuned} tuned and {loaded - tuned} baseline versions (the 'final set').")
    for key, why in predictor.not_loaded.items():
        st.warning(f"{model_params.DISPLAY_NAMES.get(key, key)} is not loaded: {why}")


def model_panel_table(predictor: predictor_module.Predictor, scores: Mapping[str, Any] | None) -> pd.DataFrame:
    """One line per model for the 'about the models' tab (version, why, training, test macro F1 of the version in use)."""
    table = predictor.model_table()
    models = (scores or {}).get("models", {})
    rows = []
    for _, r in table.iterrows():
        entry = models.get(r["key"], {})
        f1 = entry.get("tuned_macro_f1" if r["version"] == "tuned" else "baseline_macro_f1")
        rows.append(
            {
                "Model": r["model"], "Version used": r["version"], "Why": r["why"], "Loaded": bool(r["loaded"]),
                "Training rows": None if pd.isna(r["training_rows"]) else int(r["training_rows"]),
                "Class-capped subsample": r["trained_on_subsample"],
                "Tuned parameters": r["tuned_parameters"] if isinstance(r["tuned_parameters"], str) else "",
                "Test macro F1 (version used)": None if f1 is None else round(float(f1), 4),
            }
        )
    return pd.DataFrame(rows)


def render_models_panel(predictor: predictor_module.Predictor, scores: Mapping[str, Any] | None, reliability: Mapping[str, Any] | None) -> None:
    st.markdown(
        "**Which version of each model is used.** Each of the 11 models was tuned on a sample of the training rows. "
        "The tuned version is used only where the tuning cross-validation showed an improvement larger than its fold-to-fold noise; "
        "otherwise the baseline version is used. This rule was fixed before any tuned model was scored on the test set."
    )
    table = model_panel_table(predictor, scores)
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)  # tall enough to show every row
    votes = (scores or {}).get("votes", [])
    if votes:
        st.markdown("**The three majority votes on the test set**")
        frame = pd.DataFrame(votes)
        keep = [c for c in ("vote", "voters", "macro_f1", "accuracy", "detection_rate", "false_alarm_rate") if c in frame.columns]
        st.dataframe(frame[keep].rename(columns={"vote": "Vote", "voters": "Voters", "macro_f1": "Macro F1", "accuracy": "Accuracy", "detection_rate": "Detection rate", "false_alarm_rate": "False alarm rate"}).round(4), hide_index=True, width="stretch")
        st.caption("Macro F1 weights all classes equally, so the three classes with only 2 to 7 test flows move it noticeably.")
    else:
        st.caption("No test-set results found (run `python scripts\\06_evaluate.py --stage tuned`).")
    if reliability and reliability.get("per_class"):
        weak = [f"{name} ({c['support']} test flows)" for name, c in reliability["per_class"].items() if c.get("low_support")]
        if weak:
            st.markdown("**Classes with too few test flows to judge:** " + ", ".join(weak) + ".")
        if reliability.get("confusions"):
            st.markdown("**Most frequent mistakes of the final vote on the test set**")
            lines = []
            for item in reliability["confusions"][:6]:
                share = item.get("share_of_true_class")
                lines.append(f"- {item['rows']:,} '{item['true_class']}' flow(s) labelled '{item['predicted_class']}'" + (f" ({share * 100:.1f}% of that class)" if share is not None else ""))
            st.markdown("\n".join(lines))


def render_limits() -> None:
    st.markdown("**Limits**")
    st.markdown("\n".join(f"- {line}" for line in GENERAL_LIMITS))


# ---------------------------------------------------------------------------
# Demonstration (no browser needed):  python app\ui_components.py
# ---------------------------------------------------------------------------
def main() -> int:
    from nids.preprocessing import feature_schema as feature_schema_module

    try:
        schema = feature_schema_module.load_feature_schema()
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot read the feature schema: {error}")
        return 1
    names = validator.required_features(schema)
    groups = group_features(names)
    print(f"{len(names)} feature fields in {len(groups)} groups. {GROUPING_RULE}")
    for group, members in groups.items():
        print(f"  {group} ({len(members)}): " + ", ".join(members))
    medians = median_values(schema)
    print("\nFirst fields at the training median (as typed in the form):")
    for name in names[:5]:
        print(f"  {name} = {medians[name]}")
    try:
        counts = sample_counts(predictor_module.load_sample_flows())
        print(f"\nReal example flows per class ({sum(counts.values())} in all): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    except (FileNotFoundError, ValueError):
        print("\nReal example flows: not available (run scripts\\04_preprocess.py).")
    reliability, scores = load_class_reliability(), load_model_scores()
    print(f"Test-set results for the class panel: {'found for ' + str(len(reliability['per_class'])) + ' classes' if reliability else 'not found (run 06_evaluate.py --stage tuned)'}")
    print(f"Model scores for the models tab: {'found (' + str(len(scores['models'])) + ' models, ' + str(len(scores['votes'])) + ' votes)' if scores else 'not found (run 06_evaluate.py --stage tuned)'}")
    demo = io.StringIO(" Destination Port, Flow Duration, Label\n80,100,BENIGN\n")
    table, notes = read_uploaded_csv(demo)
    print(f"\nCSV reading demo: columns {list(table.columns)}; notes: {notes}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())