"""Network intrusion detection: the Streamlit application (File 63).

Start it from the project folder (virtual environment active)::

    streamlit run app\\streamlit_app.py

Five pages (``st.navigation``):

* **Detection** (default): composes the building blocks of ``ui_components.py``; the work is done by
  ``nids.prediction.predictor`` (input check, preprocessing, the 11 models, the majority vote).
* **Analysis**: data quality, exploratory analysis, statistical tests, benchmark and batch inference
  (``page_*.py``); they read small committed files in ``reports/`` and work without the trained models.

Nothing is trained or tuned here, and nothing typed in the page is stored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "import ui_components" works however the page is started

import streamlit as st  # noqa: E402

import capture_view  # noqa: E402
import page_benchmark  # noqa: E402
import page_eda  # noqa: E402
import page_quality  # noqa: E402
import page_stats  # noqa: E402
import style  # noqa: E402
import ui_components as ui  # noqa: E402
from config import model_params, settings  # noqa: E402
from nids.prediction import predictor as predictor_module  # noqa: E402
from nids.prediction import input_validator as validator  # noqa: E402

MISSING_CHOICES = {
    "Refuse a flow that has a missing value (default)": "error",
    "Fill a missing value with the training median and warn": "median",
}


def show_load_failure(error: Exception) -> None:
    st.error(f"The models could not be loaded: {error}")
    st.markdown(
        "Run these steps from the project folder, then reload the page:\n\n"
        "1. `python scripts\\04_preprocess.py` (preprocessing, schema, example flows)\n"
        "2. `python scripts\\05_train.py` (baseline models)\n"
        "3. `python scripts\\07_optimize.py` (tuned models)\n"
        "4. `python scripts\\06_evaluate.py --stage tuned` (optional: the quality panels)"
    )


def single_flow_tab(predictor: predictor_module.Predictor, samples, reliability, missing: str) -> None:
    st.markdown(
        f"Type the {len(validator.required_features(predictor.schema))} feature values of one network flow (or start from a real example) and press **Predict**. "
        "The input is checked first; then all available models vote and the majority wins."
    )
    values, pressed = ui.render_form(predictor, samples)
    if pressed:
        with st.spinner("Predicting ..."):
            try:
                st.session_state["last_prediction"] = (dict(values), missing, predictor.predict_flow(values, missing=missing))
            except predictor_module.PredictorError as error:
                st.session_state.pop("last_prediction", None)
                st.error(f"The prediction failed: {error}")
    stored = st.session_state.get("last_prediction")
    if stored is not None:
        used_values, used_missing, result = stored
        st.divider()
        if used_values != values or used_missing != missing:
            st.info("The fields or the missing-value setting changed after this result. Press **Predict** again to update it.")
        ui.render_prediction(result, reliability)


def upload_tab(predictor: predictor_module.Predictor, samples, missing: str) -> None:
    st.markdown(
        "Upload a CSV file with one flow per row and the feature names as column headers. "
        f"At most {ui.MAX_UPLOAD_ROWS:,} rows are read; a `Label` column, if present, is ignored."
    )
    if samples is not None and len(samples):
        example = samples.drop(columns=[c for c in (settings.LABEL_COLUMN, "test_row_position", "cleaned_row_position") if c in samples.columns])
        st.download_button(
            f"Download {len(example)} real example flows as a CSV (to try the upload)", data=ui.to_csv_bytes(example),
            file_name="example_flows.csv", mime="text/csv",
        )
    upload = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
    if upload is None:
        return
    try:
        table, notes = ui.read_uploaded_csv(upload)
    except ValueError as error:
        st.error(str(error))
        return
    for note in notes:
        st.info(note)
    st.caption(f"{len(table):,} rows and {table.shape[1]} columns read.")
    signature = (upload.name, upload.size, missing)
    if st.button("Predict this file", type="primary", key="csv_predict"):
        with st.spinner(f"Predicting {len(table):,} flows ..."):
            try:
                st.session_state["last_batch"] = (signature, predictor.predict_frame(table, missing=missing))
            except predictor_module.PredictorError as error:
                st.session_state.pop("last_batch", None)
                st.error(f"The prediction failed: {error}")
    stored = st.session_state.get("last_batch")
    if stored is not None and stored[0] == signature:
        st.divider()
        ui.render_batch_result(stored[1])


TABS = ["Analyze a capture", "Single flow", "Upload a CSV file", "About the models"]


def header(predictor: predictor_module.Predictor, scores) -> None:
    st.title("Network intrusion detection")
    style.hero(
        "Upload a packet capture or describe a network flow: eleven machine-learning models trained on CIC-IDS2017 "
        "vote on whether it is benign traffic or one of 14 attack types.",
        ["11 models", "majority vote", "15 classes", "CIC-IDS2017", ".pcap / .pcapng / .csv"],
    )
    final_vote = next((v for v in (scores or {}).get("votes", []) if "final" in str(v.get("key", ""))), None)
    style.kpi_row(
        [
            ("Models loaded", f"{len(predictor.models)} / {len(model_params.MODEL_KEYS)}", "final set (tuned or baseline)", style.ACCENT),
            ("Classes", f"{len(predictor.classes)}", "BENIGN + 14 attack types", None),
            ("Test macro F1", f"{final_vote['macro_f1']:.3f}" if final_vote else "n/a", "majority vote, held-out flows", None),
            ("Attacks detected", f"{final_vote['detection_rate']:.2%}" if final_vote else "n/a", f"false alarms {final_vote['false_alarm_rate']:.2%}" if final_vote else "", style.BENIGN),
        ]
    )
    st.write("")


def detection_page() -> None:
    try:
        predictor = ui.get_predictor()
    except (predictor_module.PredictorError, FileNotFoundError, ValueError) as error:
        st.title("Network intrusion detection")
        show_load_failure(error)
        st.stop()
    samples = ui.get_samples() if settings.SAMPLE_FLOWS_FILE.is_file() else None
    reliability, scores = ui.get_reliability(), ui.get_model_scores()
    header(predictor, scores)

    with st.sidebar:
        st.header(":material/monitor_heart: Status")
        ui.render_status(predictor)
        st.header(":material/help: Missing values")
        choice = st.radio("If a feature value is missing", list(MISSING_CHOICES), key="missing_choice", label_visibility="collapsed")
        st.caption("Applies to typed flows and CSV files. Typed flows normally have every value. Missing values are filled with the training median only if you choose so, and the page then says which.")
        st.caption(f"{len(validator.required_features(predictor.schema))} features are required; the other columns of the dataset are not used by the models.")
        st.header(":material/info: Good to know")
        st.caption("Packet captures always fill the undefined rates of zero-length flows with the training median, as the dataset did.")
        st.caption("A prediction is a vote of models trained on one lab network: treat it as a lead, not proof.")
    missing = MISSING_CHOICES[choice]

    tab_capture, tab_single, tab_upload, tab_models = st.tabs(TABS)
    with tab_capture:
        capture_view.capture_tab(predictor)
    with tab_single:
        single_flow_tab(predictor, samples, reliability, missing)
    with tab_upload:
        upload_tab(predictor, samples, missing)
    with tab_models:
        ui.render_models_panel(predictor, scores, reliability)
        ui.render_limits()


def main() -> None:
    st.set_page_config(page_title="Network intrusion detection", page_icon=":material/shield:", layout="wide")
    style.inject_css()
    # The analysis pages read only small committed files (reports/), so they work without the trained models.
    pages = {
        "Detection": [
            st.Page(detection_page, title="Detection (11-model vote)", icon=":material/shield:", url_path="detection", default=True),
        ],
        "Analysis": [
            st.Page(page_quality.render, title="1. Data quality", icon=":material/fact_check:", url_path="quality"),
            st.Page(page_eda.render, title="2. Exploratory analysis", icon=":material/insights:", url_path="eda"),
            st.Page(page_stats.render, title="3. Statistical tests", icon=":material/functions:", url_path="statistics"),
            st.Page(page_benchmark.render, title="4. Benchmark & batch inference", icon=":material/leaderboard:", url_path="benchmark"),
        ],
    }
    st.navigation(pages).run()


main()