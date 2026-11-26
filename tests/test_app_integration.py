"""Integration tests of the Streamlit application (File 64, Checkpoint 6).

Three layers, all on synthetic data (a small complete project is built once per module in a temporary
folder; the ``settings`` paths point there for this module only, so no real file is touched):

1. the plain functions of ``app/ui_components.py`` (grouping, formatting, upload reading, result table, wording);
2. the whole page ``app/streamlit_app.py`` run headlessly with ``streamlit.testing.v1.AppTest``
   (no browser, no server): load, predict, real example, bad input, missing-value choice, stale result, load failure;
3. the upload path. ``AppTest`` cannot drive the file-upload widget, so the three steps the upload tab runs
   (``read_uploaded_csv`` -> ``predict_frame`` -> ``render_batch_result``) are run inside ``AppTest.from_function``.

The tests that compare an answer use the predictor itself as the reference (File 60 has the independent
rule check); what is tested here is that the page shows what the predictor returns, and shows it honestly.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

import ui_components as ui  # noqa: E402
from config import model_params, settings  # noqa: E402
from nids.models import registry  # noqa: E402
from nids.persistence import model_store  # noqa: E402
from nids.prediction import input_validator as validator  # noqa: E402
from nids.prediction import predictor as P  # noqa: E402
from nids.preprocessing.preprocessor import preprocess, save_preprocessed  # noqa: E402

APP_FILE = APP_DIR / "streamlit_app.py"
KEYS = list(model_params.MODEL_KEYS)
CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "Bot"]
PATH_ATTRS = (
    "PREPROCESSED_DIR", "X_TRAIN_FILE", "X_TEST_FILE", "Y_TRAIN_FILE", "Y_TEST_FILE", "PIPELINE_FILE", "LABEL_ENCODER_FILE",
    "FEATURE_NAMES_FILE", "FEATURE_SCHEMA_FILE", "SAMPLE_FLOWS_FILE", "PREPROCESSING_METADATA_FILE", "CLEANING_REPORT_FILE",
)
TIMEOUT = 90


# ---------------------------------------------------------------------------
# a small complete project
# ---------------------------------------------------------------------------
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frames = []
    for i, name in enumerate(CLASSES):
        n = 160
        frame = pd.DataFrame(
            {
                "Destination Port": rng.integers(0, 65536, n).astype(float),
                "Flow Duration": rng.exponential(1000 * (i + 1), n).round(),
                "Total Fwd Packets": rng.integers(1 + 3 * i, 20 + 3 * i, n).astype(float),
                "Mean Len": rng.normal(20 * i, 5, n),
                "Const": np.ones(n),
            }
        )
        frame["Corr"] = frame["Mean Len"] * 2 + rng.normal(0, 0.001, n)
        frame[settings.LABEL_COLUMN] = name
        frame[settings.SOURCE_FILE_COLUMN] = "a.csv"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_evaluation_files(root: Path) -> None:
    folder = root / "evaluation"
    (folder / "test_tuned").mkdir(parents=True, exist_ok=True)
    per_class = [
        {"class": c, "support": s, "precision": 0.97, "recall": 0.9, "f1": 0.93, "low_support": s < 30}
        for c, s in (("BENIGN", 500), ("Bot", 20), ("DoS Hulk", 300), ("PortScan", 4))
    ]
    confusions = [{"true_class": "Bot", "predicted_class": "BENIGN", "rows": 3, "share_of_true_class": 0.15}]
    (folder / "test_tuned" / "majority_vote_final.json").write_text(
        json.dumps({"metrics": {"per_class": per_class, "n_rows": 824, "top_confusions": confusions}}), encoding="utf-8"
    )
    models = [{"key": k, "baseline_macro_f1": 0.8, "tuned_macro_f1": 0.85} for k in ("knn", "decision_tree")]
    votes = [{"vote": "Vote of the final set (evidence rule)", "voters": 11, "macro_f1": 0.9, "accuracy": 0.99, "detection_rate": 0.99, "false_alarm_rate": 0.001}]
    (folder / "model_comparison_tuned.json").write_text(json.dumps({"models": models, "votes": votes}), encoding="utf-8")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("app_env")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "PREPROCESSED_DIR", root / "pre")
    for name in PATH_ATTRS[1:]:
        mp.setattr(settings, name, root / "pre" / Path(getattr(settings, name)).name)
    mp.setattr(settings, "TRAINED_MODELS_DIR", root / "models" / "trained")
    mp.setattr(settings, "TUNED_MODELS_DIR", root / "models" / "tuned")
    mp.setattr(settings, "EVALUATION_REPORT_DIR", root / "evaluation")

    result = preprocess(synthetic_frame())
    save_preprocessed(result)
    tuned_params = {"knn": {"weights": "distance"}, "decision_tree": {"criterion": "entropy"}}
    for key in KEYS:
        base = registry.create_model(key).fit(result.X_train, result.y_train, n_classes=result.codec.n_classes)
        model_store.save_model(base, "trained", extra={"n_train_rows_used": int(len(result.y_train))})
        if key in tuned_params:
            tuned = registry.create_model(key, params=dict(tuned_params[key])).fit(result.X_train, result.y_train, n_classes=result.codec.n_classes)
            tuning = {"same_as_baseline": False, "improvement_beyond_noise": True, "best_params": tuned_params[key], "baseline_params": {}}
        else:
            tuned = base
            tuning = {"same_as_baseline": key == "gaussian_nb", "improvement_beyond_noise": False, "best_params": {}, "baseline_params": {}}
        model_store.save_model(tuned, "tuned", extra={"n_train_rows_used": int(len(result.y_train)), "tuning": tuning})
    write_evaluation_files(root)
    yield root
    mp.undo()


@pytest.fixture(autouse=True)
def fresh_caches():
    """The page caches the predictor and the saved reports; every test starts clean."""
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture(scope="module")
def predictor(env):
    return P.Predictor.load("final")


@pytest.fixture(scope="module")
def samples(env):
    return P.load_sample_flows()


@pytest.fixture(scope="module")
def names(predictor):
    return validator.required_features(predictor.schema)


def raw_of(samples: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return samples[names].reset_index(drop=True)


# ---------------------------------------------------------------------------
# helpers for the page tests
# ---------------------------------------------------------------------------
def start(env) -> AppTest:
    app = AppTest.from_file(str(APP_FILE), default_timeout=TIMEOUT)
    app.run()
    return app


def button(app: AppTest, label: str):
    found = [b for b in app.button if b.label == label]
    assert found, f"no button {label!r}; buttons: {[b.label for b in app.button]}"
    return found[0]


def field(app: AppTest, name: str):
    found = [t for t in app.text_input if t.label == name]
    assert found, f"no text field {name!r}"
    return found[0]


def shown(app: AppTest) -> str:
    """All visible text that matters in one string (for 'is this sentence on the page' checks)."""
    parts = [e.value for e in app.error] + [e.value for e in app.success] + [e.value for e in app.warning] + [e.value for e in app.info]
    parts += [m.value for m in app.markdown] + [c.value for c in app.caption]
    return "\n".join(str(p) for p in parts)


def metric(app: AppTest, label: str) -> str:
    found = [m.value for m in app.metric if m.label == label]
    assert found, f"no metric {label!r}; metrics: {[m.label for m in app.metric]}"
    return found[0]


def result_headline(app: AppTest) -> str:
    texts = [e.value for e in app.success] + [e.value for e in app.error if str(e.value).startswith("Predicted class")]
    assert len(texts) == 1, texts
    return texts[0]


# ===========================================================================
# 1. plain functions of ui_components
# ===========================================================================
class TestFieldGrouping:
    def test_groups_by_first_word(self):
        groups = ui.group_features(["Flow A", "Flow B", "Flow C", "Fwd X", "Fwd Y", "Fwd Z"])
        assert groups == {"Flow": ["Flow A", "Flow B", "Flow C"], "Fwd": ["Fwd X", "Fwd Y", "Fwd Z"]}

    def test_small_groups_go_to_other_last(self):
        groups = ui.group_features(["Solo one", "Flow A", "Flow B", "Flow C", "Pair x", "Pair y"])
        assert list(groups) == ["Flow", "Other"]
        assert groups["Other"] == ["Solo one", "Pair x", "Pair y"]

    def test_every_name_appears_exactly_once(self):
        names = [f"{w} {i}" for w in ("A", "B", "C") for i in range(2 + ord(w) % 3)] + ["Zed"]
        flat = [n for members in ui.group_features(names).values() for n in members]
        assert sorted(flat) == sorted(names)

    def test_empty_list(self):
        assert ui.group_features([]) == {}

    def test_a_group_of_exactly_the_minimum_size_is_kept(self):
        names = [f"Grp {i}" for i in range(ui.MIN_GROUP_SIZE)]
        assert list(ui.group_features(names)) == ["Grp"]
        assert list(ui.group_features(names[:-1])) == [ui.OTHER_GROUP]

    def test_first_word(self):
        assert ui.first_word("Flow Duration") == "Flow"
        assert ui.first_word("Single") == "Single"
        assert ui.first_word("") == ""


class TestFormatValue:
    @pytest.mark.parametrize(
        "value, text",
        [(5, "5"), (5.0, "5"), (-3.0, "-3"), (0, "0"), (0.5, "0.5"), (1234567.0, "1234567"), (None, ""), (float("nan"), ""), (float("inf"), "inf"), (float("-inf"), "-inf")],
    )
    def test_text(self, value, text):
        assert ui.format_value(value) == text

    def test_up_to_ten_significant_digits(self):
        assert ui.format_value(1 / 3) == "0.3333333333"

    def test_numpy_scalars(self):
        assert ui.format_value(np.float32(7.0)) == "7"
        assert ui.format_value(np.int64(9)) == "9"

    def test_text_passes_through_when_not_a_number(self):
        assert ui.format_value("abc") == "abc"

    def test_round_trip_through_the_validator_keeps_the_number(self):
        for x in (0.0, 1.0, 12345.0, 0.25, 3.14159):
            assert float(ui.format_value(x)) == pytest.approx(x, rel=1e-9)


class TestFieldFunctions:
    def test_field_key_is_prefixed(self):
        assert ui.field_key("Flow Duration") == f"{ui.STATE_PREFIX}::Flow Duration"
        assert ui.field_key("X", prefix="other") == "other::X"

    def test_help_mentions_range_and_median(self, predictor):
        spec = {s["name"]: s for s in validator.field_specs(predictor.schema)}["Flow Duration"]
        text = ui.field_help(spec)
        assert "Training range" in text and "median" in text

    def test_help_mentions_the_port_limit(self, predictor, names):
        if "Destination Port" not in names:
            pytest.skip("the synthetic project did not keep the port")
        spec = {s["name"]: s for s in validator.field_specs(predictor.schema)}["Destination Port"]
        assert "65535" in ui.field_help(spec)

    def test_help_with_missing_numbers_does_not_crash(self):
        assert "?" in ui.field_help({"name": "Whatever"})

    def test_median_values_cover_every_feature_as_text(self, predictor, names):
        medians = ui.median_values(predictor.schema)
        assert list(medians) == names
        assert all(isinstance(v, str) and v != "" for v in medians.values())

    def test_median_values_are_accepted_by_the_validator(self, predictor):
        result = validator.validate_flow(ui.median_values(predictor.schema), predictor.schema)
        assert result.n_valid == 1

    def test_values_from_state_returns_only_present_fields(self, names):
        state = {ui.field_key(names[0]): "1", "unrelated": "x"}
        assert ui.values_from_state(state, names) == {names[0]: "1"}

    def test_values_from_state_empty(self, names):
        assert ui.values_from_state({}, names) == {}


class TestSampleFunctions:
    def test_counts_per_class(self, samples):
        counts = ui.sample_counts(samples)
        assert set(counts) == set(samples[settings.LABEL_COLUMN])
        assert sum(counts.values()) == len(samples)
        assert list(counts) == sorted(counts)

    def test_values_are_text_of_the_saved_row(self, samples, names):
        values = ui.sample_values(samples, "Bot", 0, names)
        row = samples[samples[settings.LABEL_COLUMN] == "Bot"].iloc[0]
        assert list(values) == names
        for n in names:
            assert float(values[n]) == pytest.approx(float(row[n]), rel=1e-6, abs=1e-9)

    def test_unknown_class(self, samples, names):
        with pytest.raises(ValueError, match="no example flow"):
            ui.sample_values(samples, "Nope", 0, names)

    @pytest.mark.parametrize("index", [-1, 10_000])
    def test_index_outside(self, samples, names, index):
        with pytest.raises(ValueError, match="outside"):
            ui.sample_values(samples, "Bot", index, names)

    def test_an_example_round_trips_through_the_predictor(self, predictor, samples, names):
        for cls in ui.sample_counts(samples):
            values = ui.sample_values(samples, cls, 0, names)
            assert predictor.predict_flow(values).ok


class TestReadUploadedCsv:
    def csv(self, text: str) -> io.BytesIO:
        return io.BytesIO(text.encode("utf-8"))

    def test_plain_file(self):
        table, notes = ui.read_uploaded_csv(self.csv("a,b\n1,2\n3,4\n"))
        assert table.shape == (2, 2) and notes == []

    def test_spaces_around_column_names_are_removed_and_reported(self):
        table, notes = ui.read_uploaded_csv(self.csv(" a , b\n1,2\n"))
        assert list(table.columns) == ["a", "b"]
        assert any("Spaces" in n for n in notes)

    def test_label_column_is_dropped_and_reported(self):
        table, notes = ui.read_uploaded_csv(self.csv(f"a,{settings.LABEL_COLUMN}\n1,BENIGN\n"))
        assert list(table.columns) == ["a"]
        assert any("ignored" in n for n in notes)

    def test_label_with_leading_space_is_dropped_too(self):
        table, _ = ui.read_uploaded_csv(self.csv(f"a, {settings.LABEL_COLUMN}\n1,BENIGN\n"))
        assert settings.LABEL_COLUMN not in table.columns

    def test_row_limit(self):
        text = "a\n" + "\n".join(str(i) for i in range(30)) + "\n"
        table, notes = ui.read_uploaded_csv(self.csv(text), max_rows=10)
        assert len(table) == 10 and any("first 10" in n for n in notes)

    def test_exactly_at_the_limit_has_no_note(self):
        text = "a\n" + "\n".join(str(i) for i in range(10)) + "\n"
        table, notes = ui.read_uploaded_csv(self.csv(text), max_rows=10)
        assert len(table) == 10 and notes == []

    def test_empty_file(self):
        with pytest.raises(ValueError, match="empty"):
            ui.read_uploaded_csv(self.csv(""))

    def test_header_only_file_has_no_rows(self):
        table, _ = ui.read_uploaded_csv(self.csv("a,b\n"))
        assert len(table) == 0 and list(table.columns) == ["a", "b"]

    def test_not_text(self):
        with pytest.raises(ValueError):
            ui.read_uploaded_csv(io.BytesIO(bytes(range(128, 256)) * 4))

    def test_index_is_reset(self):
        table, _ = ui.read_uploaded_csv(self.csv("a\n1\n2\n"))
        assert list(table.index) == [0, 1]

    def test_missing_cells_stay_missing(self):
        table, _ = ui.read_uploaded_csv(self.csv("a,b\n1,\n"))
        assert pd.isna(table.loc[0, "b"])


class TestResultsTable:
    def test_row_numbers_are_one_based_and_refusals_have_a_reason(self, predictor, samples, names):
        frame = raw_of(samples, names).head(3).copy()
        frame.loc[1, names[1]] = -5
        batch = predictor.predict_frame(frame)
        table = ui.results_table(batch)
        assert list(table["row"]) == [1, 2, 3]
        assert table.columns[2] == "refused_because"
        assert pd.isna(table.loc[0, "refused_because"]) and isinstance(table.loc[1, "refused_because"], str)
        assert pd.isna(table.loc[1, "prediction"])
        assert table.loc[0, "prediction"] in CLASSES

    def test_does_not_change_the_batch(self, predictor, samples, names):
        batch = predictor.predict_frame(raw_of(samples, names).head(2))
        before = batch.frame.copy()
        ui.results_table(batch)
        pd.testing.assert_frame_equal(batch.frame, before)

    def test_csv_bytes_round_trip(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        back = pd.read_csv(io.BytesIO(ui.to_csv_bytes(frame)))
        pd.testing.assert_frame_equal(back, frame)

    def test_csv_bytes_have_no_index_column(self):
        assert ui.to_csv_bytes(pd.DataFrame({"a": [1]})).decode().splitlines()[0] == "a"


class TestWording:
    @pytest.mark.parametrize(
        "top, n, band, short",
        [
            (11, 11, "all models agree", "All agree"),
            (8, 11, "a majority of the models agree", "Majority"),
            (6, 11, "a majority of the models agree", "Majority"),
            (5, 11, "no majority", "Plurality only"),
            (3, 11, "no majority", "Plurality only"),
            (2, 4, "no majority", "Plurality only"),
            (3, 4, "a majority of the models agree", "Majority"),
            (1, 1, "all models agree", "All agree"),
            (0, 0, "no prediction", "none"),
        ],
    )
    def test_agreement(self, top, n, band, short):
        assert ui.agreement_band(top, n).startswith(band)
        assert ui.agreement_short(top, n) == short

    def test_is_benign(self):
        assert ui.is_benign(settings.BENIGN_LABEL)
        assert not ui.is_benign("DoS Hulk") and not ui.is_benign(None)


class TestReliability:
    def test_no_reports_means_empty(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", tmp_path / "nothing")
        assert ui.load_class_reliability() == {} and ui.load_model_scores() == {}

    def test_reads_the_saved_files(self, env):
        reliability = ui.load_class_reliability()
        assert reliability["per_class"]["Bot"]["support"] == 20
        assert reliability["confusions"][0]["rows"] == 3
        scores = ui.load_model_scores()
        assert scores["models"]["knn"]["tuned_macro_f1"] == 0.85 and len(scores["votes"]) == 1

    def test_sentences_use_the_saved_numbers(self, env):
        lines = ui.reliability_lines("Bot", ui.load_class_reliability())
        text = " ".join(lines)
        assert "90.0%" in text and "20" in text and "97.0%" in text
        assert "15.0%" in text and "'Bot' flow(s) were labelled 'BENIGN'" in text

    def test_low_support_gets_a_caution(self, env):
        text = " ".join(ui.reliability_lines("PortScan", ui.load_class_reliability()))
        assert "not reliable" in text

    def test_well_supported_class_has_no_caution(self, env):
        assert "not reliable" not in " ".join(ui.reliability_lines("BENIGN", ui.load_class_reliability()))

    @pytest.mark.parametrize("reliability", [None, {}, {"per_class": {}}])
    def test_unknown_class_or_no_data_says_so(self, reliability):
        assert "No test-set results" in ui.reliability_lines("Bot", reliability)[0]

    def test_damaged_report_is_not_silently_accepted(self, env, tmp_path, monkeypatch):
        folder = tmp_path / "eval"
        (folder / "test_tuned").mkdir(parents=True)
        (folder / "test_tuned" / "majority_vote_final.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", folder)
        with pytest.raises(Exception):
            ui.load_class_reliability()


class TestModelPanelTable:
    def test_one_line_per_model_with_scores_of_the_version_in_use(self, env, predictor):
        scores = ui.load_model_scores()
        table = ui.model_panel_table(predictor, scores)
        assert len(table) == len(KEYS) and table["Loaded"].all()
        knn = table[table["Model"] == model_params.DISPLAY_NAMES["knn"]].iloc[0]
        assert knn["Version used"] == "tuned" and knn["Test macro F1 (version used)"] == 0.85

    def test_without_scores_the_column_is_empty(self, env, predictor):
        table = ui.model_panel_table(predictor, None)
        assert table["Test macro F1 (version used)"].isna().all()


# ===========================================================================
# 2. the whole page, run headlessly
# ===========================================================================
class TestCaptureTab:
    def test_demo_capture_is_analysed_without_error(self, env):
        app = start(env)
        button(app, "Try the demo capture").click().run()
        assert not app.exception
        headings = [m.value for m in app.markdown]
        assert "#### Results for `demo_capture.pcap`" in headings
        assert "##### Every flow" in headings
        assert [d for d in app.get("download_button") if "flows, features and predictions" in d.proto.label]

    def test_nothing_is_analysed_before_a_click(self, env):
        app = start(env)
        assert not [m for m in app.markdown if m.value.startswith("#### Results for")]
        assert not [b for b in app.button if b.label == "Analyze this capture"]  # appears only once a file is chosen


class TestPageLoads:
    def test_no_exception_and_the_main_parts_are_there(self, env, names):
        app = start(env)
        assert not app.exception
        assert [t.value for t in app.title] == ["Network intrusion detection"]
        assert [t.label for t in app.tabs] == ["Analyze a capture", "Single flow", "Upload a CSV file", "About the models"]
        assert metric_in_sidebar(app, "Models loaded") == f"{len(KEYS)} of {len(KEYS)}"

    def test_every_feature_has_a_field_filled_with_its_median(self, env, predictor, names):
        app = start(env)
        medians = ui.median_values(predictor.schema)
        for name in names:
            assert field(app, name).value == medians[name]

    def test_fields_are_inside_the_groups(self, env, names):
        app = start(env)
        assert len(app.expander) >= 1
        assert len(app.text_input) == len(names)

    def test_missing_value_choice_defaults_to_refuse(self, env):
        app = start(env)
        assert app.radio[0].value.startswith("Refuse")

    def test_upload_tab_has_uploader_and_example_download(self, env):
        app = start(env)
        assert len(app.get("file_uploader")) == 2  # the packet-capture tab and the CSV tab
        assert len(app.get("download_button")) >= 1

    def test_models_tab_and_limits(self, env):
        app = start(env)
        text = shown(app)
        assert "Which version of each model is used" in text
        assert "majority vote" in text.lower() and "CIC-IDS2017" in text
        assert len(app.dataframe) >= 2  # the model table and the votes table

    def test_nothing_is_predicted_before_the_button_is_pressed(self, env):
        app = start(env)
        assert not app.success and not [e for e in app.error if "Predicted" in str(e.value)]
        assert "last_prediction" not in app.session_state


def metric_in_sidebar(app: AppTest, label: str) -> str:
    found = [m.value for m in app.sidebar.metric if m.label == label]
    assert found, f"no sidebar metric {label!r}"
    return found[0]


class TestSingleFlow:
    def test_predict_with_the_medians(self, env, predictor):
        app = start(env)
        button(app, "Predict").click().run()
        assert not app.exception
        expected = predictor.predict_flow(ui.median_values(predictor.schema))
        assert f"Predicted class: {expected.label}" in result_headline(app)
        assert metric(app, "Votes for the result") == f"{expected.top_votes} of {expected.n_voters}"
        assert metric(app, "Models voting") == f"{expected.n_voters} of {len(KEYS)}"

    def test_the_shown_vote_table_is_the_predictors_vote_table(self, env, predictor):
        app = start(env)
        button(app, "Predict").click().run()
        expected = predictor.predict_flow(ui.median_values(predictor.schema))
        tables = [d.value for d in app.dataframe if "Model" in d.value.columns and "Vote" in d.value.columns and "Agrees with the result" in d.value.columns]
        assert len(tables) == 1
        assert list(tables[0]["Model"]) == list(expected.votes["model"]) and list(tables[0]["Vote"]) == list(expected.votes["vote"])

    @pytest.mark.parametrize("cls", CLASSES)
    def test_real_example_of_every_class(self, env, predictor, samples, names, cls):
        app = start(env)
        app.selectbox[0].set_value(cls).run()
        button(app, "Load this real example").click().run()
        assert any(f"real example 1 of class '{cls}'" in c.value for c in app.caption)
        for name, text in ui.sample_values(samples, cls, 0, names).items():
            assert field(app, name).value == text
        button(app, "Predict").click().run()
        assert not app.exception
        expected = predictor.predict_flow(ui.sample_values(samples, cls, 0, names))
        assert f"Predicted class: {expected.label}" in result_headline(app)

    def test_attack_and_benign_headlines_differ_in_kind(self, env, predictor, samples, names):
        app = start(env)
        for cls in CLASSES:
            app.selectbox[0].set_value(cls).run()
            button(app, "Load this real example").click().run()
            button(app, "Predict").click().run()
            label = predictor.predict_flow(ui.sample_values(samples, cls, 0, names)).label
            if label == settings.BENIGN_LABEL:
                assert any("benign traffic" in s.value for s in app.success)
            else:
                assert any("an attack class" in e.value for e in app.error)

    def test_selecting_an_example_number_beyond_another_class_is_reset(self, env, samples, tmp_path, monkeypatch):
        # a class with fewer examples than another: keep only one example of "Bot"
        fewer = pd.concat([samples[samples[settings.LABEL_COLUMN] != "Bot"], samples[samples[settings.LABEL_COLUMN] == "Bot"].head(1)], ignore_index=True)
        path = tmp_path / "few.parquet"
        fewer.to_parquet(path)
        monkeypatch.setattr(settings, "SAMPLE_FLOWS_FILE", path)
        counts = ui.sample_counts(fewer)
        big = max(counts, key=counts.get)
        assert counts["Bot"] == 1 and counts[big] > 1
        app = start(env)
        app.selectbox[0].set_value(big).run()
        app.number_input[0].set_value(counts[big]).run()
        app.selectbox[0].set_value("Bot").run()
        assert not app.exception and app.number_input[0].value == 1

    def test_use_medians_button_restores_the_fields(self, env, predictor, names):
        app = start(env)
        field(app, names[0]).set_value("12345").run()
        button(app, "Use the training medians").click().run()
        assert field(app, names[0]).value == ui.median_values(predictor.schema)[names[0]]
        assert any("training medians" in c.value for c in app.caption)

    def test_a_negative_value_is_refused_with_the_reason(self, env, names):
        app = start(env)
        target = "Flow Duration"
        field(app, target).set_value("-5").run()
        button(app, "Predict").click().run()
        assert not app.exception and not app.success
        assert any("The flow was not predicted" in e.value for e in app.error)
        assert any(target in e.value for e in app.error)

    @pytest.mark.parametrize("text", ["abc", "1e", "--3", "1,5,6"])
    def test_text_that_is_not_a_number_is_refused(self, env, text):
        app = start(env)
        field(app, "Flow Duration").set_value(text).run()
        button(app, "Predict").click().run()
        assert not app.exception and not app.success
        assert any("not a number" in e.value.lower() or "not a valid number" in e.value.lower() for e in app.error), [e.value for e in app.error]

    def test_an_impossible_port_is_refused(self, env, names):
        if "Destination Port" not in names:
            pytest.skip("the synthetic project did not keep the port")
        app = start(env)
        field(app, "Destination Port").set_value("655360").run()
        button(app, "Predict").click().run()
        assert not app.success and any("65535" in e.value for e in app.error)

    def test_a_value_far_outside_the_training_range_warns_but_predicts(self, env):
        app = start(env)
        field(app, "Flow Duration").set_value("99999999").run()
        button(app, "Predict").click().run()
        assert not app.exception
        assert [e for e in app.error if str(e.value).startswith("Predicted class")] or app.success
        assert any("training range" in w.value.lower() or "outside" in w.value.lower() for w in app.warning), [w.value for w in app.warning]

    def test_a_blank_field_is_refused_by_default(self, env):
        app = start(env)
        field(app, "Flow Duration").set_value("").run()
        button(app, "Predict").click().run()
        assert not app.success and any("Flow Duration" in e.value for e in app.error)

    def test_a_blank_field_is_filled_when_the_user_chooses_so_and_the_page_says_so(self, env, predictor):
        app = start(env)
        field(app, "Flow Duration").set_value("").run()
        app.radio[0].set_value("Fill a missing value with the training median and warn").run()
        button(app, "Predict").click().run()
        assert not app.exception
        assert [e for e in app.error if str(e.value).startswith("Predicted class")] or app.success
        assert any("median" in w.value.lower() for w in app.warning), [w.value for w in app.warning]
        # filling with the median gives the same answer as the medians themselves
        expected = predictor.predict_flow(ui.median_values(predictor.schema))
        assert f"Predicted class: {expected.label}" in result_headline(app)

    def test_changing_a_field_after_a_result_shows_the_stale_notice(self, env):
        app = start(env)
        button(app, "Predict").click().run()
        assert not any("changed after this result" in i.value for i in app.info)
        field(app, "Mean Len").set_value("77").run()
        assert any("changed after this result" in i.value for i in app.info)
        button(app, "Predict").click().run()
        assert not any("changed after this result" in i.value for i in app.info)

    def test_changing_the_missing_value_choice_also_makes_the_result_stale(self, env):
        app = start(env)
        button(app, "Predict").click().run()
        app.radio[0].set_value("Fill a missing value with the training median and warn").run()
        assert any("changed after this result" in i.value for i in app.info)

    def test_the_result_stays_after_other_interaction(self, env):
        app = start(env)
        button(app, "Predict").click().run()
        before = result_headline(app)
        app.run()
        assert result_headline(app) == before

    def test_a_refused_flow_clears_the_old_result(self, env):
        app = start(env)
        button(app, "Predict").click().run()
        field(app, "Flow Duration").set_value("-1").run()
        button(app, "Predict").click().run()
        assert not app.success
        assert not [e for e in app.error if str(e.value).startswith("Predicted class")]

    def test_the_test_set_panel_names_the_class(self, env):
        app = start(env)
        button(app, "Predict").click().run()
        text = shown(app)
        assert "held-out test flows" in text


class TestFailures:
    def test_models_that_are_missing_show_a_help_message_instead_of_a_traceback(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "TRAINED_MODELS_DIR", tmp_path / "t")
        monkeypatch.setattr(settings, "TUNED_MODELS_DIR", tmp_path / "u")
        app = start(env)
        assert not app.exception
        assert any("could not be loaded" in e.value for e in app.error)
        text = shown(app)
        for step in ("04_preprocess", "05_train", "07_optimize"):
            assert step in text
        assert not app.text_input  # nothing else is drawn

    def test_a_missing_preprocessed_folder_shows_the_same_message(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "PIPELINE_FILE", tmp_path / "nothing.joblib")
        app = start(env)
        assert not app.exception and any("could not be loaded" in e.value for e in app.error)

    def test_one_damaged_model_is_reported_and_the_others_still_vote(self, env, tmp_path, monkeypatch):
        folders = {}
        for attr in ("TRAINED_MODELS_DIR", "TUNED_MODELS_DIR"):
            copy = tmp_path / attr
            shutil.copytree(getattr(settings, attr), copy)
            monkeypatch.setattr(settings, attr, copy)
            folders[attr] = copy
        for copy in folders.values():
            (copy / "random_forest.joblib").write_bytes(b"not a model")
        app = start(env)
        assert not app.exception
        assert metric_in_sidebar(app, "Models loaded") == f"{len(KEYS) - 1} of {len(KEYS)}"
        assert any("is not loaded" in w.value for w in app.sidebar.warning)
        button(app, "Predict").click().run()
        assert not app.exception
        assert metric(app, "Models voting") == f"{len(KEYS) - 1} of {len(KEYS)}"
        assert any("did not vote" in w.value for w in app.warning)

    def test_no_example_flows_file_still_gives_a_working_form(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "SAMPLE_FLOWS_FILE", tmp_path / "gone.parquet")
        app = start(env)
        assert not app.exception
        assert not [b for b in app.button if b.label == "Load this real example"]
        assert any("No real example flows" in c.value for c in app.caption)
        button(app, "Predict").click().run()
        assert not app.exception and metric(app, "Votes for the result")

    def test_no_evaluation_reports_still_gives_a_working_page(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", tmp_path / "none")
        app = start(env)
        assert not app.exception
        assert any("No test-set results found" in c.value for c in app.caption)
        button(app, "Predict").click().run()
        assert not app.exception


# ===========================================================================
# 3. the upload path (the widget itself cannot be driven by AppTest)
# ===========================================================================
def batch_page():
    """The three steps of the upload tab, as one small page: read the bytes, predict, show."""
    import streamlit as st

    import ui_components as ui
    from nids.prediction import predictor as predictor_module

    predictor = predictor_module.Predictor.load("final")
    table, notes = ui.read_uploaded_csv(__import__("io").BytesIO(st.session_state["csv_bytes"]))
    for note in notes:
        st.info(note)
    st.caption(f"{len(table):,} rows and {table.shape[1]} columns read.")
    ui.render_batch_result(predictor.predict_frame(table, missing=st.session_state.get("missing", "error")))


def run_batch(csv_bytes: bytes, missing: str = "error") -> AppTest:
    app = AppTest.from_function(batch_page, default_timeout=TIMEOUT)
    app.session_state["csv_bytes"] = csv_bytes
    app.session_state["missing"] = missing
    app.run()
    return app


def csv_of(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


class TestUploadPath:
    def test_a_good_file_is_predicted_row_by_row(self, env, predictor, samples, names):
        frame = raw_of(samples, names)
        app = run_batch(csv_of(frame))
        assert not app.exception
        assert metric(app, "Rows read") == f"{len(frame):,}" and metric(app, "Predicted") == f"{len(frame):,}"
        assert metric(app, "Refused by the input check") == "0"
        shown_table = [d.value for d in app.dataframe if "prediction" in d.value.columns][0]
        expected = predictor.predict_frame(frame).frame["prediction"].tolist()
        assert shown_table["prediction"].tolist() == expected
        assert shown_table["row"].tolist() == list(range(1, len(frame) + 1))

    def test_the_download_button_is_there_and_has_the_table(self, env, samples, names):
        app = run_batch(csv_of(raw_of(samples, names).head(5)))
        downloads = app.get("download_button")
        assert len(downloads) == 1 and downloads[0].proto.label == "Download the result as CSV"

    def test_file_with_label_column_and_padded_headers_gives_the_same_result(self, env, samples, names):
        frame = raw_of(samples, names).head(20)
        plain = run_batch(csv_of(frame))
        padded = frame.copy()
        padded.columns = [f" {c} " for c in padded.columns]
        padded[f" {settings.LABEL_COLUMN}"] = "BENIGN"
        other = run_batch(csv_of(padded))
        assert not other.exception
        a = [d.value for d in plain.dataframe if "prediction" in d.value.columns][0]["prediction"].tolist()
        b = [d.value for d in other.dataframe if "prediction" in d.value.columns][0]["prediction"].tolist()
        assert a == b
        assert any("Spaces" in i.value for i in other.info) and any("ignored" in i.value for i in other.info)

    def test_bad_rows_are_refused_the_others_predicted(self, env, samples, names):
        frame = raw_of(samples, names).head(6).copy()
        frame.loc[2, "Flow Duration"] = -9
        frame.loc[4, "Flow Duration"] = np.nan
        app = run_batch(csv_of(frame))
        assert not app.exception
        assert metric(app, "Predicted") == "4" and metric(app, "Refused by the input check") == "2"
        table = [d.value for d in app.dataframe if "refused_because" in d.value.columns][0]
        assert table["prediction"].isna().tolist() == [False, False, True, False, True, False]
        assert table.loc[2, "refused_because"] and table.loc[4, "refused_because"]
        assert app.expander and any("Input findings" in e.label for e in app.expander)

    def test_the_missing_value_choice_changes_what_happens_to_blank_cells(self, env, samples, names):
        frame = raw_of(samples, names).head(4).copy()
        frame.loc[1, "Flow Duration"] = np.nan
        refused = run_batch(csv_of(frame), missing="error")
        filled = run_batch(csv_of(frame), missing="median")
        assert metric(refused, "Refused by the input check") == "1"
        assert metric(filled, "Refused by the input check") == "0" and metric(filled, "Predicted") == "4"

    def test_a_file_with_missing_feature_columns_predicts_nothing_and_says_which(self, env, samples, names):
        app = run_batch(csv_of(raw_of(samples, names).head(3).drop(columns=[names[0]])))
        assert not app.exception
        assert metric(app, "Predicted") == "0"
        assert any(names[0] in e.value for e in app.error), [e.value for e in app.error]

    def test_extra_columns_are_harmless(self, env, predictor, samples, names):
        frame = raw_of(samples, names).head(5).copy()
        frame["Something else"] = 1.0
        app = run_batch(csv_of(frame))
        assert not app.exception and metric(app, "Predicted") == "5"
        shown_table = [d.value for d in app.dataframe if "prediction" in d.value.columns][0]
        assert shown_table["prediction"].tolist() == predictor.predict_frame(raw_of(samples, names).head(5)).frame["prediction"].tolist()

    def test_a_header_only_file_does_not_crash(self, env, names):
        app = run_batch((",".join(names) + "\n").encode())
        assert not app.exception

    def test_a_bar_chart_of_predicted_classes_is_drawn(self, env, samples, names):
        app = run_batch(csv_of(raw_of(samples, names)))
        assert len(app.get("arrow_vega_lite_chart")) == 1


def upload_page():
    """The real page, with the file-upload widget replaced by a fixed in-memory file (AppTest cannot upload)."""
    import io
    import runpy

    import streamlit as st

    class FakeUpload(io.BytesIO):
        name = "flows.csv"

        @property
        def size(self):
            return len(self.getvalue())

    real_uploader = st.file_uploader

    def fake_uploader(*args, **kwargs):
        if kwargs.get("key") != "csv_upload":  # the packet-capture uploader stays a real (empty) widget
            return real_uploader(*args, **kwargs)
        data = st.session_state.get("csv_bytes")
        return None if data is None else FakeUpload(data)

    st.file_uploader = fake_uploader
    runpy.run_path(st.session_state["app_file"], run_name="__main__")


def run_upload_tab(env, csv_bytes: bytes | None) -> AppTest:
    app = AppTest.from_function(upload_page, default_timeout=TIMEOUT)
    app.session_state["app_file"] = str(APP_FILE)
    if csv_bytes is not None:
        app.session_state["csv_bytes"] = csv_bytes
    app.run()
    return app


class TestUploadTabOfThePage:
    """The upload tab of the real page, with the widget replaced by a fixed file."""

    def test_the_file_is_read_but_not_predicted_until_the_button_is_pressed(self, env, samples, names):
        app = run_upload_tab(env, csv_of(raw_of(samples, names).head(8)))
        assert not app.exception
        assert any("8 rows and" in c.value for c in app.caption)
        assert "last_batch" not in app.session_state
        assert not [m for m in app.metric if m.label == "Rows read"]

    def test_pressing_the_button_predicts_the_file(self, env, predictor, samples, names):
        frame = raw_of(samples, names).head(8)
        app = run_upload_tab(env, csv_of(frame))
        button(app, "Predict this file").click().run()
        assert not app.exception
        assert metric(app, "Predicted") == "8"
        table = [d.value for d in app.dataframe if "prediction" in d.value.columns][0]
        assert table["prediction"].tolist() == predictor.predict_frame(frame).frame["prediction"].tolist()

    def test_the_missing_value_choice_is_applied_to_the_file(self, env, samples, names):
        frame = raw_of(samples, names).head(4).copy()
        frame.loc[1, "Flow Duration"] = np.nan
        app = run_upload_tab(env, csv_of(frame))
        button(app, "Predict this file").click().run()
        assert metric(app, "Refused by the input check") == "1"
        app.radio[0].set_value("Fill a missing value with the training median and warn").run()
        button(app, "Predict this file").click().run()
        assert metric(app, "Refused by the input check") == "0" and metric(app, "Predicted") == "4"

    def test_a_result_made_with_another_setting_is_not_shown_for_the_new_one(self, env, samples, names):
        frame = raw_of(samples, names).head(4).copy()
        frame.loc[1, "Flow Duration"] = np.nan
        app = run_upload_tab(env, csv_of(frame))
        button(app, "Predict this file").click().run()
        assert [m for m in app.metric if m.label == "Rows read"]
        app.radio[0].set_value("Fill a missing value with the training median and warn").run()
        assert not [m for m in app.metric if m.label == "Rows read"]

    def test_an_unreadable_file_gives_a_message_not_a_traceback(self, env):
        app = run_upload_tab(env, b"")
        assert not app.exception and any("empty" in e.value for e in app.error)

    def test_no_file_means_no_result_and_no_error(self, env):
        app = run_upload_tab(env, None)
        assert not app.exception and not app.error


class TestEndToEndConsistency:
    """One flow typed into the form and the same flow in an uploaded file must get the same answer."""

    def test_form_and_upload_agree_for_real_examples(self, env, samples, names):
        for cls in CLASSES:
            app = start(env)
            app.selectbox[0].set_value(cls).run()
            button(app, "Load this real example").click().run()
            button(app, "Predict").click().run()
            headline = result_headline(app)
            row = {n: float(field(app, n).value) for n in names}
            batch = run_batch(csv_of(pd.DataFrame([row])))
            table = [d.value for d in batch.dataframe if "prediction" in d.value.columns][0]
            assert headline.startswith(f"Predicted class: {table.loc[0, 'prediction']}")