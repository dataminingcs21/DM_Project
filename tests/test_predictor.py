"""Tests for the predictor (File 60).

Synthetic data only.  A small complete project is built once per test module (preprocessing with the
project's own ``preprocess`` and ``save_preprocessed``, 11 baseline and 11 tuned models) in a temporary
folder; the ``settings`` paths are redirected there for the duration of this module, so no other test
and no real file is touched.  Tests that damage files work on a copy.

The main correctness tests compare the predictor with an independent, plain-Python computation of the
written rule (pipeline -> each model's own ``predict`` -> count the votes -> tie rule).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import model_params, settings
from nids.models import registry
from nids.persistence import model_store
from nids.prediction import input_validator as validator
from nids.prediction import predictor as P
from nids.preprocessing.preprocessor import preprocess, save_preprocessed
from nids.utils.io_utils import load_array, load_joblib, save_array, write_json

KEYS = list(model_params.MODEL_KEYS)
CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "Bot"]
TUNED_PARAMS = {"knn": {"weights": "distance"}, "decision_tree": {"criterion": "entropy"}}
PATH_ATTRS = (
    "PREPROCESSED_DIR", "X_TRAIN_FILE", "X_TEST_FILE", "Y_TRAIN_FILE", "Y_TEST_FILE", "PIPELINE_FILE", "LABEL_ENCODER_FILE",
    "FEATURE_NAMES_FILE", "FEATURE_SCHEMA_FILE", "SAMPLE_FLOWS_FILE", "PREPROCESSING_METADATA_FILE", "CLEANING_REPORT_FILE",
)


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


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Build the small project once; yield its root while the settings point there."""
    root = tmp_path_factory.mktemp("predictor_env")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "PREPROCESSED_DIR", root / "pre")
    for name in PATH_ATTRS[1:]:
        mp.setattr(settings, name, root / "pre" / Path(getattr(settings, name)).name)
    mp.setattr(settings, "TRAINED_MODELS_DIR", root / "models" / "trained")
    mp.setattr(settings, "TUNED_MODELS_DIR", root / "models" / "tuned")
    mp.setattr(settings, "EVALUATION_REPORT_DIR", root / "evaluation")

    result = preprocess(synthetic_frame())
    save_preprocessed(result)
    for key in KEYS:
        base = registry.create_model(key).fit(result.X_train, result.y_train, n_classes=result.codec.n_classes)
        model_store.save_model(base, "trained", extra={"n_train_rows_used": int(len(result.y_train))})
        if key in TUNED_PARAMS:
            tuned = registry.create_model(key, params=dict(TUNED_PARAMS[key])).fit(result.X_train, result.y_train, n_classes=result.codec.n_classes)
            tuning = {"same_as_baseline": False, "improvement_beyond_noise": True, "best_params": TUNED_PARAMS[key], "baseline_params": {}}
        else:
            tuned = base
            tuning = {"same_as_baseline": key == "gaussian_nb", "improvement_beyond_noise": False, "best_params": {}, "baseline_params": {}}
        model_store.save_model(tuned, "tuned", extra={"n_train_rows_used": int(len(result.y_train)), "tuning": tuning})
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def predictor(env):
    return P.Predictor.load("final")


@pytest.fixture
def damaged(env, tmp_path, monkeypatch):
    """A copy of the model folders (and preprocessed folder) that a test may damage."""
    for src, attr in ((settings.TRAINED_MODELS_DIR, "TRAINED_MODELS_DIR"), (settings.TUNED_MODELS_DIR, "TUNED_MODELS_DIR")):
        dst = tmp_path / attr
        shutil.copytree(src, dst)
        monkeypatch.setattr(settings, attr, dst)
    pre = tmp_path / "pre"
    shutil.copytree(settings.PREPROCESSED_DIR, pre)
    monkeypatch.setattr(settings, "PREPROCESSED_DIR", pre)
    for name in PATH_ATTRS[1:]:
        monkeypatch.setattr(settings, name, pre / Path(getattr(settings, name)).name)
    return tmp_path


@pytest.fixture(scope="module")
def samples(env):
    return P.load_sample_flows()


def raw_of(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.drop(columns=[settings.LABEL_COLUMN, "test_row_position", "cleaned_row_position"], errors="ignore")


def reference_label(predictor: P.Predictor, X: np.ndarray, row: int, rule: str = "prefer_non_benign") -> int:
    """The written rule, as simply as possible: each model's own predict, count, leaders, tie rule."""
    labels = [int(model.predict(X[row : row + 1])[0]) for model in predictor.models.values()]
    counts = Counter(labels)
    top = max(counts.values())
    leaders = sorted(c for c, n in counts.items() if n == top)
    if rule == "prefer_non_benign":
        attacks = [c for c in leaders if c != predictor.codec.benign_index]
        leaders = attacks or leaders
    return leaders[0]


class FakeModel:
    """Votes a fixed class for every row (or raises)."""

    def __init__(self, label=None, error=None):
        self.label, self.error = label, error

    def predict(self, X):
        if self.error:
            raise RuntimeError(self.error)
        return np.full(len(X), self.label, dtype="int64")


def with_models(predictor: P.Predictor, models: dict) -> P.Predictor:
    return P.Predictor(
        predictor.pipeline, predictor.codec, predictor.schema, models,
        {k: "baseline" for k in KEYS}, {k: "test" for k in KEYS}, {}, "final",
    )


# ---------------------------------------------------------------------------
# The evidence rule
# ---------------------------------------------------------------------------
class TestFinalSet:
    def test_rule_on_the_synthetic_tuning_records(self, env):
        version, reason = P.final_set()
        assert {k for k, v in version.items() if v == "tuned"} == set(TUNED_PARAMS)
        assert all(v == "baseline" for k, v in version.items() if k not in TUNED_PARAMS)
        assert reason["knn"] == "tuning improvement beyond fold noise"
        assert reason["gaussian_nb"] == "the baseline setting was already the best in tuning"
        assert reason["xgboost"] == "tuning improvement within fold noise"

    def test_covers_all_11_models(self, env):
        version, reason = P.final_set()
        assert list(version) == KEYS and list(reason) == KEYS

    def test_subset_of_keys(self, env):
        assert list(P.final_set(["knn", "xgboost"])[0]) == ["knn", "xgboost"]

    def test_no_tuned_model_means_baseline(self, damaged):
        (settings.TUNED_MODELS_DIR / "knn.json").unlink()
        version, reason = P.final_set()
        assert version["knn"] == "baseline" and reason["knn"] == "no tuned model saved"

    @pytest.mark.parametrize(
        "tuning, expected",
        [
            ({"same_as_baseline": False, "improvement_beyond_noise": True}, "tuned"),
            ({"same_as_baseline": False, "improvement_beyond_noise": False}, "baseline"),
            ({"same_as_baseline": True, "improvement_beyond_noise": True}, "baseline"),
            ({}, "baseline"),
        ],
    )
    def test_decision_table(self, damaged, tuning, expected):
        path = settings.TUNED_MODELS_DIR / "xgboost.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["extra"]["tuning"] = tuning
        path.write_text(json.dumps(meta), encoding="utf-8")
        assert P.final_set(["xgboost"])[0]["xgboost"] == expected

    def test_agrees_with_the_evaluation_script(self, env):
        spec = importlib.util.spec_from_file_location("evaluate06", Path(__file__).resolve().parents[1] / "scripts" / "06_evaluate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.final_set(KEYS)[0] == P.final_set(KEYS)[0]  # the decisions; the wording of the reasons may differ


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class TestLoad:
    def test_loads_all_models_in_the_final_versions(self, predictor):
        assert predictor.voters == KEYS and predictor.not_loaded == {}
        assert {k for k, v in predictor.versions.items() if v == "tuned"} == set(TUNED_PARAMS)
        assert predictor.classes == tuple(sorted(CLASSES))

    def test_the_tuned_models_really_are_the_tuned_ones(self, predictor):
        assert predictor.models["knn"].params.get("weights") == "distance"
        assert predictor.models["decision_tree"].params.get("criterion") == "entropy"

    @pytest.mark.parametrize("selection", ["baseline", "tuned"])
    def test_other_selections(self, env, selection):
        p = P.Predictor.load(selection)
        assert set(p.versions.values()) == {selection} and len(p.models) == 11

    def test_unknown_selection(self, env):
        with pytest.raises(ValueError, match="selection must be one of"):
            P.Predictor.load("best")

    def test_model_table(self, predictor):
        table = predictor.model_table()
        assert list(table["key"]) == KEYS and table["loaded"].all()
        assert set(table.columns) >= {"model", "version", "why", "loaded", "problem", "training_rows", "trained_on_subsample", "tuned_parameters", "file_size_mb"}
        row = table.set_index("key").loc["knn"]
        assert row["version"] == "tuned" and row["tuned_parameters"] == "weights=distance"
        assert pd.isna(table.set_index("key").loc["xgboost", "tuned_parameters"])
        assert (table["training_rows"] > 0).all()

    def test_a_missing_model_is_reported_not_replaced(self, damaged):
        (settings.TUNED_MODELS_DIR / "knn.joblib").unlink()
        p = P.Predictor.load("final")
        assert "knn" not in p.models and len(p.models) == 10
        assert "tuned version not loaded" in p.not_loaded["knn"] and p.versions["knn"] == "tuned"  # not swapped for the baseline
        row = p.model_table().set_index("key").loc["knn"]
        assert not row["loaded"] and "knn.joblib" in row["problem"]

    def test_a_damaged_file_is_refused(self, damaged):
        path = settings.TRAINED_MODELS_DIR / "random_forest.joblib"
        path.write_bytes(path.read_bytes()[:-50] + b"x" * 50)
        p = P.Predictor.load("final")
        assert "checksum" in p.not_loaded["random_forest"]

    def test_a_stale_model_is_refused(self, damaged):
        path = settings.TRAINED_MODELS_DIR / "xgboost.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["model_params_hash"] = "0" * 16
        path.write_text(json.dumps(meta), encoding="utf-8")
        p = P.Predictor.load("final")
        assert "stale" in p.not_loaded["xgboost"]

    def test_require_all(self, damaged):
        (settings.TUNED_MODELS_DIR / "knn.joblib").unlink()
        with pytest.raises(P.PredictorError, match="not all models could be loaded"):
            P.Predictor.load("final", require_all=True)

    def test_no_model_at_all(self, damaged):
        shutil.rmtree(settings.TUNED_MODELS_DIR)
        shutil.rmtree(settings.TRAINED_MODELS_DIR)
        with pytest.raises(P.PredictorError, match="no model is available"):
            P.Predictor.load("final")

    def test_constructor_needs_a_model(self, predictor):
        with pytest.raises(P.PredictorError, match="no model is available"):
            P.Predictor(predictor.pipeline, predictor.codec, predictor.schema, {}, {}, {}, {"knn": "gone"})

    def test_stale_preprocessing(self, damaged):
        meta = json.loads(settings.PREPROCESSING_METADATA_FILE.read_text(encoding="utf-8"))
        meta["config_hash"] = "0" * 16
        settings.PREPROCESSING_METADATA_FILE.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(P.PredictorError, match="preprocessing settings changed"):
            P.Predictor.load("final")

    def test_missing_pipeline_file(self, damaged):
        settings.PIPELINE_FILE.unlink()
        with pytest.raises(P.PredictorError):
            P.Predictor.load("final")

    def test_pipeline_and_schema_disagree(self, damaged):
        # a schema that is internally consistent but names a model feature differently from the fitted pipeline
        text = settings.FEATURE_SCHEMA_FILE.read_text(encoding="utf-8")
        settings.FEATURE_SCHEMA_FILE.write_text(text.replace('"Mean Len"', '"Mean Len X"'), encoding="utf-8")
        with pytest.raises(P.PredictorError, match="disagree"):
            P.Predictor.load("final")

    def test_a_model_with_another_number_of_classes(self, predictor, monkeypatch):
        model = load_joblib(model_store.model_file("knn", "trained"))
        monkeypatch.setattr(model, "n_classes_", 99, raising=False)
        monkeypatch.setattr(model_store, "load_model", lambda key, stage, **kw: model)
        with pytest.raises(P.PredictorError, match="classes"):
            P.Predictor.load("baseline")


# ---------------------------------------------------------------------------
# One flow
# ---------------------------------------------------------------------------
class TestPredictFlow:
    def test_typical_flow(self, predictor):
        r = predictor.predict_flow(validator.typical_flow(predictor.schema))
        assert r.ok and r.label in predictor.classes and predictor.codec.classes[r.code] == r.label
        assert r.n_voters == 11 and 1 <= r.top_votes <= 11 and r.missing == {}
        assert r.validation.ok and r.seconds >= 0

    def test_votes_table(self, predictor):
        r = predictor.predict_flow(validator.typical_flow(predictor.schema))
        assert list(r.votes.columns) == ["model", "version", "vote"] and len(r.votes) == 11
        assert set(r.votes["version"]) == {"tuned", "baseline"}
        assert (r.votes["vote"].isin(predictor.classes)).all()
        assert r.votes.set_index("model").loc["KNN", "version"] == "tuned"
        assert r.votes.set_index("model").loc["XGBoost", "version"] == "baseline"

    def test_tally_and_agreement(self, predictor):
        r = predictor.predict_flow(validator.typical_flow(predictor.schema))
        assert int(r.tally["votes"].sum()) == 11 and r.tally.loc[r.tally["winner"], "class"].iloc[0] == r.label
        assert r.agreement == f"{r.top_votes} of 11 models voted for {r.label}" or "tied" in r.agreement
        assert r.votes["vote"].value_counts().max() == r.top_votes

    def test_summary_is_json_safe(self, predictor):
        s = predictor.predict_flow(validator.typical_flow(predictor.schema)).summary()
        json.dumps(s)
        assert s["ok"] and s["n_voters"] == 11 and s["input"]["ok"]

    def test_matches_the_written_rule_on_real_looking_flows(self, predictor, samples):
        batch_input = raw_of(samples)
        X = predictor._features(validator.validate_frame(batch_input, predictor.schema))
        assert len(X) == len(samples)
        for row in range(len(samples)):
            r = predictor.predict_flow(batch_input.iloc[row].to_dict())
            assert r.ok
            assert r.code == reference_label(predictor, X, row), f"row {row}"

    def test_refused_input_is_not_predicted(self, predictor):
        r = predictor.predict_flow({**validator.typical_flow(predictor.schema), "Flow Duration": -1})
        assert not r.ok and r.label is None and r.code is None and r.votes.empty and r.n_voters == 0
        assert r.validation.errors and "negative" in r.validation.errors[0].message
        assert r.agreement == "no prediction"

    def test_impossible_port_is_refused(self, predictor):
        assert not predictor.predict_flow({**validator.typical_flow(predictor.schema), "Destination Port": 70000}).ok

    def test_warnings_pass_through(self, predictor):
        r = predictor.predict_flow({**validator.typical_flow(predictor.schema), "Mean Len": 1e6})
        assert r.ok and any(w.code == "out_of_range" for w in r.validation.warnings)

    def test_missing_policy(self, predictor):
        flow = {**validator.typical_flow(predictor.schema), "Mean Len": None}
        assert not predictor.predict_flow(flow).ok
        assert predictor.predict_flow(flow, missing="median").ok

    def test_same_flow_same_answer(self, predictor):
        flow = validator.typical_flow(predictor.schema)
        a, b = predictor.predict_flow(flow), predictor.predict_flow(dict(reversed(list(flow.items()))))
        assert a.label == b.label and a.votes.equals(b.votes)

    def test_the_input_dict_is_not_modified(self, predictor):
        flow = validator.typical_flow(predictor.schema)
        before = dict(flow)
        predictor.predict_flow(flow)
        assert flow == before


# ---------------------------------------------------------------------------
# Votes with controlled models
# ---------------------------------------------------------------------------
class TestVotingBehaviour:
    def flow(self, predictor):
        return validator.typical_flow(predictor.schema)

    def test_clear_majority(self, predictor):
        c = {n: i for i, n in enumerate(predictor.classes)}
        models = {k: FakeModel(c["Bot"]) for k in KEYS[:7]} | {k: FakeModel(c["BENIGN"]) for k in KEYS[7:]}
        r = with_models(predictor, models).predict_flow(self.flow(predictor))
        assert r.label == "Bot" and r.top_votes == 7 and not r.tied and r.agreement == "7 of 11 models voted for Bot"

    def test_tie_prefers_the_attack_and_says_so(self, predictor):
        c = {n: i for i, n in enumerate(predictor.classes)}
        models = {k: FakeModel(c["BENIGN"]) for k in KEYS[:5]} | {k: FakeModel(c["PortScan"]) for k in KEYS[5:10]} | {KEYS[10]: FakeModel(c["Bot"])}
        r = with_models(predictor, models).predict_flow(self.flow(predictor))
        assert r.tied and r.label == "PortScan" and "tied" in r.agreement and settings.TIE_BREAK_RULE in r.agreement

    def test_a_failing_model_is_listed_and_the_others_vote(self, predictor):
        c = {n: i for i, n in enumerate(predictor.classes)}
        models = {k: FakeModel(c["Bot"]) for k in KEYS}
        models["svm"] = FakeModel(error="boom")
        r = with_models(predictor, models).predict_flow(self.flow(predictor))
        assert r.ok and r.n_voters == 10 and "svm" in r.missing and "boom" in r.missing["svm"]
        assert "SVC (RBF kernel)" not in set(r.votes["model"])

    def test_models_that_were_not_loaded_are_listed(self, damaged):
        (settings.TUNED_MODELS_DIR / "knn.joblib").unlink()
        p = P.Predictor.load("final")
        r = p.predict_flow(validator.typical_flow(p.schema))
        assert r.ok and r.n_voters == 10 and "knn" in r.missing

    def test_every_model_failing(self, predictor):
        models = {k: FakeModel(error="boom") for k in KEYS}
        with pytest.raises(P.PredictorError, match="every model failed"):
            with_models(predictor, models).predict_flow(self.flow(predictor))


# ---------------------------------------------------------------------------
# A table of flows
# ---------------------------------------------------------------------------
class TestPredictFrame:
    def test_samples(self, predictor, samples):
        b = predictor.predict_frame(raw_of(samples))
        f = b.frame
        assert len(f) == len(samples) and b.n_predicted == len(samples)
        assert list(f.columns[:4]) == ["prediction", "top_votes", "n_voters", "tied"]
        assert [model_params.DISPLAY_NAMES[k] for k in KEYS] == list(f.columns[4:])
        assert f["prediction"].isin(predictor.classes).all() and (f["n_voters"] == 11).all()

    def test_equals_the_single_flow_answers(self, predictor, samples):
        raw = raw_of(samples)
        b = predictor.predict_frame(raw)
        for row in range(0, len(raw), 7):
            r = predictor.predict_flow(raw.iloc[row].to_dict())
            assert b.frame["prediction"].iloc[row] == r.label and int(b.frame["top_votes"].iloc[row]) == r.top_votes

    def test_equals_the_written_rule(self, predictor, samples):
        raw = raw_of(samples)
        X = predictor._features(validator.validate_frame(raw, predictor.schema))
        codes = predictor.codec.encode(predictor.predict_frame(raw).frame["prediction"].to_numpy(dtype=object))
        expected = [reference_label(predictor, X, i) for i in range(len(raw))]
        assert codes.tolist() == expected

    def test_refused_rows_keep_their_place(self, predictor, samples):
        raw = raw_of(samples).iloc[:6].copy()
        raw.index = [100, 101, 102, 103, 104, 105]
        raw.loc[102, "Flow Duration"] = -1.0
        raw.loc[104, "Destination Port"] = 99999.0
        b = predictor.predict_frame(raw)
        assert list(b.frame.index) == [100, 101, 102, 103, 104, 105]
        assert b.frame["prediction"].isna().tolist() == [False, False, True, False, True, False]
        assert b.frame.loc[102, "top_votes"] is pd.NA or pd.isna(b.frame.loc[102, "top_votes"])
        assert b.n_predicted == 4 and b.summary()["n_refused"] == 2 and len(b.validation.errors) == 2
        assert predictor.predict_frame(raw.loc[[100, 101, 103, 105]]).frame["prediction"].tolist() == b.frame["prediction"].dropna().tolist()

    def test_a_column_error_refuses_everything_and_runs_no_model(self, predictor, samples, monkeypatch):
        called = []
        monkeypatch.setattr(P.Predictor, "_vote", lambda self, X: called.append(1))
        b = predictor.predict_frame(raw_of(samples).drop(columns=["Mean Len"]))
        assert b.n_predicted == 0 and not called and not b.validation.column_ok

    def test_no_rows(self, predictor, samples):
        b = predictor.predict_frame(raw_of(samples).iloc[0:0])
        assert len(b.frame) == 0 and b.n_predicted == 0 and not b.validation.ok

    def test_summary_is_json_safe(self, predictor, samples):
        s = predictor.predict_frame(raw_of(samples)).summary()
        json.dumps(s)
        assert s["n_rows"] == len(samples) and sum(s["class_counts"].values()) == len(samples) and s["voters"] == KEYS

    def test_a_failing_model_in_a_batch(self, predictor, samples):
        c = {n: i for i, n in enumerate(predictor.classes)}
        models = {k: FakeModel(c["Bot"]) for k in KEYS}
        models["knn"] = FakeModel(error="boom")
        b = with_models(predictor, models).predict_frame(raw_of(samples).iloc[:5])
        assert "knn" in b.missing and (b.frame["n_voters"] == 10).all() and "KNN" not in b.frame.columns


# ---------------------------------------------------------------------------
# Same arithmetic as the preprocessing step
# ---------------------------------------------------------------------------
class TestSameArithmeticAsPreprocessing:
    def test_typed_flow_path_reproduces_the_saved_test_matrix_exactly(self, predictor, samples):
        validation = validator.validate_frame(raw_of(samples), predictor.schema)
        ours = predictor._features(validation)
        theirs = load_array(settings.X_TEST_FILE, mmap_mode="r")[samples["test_row_position"].to_numpy()]
        assert ours.dtype == theirs.dtype == np.dtype(settings.FLOAT_DTYPE)
        assert np.array_equal(ours, np.asarray(theirs))  # bit-identical (float64 input differed by about 1e-6)

    def test_float64_input_would_not_be_identical(self, predictor, samples):
        # documents WHY the cast exists: the same pipeline on float64 values gives slightly different numbers
        raw64 = validator.validate_frame(raw_of(samples), predictor.schema).valid_frame
        out64 = predictor.pipeline.transform(raw64).to_numpy(dtype="float64")
        out32 = predictor._features(validator.validate_frame(raw_of(samples), predictor.schema)).astype("float64")
        assert np.abs(out64 - out32).max() > 0

    def test_input_frame_of_the_validator_is_not_changed_by_the_cast(self, predictor, samples):
        validation = validator.validate_frame(raw_of(samples), predictor.schema)
        before = validation.frame.copy(deep=True)
        predictor._features(validation)
        pd.testing.assert_frame_equal(validation.frame, before)
        assert validation.frame.dtypes.eq("float64").all()


# ---------------------------------------------------------------------------
# Sample flows
# ---------------------------------------------------------------------------
class TestSampleFlows:
    def test_columns_and_content(self, samples):
        assert settings.LABEL_COLUMN in samples.columns and "test_row_position" in samples.columns
        assert set(samples[settings.LABEL_COLUMN]) == set(CLASSES)

    def test_every_sample_flow_passes_the_input_check(self, predictor, samples):
        assert validator.validate_frame(raw_of(samples), predictor.schema).ok

    def test_missing_file(self, damaged):
        settings.SAMPLE_FLOWS_FILE.unlink()
        with pytest.raises((FileNotFoundError, ValueError)):
            P.load_sample_flows()


# ---------------------------------------------------------------------------
# The demonstration
# ---------------------------------------------------------------------------
def write_evaluation_vote(predictor: P.Predictor, tamper: bool = False) -> None:
    X = load_array(settings.X_TEST_FILE)
    labels = np.array([reference_label(predictor, X, i) for i in range(len(X))], dtype="int16")
    if tamper:
        labels[:] = (labels + 1) % predictor.codec.n_classes
    folder = settings.EVALUATION_REPORT_DIR / "test_tuned"
    folder.mkdir(parents=True, exist_ok=True)
    save_array(labels, folder / "majority_vote_final_pred.npy")


class TestDemo:
    def test_runs_and_both_checks_pass(self, predictor, capsys):
        write_evaluation_vote(predictor)
        assert P.main() == 0
        out = capsys.readouterr().out
        assert "OK (identical)" in out and "identical -> OK" in out and "MISMATCH" not in out
        assert "Typical flow" in out and "Sample flows:" in out

    def test_a_different_saved_vote_is_a_mismatch(self, predictor, capsys):
        write_evaluation_vote(predictor, tamper=True)
        assert P.main() == 2
        assert "MISMATCH" in capsys.readouterr().out

    def test_check_two_is_skipped_without_the_saved_vote(self, damaged, capsys):
        folder = settings.EVALUATION_REPORT_DIR / "test_tuned"
        if folder.exists():
            shutil.rmtree(folder)
        assert P.main() == 0
        assert "check 2 skipped" in capsys.readouterr().out

    def test_a_model_that_did_not_load_is_a_failure(self, damaged, capsys):
        (settings.TUNED_MODELS_DIR / "knn.joblib").unlink()
        assert P.main() == 2
        assert "NOT LOADED" in capsys.readouterr().out

    def test_missing_artifacts(self, damaged, capsys):
        settings.PIPELINE_FILE.unlink()
        assert P.main() == 1