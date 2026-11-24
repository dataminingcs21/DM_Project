"""Tests for the preprocessing stage: feature filter (22), pipeline (23), schema (24), preprocessor (25).

Synthetic data only. The column names ("Feature A", ...) are placeholders, not real dataset columns.
"""

from __future__ import annotations

import copy
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from config import settings
from nids.preprocessing import feature_schema as fs
from nids.preprocessing.feature_filter import FeatureFilter
from nids.preprocessing.pipeline_builder import SignedLog1p, build_pipeline, describe_pipeline
from nids.preprocessing.preprocessor import (
    check_preprocessed_fresh,
    pick_sample_flows,
    preprocess,
    save_preprocessed,
    select_feature_columns,
)
from nids.preprocessing.splitter import stratified_split
from nids.utils.io_utils import load_array, load_joblib

LABEL = settings.LABEL_COLUMN
SIZES = {"BENIGN": 600, "DoS": 150, "PortScan": 60}


@pytest.fixture(autouse=True)
def clean_preprocessed_dir():
    """Each test starts and ends without saved preprocessing files (they live in the temporary root)."""
    shutil.rmtree(settings.PREPROCESSED_DIR, ignore_errors=True)
    yield
    shutil.rmtree(settings.PREPROCESSED_DIR, ignore_errors=True)


def make_raw(sizes: dict[str, int] | None = None, seed: int = 0) -> pd.DataFrame:
    """Synthetic cleaned data: a skewed column, a correlated copy, a class-dependent column, counts, a constant."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, (name, n) in enumerate((sizes or SIZES).items()):
        a = np.abs(rng.normal(i * 2.0, 1.0, n)) * 100
        frames.append(
            pd.DataFrame(
                {
                    "Feature A": a,
                    "Feature B": a * 3 + rng.normal(0, 0.5, n),  # correlated with A (|r| > 0.95)
                    "Feature C": rng.normal(i * 3.0, 1.0, n),
                    "Feature D": rng.integers(0, 6, n).astype(float),
                    "Const": 7.0,
                    LABEL: name,
                    settings.SOURCE_FILE_COLUMN: f"file_{i % 2}.csv",
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def feature_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Float32 feature columns only: a, b (~ a), c (~ b but weaker vs a), d, constant, all-missing."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + 0.2 * rng.normal(size=n)  # r(a,b) ~ 0.98
    c = b + 0.5 * rng.normal(size=n)  # r(b,c) ~ 0.90, r(a,c) ~ 0.88
    d = rng.normal(size=n)
    return pd.DataFrame(
        {"a": a, "b": b, "c": c, "d": d, "const": 5.0, "empty": np.nan}
    ).astype("float32")


# ---------------------------------------------------------------------------
# FeatureFilter
# ---------------------------------------------------------------------------
class TestFeatureFilter:
    def test_drops_constant_and_all_missing_columns(self):
        flt = FeatureFilter().fit(feature_frame())
        assert set(flt.constant_features_) == {"const", "empty"}
        assert not {"const", "empty"} & set(flt.get_feature_names_out())

    def test_keeps_the_first_of_two_correlated_columns(self):
        flt = FeatureFilter().fit(feature_frame())
        assert flt.correlated_features_["b"]["kept_instead"] == "a"
        assert "a" in flt.get_feature_names_out() and "b" not in flt.get_feature_names_out()

    def test_column_order_decides_which_is_kept(self):
        frame = feature_frame()[["b", "a", "d"]]
        flt = FeatureFilter().fit(frame)
        assert list(flt.get_feature_names_out()) == ["b", "d"]

    def test_negative_correlation_counts(self):
        frame = pd.DataFrame({"x": np.arange(100.0), "y": -np.arange(100.0) * 2 + 1})
        flt = FeatureFilter().fit(frame)
        assert list(flt.get_feature_names_out()) == ["x"]
        assert flt.correlated_features_["y"]["correlation"] < 0

    def test_greedy_rule_compares_with_kept_columns_only(self):
        flt = FeatureFilter().fit(feature_frame())
        # b is dropped (kept a); c is below the threshold against the kept column a, so c stays
        assert list(flt.get_feature_names_out()) == ["a", "c", "d"]

    def test_threshold_parameter_and_setting(self, monkeypatch):
        frame = feature_frame()
        assert "c" not in FeatureFilter(correlation_threshold=0.85).fit(frame).get_feature_names_out()
        monkeypatch.setattr(settings, "CORRELATION_THRESHOLD", 0.85)
        assert "c" not in FeatureFilter().fit(frame).get_feature_names_out()

    def test_missing_and_infinite_values_are_ignored(self):
        frame = feature_frame()
        frame.loc[0, "a"] = np.inf
        frame.loc[1, "b"] = np.nan
        flt = FeatureFilter().fit(frame)
        assert "b" in flt.correlated_features_

    def test_transform_returns_the_kept_columns(self):
        frame = feature_frame()
        flt = FeatureFilter().fit(frame)
        out = flt.transform(frame)
        assert out.shape == (len(frame), 3)
        assert np.array_equal(out[:, 0], frame["a"].to_numpy())

    def test_works_on_plain_arrays(self):
        flt = FeatureFilter().fit(feature_frame().to_numpy())
        assert list(flt.feature_names_in_[:2]) == ["x0", "x1"]
        assert flt.transform(feature_frame().to_numpy()).shape[1] == 3

    def test_transform_is_refused_for_other_columns(self):
        frame = feature_frame()
        flt = FeatureFilter().fit(frame)
        with pytest.raises(ValueError, match="columns"):
            flt.transform(frame[["a", "b"]])
        with pytest.raises(ValueError, match="names or their order"):
            flt.transform(frame[list(reversed(frame.columns))])

    def test_not_fitted_and_bad_input(self):
        with pytest.raises(NotFittedError):
            FeatureFilter().transform(feature_frame())
        with pytest.raises(ValueError, match="2-dimensional"):
            FeatureFilter().fit(np.zeros(5))
        with pytest.raises(ValueError, match="correlation_threshold"):
            FeatureFilter(correlation_threshold=0).fit(feature_frame())
        with pytest.raises(ValueError, match="every column"):
            FeatureFilter().fit(feature_frame()[["const", "empty"]])

    def test_the_labels_are_never_used(self):
        frame = feature_frame()
        a = FeatureFilter().fit(frame, y=np.zeros(len(frame)))
        b = FeatureFilter().fit(frame, y=np.arange(len(frame)))
        assert np.array_equal(a.support_, b.support_)

    def test_sampling_is_reproducible(self):
        frame = feature_frame()
        r1 = FeatureFilter(max_rows=1000).fit(frame).report()
        r2 = FeatureFilter(max_rows=1000).fit(frame).report()
        assert r1 == r2 and r1["rows_used_for_correlation"] == 1000

    def test_report_is_json_safe_and_consistent(self):
        report = FeatureFilter().fit(feature_frame()).report()
        json.dumps(report)
        assert report["n_features_in"] == 6 and report["n_features_out"] == 3


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestSignedLog1p:
    def test_sign_zero_order_and_missing(self):
        x = np.array([[-5.0], [-1.0], [0.0], [1.0], [1e9]])
        out = SignedLog1p().fit_transform(x).ravel()
        assert out[2] == 0 and (np.diff(out) > 0).all() and out[0] < 0 < out[3]
        assert np.isnan(SignedLog1p().fit_transform(np.array([[np.nan]]))[0, 0])

    def test_compresses_huge_values(self):
        out = SignedLog1p().fit_transform(np.array([[1e9]]))
        assert out[0, 0] < 25


class TestPipeline:
    def test_default_step_order(self):
        assert [name for name, _ in build_pipeline().steps] == ["inf_to_nan", "imputer", "log", "filter", "scaler"]

    def test_log_can_be_switched_off_by_argument_and_setting(self, monkeypatch):
        assert "log" not in build_pipeline(log_transform=False).named_steps
        monkeypatch.setattr(settings, "LOG_TRANSFORM", False)
        assert "log" not in build_pipeline().named_steps

    def test_unknown_scaler_is_rejected(self):
        with pytest.raises(ValueError, match="scaler_type"):
            build_pipeline(scaler_type="magic")

    def test_train_output_is_centred_scaled_and_finite(self):
        frame = feature_frame()
        frame.loc[3, "a"] = np.inf
        frame.loc[4, "d"] = np.nan
        out = build_pipeline().fit_transform(frame)
        assert np.isfinite(out.to_numpy()).all()
        assert np.allclose(out.mean(), 0, atol=1e-3) and np.allclose(out.std(ddof=0), 1, atol=1e-2)
        assert list(out.columns) == ["a", "c", "d"]

    def test_names_out_match_the_output_columns(self):
        pipeline = build_pipeline()
        out = pipeline.fit_transform(feature_frame())
        assert list(pipeline.get_feature_names_out()) == list(out.columns)

    def test_missing_values_use_the_training_median(self):
        train = feature_frame()
        train.loc[:50, "d"] = np.nan
        pipeline = build_pipeline().fit(train)
        median = float(np.nanmedian(train["d"].to_numpy(dtype="float64")))
        row_a = train.iloc[[100]].copy()
        row_b = row_a.copy()
        row_a["d"] = np.nan
        row_b["d"] = np.float32(median)
        assert np.allclose(pipeline.transform(row_a).to_numpy(), pipeline.transform(row_b).to_numpy(), atol=1e-5)

    def test_infinity_is_treated_as_missing(self):
        train = feature_frame()
        pipeline = build_pipeline().fit(train)
        row_a, row_b = train.iloc[[10]].copy(), train.iloc[[10]].copy()
        row_a["d"], row_b["d"] = np.inf, np.nan
        assert np.allclose(pipeline.transform(row_a).to_numpy(), pipeline.transform(row_b).to_numpy())

    def test_the_input_is_not_modified(self):
        frame = feature_frame()
        frame.loc[2, "a"] = np.inf
        before = frame.copy()
        build_pipeline().fit_transform(frame)
        assert frame.equals(before) and np.isinf(frame.loc[2, "a"])

    def test_test_rows_do_not_change_what_was_learned(self):
        frame = feature_frame()
        train, test = frame.iloc[:3000], frame.iloc[3000:].copy()
        first = build_pipeline().fit(train)
        test.iloc[:, :4] = test.iloc[:, :4] * 1000 + 50  # extreme test values
        first.transform(test)
        second = build_pipeline().fit(train)
        assert np.allclose(first.transform(train).to_numpy(), second.transform(train).to_numpy())

    def test_transform_refuses_other_columns(self):
        frame = feature_frame()
        pipeline = build_pipeline().fit(frame)
        with pytest.raises(ValueError, match="columns"):
            pipeline.transform(frame[["a", "b"]])
        with pytest.raises(ValueError, match="names or their order"):
            pipeline.transform(frame[list(reversed(frame.columns))])

    def test_minmax_scaler_maps_training_data_to_0_1(self):
        out = build_pipeline(scaler_type="minmax").fit_transform(feature_frame())
        assert out.min().min() >= -1e-6 and out.max().max() <= 1 + 1e-6

    def test_robust_scaler_runs(self):
        out = build_pipeline(scaler_type="robust").fit_transform(feature_frame())
        assert np.isfinite(out.to_numpy()).all()

    def test_pipeline_survives_joblib(self, tmp_path):
        frame = feature_frame()
        pipeline = build_pipeline().fit(frame)
        joblib.dump(pipeline, tmp_path / "p.joblib")
        again = joblib.load(tmp_path / "p.joblib")
        assert np.allclose(pipeline.transform(frame).to_numpy(), again.transform(frame).to_numpy())

    def test_description_is_json_safe(self):
        pipeline = build_pipeline().fit(feature_frame())
        info = describe_pipeline(pipeline)
        json.dumps(info)
        assert info["log_transform"] is True and info["feature_filter"]["n_features_out"] == 3


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
@pytest.fixture
def fitted():
    frame = feature_frame()
    pipeline = build_pipeline().fit(frame)
    return frame, pipeline, fs.build_feature_schema(frame, pipeline)


class TestFeatureSchema:
    def test_structure_and_counts(self, fitted):
        _, _, schema = fitted
        assert schema["n_raw_features"] == 6 and schema["n_model_features"] == 3
        assert schema["model_feature_names"] == ["a", "c", "d"]
        assert [f["name"] for f in schema["features"]] == schema["raw_feature_names"]

    def test_reasons_for_dropped_features(self, fitted):
        _, _, schema = fitted
        by_name = {f["name"]: f for f in schema["features"]}
        assert by_name["a"]["dropped_because"] is None
        assert "constant" in by_name["const"]["dropped_because"]
        assert "correlated with 'a'" in by_name["b"]["dropped_because"]

    def test_statistics_come_from_the_training_data(self, fitted):
        frame, _, schema = fitted
        a = next(f for f in schema["features"] if f["name"] == "a")
        values = frame["a"].to_numpy(dtype="float64")
        assert a["min"] == pytest.approx(values.min()) and a["max"] == pytest.approx(values.max())
        assert a["median"] == pytest.approx(np.median(values))
        assert a["p01"] <= a["median"] <= a["p99"]

    def test_missing_and_infinite_values_are_counted_not_used(self):
        frame = feature_frame()
        frame.loc[0, "d"] = np.inf
        frame.loc[1, "d"] = np.nan
        schema = fs.build_feature_schema(frame, build_pipeline().fit(frame))
        d = next(f for f in schema["features"] if f["name"] == "d")
        assert d["n_missing_or_infinite_in_train"] == 2 and np.isfinite(d["max"])

    def test_all_missing_column_has_no_statistics(self, fitted):
        _, _, schema = fitted
        empty = next(f for f in schema["features"] if f["name"] == "empty")
        assert empty["median"] is None and empty["all_whole_numbers"] is False

    def test_whole_number_flag(self):
        frame = pd.DataFrame({"count": np.tile([0.0, 1.0, 2.0, 3.0], 50), "x": np.random.default_rng(0).normal(size=200)}).astype("float32")
        schema = fs.build_feature_schema(frame, build_pipeline().fit(frame))
        flags = {f["name"]: f["all_whole_numbers"] for f in schema["features"]}
        assert flags == {"count": True, "x": False}

    def test_non_negative_columns_come_from_the_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "NON_NEGATIVE_COLUMNS", {"a": "a test reason"})
        frame = feature_frame()
        schema = fs.build_feature_schema(frame, build_pipeline().fit(frame))
        a = next(f for f in schema["features"] if f["name"] == "a")
        assert a["must_be_non_negative"] and a["non_negative_reason"] == "a test reason"

    def test_validation_detects_inconsistencies(self, fitted):
        _, _, schema = fitted
        bad = copy.deepcopy(schema)
        bad["features"][0]["name"] = "other"
        with pytest.raises(ValueError, match="does not match"):
            fs.validate_schema(bad)
        bad = copy.deepcopy(schema)
        bad["model_feature_names"] = ["a", "zzz"]
        with pytest.raises(ValueError):
            fs.validate_schema(bad)
        bad = copy.deepcopy(schema)
        bad["n_raw_features"] = 99
        with pytest.raises(ValueError, match="counts"):
            fs.validate_schema(bad)
        with pytest.raises(ValueError, match="missing keys"):
            fs.validate_schema({})

    def test_bad_inputs_are_rejected(self, fitted):
        frame, pipeline, _ = fitted
        with pytest.raises(ValueError, match="DataFrame"):
            fs.build_feature_schema(frame.to_numpy(), pipeline)
        with pytest.raises(ValueError, match="differ"):
            fs.build_feature_schema(frame[list(reversed(frame.columns))], pipeline)
        with pytest.raises(ValueError, match="fitted"):
            fs.build_feature_schema(frame, build_pipeline())

    def test_save_and_load_round_trip(self, fitted):
        _, _, schema = fitted
        fs.save_feature_schema(schema)
        assert json.loads(settings.FEATURE_NAMES_FILE.read_text())["model_feature_names"] == ["a", "c", "d"]
        assert fs.load_feature_schema() == json.loads(json.dumps(schema))

    def test_table_and_column_check(self, fitted):
        _, _, schema = fitted
        assert fs.schema_table(schema).shape[0] == 6
        check = fs.check_columns(schema, ["a", "b", "extra"])
        assert check["unexpected"] == ["extra"] and "c" in check["missing"]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result():
    return preprocess(make_raw())


class TestPreprocess:
    def test_shapes_types_and_split(self, result):
        n = sum(SIZES.values())
        assert result.X_train.shape[0] + result.X_test.shape[0] == n
        assert result.X_train.shape[1] == result.X_test.shape[1] == len(result.feature_names)
        assert result.X_train.dtype == np.float32 and result.y_train.dtype == np.int64
        assert len(result.y_train) == len(result.X_train) and len(result.y_test) == len(result.X_test)

    def test_split_is_the_stratified_split_of_the_target(self, result):
        df = make_raw()
        train, test = stratified_split(df[LABEL])
        assert np.array_equal(result.train_idx, train) and np.array_equal(result.test_idx, test)

    def test_training_data_is_centred_and_everything_is_finite(self, result):
        assert np.isfinite(result.X_train).all() and np.isfinite(result.X_test).all()
        assert np.abs(result.X_train.mean(axis=0)).max() < 1e-3

    def test_codes_decode_to_the_original_labels(self, result):
        df = make_raw()
        assert (result.codec.decode(result.y_train) == df[LABEL].to_numpy()[result.train_idx]).all()
        assert (result.codec.decode(result.y_test) == df[LABEL].to_numpy()[result.test_idx]).all()
        assert result.codec.classes[0] == settings.BENIGN_LABEL

    def test_constant_and_correlated_columns_are_removed(self, result):
        assert result.feature_names == ["Feature A", "Feature C", "Feature D"]
        assert result.schema["n_raw_features"] == 5

    def test_the_label_and_metadata_are_never_features(self, result):
        assert LABEL not in result.schema["raw_feature_names"]
        assert settings.SOURCE_FILE_COLUMN not in result.schema["raw_feature_names"]

    def test_test_rows_cannot_influence_the_training_output(self, result):
        df = make_raw()
        df.loc[result.test_idx, ["Feature A", "Feature C"]] = [1e9, -1e9]  # absurd test values
        again = preprocess(df)
        assert np.array_equal(again.X_train, result.X_train)

    def test_input_is_not_modified(self):
        df = make_raw()
        df.loc[3, "Feature C"] = np.nan
        before = df.copy()
        preprocess(df)
        assert df.equals(before)

    def test_sample_flows_are_real_test_rows_and_reproduce_through_the_pipeline(self, result):
        df = make_raw()
        samples = result.sample_flows
        assert set(samples[LABEL]) == set(SIZES)
        assert samples.groupby(LABEL).size().max() <= 5
        pos = samples["test_row_position"].to_numpy()
        cleaned_pos = samples["cleaned_row_position"].to_numpy()
        assert np.array_equal(result.test_idx[pos], cleaned_pos)
        assert (df.loc[cleaned_pos, LABEL].to_numpy() == samples[LABEL].to_numpy()).all()
        raw = samples[result.schema["raw_feature_names"]].astype("float32")
        assert np.allclose(result.pipeline.transform(raw).to_numpy(), result.X_test[pos], atol=1e-5)

    def test_pick_sample_flows_is_deterministic_and_capped(self):
        raw = make_raw()[["Feature A"]].astype("float32")
        target = make_raw()[LABEL]
        a, b = pick_sample_flows(raw, target, per_class=2), pick_sample_flows(raw, target, per_class=2)
        assert a.equals(b) and len(a) == 6

    def test_excluded_features_are_removed(self, monkeypatch):
        monkeypatch.setattr(settings, "EXCLUDED_FEATURES", {"Feature C": "test reason"})
        assert "Feature C" not in select_feature_columns(make_raw())

    def test_binary_mode(self, monkeypatch):
        monkeypatch.setattr(settings, "TARGET_MODE", "binary")
        res = preprocess(make_raw())
        assert res.codec.classes == (settings.BENIGN_LABEL, settings.MALICIOUS_LABEL)
        assert set(np.unique(res.y_train)) == {0, 1}

    def test_bad_inputs_are_rejected(self):
        df = make_raw()
        with pytest.raises(ValueError, match="label column"):
            preprocess(df.drop(columns=[LABEL]))
        with pytest.raises(ValueError, match="non-numeric"):
            preprocess(df.assign(**{"Feature A": df["Feature A"].astype(str)}))

    def test_report_is_json_safe_and_complete(self, result):
        json.dumps(result.report)
        assert result.report["n_model_features"] == 3
        counts = pd.DataFrame(result.report["class_counts"])
        assert counts["train"].sum() == len(result.y_train) and counts["test"].sum() == len(result.y_test)


class TestSaveAndFreshness:
    def test_every_file_is_written_and_reloads_identically(self, result):
        metadata = save_preprocessed(result)
        for path in (
            settings.X_TRAIN_FILE, settings.X_TEST_FILE, settings.Y_TRAIN_FILE, settings.Y_TEST_FILE,
            settings.PIPELINE_FILE, settings.LABEL_ENCODER_FILE, settings.FEATURE_SCHEMA_FILE,
            settings.FEATURE_NAMES_FILE, settings.SAMPLE_FLOWS_FILE, settings.PREPROCESSING_METADATA_FILE,
        ):
            assert path.is_file(), path
        assert np.array_equal(load_array(settings.X_TRAIN_FILE), result.X_train)
        assert np.array_equal(load_array(settings.Y_TEST_FILE), result.y_test)
        assert load_joblib(settings.LABEL_ENCODER_FILE) == result.codec
        assert metadata["config_hash"] and metadata["X_train_shape"] == list(result.X_train.shape)

    def test_saved_pipeline_reproduces_the_saved_arrays(self, result):
        save_preprocessed(result)
        pipeline = load_joblib(settings.PIPELINE_FILE)
        pos = result.sample_flows["test_row_position"].to_numpy()
        raw = result.sample_flows[result.schema["raw_feature_names"]].astype("float32")
        assert np.allclose(pipeline.transform(raw).to_numpy(), load_array(settings.X_TEST_FILE)[pos], atol=1e-5)

    def test_fresh_after_saving(self, result):
        save_preprocessed(result)
        assert check_preprocessed_fresh() == (True, "")

    def test_stale_when_a_setting_changes(self, result, monkeypatch):
        save_preprocessed(result)
        monkeypatch.setattr(settings, "LOG_TRANSFORM", False)
        fresh, reason = check_preprocessed_fresh()
        assert not fresh and "settings changed" in reason

    def test_not_fresh_without_metadata(self):
        fresh, reason = check_preprocessed_fresh()
        assert not fresh and "no preprocessed data" in reason

    def test_metadata_is_written_last(self, result):
        save_preprocessed(result)
        meta_time = settings.PREPROCESSING_METADATA_FILE.stat().st_mtime_ns
        others = [settings.X_TRAIN_FILE, settings.PIPELINE_FILE, settings.FEATURE_SCHEMA_FILE, settings.SAMPLE_FLOWS_FILE]
        assert all(p.stat().st_mtime_ns <= meta_time for p in others)