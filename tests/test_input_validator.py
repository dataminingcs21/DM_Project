"""Tests for the input validator (File 58).

Synthetic data only.  The schema is built from a small synthetic table by the project's own
``build_pipeline`` / ``build_feature_schema`` (so the tests follow the real schema format), using
feature names that appear in ``settings.NON_NEGATIVE_COLUMNS`` or ``HARD_LIMITS``.  A final group
checks that the validator's output really works with the fitted preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.prediction import input_validator as V
from nids.preprocessing import feature_schema as fs
from nids.preprocessing import pipeline_builder as pb

N = 3000
MODEL_FEATURES = ["Destination Port", "Flow Duration", "Flow Bytes/s", "Total Fwd Packets", "Mean Len"]
UNUSED = ["Const", "Corr"]


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Destination Port": rng.integers(0, 65536, N).astype(float),
            "Flow Duration": rng.integers(1, 10**6, N).astype(float),
            "Flow Bytes/s": rng.exponential(1000, N),
            "Total Fwd Packets": rng.integers(1, 50, N).astype(float),
            "Mean Len": rng.normal(50, 10, N),
            "Const": np.ones(N),
        }
    )
    frame["Corr"] = frame["Mean Len"] * 2 + rng.normal(0, 0.001, N)
    frame.loc[:20, "Flow Bytes/s"] = np.inf  # the training data had infinite values in this column
    return frame


@pytest.fixture(scope="module")
def pipeline(table):
    return pb.build_pipeline().fit(table)


@pytest.fixture(scope="module")
def schema(table, pipeline):
    return fs.build_feature_schema(table, pipeline)


@pytest.fixture
def typical(schema):
    return V.typical_flow(schema)


def codes(result: V.ValidationResult) -> set[str]:
    return set(result.counts)


# ---------------------------------------------------------------------------
# The synthetic schema is what the tests assume
# ---------------------------------------------------------------------------
class TestSchemaAssumptions:
    def test_model_and_unused_features(self, schema):
        assert schema["model_feature_names"] == MODEL_FEATURES
        assert [f["name"] for f in schema["features"] if not f["used_by_models"]] == UNUSED

    def test_non_negative_and_seen_infinite(self, schema):
        by = {f["name"]: f for f in schema["features"]}
        assert by["Flow Duration"]["must_be_non_negative"] and by["Flow Bytes/s"]["must_be_non_negative"]
        assert not by["Destination Port"]["must_be_non_negative"]
        assert by["Flow Bytes/s"]["n_missing_or_infinite_in_train"] > 0
        assert by["Flow Duration"]["n_missing_or_infinite_in_train"] == 0


# ---------------------------------------------------------------------------
# A good flow
# ---------------------------------------------------------------------------
class TestValidFlow:
    def test_typical_flow_is_ok(self, schema, typical):
        r = V.validate_flow(typical, schema)
        assert r.ok and r.issues == [] and r.n_rows == 1 and r.n_valid == 1

    def test_frame_has_all_raw_columns_in_pipeline_order(self, schema, typical):
        r = V.validate_flow(typical, schema)
        assert list(r.frame.columns) == schema["raw_feature_names"]
        assert r.frame.dtypes.eq("float64").all()

    def test_unused_columns_are_nan_when_not_given(self, schema, typical):
        r = V.validate_flow(typical, schema)
        assert r.frame[UNUSED].isna().all().all()
        assert r.frame[MODEL_FEATURES].notna().all().all()

    def test_values_are_kept_exactly(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": 55.25}, schema)
        assert r.frame.loc[0, "Mean Len"] == 55.25

    def test_numeric_text_is_accepted_with_spaces(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": " 55.25 "}, schema)
        assert r.ok and r.frame.loc[0, "Mean Len"] == 55.25

    def test_scientific_notation_text(self, schema, typical):
        assert V.validate_flow({**typical, "Flow Duration": "1e3"}, schema).frame.loc[0, "Flow Duration"] == 1000.0

    def test_integers_and_numpy_numbers(self, schema, typical):
        r = V.validate_flow({**typical, "Total Fwd Packets": np.int64(7), "Mean Len": np.float32(50.5)}, schema)
        assert r.ok and r.frame.loc[0, "Total Fwd Packets"] == 7.0

    def test_column_order_does_not_matter(self, schema, typical):
        r = V.validate_flow(dict(reversed(list(typical.items()))), schema)
        assert r.ok and list(r.frame.columns) == schema["raw_feature_names"]

    def test_unused_columns_given_is_info_only(self, schema, typical):
        r = V.validate_flow({**typical, "Const": 1.0, "Corr": 3.0}, schema)
        assert r.ok and codes(r) == {"unused_given"}
        assert r.issues[0].level == V.INFO and r.unused_given == ["Const", "Corr"]
        assert r.frame.loc[0, "Corr"] == 3.0

    def test_text_in_an_unused_column_is_ignored(self, schema, typical):
        r = V.validate_flow({**typical, "Const": "abc"}, schema)
        assert r.ok and np.isnan(r.frame.loc[0, "Const"])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TestErrors:
    @pytest.mark.parametrize("bad", ["abc", "1,000", "1_000", "12 34", "--5", "0x10", [1, 2], {"a": 1}])
    def test_not_a_number(self, schema, typical, bad):
        r = V.validate_flow({**typical, "Mean Len": bad}, schema)
        assert not r.ok and codes(r) == {"not_a_number"} and not r.valid_mask[0]
        assert np.isnan(r.frame.loc[0, "Mean Len"])

    @pytest.mark.parametrize("bad", [True, False, np.bool_(True)])
    def test_boolean_is_not_a_number(self, schema, typical, bad):
        assert codes(V.validate_flow({**typical, "Mean Len": bad}, schema)) == {"not_a_number"}

    def test_huge_integer_is_not_a_number(self, schema, typical):
        assert codes(V.validate_flow({**typical, "Mean Len": 10**400}, schema)) == {"not_a_number"}

    @pytest.mark.parametrize("blank", [None, "", "   ", float("nan"), "nan", pd.NA])
    def test_blank_required_value(self, schema, typical, blank):
        r = V.validate_flow({**typical, "Mean Len": blank}, schema)
        assert not r.ok and codes(r) == {"missing"}

    def test_required_feature_not_given(self, schema, typical):
        r = V.validate_flow({k: v for k, v in typical.items() if k != "Mean Len"}, schema)
        assert not r.ok and codes(r) == {"missing_column"} and not r.column_ok
        assert "Mean Len" in r.issues[0].message

    def test_empty_flow_lists_every_required_feature(self, schema):
        r = V.validate_flow({}, schema)
        assert r.counts == {"missing_column": len(MODEL_FEATURES)} and not r.ok

    def test_negative_in_non_negative_feature(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Duration": -5}, schema)
        assert not r.ok and codes(r) == {"negative"}
        assert "duration cannot be negative" in r.issues[0].message
        assert np.isnan(r.frame.loc[0, "Flow Duration"])

    def test_negative_allowed_where_training_allowed_it(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": -5.0}, schema)  # not in NON_NEGATIVE_COLUMNS; outside the range only
        assert r.ok and codes(r) == {"out_of_range"}

    def test_infinity_where_training_never_had_it(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Duration": float("inf")}, schema)
        assert not r.ok and codes(r) == {"non_finite"}
        r = V.validate_flow({**typical, "Flow Duration": "-inf"}, schema)
        assert codes(r) == {"non_finite"}

    def test_missing_policy_does_not_excuse_infinity(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Duration": float("inf")}, schema, missing="median")
        assert not r.ok and codes(r) == {"non_finite"}


# ---------------------------------------------------------------------------
# Protocol limits
# ---------------------------------------------------------------------------
class TestHardLimits:
    @pytest.mark.parametrize("port", [65536, 655360, -1, 1e9])
    def test_port_outside_16_bits_is_an_error(self, schema, typical, port):
        r = V.validate_flow({**typical, "Destination Port": port}, schema)
        assert not r.ok and "impossible_value" in codes(r)
        assert "16-bit" in next(i.message for i in r.issues if i.code == "impossible_value")
        assert np.isnan(r.frame.loc[0, "Destination Port"])

    @pytest.mark.parametrize("port", [0, 80, 443, 65535])
    def test_valid_ports(self, schema, typical, port):
        assert V.validate_flow({**typical, "Destination Port": port}, schema).ok

    def test_fractional_port_is_only_a_warning(self, schema, typical):
        r = V.validate_flow({**typical, "Destination Port": 80.5}, schema)
        assert r.ok and codes(r) == {"not_whole"}

    def test_limit_for_a_name_not_in_the_schema_is_ignored(self, schema, typical, monkeypatch):
        monkeypatch.setitem(V.HARD_LIMITS, "No Such Feature", (0.0, 1.0, "test"))
        assert V.validate_flow(typical, schema).ok

    def test_limit_can_be_added(self, schema, typical, monkeypatch):
        monkeypatch.setitem(V.HARD_LIMITS, "Mean Len", (0.0, 100.0, "test limit"))
        assert "impossible_value" in codes(V.validate_flow({**typical, "Mean Len": 101}, schema))


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
class TestWarnings:
    def test_outside_training_range(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": 1e6}, schema)
        assert r.ok and codes(r) == {"out_of_range"}
        assert "not evidence of an attack" in r.issues[0].message
        assert r.frame.loc[0, "Mean Len"] == 1e6  # the value is kept

    def test_range_edges_are_inside(self, schema, typical):
        by = {f["name"]: f for f in schema["features"]}
        for edge in ("min", "max"):
            assert V.validate_flow({**typical, "Mean Len": by["Mean Len"][edge]}, schema).issues == []

    def test_fractional_value_in_whole_number_feature(self, schema, typical):
        r = V.validate_flow({**typical, "Total Fwd Packets": 2.5}, schema)
        assert r.ok and codes(r) == {"not_whole"}

    def test_infinity_where_training_had_it_is_a_warning(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Bytes/s": float("inf")}, schema)
        assert r.ok and codes(r) == {"non_finite_like_training"}
        assert np.isnan(r.frame.loc[0, "Flow Bytes/s"])  # the pipeline imputes it

    def test_blank_where_training_had_it_is_a_warning(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Bytes/s": ""}, schema)
        assert r.ok and codes(r) == {"missing_like_training"}

    def test_unexpected_column_with_suggestion(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Lenx": 1}, schema)
        assert r.ok and codes(r) == {"unexpected_column"} and r.unexpected == ["Mean Lenx"]
        assert "did you mean 'Mean Len'" in r.issues[0].message

    def test_unexpected_column_without_suggestion(self, schema, typical):
        r = V.validate_flow({**typical, "Zzzzzz": 1}, schema)
        assert "did you mean" not in r.issues[0].message

    def test_warnings_do_not_invalidate_the_row(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": 1e6, "Total Fwd Packets": 2.5}, schema)
        assert r.ok and len(r.warnings) == 2 and r.errors == []


# ---------------------------------------------------------------------------
# Missing-value policy
# ---------------------------------------------------------------------------
class TestMissingPolicy:
    def test_median_policy_fills_and_names_it(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": None}, schema, missing="median")
        assert r.ok and codes(r) == {"missing_filled"} and np.isnan(r.frame.loc[0, "Mean Len"])

    def test_median_policy_for_a_column_not_given(self, schema, typical):
        r = V.validate_flow({k: v for k, v in typical.items() if k != "Mean Len"}, schema, missing="median")
        assert r.ok and codes(r) == {"missing_column_filled"}

    def test_median_policy_still_rejects_text(self, schema, typical):
        assert not V.validate_flow({**typical, "Mean Len": "abc"}, schema, missing="median").ok

    def test_unknown_policy(self, schema, typical):
        with pytest.raises(ValueError, match="missing must be one of"):
            V.validate_flow(typical, schema, missing="zero")


# ---------------------------------------------------------------------------
# Tables of flows
# ---------------------------------------------------------------------------
class TestFrames:
    def make(self, typical, n=5):
        return pd.DataFrame([typical] * n).astype(object)

    def test_clean_table(self, schema, typical):
        r = V.validate_frame(self.make(typical), schema)
        assert r.ok and r.n_rows == 5 and r.n_valid == 5 and r.valid_frame.shape == (5, len(schema["raw_feature_names"]))

    def test_only_bad_rows_are_invalid_and_numbered_from_one(self, schema, typical):
        df = self.make(typical)
        df.loc[1, "Flow Duration"] = -1
        df.loc[3, "Mean Len"] = "x"
        r = V.validate_frame(df, schema)
        assert not r.ok and r.valid_mask.tolist() == [True, False, True, False, True]
        assert sorted(i.row for i in r.errors) == [2, 4] and r.n_valid == 3
        assert r.valid_frame.shape[0] == 3

    def test_index_is_kept(self, schema, typical):
        df = self.make(typical, 3)
        df.index = [10, 20, 30]
        assert list(V.validate_frame(df, schema).frame.index) == [10, 20, 30]

    def test_numeric_dtype_columns(self, schema, table):
        df = table[MODEL_FEATURES].iloc[100:110].reset_index(drop=True)
        r = V.validate_frame(df, schema)
        assert r.ok

    def test_integer_dtype_and_nullable_integers(self, schema, typical):
        df = pd.DataFrame({k: [int(v)] * 2 for k, v in typical.items()})
        df["Total Fwd Packets"] = df["Total Fwd Packets"].astype("Int64")
        assert V.validate_frame(df, schema).ok

    def test_boolean_column_is_all_errors(self, schema, typical):
        df = self.make(typical, 3)
        df["Mean Len"] = [True, False, True]
        r = V.validate_frame(df, schema)
        assert r.counts["not_a_number"] == 3 and r.n_valid == 0

    def test_no_rows(self, schema, typical):
        r = V.validate_frame(self.make(typical).iloc[0:0], schema)
        assert not r.ok and codes(r) == {"no_rows"} and r.n_rows == 0

    def test_duplicate_column(self, schema, typical):
        df = self.make(typical, 2)
        df = pd.concat([df, df[["Mean Len"]]], axis=1)
        r = V.validate_frame(df, schema)
        assert not r.ok and "duplicate_column" in codes(r) and not r.column_ok
        assert r.valid_frame.empty

    def test_missing_column_makes_every_row_unusable(self, schema, typical):
        r = V.validate_frame(self.make(typical).drop(columns=["Mean Len"]), schema)
        assert not r.ok and not r.column_ok and r.valid_frame.empty

    def test_many_findings_are_counted_but_not_all_listed(self, schema, typical):
        df = pd.DataFrame([typical] * 2000)
        df["Flow Duration"] = -1.0
        r = V.validate_frame(df, schema)
        assert r.counts["negative"] == 2000 and len(r.issues) == V.MAX_LISTED_ISSUES
        assert r.summary()["issues_not_listed"] == 2000 - V.MAX_LISTED_ISSUES
        assert r.n_valid == 0

    def test_input_is_not_modified(self, schema, typical):
        df = self.make(typical)
        df.loc[0, "Mean Len"] = " 5 "
        before = df.copy(deep=True)
        V.validate_frame(df, schema)
        pd.testing.assert_frame_equal(df, before)

    def test_not_a_dataframe(self, schema):
        with pytest.raises(TypeError):
            V.validate_frame([{"a": 1}], schema)
        with pytest.raises(TypeError):
            V.validate_flow([1, 2], schema)


# ---------------------------------------------------------------------------
# Results, tables, helpers
# ---------------------------------------------------------------------------
class TestResultAndHelpers:
    def test_summary_is_json_safe(self, schema, typical):
        import json

        r = V.validate_flow({**typical, "Mean Len": 1e6, "Flow Duration": -1}, schema)
        s = r.summary()
        json.dumps(s)
        assert s["errors"] == 1 and s["warnings"] == 1 and s["ok"] is False

    def test_single_flow_findings_have_no_row_number(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Duration": -1}, schema)
        assert r.issues[0].row is None and "row" not in str(r.issues[0])

    def test_frame_findings_have_a_row_number(self, schema, typical):
        df = pd.DataFrame([typical] * 2)
        df.loc[1, "Flow Duration"] = -1.0
        assert "row 2" in str(V.validate_frame(df, schema).issues[0])

    def test_issues_table(self, schema, typical):
        r = V.validate_flow({**typical, "Flow Duration": -1}, schema)
        t = V.issues_table(r)
        assert list(t.columns) == ["level", "row", "field", "code", "message", "value"]
        assert t.loc[0, "code"] == "negative" and t.loc[0, "field"] == "Flow Duration"
        assert V.issues_table(V.validate_flow(typical, schema)).empty

    def test_long_values_are_shortened(self, schema, typical):
        r = V.validate_flow({**typical, "Mean Len": "x" * 500}, schema)
        assert len(r.issues[0].value) <= 40

    def test_field_specs(self, schema):
        specs = V.field_specs(schema)
        assert [s["name"] for s in specs] == MODEL_FEATURES
        by = {s["name"]: s for s in specs}
        assert by["Flow Duration"]["must_be_non_negative"] and by["Destination Port"]["all_whole_numbers"]
        assert by["Mean Len"]["min"] < by["Mean Len"]["median"] < by["Mean Len"]["max"]

    def test_typical_flow_has_the_median_of_every_model_feature(self, schema):
        t = V.typical_flow(schema)
        assert list(t) == MODEL_FEATURES
        assert t["Total Fwd Packets"] == next(s["median"] for s in V.field_specs(schema) if s["name"] == "Total Fwd Packets")

    def test_required_features(self, schema):
        assert V.required_features(schema) == MODEL_FEATURES

    def test_default_schema_is_read_from_disk(self, monkeypatch, schema, typical):
        monkeypatch.setattr(V.feature_schema_module, "load_feature_schema", lambda: schema)
        assert V.validate_flow(typical).ok

    def test_demo_runs(self, monkeypatch, schema, capsys):
        monkeypatch.setattr(V.feature_schema_module, "load_feature_schema", lambda: schema)
        assert V.main() == 0
        out = capsys.readouterr().out
        assert "typical flow" in out and "ok=True" in out and "ok=False" in out and "protocol limit" in out

    def test_demo_reports_a_missing_schema(self, monkeypatch):
        def boom():
            raise FileNotFoundError("run scripts/04_preprocess.py first")

        monkeypatch.setattr(V.feature_schema_module, "load_feature_schema", boom)
        assert V.main() == 1


# ---------------------------------------------------------------------------
# The validator's output works with the real fitted pipeline
# ---------------------------------------------------------------------------
class TestWorksWithThePipeline:
    def test_only_the_model_features_give_the_same_output_as_all_columns(self, schema, pipeline, table):
        rows = table.iloc[100:110]
        full = pipeline.transform(rows)
        r = V.validate_frame(rows[MODEL_FEATURES].reset_index(drop=True), schema)
        assert r.ok
        np.testing.assert_allclose(pipeline.transform(r.frame).to_numpy(), full.to_numpy())

    def test_output_columns_are_the_model_features(self, schema, pipeline, typical):
        out = pipeline.transform(V.validate_flow(typical, schema).frame)
        assert list(out.columns) == schema["model_feature_names"] and np.isfinite(out.to_numpy()).all()

    def test_an_infinite_value_seen_in_training_is_imputed_like_in_training(self, schema, pipeline, typical):
        r = V.validate_flow({**typical, "Flow Bytes/s": float("inf")}, schema)
        out = pipeline.transform(r.frame)
        median_row = pipeline.transform(V.validate_flow({**typical, "Flow Bytes/s": None}, schema).frame)
        np.testing.assert_allclose(out.to_numpy(), median_row.to_numpy())
        assert np.isfinite(out.to_numpy()).all()