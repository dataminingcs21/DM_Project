"""Tests for the statistical tests, the data-quality checks and the hermetic benchmark (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.data import quality
from nids.eda import stat_tests as T
from nids.evaluation import pipeline_benchmark as B

LABEL = settings.LABEL_COLUMN


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    labels = np.array(["BENIGN"] * n + ["DoS"] * n + ["PortScan"] * n)
    shift = np.repeat([0.0, 3.0, 6.0], n)
    return pd.DataFrame(
        {
            "separating": rng.lognormal(mean=shift, sigma=0.5),  # depends strongly on the class
            "noise": rng.normal(size=3 * n),                      # independent of the class
            "normal": np.random.default_rng(1).normal(loc=10, scale=2, size=3 * n),  # own seed: a stable normal draw
            "constant": np.ones(3 * n),
            LABEL: labels,
            settings.SOURCE_FILE_COLUMN: "day.csv",
        }
    )


# ---------------------------------------------------------------------------
# statistical tests
# ---------------------------------------------------------------------------
class TestHolm:
    def test_matches_the_definition(self):
        adjusted = T.holm([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_nan_stays_nan_and_values_are_capped(self):
        adjusted = T.holm([np.nan, 0.9, 0.8])
        assert np.isnan(adjusted[0]) and np.nanmax(adjusted) <= 1.0


class TestStatTests:
    def test_separating_feature_is_detected_by_every_class_test(self, flows):
        for test in ("chi2", "anova", "kruskal", "mannwhitney"):
            r = T.run_test(test, flows, "separating")
            assert r["p_value"] < 1e-10, test
        assert T.run_test("kruskal", flows, "separating")["effect_size"] > 0.5
        assert T.run_test("chi2", flows, "separating")["effect_size"] > 0.5

    def test_noise_feature_has_a_negligible_effect(self, flows):
        r = T.run_test("kruskal", flows, "noise")
        assert T.interpret_effect("kruskal", r["effect_size"]) == "negligible"

    def test_mann_whitney_sign_says_attacks_are_larger(self, flows):
        assert T.run_test("mannwhitney", flows, "separating")["effect_size"] > 0.9

    def test_shapiro_tells_normal_from_skewed(self, flows):
        assert T.run_test("shapiro", flows, "normal")["p_value"] > 0.01
        skewed = T.run_test("shapiro", flows, "separating")
        assert skewed["p_value"] < 1e-6 and skewed["statistic_log"] > skewed["statistic"]

    def test_constant_feature_does_not_crash(self, flows):
        for test in T.TESTS:
            r = T.run_test(test, flows, "constant")
            assert np.isnan(r["p_value"])

    def test_infinite_values_are_ignored(self, flows):
        df = flows.copy()
        df.loc[:5, "separating"] = np.inf
        r = T.run_test("kruskal", df, "separating")
        assert r["n"] == len(df) - 6

    def test_unknown_test(self, flows):
        with pytest.raises(ValueError):
            T.run_test("t-test", flows, "noise")

    def test_run_all_and_summary(self, flows):
        results = T.run_all_tests(flows, ["separating", "noise", "normal", "constant"])
        assert set(results) == set(T.TESTS)
        assert results["kruskal"].iloc[0]["feature"] == "separating"
        summary = T.summary_table(results)
        assert list(summary["test"]) == list(T.TESTS)
        assert summary.set_index("test").loc["kruskal", "reject_h0_holm"] >= 1

    def test_input_is_not_modified(self, flows):
        before = flows.copy()
        T.run_all_tests(flows, ["separating", "noise"])
        pd.testing.assert_frame_equal(flows, before)


# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------
class TestQuality:
    def test_finds_every_kind_of_problem(self, flows):
        df = flows.copy()
        df.loc[0, "noise"] = np.nan
        df.loc[1, "noise"] = np.inf
        df = pd.concat([df, df.iloc[[2, 3]]], ignore_index=True)
        df["Flow Duration"] = 1.0
        df.loc[4, "Flow Duration"] = -5.0
        rep = quality.quality_report(df)
        assert rep["duplicates"]["n_duplicate_rows"] == 2
        cols = rep["columns"].set_index("column")
        assert cols.loc["noise", "n_missing"] == 1 and cols.loc["noise", "n_inf"] == 1
        assert bool(cols.loc["constant", "is_constant"])
        assert cols.loc["Flow Duration", "n_negative"] == 1 and bool(cols.loc["Flow Duration", "negative_impossible"])
        text = " ".join(rep["issues"])
        for word in ("duplicate", "missing", "infinite", "negative", "constant"):
            assert word in text

    def test_conflicting_labels(self):
        df = pd.DataFrame({"a": [1, 1, 2], LABEL: ["BENIGN", "DoS", "BENIGN"]})
        assert quality.duplicate_summary(df)["n_conflicting_feature_vectors"] == 1

    def test_balance(self, flows):
        bal = quality.class_balance(flows)
        assert bal["rows"].sum() == len(flows) and bal["share_pct"].sum() == pytest.approx(100)

    def test_clean_table_has_no_issue_but_the_constant(self, flows):
        rep = quality.quality_report(flows.drop(columns=["constant"]))
        assert rep["issues"] == []


# ---------------------------------------------------------------------------
# hermetic benchmark
# ---------------------------------------------------------------------------
class TestPipelineBenchmark:
    def test_pipeline_drops_metadata_and_handles_infinity(self, flows):
        df = flows.copy()
        df.loc[0, "separating"] = np.inf
        features = B.feature_columns(df)
        assert LABEL not in features and settings.SOURCE_FILE_COLUMN not in features
        pipe = B.build_model_pipeline("decision_tree", features)
        y = df[LABEL].astype("category").cat.codes
        pipe.fit(df, y)
        assert len(pipe.predict(df)) == len(df)

    def test_preprocessing_is_refitted_inside_each_fold(self, flows):
        result = B.cross_validate_models(flows, models=["dummy", "decision_tree"], n_splits=3, n_jobs=1)
        assert set(result.folds["model"]) == {"dummy", "decision_tree"}
        assert len(result.folds) == 6
        assert result.summary.iloc[0]["model"] == "decision_tree"
        assert result.summary.set_index("model").loc["decision_tree", "macro_f1_mean"] > 0.9
        assert (result.oof_predictions["decision_tree"] >= 0).all()  # every row predicted exactly once
        per_class = result.per_class_f1()
        assert list(per_class.index) == ["BENIGN", "DoS", "PortScan"]
        assert result.confusion("decision_tree").to_numpy().sum() == len(flows)

    def test_paired_comparison(self, flows):
        result = B.cross_validate_models(flows, models=["dummy", "decision_tree"], n_splits=3, n_jobs=1)
        paired = B.paired_comparison(result.folds, "dummy")
        row = paired.set_index("model").loc["decision_tree"]
        assert row["mean_diff_vs_dummy"] > 0 and row["folds_better"] == 3
        with pytest.raises(ValueError):
            B.paired_comparison(result.folds, "svm")

    def test_holdout_scores_refuse_unknown_labels(self, flows):
        features = B.feature_columns(flows)
        pipe = B.build_model_pipeline("decision_tree", features).fit(flows, flows[LABEL].astype("category").cat.codes)
        scores = B.holdout_scores(pipe, flows, ["BENIGN", "DoS", "PortScan"])
        assert scores["macro_f1"] > 0.9 and 0 <= scores["false_alarm_rate"] <= 1
        bad = flows.copy()
        bad.loc[0, LABEL] = "Heartbleed"
        with pytest.raises(ValueError):
            B.holdout_scores(pipe, bad, ["BENIGN", "DoS", "PortScan"])

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            B.build_model_pipeline("catboost", ["a"])

    def test_available_models_contains_the_sklearn_ones(self):
        keys = B.available_models()
        for key in ("dummy", "logistic_regression", "decision_tree", "random_forest", "extra_trees"):
            assert key in keys
