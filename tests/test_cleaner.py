"""Tests for ``nids.cleaning.cleaner`` (File 16) and the cleaning settings (File 04).

Only synthetic data is used.  Settings that name real dataset columns are
replaced by synthetic names with ``monkeypatch``, so the tests check the
cleaning *logic*, not the real column list.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.cleaning import cleaner

LABEL = settings.LABEL_COLUMN
SOURCE = settings.SOURCE_FILE_COLUMN


@pytest.fixture(autouse=True)
def synthetic_cleaning_settings(monkeypatch):
    """Cleaning rules for synthetic columns, with the default policies."""
    monkeypatch.setattr(settings, "LABEL_MAP", {"Bad � Label": "Bad - Label"})
    monkeypatch.setattr(settings, "NON_NEGATIVE_COLUMNS", {"Feature A": "test", "Feature B": "test", "Absent": "test"})
    monkeypatch.setattr(settings, "NEGATIVE_VALUE_POLICY", "to_nan")
    monkeypatch.setattr(settings, "DROP_IDENTICAL_COLUMNS", True)
    monkeypatch.setattr(settings, "CONFLICT_POLICY", "keep")


def frame(rows: list[tuple], columns=("Feature A", "Feature B", "Feature C")) -> pd.DataFrame:
    """Build a frame from tuples ``(a, b, c, label)``; every row comes from ``f.csv``."""
    df = pd.DataFrame([r[:-1] for r in rows], columns=list(columns), dtype="float64")
    df[LABEL] = [r[-1] for r in rows]
    df[SOURCE] = "f.csv"
    return df


# ---------------------------------------------------------------------------
# Step 1 and 2: labels
# ---------------------------------------------------------------------------
class TestLabels:
    def test_missing_labels_are_dropped(self):
        df = frame([(1, 1, 1, "BENIGN"), (2, 2, 2, None), (3, 3, 3, "DoS")])
        out, report, _ = cleaner.clean_dataset(df)
        assert len(out) == 2
        assert report["steps"][0]["rows_removed"] == 1

    def test_label_map_replaces_whole_labels_and_reports(self):
        df = frame([(1, 1, 1, "Bad � Label"), (2, 2, 2, "BENIGN")])
        out, report, _ = cleaner.clean_dataset(df)
        assert sorted(out[LABEL]) == ["BENIGN", "Bad - Label"]
        section = next(s for s in report["steps"] if s["step"] == "apply_label_map")
        assert section["rows_changed"] == 1
        assert section["applied"]["Bad � Label"] == {"new": "Bad - Label", "rows": 1}

    def test_unused_map_entries_are_listed(self):
        out, report, _ = cleaner.clean_dataset(frame([(1, 1, 1, "BENIGN")]))
        section = next(s for s in report["steps"] if s["step"] == "apply_label_map")
        assert section["map_entries_not_found"] == ["Bad � Label"]

    def test_unmapped_odd_labels_are_reported_not_changed(self):
        df = frame([(1, 1, 1, "Other � Label"), (2, 2, 2, "BENIGN")])
        out, report, _ = cleaner.clean_dataset(df)
        assert "Other � Label" in out[LABEL].tolist()
        section = next(s for s in report["steps"] if s["step"] == "apply_label_map")
        assert section["odd_labels_remaining"] == ["Other � Label"]


# ---------------------------------------------------------------------------
# Step 3: identical columns
# ---------------------------------------------------------------------------
class TestIdenticalColumns:
    def _with_copy(self):
        df = frame([(1, 5, 7, "BENIGN"), (2, 6, 8, "DoS"), (3, np.nan, 9, "BENIGN")])
        df["Feature C copy"] = df["Feature C"]
        df["Feature B copy"] = df["Feature B"]  # contains NaN: NaN must equal NaN
        return df

    def test_identical_columns_are_found_and_the_first_is_kept(self):
        out, report, _ = cleaner.clean_dataset(self._with_copy())
        section = next(s for s in report["steps"] if s["step"] == "drop_identical_columns")
        assert sorted(map(tuple, section["identical_groups"])) == [("Feature B", "Feature B copy"), ("Feature C", "Feature C copy")]
        assert "Feature C" in out.columns and "Feature C copy" not in out.columns
        assert "Feature B" in out.columns and "Feature B copy" not in out.columns
        assert report["n_feature_columns_in"] - report["n_feature_columns_out"] == 2

    def test_similar_but_different_columns_are_kept(self):
        df = frame([(1, 5, 7, "BENIGN"), (2, 6, 8, "DoS")])
        df["Almost"] = [7.0, 8.000001]
        out, _, _ = cleaner.clean_dataset(df)
        assert "Almost" in out.columns

    def test_can_be_switched_off(self, monkeypatch):
        monkeypatch.setattr(settings, "DROP_IDENTICAL_COLUMNS", False)
        out, report, _ = cleaner.clean_dataset(self._with_copy())
        assert "Feature C copy" in out.columns
        section = next(s for s in report["steps"] if s["step"] == "drop_identical_columns")
        assert section["columns_dropped"] == [] and section["identical_groups"]  # still reported


# ---------------------------------------------------------------------------
# Step 4: impossible negatives
# ---------------------------------------------------------------------------
class TestNegativeValues:
    def _df(self):
        return frame([(-1, 5, -9, "BENIGN"), (2, -6, -8, "DoS"), (3, 7, -7, "DoS"), (4, 8, 1, "BENIGN")])

    def test_listed_columns_are_blanked_others_untouched(self):
        out, report, _ = cleaner.clean_dataset(self._df())
        assert out["Feature A"].isna().sum() == 1 and out["Feature B"].isna().sum() == 1
        assert (out["Feature C"] < 0).sum() == 3  # not in NON_NEGATIVE_COLUMNS: kept as is

    def test_report_counts_per_column_and_per_class(self):
        _, report, _ = cleaner.clean_dataset(self._df())
        section = next(s for s in report["steps"] if s["step"] == "handle_negative_values")
        assert section["values_per_column"] == {"Feature A": 1, "Feature B": 1}
        assert section["rows_affected"] == 2
        assert section["rows_affected_per_class"] == {"BENIGN": 1, "DoS": 1}
        assert section["columns_not_present"] == ["Absent"]

    def test_keep_policy_only_reports(self, monkeypatch):
        monkeypatch.setattr(settings, "NEGATIVE_VALUE_POLICY", "keep")
        out, report, _ = cleaner.clean_dataset(self._df())
        assert out["Feature A"].min() == -1
        section = next(s for s in report["steps"] if s["step"] == "handle_negative_values")
        assert section["rows_affected"] == 2

    def test_zero_is_not_negative(self):
        out, _, _ = cleaner.clean_dataset(frame([(0, 0, 0, "BENIGN"), (1, 1, 1, "DoS")]))
        assert out["Feature A"].notna().all()


# ---------------------------------------------------------------------------
# Step 5: infinite values
# ---------------------------------------------------------------------------
class TestInfiniteValues:
    def test_inf_and_minus_inf_become_missing(self):
        # Feature C has no non-negative rule, so both infinities are handled by this step
        df = frame([(1, 1, np.inf, "BENIGN"), (2, 2, -np.inf, "DoS"), (3, 3, 3, "DoS")])
        out, report, _ = cleaner.clean_dataset(df)
        features = [c for c in out.columns if c not in (LABEL, SOURCE)]
        assert not np.isinf(out[features].to_numpy()).any()
        section = next(s for s in report["steps"] if s["step"] == "handle_infinite_values")
        assert section["values_per_column"] == {"Feature C": 2}
        assert section["rows_affected_per_class"] == {"BENIGN": 1, "DoS": 1}

    def test_minus_inf_in_a_non_negative_column_is_counted_as_a_negative_value(self):
        """The negatives step runs first, so -inf in such a column is blanked and counted there."""
        df = frame([(1, -np.inf, 1, "BENIGN"), (2, 2, 2, "DoS")])
        out, report, _ = cleaner.clean_dataset(df)
        negatives = next(s for s in report["steps"] if s["step"] == "handle_negative_values")
        infinite = next(s for s in report["steps"] if s["step"] == "handle_infinite_values")
        assert negatives["values_per_column"] == {"Feature B": 1}
        assert infinite["values_per_column"] == {}
        assert out["Feature B"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Step 6: exact duplicates
# ---------------------------------------------------------------------------
class TestDuplicates:
    def test_exact_duplicates_removed_first_kept(self):
        df = frame([(1, 2, 3, "BENIGN"), (1, 2, 3, "BENIGN"), (4, 5, 6, "DoS")])
        df.loc[1, SOURCE] = "g.csv"  # source_file is ignored when judging duplicates
        out, report, _ = cleaner.clean_dataset(df)
        assert len(out) == 2
        assert out[SOURCE].iloc[0] == "f.csv"
        section = next(s for s in report["steps"] if s["step"] == "remove_exact_duplicates")
        assert section["rows_removed"] == 1 and section["removed_per_class"] == {"BENIGN": 1}

    def test_same_features_with_a_different_label_is_not_a_duplicate(self):
        df = frame([(1, 2, 3, "BENIGN"), (1, 2, 3, "DoS")])
        out, _, _ = cleaner.clean_dataset(df)
        assert len(out) == 2

    def test_values_made_missing_count_as_equal(self):
        """inf, a negative blanked by the rules and NaN all end up as NaN, so these rows are duplicates."""
        df = frame([(np.inf, 2, 3, "BENIGN"), (-5, 2, 3, "BENIGN"), (np.nan, 2, 3, "BENIGN")])
        out, report, _ = cleaner.clean_dataset(df)
        assert len(out) == 1
        assert report["steps"][-2]["rows_removed"] == 2


# ---------------------------------------------------------------------------
# Step 7: conflicting vectors
# ---------------------------------------------------------------------------
class TestConflictingVectors:
    def _df(self):
        return frame([(1, 2, 3, "BENIGN"), (1, 2, 3, "DoS"), (4, 5, 6, "DoS"), (7, 8, 9, "BENIGN")])

    def test_keep_policy_reports_only(self):
        out, report, _ = cleaner.clean_dataset(self._df())
        assert len(out) == 4
        section = report["steps"][-1]
        assert section["n_conflicting_vectors"] == 1
        assert section["rows_in_conflicting_vectors"] == 2
        assert section["label_combinations"] == {"BENIGN | DoS": 1}
        assert section["rows_per_class"] == {"BENIGN": 1, "DoS": 1}

    def test_drop_policy_removes_every_row_of_such_vectors(self, monkeypatch):
        monkeypatch.setattr(settings, "CONFLICT_POLICY", "drop")
        out, report, _ = cleaner.clean_dataset(self._df())
        assert len(out) == 2
        assert report["steps"][-1]["rows_removed"] == 2

    def test_no_conflicts(self):
        out, report, _ = cleaner.clean_dataset(frame([(1, 2, 3, "BENIGN"), (4, 5, 6, "DoS")]))
        assert report["steps"][-1]["n_conflicting_vectors"] == 0
        assert report["steps"][-1]["label_combinations"] == {}


# ---------------------------------------------------------------------------
# clean_dataset as a whole
# ---------------------------------------------------------------------------
class TestCleanDataset:
    def test_input_is_not_modified_and_index_is_reset(self, make_flow_frame):
        df = make_flow_frame()
        df.loc[3, "Feature A"] = np.inf
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # duplicates
        before = df.copy()
        out, _, _ = cleaner.clean_dataset(df)
        pd.testing.assert_frame_equal(df, before)
        assert list(out.index) == list(range(len(out)))

    def test_missing_values_are_left_for_the_imputer(self):
        df = frame([(np.inf, 1, 1, "BENIGN"), (2, 2, 2, "DoS")])
        out, report, _ = cleaner.clean_dataset(df)
        assert out["Feature A"].isna().sum() == 1  # not filled
        assert report["missing_values_after_cleaning"]["n_missing"] == 1
        assert report["missing_values_after_cleaning"]["per_column"] == {"Feature A": 1}

    def test_report_is_json_safe_and_consistent(self, make_flow_frame):
        df = make_flow_frame()
        out, report, _ = cleaner.clean_dataset(df)
        json.dumps(report)  # must not raise
        assert report["rows_in"] == len(df) and report["rows_out"] == len(out)
        assert sum(report["class_counts_after"].values()) == len(out)
        assert report["config"] == settings.cleaning_config()

    def test_class_table_uses_the_fixed_labels(self):
        df = frame([(1, 1, 1, "Bad � Label"), (1, 1, 1, "Bad � Label"), (2, 2, 2, "BENIGN")])
        _, _, tables = cleaner.clean_dataset(df)
        table = tables["class_counts"].set_index("label")
        assert "Bad � Label" not in table.index
        assert table.loc["Bad - Label", ["rows_before", "rows_after", "rows_removed"]].tolist() == [2, 1, 1]

    def test_classes_below_the_minimum_are_flagged(self):
        rows = [(i, i + 1, i + 2, "BENIGN") for i in range(20)] + [(100, 100, 100, "Tiny")]
        _, report, tables = cleaner.clean_dataset(frame(rows))
        assert report["classes_below_min_class_samples"] == ["Tiny"]
        assert bool(tables["class_counts"].set_index("label").loc["Tiny", "below_min_class_samples"])

    def test_non_numeric_feature_is_rejected(self):
        df = frame([(1, 1, 1, "BENIGN")])
        df["Feature A"] = df["Feature A"].astype(str)
        with pytest.raises(ValueError, match="Non-numeric"):
            cleaner.clean_dataset(df)

    def test_missing_label_column_is_rejected(self):
        with pytest.raises(ValueError, match="Label column"):
            cleaner.clean_dataset(frame([(1, 1, 1, "BENIGN")]).drop(columns=LABEL))


# ---------------------------------------------------------------------------
# The real cleaning settings (File 04) are well formed
# ---------------------------------------------------------------------------
class TestDefaultSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.undo()  # look at the real values, not the synthetic ones
        assert settings.LABEL_MAP and all("�" in k and "�" not in v for k, v in settings.LABEL_MAP.items())
        assert all(reason.strip() for reason in settings.NON_NEGATIVE_COLUMNS.values())
        assert not any(c.startswith("Init_Win_bytes") for c in settings.NON_NEGATIVE_COLUMNS)
        config = settings.cleaning_config()
        assert config["conflict_policy"] in settings.ALLOWED_CONFLICT_POLICIES
        json.dumps(config)