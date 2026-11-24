"""Tests for the target builder (File 19) and the splitter (File 20).

Synthetic data only. The class names "BENIGN" (the project's benign label) and
"A", "B", ... are placeholders, not claims about the real dataset.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.preprocessing.splitter import split_report, stratified_split
from nids.preprocessing.target_builder import (
    ATTACK_LABEL,
    LabelCodec,
    build_target,
    make_codec,
    merge_rare_classes,
    to_binary,
)

BENIGN = settings.BENIGN_LABEL


def _labels(sizes: dict[str, int], seed: int = 0) -> pd.Series:
    values = np.repeat(list(sizes), list(sizes.values()))
    return pd.Series(values).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target builder
# ---------------------------------------------------------------------------
class TestBuildTarget:
    def test_multiclass_keep_leaves_labels_unchanged(self):
        labels = _labels({BENIGN: 50, "A": 20, "Rare": 3})
        target, report = build_target(labels, mode="multiclass", rare_policy="keep")
        assert target.equals(labels)
        assert report["n_classes"] == 3
        assert report["merged_classes"] == []

    def test_keep_flags_classes_below_the_minimum(self):
        labels = _labels({BENIGN: 50, "A": 20, "Rare": 3})
        _, report = build_target(labels, mode="multiclass", rare_policy="keep")
        assert report["classes_below_min_class_samples"] == ["Rare"]

    def test_merge_combines_rare_classes(self):
        labels = _labels({BENIGN: 50, "A": 20, "R1": 3, "R2": 4})
        target, report = build_target(labels, mode="multiclass", rare_policy="merge")
        assert report["merged_classes"] == ["R1", "R2"]
        assert report["counts_after"][settings.RARE_CLASS_MERGE_LABEL] == 7
        assert "R1" not in set(target)
        assert report["counts_before"]["R1"] == 3  # the report keeps the original counts

    def test_merge_never_touches_benign(self):
        labels = _labels({BENIGN: 3, "A": 20, "R1": 3, "R2": 4})
        target, report = build_target(labels, mode="multiclass", rare_policy="merge")
        assert BENIGN in set(target)
        assert BENIGN not in report["merged_classes"]

    def test_merge_refused_when_merged_class_still_too_small(self):
        labels = _labels({BENIGN: 50, "A": 20, "R1": 2, "R2": 1})
        with pytest.raises(ValueError, match="merging does not help"):
            build_target(labels, mode="multiclass", rare_policy="merge")

    def test_merge_without_rare_classes_changes_nothing(self):
        labels = _labels({BENIGN: 50, "A": 20})
        merged, names = merge_rare_classes(labels)
        assert names == [] and merged.equals(labels)

    def test_binary_mode(self):
        labels = _labels({BENIGN: 50, "A": 20, "B": 10})
        target, report = build_target(labels, mode="binary")
        assert set(target) == {BENIGN, ATTACK_LABEL}
        assert report["counts_after"] == {BENIGN: 50, ATTACK_LABEL: 30}
        assert report["rare_class_policy"].startswith("not applicable")

    def test_binary_ignores_the_rare_policy(self):
        labels = _labels({BENIGN: 50, "A": 20, "R1": 3})
        target, _ = build_target(labels, mode="binary", rare_policy="merge")
        assert set(target) == {BENIGN, ATTACK_LABEL}

    def test_to_binary_keeps_the_index(self):
        labels = pd.Series([BENIGN, "A"], index=[10, 20])
        assert to_binary(labels).index.tolist() == [10, 20]

    def test_defaults_come_from_settings(self, monkeypatch):
        labels = _labels({BENIGN: 50, "A": 20})
        monkeypatch.setattr(settings, "TARGET_MODE", "binary")
        target, report = build_target(labels)
        assert report["mode"] == "binary" and set(target) == {BENIGN, ATTACK_LABEL}

    def test_index_and_length_are_preserved(self):
        labels = pd.Series([BENIGN, "A", BENIGN], index=[5, 6, 7])
        target, _ = build_target(labels, mode="multiclass", rare_policy="keep")
        assert target.index.tolist() == [5, 6, 7]

    def test_input_is_not_modified(self):
        labels = _labels({BENIGN: 50, "A": 20, "R1": 3, "R2": 4})
        before = labels.copy()
        build_target(labels, mode="multiclass", rare_policy="merge")
        assert labels.equals(before)

    def test_invalid_inputs_are_rejected(self):
        labels = _labels({BENIGN: 50, "A": 20})
        with pytest.raises(ValueError, match="mode"):
            build_target(labels, mode="three_classes")
        with pytest.raises(ValueError, match="rare_policy"):
            build_target(labels, rare_policy="delete")
        with pytest.raises(ValueError, match="missing"):
            build_target(pd.Series([BENIGN, None, "A"]))
        with pytest.raises(ValueError, match=BENIGN):
            build_target(pd.Series(["A"] * 10))

    def test_report_is_json_safe(self):
        import json

        _, report = build_target(_labels({BENIGN: 50, "A": 20, "R1": 3}))
        json.dumps(report)


# ---------------------------------------------------------------------------
# Label codec
# ---------------------------------------------------------------------------
class TestLabelCodec:
    def test_benign_is_index_zero_then_alphabetical(self):
        codec = make_codec(pd.Series(["Zeta", "A", BENIGN, "Mid"]))
        assert codec.classes == (BENIGN, "A", "Mid", "Zeta")
        assert codec.benign_index == 0

    def test_codec_without_benign_has_no_benign_index(self):
        codec = make_codec(pd.Series(["A", "B"]))
        assert codec.benign_index is None and codec.classes == ("A", "B")

    def test_encode_decode_round_trip(self):
        target = _labels({BENIGN: 10, "A": 5, "B": 3})
        codec = make_codec(target)
        codes = codec.encode(target)
        assert codes.dtype == np.int64
        assert (codec.decode(codes) == target.to_numpy()).all()

    def test_encode_is_independent_of_the_row_index(self):
        codec = LabelCodec((BENIGN, "A"))
        series = pd.Series(["A", BENIGN], index=[100, 200])
        assert codec.encode(series).tolist() == [1, 0]

    def test_unknown_name_is_rejected_not_guessed(self):
        codec = LabelCodec((BENIGN, "A"))
        with pytest.raises(ValueError, match="unknown class name"):
            codec.encode(["A", "Other"])

    def test_out_of_range_code_is_rejected(self):
        codec = LabelCodec((BENIGN, "A"))
        with pytest.raises(ValueError, match="class codes"):
            codec.decode([0, 2])
        with pytest.raises(ValueError, match="class codes"):
            codec.decode([-1])

    def test_duplicate_or_empty_classes_are_rejected(self):
        with pytest.raises(ValueError):
            LabelCodec((BENIGN, BENIGN))
        with pytest.raises(ValueError):
            LabelCodec(())

    def test_codec_survives_joblib(self, tmp_path):
        codec = make_codec(_labels({BENIGN: 10, "A": 5}))
        path = tmp_path / "codec.joblib"
        joblib.dump(codec, path)
        assert joblib.load(path) == codec

    def test_to_dict(self):
        assert LabelCodec((BENIGN, "A")).to_dict() == {BENIGN: 0, "A": 1}


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
class TestStratifiedSplit:
    SIZES = {BENIGN: 400, "A": 120, "B": 60, "Rare": 11}

    def test_parts_are_disjoint_and_cover_all_rows(self):
        target = _labels(self.SIZES)
        train, test = stratified_split(target)
        assert np.intersect1d(train, test).size == 0
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(target)))

    def test_every_class_is_on_both_sides(self):
        target = _labels(self.SIZES)
        train, test = stratified_split(target)
        assert set(target.iloc[train]) == set(target) == set(target.iloc[test])

    def test_class_proportions_are_kept(self):
        target = _labels(self.SIZES)
        train, test = stratified_split(target, test_size=0.25)
        for name, size in self.SIZES.items():
            in_test = int((target.iloc[test] == name).sum())
            assert abs(in_test - 0.25 * size) <= 1

    def test_test_fraction_follows_the_setting(self):
        target = _labels(self.SIZES)
        _, test = stratified_split(target)
        assert abs(len(test) / len(target) - settings.TEST_SIZE) < 0.01

    def test_same_seed_same_split_different_seed_different_split(self):
        target = _labels(self.SIZES)
        a_train, a_test = stratified_split(target, seed=1)
        b_train, b_test = stratified_split(target, seed=1)
        c_train, _ = stratified_split(target, seed=2)
        assert np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)
        assert not np.array_equal(a_train, c_train)

    def test_indices_are_sorted_integers(self):
        train, test = stratified_split(_labels(self.SIZES))
        assert np.issubdtype(train.dtype, np.integer)
        assert (np.diff(train) > 0).all() and (np.diff(test) > 0).all()

    def test_accepts_numpy_codes(self):
        codes = _labels(self.SIZES).map({BENIGN: 0, "A": 1, "B": 2, "Rare": 3}).to_numpy()
        train, test = stratified_split(codes)
        assert len(train) + len(test) == len(codes)

    def test_class_with_one_row_is_rejected(self):
        with pytest.raises(ValueError, match="fewer than 2 rows"):
            stratified_split(_labels({BENIGN: 50, "Single": 1}))

    def test_invalid_inputs_are_rejected(self):
        target = _labels(self.SIZES)
        with pytest.raises(ValueError, match="test_size"):
            stratified_split(target, test_size=1.0)
        with pytest.raises(ValueError, match="test_size"):
            stratified_split(target, test_size=0.0)
        with pytest.raises(ValueError, match="non-empty"):
            stratified_split(np.array([]))
        with pytest.raises(ValueError, match="missing"):
            stratified_split(pd.Series([BENIGN, None] * 10))

    def test_works_with_the_shared_synthetic_frame(self, make_flow_frame):
        frame = make_flow_frame()
        train, test = stratified_split(frame[settings.LABEL_COLUMN])
        assert len(train) + len(test) == len(frame)


class TestSplitReport:
    def test_report_confirms_a_valid_split(self):
        target = _labels({BENIGN: 400, "A": 120, "Rare": 11})
        train, test = stratified_split(target)
        summary, table = split_report(target, train, test)
        assert summary["train_test_disjoint"] and summary["covers_all_rows"]
        assert summary["n_train"] == len(train) and summary["n_test"] == len(test)
        assert summary["classes_missing_from_test"] == [] == summary["classes_missing_from_train"]
        assert table[["train", "test"]].sum(axis=1).eq(table["total"]).all()

    def test_report_detects_overlap_and_gaps(self):
        target = _labels({BENIGN: 40, "A": 10})
        summary, _ = split_report(target, np.arange(0, 30), np.arange(25, 50))
        assert not summary["train_test_disjoint"]
        summary, _ = split_report(target, np.arange(0, 20), np.arange(30, 50))
        assert not summary["covers_all_rows"]

    def test_report_lists_classes_missing_from_a_side(self):
        target = pd.Series([BENIGN] * 10 + ["A"] * 10)
        train, test = np.arange(0, 15), np.arange(15, 20)  # test holds only "A"
        summary, _ = split_report(target, train, test)
        assert summary["classes_missing_from_test"] == [BENIGN]

    def test_report_flags_classes_too_small_for_cv(self):
        target = _labels({BENIGN: 400, "Tiny": 4})  # about 3 rows left in train, fewer than 5 folds
        train, test = stratified_split(target)
        summary, _ = split_report(target, train, test)
        assert summary["classes_train_below_cv_folds"] == ["Tiny"]

    def test_summary_is_json_safe(self):
        import json

        target = _labels({BENIGN: 100, "A": 30})
        train, test = stratified_split(target)
        summary, _ = split_report(target, train, test)
        json.dumps(summary)