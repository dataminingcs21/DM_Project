"""Tests for ``nids.data.schema`` (File 07) and ``nids.data.loader`` (File 08).

Only synthetic data is used (see ``conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.data import schema
from nids.data.loader import list_raw_files, load_raw_data, read_csv_file


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------
class TestNormalization:
    def test_column_names_are_stripped_and_collapsed(self):
        assert schema.normalize_columns([" Flow Duration", "Label ", "a   b"]) == ["Flow Duration", "Label", "a b"]

    def test_label_value_whitespace(self):
        assert schema.normalize_label_value("  DoS   Hulk ") == "DoS Hulk"

    def test_missing_label_values_stay_missing(self):
        assert schema.normalize_label_value(None) is None
        assert np.isnan(schema.normalize_label_value(np.nan))

    def test_duplicates_found_after_normalization(self):
        assert schema.find_duplicate_columns(["A", " A", "B"]) == ["A"]
        assert schema.find_duplicate_columns(["A", "B"]) == []


class TestColumnRoles:
    def test_feature_columns_exclude_label_and_metadata(self):
        cols = [" Feature A", " Label", settings.SOURCE_FILE_COLUMN, "Feature B"]
        assert schema.get_feature_columns(cols) == ["Feature A", "Feature B"]

    def test_has_label(self):
        assert schema.has_label([" Label"])
        assert not schema.has_label(["Other"])

    def test_metadata_columns_contains_source_file(self):
        assert settings.SOURCE_FILE_COLUMN in schema.METADATA_COLUMNS


class TestCompareColumns:
    def test_identical_ignoring_spaces(self):
        assert schema.compare_columns([" a", "b"], ["a", "b"]).is_identical

    def test_missing_and_extra(self):
        result = schema.compare_columns(["a", "b", "z"], ["a", "b", "c"])
        assert result.missing == ["c"]
        assert result.extra == ["z"]
        assert not result.is_identical

    def test_same_columns_different_order(self):
        result = schema.compare_columns(["b", "a"], ["a", "b"])
        assert not result.missing and not result.extra
        assert not result.same_order
        assert not result.is_identical


class TestCoerceNumeric:
    def test_counts_unconvertible_values_and_keeps_input(self):
        df = pd.DataFrame({"x": ["1", "2", "Infinity", "bad", None], "y": [1, 2, 3, 4, 5]})
        out, report = schema.coerce_numeric(df, ["x", "y"])
        assert report == {"x": 1}  # only "bad"; None was already missing
        assert pd.api.types.is_numeric_dtype(out["x"])
        assert np.isinf(out["x"].iloc[2])  # "Infinity" becomes inf, not NaN
        assert df["x"].dtype == object  # the input is not modified

    def test_numeric_columns_are_skipped(self):
        df = pd.DataFrame({"y": [1, 2]})
        out, report = schema.coerce_numeric(df, ["y"])
        assert report == {}
        assert out.equals(df)


# ---------------------------------------------------------------------------
# loader.py
# ---------------------------------------------------------------------------
class TestListRawFiles:
    def test_only_csv_files_sorted(self, raw_csv_dir):
        assert [p.name for p in list_raw_files(raw_csv_dir)] == ["a.csv", "b.csv"]

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Raw data folder not found"):
            list_raw_files(tmp_path / "nope")

    def test_empty_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files matching"):
            list_raw_files(tmp_path)


class TestReadCsvFile:
    def test_normalizes_and_adds_metadata(self, raw_csv_dir):
        df, report = read_csv_file(raw_csv_dir / "a.csv")
        assert list(df.columns) == ["Feature A", "Feature B", "Label", settings.SOURCE_FILE_COLUMN]
        assert df["Label"].tolist() == ["BENIGN", "DoS Hulk", "BENIGN"]
        assert (df[settings.SOURCE_FILE_COLUMN] == "a.csv").all()
        assert report["rows"] == 3 and report["encoding"] == "utf-8"
        assert report["non_numeric_values_coerced"] == {"Feature B": 1}

    def test_infinity_is_kept_for_the_cleaner(self, raw_csv_dir):
        df, _ = read_csv_file(raw_csv_dir / "a.csv")
        assert np.isinf(df["Feature B"].iloc[0])
        assert np.isnan(df["Feature B"].iloc[2])

    def test_windows_1252_file_is_decoded(self, raw_csv_dir):
        df, report = read_csv_file(raw_csv_dir / "b.csv")
        assert report["encoding"] == "cp1252"
        assert "Web Attack – XSS" in df["Label"].tolist()

    def test_nrows(self, raw_csv_dir):
        df, _ = read_csv_file(raw_csv_dir / "a.csv", nrows=1)
        assert len(df) == 1

    def test_missing_label_column(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("Feature A,Other\n1,2\n")
        with pytest.raises(ValueError, match="label column"):
            read_csv_file(path)

    def test_duplicate_columns_after_normalization(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("Feature A, Feature A,Label\n1,2,BENIGN\n")
        with pytest.raises(ValueError, match="duplicate column names"):
            read_csv_file(path)


class TestLoadRawData:
    def test_combines_files_in_reference_column_order(self, raw_csv_dir):
        df, report = load_raw_data(raw_csv_dir)
        assert list(df.columns) == ["Feature A", "Feature B", "Label", settings.SOURCE_FILE_COLUMN]
        assert len(df) == 5
        assert df[settings.SOURCE_FILE_COLUMN].tolist() == ["a.csv"] * 3 + ["b.csv"] * 2
        # b.csv has its columns in another order: values must still be aligned by name
        assert df.loc[df["Label"] == "BENIGN", "Feature A"].tolist()[-1] == 6

    def test_report_content(self, raw_csv_dir):
        _, report = load_raw_data(raw_csv_dir)
        assert report["n_files"] == 2
        assert report["n_rows"] == 5
        assert report["n_feature_columns"] == 2
        assert [f["file"] for f in report["files"]] == ["a.csv", "b.csv"]

    def test_nrows_applies_to_each_file(self, raw_csv_dir):
        df, _ = load_raw_data(raw_csv_dir, nrows=1)
        assert len(df) == 2

    def test_column_mismatch_between_files_is_an_error(self, raw_csv_dir):
        (raw_csv_dir / "c.csv").write_text("Feature A,Label\n1,BENIGN\n")
        with pytest.raises(ValueError, match="columns differ"):
            load_raw_data(raw_csv_dir)

    def test_no_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path)

    def test_default_folder_is_settings_raw_dir(self, test_project_root):
        # The temporary project has no raw data: the error must point at the settings folder.
        with pytest.raises(FileNotFoundError) as error:
            load_raw_data()
        assert str(settings.RAW_DATA_DIR) in str(error.value)