"""Tests for the model store (File 43): saving, loading, damage detection and staleness.

Synthetic data only.  Most tests write into pytest's temporary folders; the two
that use the default folders write into the temporary project root created by
``tests/conftest.py`` (never into your real ``artifacts/``).
"""

from __future__ import annotations

import json
import shutil
import warnings

import joblib
import numpy as np
import pytest

from config import model_params, settings
from nids.models import registry
from nids.persistence import model_store as ms
from nids.utils import io_utils

KEYS = list(model_params.MODEL_KEYS)
N_CLASSES = 4


def make_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, size=n)
    y[:N_CLASSES] = np.arange(N_CLASSES)
    centers = np.random.default_rng(1).normal(0, 2.0, size=(N_CLASSES, 5))
    X = (centers[y] + rng.normal(0, 1.0, size=(n, 5))).astype("float32")
    return X, y.astype("int64")


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture()
def tree(data):
    X, y = data
    return registry.create_model("decision_tree").fit(X, y, n_classes=N_CLASSES)


def rewrite_metadata(directory, key, **changes):
    path = directory / f"{key}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    path.write_text(json.dumps(meta), encoding="utf-8")


# --------------------------------------------------------------------------- save and load
class TestSaveAndLoad:
    @pytest.mark.parametrize("key", KEYS)
    def test_round_trip_for_every_model(self, key, data, tmp_path):
        X, y = data
        model = registry.create_model(key).fit(X, y, n_classes=N_CLASSES)
        ms.save_model(model, directory=tmp_path)
        loaded = ms.load_model(key, directory=tmp_path)
        assert type(loaded) is type(model) and loaded.is_fitted
        assert np.array_equal(loaded.predict(X), model.predict(X))
        assert np.allclose(loaded.predict_scores(X), model.predict_scores(X))

    def test_metadata_content(self, tree, tmp_path):
        meta = ms.save_model(tree, extra={"subsample": 1234}, directory=tmp_path)
        assert meta["key"] == "decision_tree" and meta["stage"] == "trained"
        assert meta["n_features_in"] == 5
        assert meta["extra"] == {"subsample": 1234}
        assert meta["model_info"]["n_train_rows"] == 300 and meta["model_info"]["is_fitted"] is True
        assert meta["sha256"] and len(meta["sha256"]) == 64
        assert meta["preprocessing_hash"] == ms.current_hashes()["preprocessing_hash"]
        assert meta["model_params_hash"] == ms.current_hashes()["model_params_hash"]
        assert set(meta["libraries"]) >= {"python", "scikit-learn", "numpy", "joblib"}
        assert ms.load_metadata("decision_tree", directory=tmp_path) == json.loads(
            json.dumps(meta)
        )  # what is on disk is what was returned

    def test_default_folders(self, tree):
        ms.save_model(tree)
        ms.save_model(tree, stage="tuned")
        try:
            assert (settings.TRAINED_MODELS_DIR / "decision_tree.joblib").is_file()
            assert (settings.TRAINED_MODELS_DIR / "decision_tree.json").is_file()
            assert (settings.TUNED_MODELS_DIR / "decision_tree.joblib").is_file()
            assert ms.load_model("decision_tree").is_fitted
            assert ms.load_metadata("decision_tree", stage="tuned")["stage"] == "tuned"
        finally:
            ms.delete_model("decision_tree")
            ms.delete_model("decision_tree", stage="tuned")
        assert not (settings.TRAINED_MODELS_DIR / "decision_tree.joblib").exists()

    def test_overwrite_replaces_the_model(self, data, tmp_path):
        X, y = data
        first = registry.create_model("decision_tree", params={"max_depth": 1}).fit(X, y)
        second = registry.create_model("decision_tree", params={"max_depth": 6}).fit(X, y)
        ms.save_model(first, directory=tmp_path)
        ms.save_model(second, directory=tmp_path)
        loaded = ms.load_model("decision_tree", directory=tmp_path)
        assert loaded.params["max_depth"] == 6
        assert ms.load_metadata("decision_tree", directory=tmp_path)["model_info"]["params"]["max_depth"] == 6

    def test_bad_arguments(self, tree, tmp_path):
        with pytest.raises(ValueError, match="stage"):
            ms.stage_dir("final")
        with pytest.raises(ValueError, match="unknown model key"):
            ms.model_file("kmeans")
        with pytest.raises(ms.ModelStoreError):
            ms.save_model(registry.create_model("knn"), directory=tmp_path)  # not fitted
        with pytest.raises(TypeError):
            ms.save_model(object(), directory=tmp_path)  # not a BaseModel

    def test_missing_model(self, tmp_path):
        with pytest.raises(ms.ModelStoreError, match="no saved"):
            ms.load_model("knn", directory=tmp_path)


# --------------------------------------------------------------------------- damage and interruption
class TestDamage:
    def test_checksum_detects_a_replaced_file(self, data, tree, tmp_path):
        X, y = data
        ms.save_model(tree, directory=tmp_path)
        other = registry.create_model("decision_tree", params={"max_depth": 2}).fit(X, y)
        joblib.dump(other, tmp_path / "decision_tree.joblib")  # a valid model, but not the saved one
        with pytest.raises(ms.ModelStoreError, match="checksum"):
            ms.load_model("decision_tree", directory=tmp_path)
        assert ms.load_model("decision_tree", directory=tmp_path, verify=False).is_fitted  # check skipped on request

    def test_truncated_file(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        path = tmp_path / "decision_tree.joblib"
        path.write_bytes(path.read_bytes()[:-10])
        with pytest.raises(ms.ModelStoreError, match="checksum"):
            ms.load_model("decision_tree", directory=tmp_path)

    def test_file_with_the_wrong_model_inside(self, data, tree, tmp_path):
        X, y = data
        ms.save_model(registry.create_model("knn").fit(X, y), directory=tmp_path)
        ms.save_model(tree, directory=tmp_path)  # correct metadata for the tree
        shutil.copy(tmp_path / "knn.joblib", tmp_path / "decision_tree.joblib")  # then a KNN is placed under its name
        with pytest.raises(ms.ModelStoreError):
            ms.load_model("decision_tree", directory=tmp_path, verify=False)

    def test_not_a_model_inside(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        joblib.dump({"not": "a model"}, tmp_path / "decision_tree.joblib")
        with pytest.raises(ms.ModelStoreError, match="does not contain"):
            ms.load_model("decision_tree", directory=tmp_path, verify=False)

    def test_metadata_without_model_file(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        (tmp_path / "decision_tree.joblib").unlink()
        with pytest.raises(ms.ModelStoreError, match="missing"):
            ms.load_model("decision_tree", directory=tmp_path)
        assert "metadata without model file" in ms.list_saved(directory=tmp_path)["decision_tree"]["problems"][0]

    def test_model_file_without_metadata(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        (tmp_path / "decision_tree.json").unlink()
        with pytest.raises(ms.ModelStoreError, match="no saved"):
            ms.load_model("decision_tree", directory=tmp_path)
        report = ms.list_saved(directory=tmp_path)["decision_tree"]
        assert report["saved"] is False and "interrupted" in report["problems"][0]

    def test_interrupted_save_leaves_no_old_metadata(self, data, tmp_path, monkeypatch):
        X, y = data
        old = registry.create_model("decision_tree", params={"max_depth": 1}).fit(X, y)
        new = registry.create_model("decision_tree", params={"max_depth": 6}).fit(X, y)
        ms.save_model(old, directory=tmp_path)

        def broken_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(ms, "write_json", broken_write)
        with pytest.raises(OSError):
            ms.save_model(new, directory=tmp_path)
        monkeypatch.undo()
        # the new model file exists but there is no metadata: the save is not considered complete
        with pytest.raises(ms.ModelStoreError, match="no saved"):
            ms.load_model("decision_tree", directory=tmp_path)

    def test_corrupt_metadata_json_is_reported_not_crashed(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        (tmp_path / "decision_tree.json").write_text("{not json", encoding="utf-8")
        report = ms.list_saved(directory=tmp_path)["decision_tree"]
        assert report["saved"] is False and report["fresh"] is False


# --------------------------------------------------------------------------- staleness and versions
class TestFreshness:
    def test_fresh_model_loads(self, tree, tmp_path):
        meta = ms.save_model(tree, directory=tmp_path)
        assert ms.check_fresh(meta) == []
        assert ms.list_saved(directory=tmp_path)["decision_tree"]["fresh"] is True

    def test_preprocessing_change_makes_models_stale(self, tree, tmp_path, monkeypatch):
        ms.save_model(tree, directory=tmp_path)
        monkeypatch.setattr(settings, "TEST_SIZE", 0.30)
        with pytest.raises(ms.StaleModelError, match="preprocessing"):
            ms.load_model("decision_tree", directory=tmp_path)
        report = ms.list_saved(directory=tmp_path)["decision_tree"]
        assert report["saved"] is True and report["fresh"] is False

    def test_model_settings_change_makes_models_stale(self, tree, tmp_path, monkeypatch):
        ms.save_model(tree, directory=tmp_path)
        monkeypatch.setitem(settings.SUBSAMPLE_SIZES, "knn", settings.SUBSAMPLE_SIZES["knn"] + 1)
        with pytest.raises(ms.StaleModelError, match="model settings"):
            ms.load_model("decision_tree", directory=tmp_path)

    def test_class_weight_mode_change_makes_models_stale(self, tree, tmp_path, monkeypatch):
        ms.save_model(tree, directory=tmp_path)
        monkeypatch.setattr(settings, "CLASS_WEIGHT_MODE", "balanced")
        with pytest.raises(ms.StaleModelError):
            ms.load_model("decision_tree", directory=tmp_path)

    def test_hardware_setting_does_not_make_models_stale(self, tree, tmp_path, monkeypatch):
        ms.save_model(tree, directory=tmp_path)
        monkeypatch.setattr(settings, "N_JOBS", 2)
        assert ms.load_model("decision_tree", directory=tmp_path).is_fitted

    def test_allow_stale_loads_with_a_warning(self, tree, tmp_path, caplog):
        ms.save_model(tree, directory=tmp_path)
        rewrite_metadata(tmp_path, "decision_tree", model_params_hash="old")
        with pytest.raises(ms.StaleModelError):
            ms.load_model("decision_tree", directory=tmp_path)
        assert ms.load_model("decision_tree", directory=tmp_path, allow_stale=True).is_fitted

    def test_stale_error_is_a_store_error(self):
        assert issubclass(ms.StaleModelError, ms.ModelStoreError)

    def test_library_version_change_warns(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        meta = ms.load_metadata("decision_tree", directory=tmp_path)
        libs = dict(meta["libraries"], **{"scikit-learn": "0.0.1"})
        rewrite_metadata(tmp_path, "decision_tree", libraries=libs)
        with pytest.warns(RuntimeWarning, match="other library versions"):
            ms.load_model("decision_tree", directory=tmp_path)
        with pytest.raises(ms.ModelStoreError, match="other library versions"):
            ms.load_model("decision_tree", directory=tmp_path, strict_versions=True)

    def test_same_versions_do_not_warn(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ms.load_model("decision_tree", directory=tmp_path)


# --------------------------------------------------------------------------- listing and deleting
class TestListAndDelete:
    def test_list_saved_covers_all_eleven_keys(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        report = ms.list_saved(directory=tmp_path)
        assert list(report) == KEYS
        assert [k for k, v in report.items() if v["saved"]] == ["decision_tree"]
        assert report["knn"]["problems"] == ["not saved"]
        assert report["decision_tree"]["file_size_mb"] is not None

    def test_delete(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        assert ms.delete_model("decision_tree", directory=tmp_path) is True
        assert not any(tmp_path.iterdir())
        assert ms.delete_model("decision_tree", directory=tmp_path) is False

    def test_metadata_is_plain_json(self, tree, tmp_path):
        ms.save_model(tree, directory=tmp_path)
        text = (tmp_path / "decision_tree.json").read_text(encoding="utf-8")
        assert json.loads(text)["sha256"] == io_utils.read_json(tmp_path / "decision_tree.json")["sha256"]