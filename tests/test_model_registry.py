"""Tests for the model registry (File 41) and the 11 models (Files 30-40), all together.

Synthetic data only: 4 classes, standardized features, so every test runs in seconds.
These tests check the common interface of ``BaseModel`` on every model; they say
nothing about how well a model performs on the real data.
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from config import model_params
from nids.models import registry
from nids.models.base import BaseModel

KEYS = list(model_params.MODEL_KEYS)
DECISION_MODELS = {"svm", "sgd"}  # no probabilities
N_CLASSES = 4


def make_data(n: int = 600, n_features: int = 6, seed: int = 0):
    """Standardized synthetic features, 4 imbalanced classes, class codes 0..3 all present."""
    rng = np.random.default_rng(seed)
    y = rng.choice(N_CLASSES, size=n, p=[0.55, 0.25, 0.15, 0.05])
    y[:N_CLASSES] = np.arange(N_CLASSES)
    centers = np.random.default_rng(1).normal(0, 2.0, size=(N_CLASSES, n_features))
    X = centers[y] + rng.normal(0, 1.0, size=(n, n_features))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype("float32"), y.astype("int64")


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def fitted(data):
    """One fitted model per key (module-scoped: fitting 11 models once is enough)."""
    X, y = data
    return {key: registry.create_model(key).fit(X, y, n_classes=N_CLASSES) for key in KEYS}


# --------------------------------------------------------------------------- registry
class TestRegistry:
    def test_keys_are_the_eleven_in_brief_order(self):
        assert registry.all_model_keys() == tuple(model_params.MODEL_KEYS)
        assert len(registry.all_model_keys()) == 11

    def test_all_models_available(self):
        report = registry.availability()
        assert set(report) == set(KEYS)
        assert all(v["available"] and v["error"] is None for v in report.values())

    @pytest.mark.parametrize("key", KEYS)
    def test_create_model(self, key):
        model = registry.create_model(key)
        assert isinstance(model, BaseModel)
        assert model.key == key
        assert not model.is_fitted
        assert model.display_name == registry.display_name(key)

    def test_create_all_models_order_and_unfitted(self):
        models = registry.create_all_models()
        assert list(models) == KEYS
        assert all(not m.is_fitted for m in models.values())

    def test_params_override(self):
        model = registry.create_model("decision_tree", params={"max_depth": 3})
        assert model.params["max_depth"] == 3
        assert model.params["random_state"] == model_params.default_params("decision_tree")["random_state"]

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown model key"):
            registry.create_model("kmeans")
        with pytest.raises(ValueError):
            registry.get_model_class("apriori")

    def test_missing_library_makes_only_that_model_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "xgboost", None)  # import xgboost now fails
        monkeypatch.delitem(sys.modules, "nids.models.xgboost_model", raising=False)
        report = registry.availability()
        assert [k for k, v in report.items() if not v["available"]] == ["xgboost"]
        assert "xgboost" in report["xgboost"]["error"]
        with pytest.raises(registry.ModelUnavailableError):
            registry.create_all_models()
        partial = registry.create_all_models(skip_unavailable=True)
        assert list(partial) == [k for k in KEYS if k != "xgboost"]

    def test_class_key_mismatch_is_detected(self, monkeypatch):
        entries = dict(registry._REGISTRY)
        entries["knn"] = ("nids.models.decision_tree_model", "DecisionTreeModel")
        monkeypatch.setattr(registry, "_REGISTRY", entries)
        with pytest.raises(registry.ModelUnavailableError, match="registry expects"):
            registry.get_model_class("knn")

    def test_registry_and_model_keys_must_agree(self, monkeypatch):
        entries = dict(registry._REGISTRY)
        del entries["sgd"]
        monkeypatch.setattr(registry, "_REGISTRY", entries)
        with pytest.raises(ValueError, match="differ"):
            registry._validate()


# --------------------------------------------------------------------------- common interface
@pytest.mark.parametrize("key", KEYS)
class TestEveryModel:
    def test_predict(self, key, fitted, data):
        X, _ = data
        pred = fitted[key].predict(X)
        assert pred.dtype == np.int64 and pred.shape == (len(X),)
        assert set(np.unique(pred)) <= set(range(N_CLASSES))

    def test_scores(self, key, fitted, data):
        X, _ = data
        model = fitted[key]
        scores = model.predict_scores(X)
        assert scores.shape == (len(X), N_CLASSES)
        assert np.isfinite(scores).all()
        if key in DECISION_MODELS:
            assert model.score_kind == "decision"
        else:
            assert model.score_kind == "probability"
            assert scores.dtype == np.float64
            assert np.allclose(scores.sum(axis=1), 1.0)
            assert (scores >= 0).all()
            # scikit-learn's multiclass ROC-AUC accepts these rows only if they sum to 1
            assert 0.0 <= roc_auc_score(data[1], scores, multi_class="ovr") <= 1.0

    def test_top_score_matches_prediction(self, key, fitted, data):
        X, _ = data
        model = fitted[key]
        top = model.predict_scores(X).argmax(axis=1)
        agreement = float((top == model.predict(X)).mean())
        # SVC derives its scores from one-vs-one votes: rare ties may differ (seen on real data: 99.98%)
        assert agreement >= (0.99 if key == "svm" else 1.0)

    def test_learns_something(self, key, fitted, data):
        # the test data is easy by construction: every model must beat the majority class (55%)
        X, y = data
        assert float((fitted[key].predict(X) == y).mean()) > 0.60

    def test_save_and_load(self, key, fitted, data, tmp_path):
        X, _ = data
        path = tmp_path / f"{key}.joblib"
        joblib.dump(fitted[key], path)
        loaded = joblib.load(path)
        assert np.array_equal(loaded.predict(X[:100]), fitted[key].predict(X[:100]))

    def test_info_is_json_safe(self, key, fitted):
        info = fitted[key].info()
        json.dumps(info)
        assert info["key"] == key and info["is_fitted"] is True
        assert info["n_classes"] == N_CLASSES and info["n_train_rows"] == 600

    def test_not_fitted_error(self, key, data):
        with pytest.raises(NotFittedError):
            registry.create_model(key).predict(data[0])

    def test_with_params_returns_new_unfitted_model(self, key):
        model = registry.create_model(key)
        before = dict(model.params)
        other = model.with_params(**{k: v for k, v in list(before.items())[:1]})
        assert other is not model and not other.is_fitted and type(other) is type(model)
        assert model.params == before

    def test_unknown_parameter_is_an_error(self, key, data):
        X, y = data
        with pytest.raises(TypeError):
            registry.create_model(key, params={"no_such_parameter": 1}).fit(X, y)

    def test_rejects_bad_data(self, key, data):
        X, y = data
        bad = X.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="missing or infinite"):
            registry.create_model(key).fit(bad, y)
        with pytest.raises(ValueError, match="class codes"):
            registry.create_model(key).fit(X[y != 2], y[y != 2])  # a class is missing (gap in the codes)
        with pytest.raises(ValueError, match="classes"):
            registry.create_model(key).fit(X, y, n_classes=N_CLASSES + 1)

    def test_balanced_mode(self, key, data):
        X, y = data
        model = registry.create_model(key, class_weight_mode="balanced").fit(X, y)
        assert model.info()["class_weight_applied"] is (key != "knn")  # KNN cannot use weights

    def test_default_mode_applies_no_weights(self, key, fitted):
        assert fitted[key].info()["class_weight_applied"] is False


# --------------------------------------------------------------------------- model-specific behaviour
class TestSpecificBehaviour:
    def test_knn_refuses_sample_weight(self, data):
        X, y = data
        with pytest.raises(ValueError, match="sample weights"):
            registry.create_model("knn").fit(X, y, sample_weight=np.ones(len(y)))

    def test_negative_sample_weight_refused(self, data):
        X, y = data
        w = np.ones(len(y))
        w[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            registry.create_model("decision_tree").fit(X, y, sample_weight=w)

    def test_gaussian_nb_uses_float64(self):
        """Real-data finding (File 33): float32 broke Gaussian NB when within-class variance is tiny."""
        assert registry.get_model_class("gaussian_nb").input_dtype == "float64"
        rng = np.random.default_rng(3)
        y = np.repeat(np.arange(N_CLASSES), 200)
        X = (y[:, None] * 50.0 + rng.normal(0, 1e-3, size=(len(y), 5))).astype("float32")  # near-constant inside classes
        model = registry.create_model("gaussian_nb").fit(X, y, n_classes=N_CLASSES)
        scores = model.predict_scores(X)
        assert np.isfinite(scores).all() and np.allclose(scores.sum(axis=1), 1.0)

    def test_bernoulli_nb_probabilities_are_finite_with_huge_values(self):
        X, y = make_data()
        X[:, 0] *= 1e6
        scores = registry.create_model("bernoulli_nb").fit(X, y).predict_scores(X)
        assert np.isfinite(scores).all() and np.allclose(scores.sum(axis=1), 1.0)

    @pytest.mark.parametrize("key", ["logistic_regression", "sgd"])
    def test_convergence_is_reported(self, key, data):
        X, y = data
        converged = registry.create_model(key).fit(X, y)
        assert converged.info()["converged"] is True
        assert converged.info()["n_iter"] < converged.info()["max_iter"]
        stopped = registry.create_model(key, params={"max_iter": 1}).fit(X, y)
        assert stopped.converged is False and stopped.info()["converged"] is False

    def test_xgboost_device(self, data, monkeypatch):
        from nids.models import xgboost_model as xm

        assert xm.resolve_device("cpu") == "cpu"
        assert xm.resolve_device("cuda") == "cuda"
        with pytest.raises(ValueError):
            xm.resolve_device("tpu")
        X, y = data
        model = registry.create_model("xgboost", params={"device": "cpu"}).fit(X, y)
        info = model.info()
        assert info["train_device"].startswith("cpu") and info["n_rounds"] == 100
        # whatever the training device was, predictions are made on the CPU
        assert model.estimator_.get_params()["device"] == "cpu"

    def test_xgboost_explicit_cuda_without_gpu_is_an_error(self, data):
        from nids.models import xgboost_model as xm

        if xm._gpu_works():
            pytest.skip("this computer has a usable GPU")
        X, y = data
        with pytest.raises(RuntimeError, match="GPU"):
            registry.create_model("xgboost", params={"device": "cuda"}).fit(X, y)

    def test_random_forest_and_tree_are_reproducible(self, data):
        X, y = data
        for key in ("decision_tree", "random_forest", "xgboost"):
            a = registry.create_model(key, params={"device": "cpu"} if key == "xgboost" else None).fit(X, y)
            b = registry.create_model(key, params={"device": "cpu"} if key == "xgboost" else None).fit(X, y)
            assert np.array_equal(a.predict_scores(X), b.predict_scores(X)), key

    def test_binary_scores_have_two_columns(self):
        """Two classes: decision models turn the single score into two columns."""
        X, y = make_data()
        y2 = (y > 0).astype("int64")
        for key in ("sgd", "svm"):
            scores = registry.create_model(key).fit(X, y2).predict_scores(X)
            assert scores.shape == (len(X), 2)