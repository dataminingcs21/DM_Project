"""Tests of the data mining extensions (clustering, anomaly, association rules, importance) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.mining import anomaly as A
from nids.mining import association as R
from nids.mining import clustering as C
from nids.mining import importance as I
from nids.mining.reporting import markdown_table


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def blobs(seed: int = 0, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    y = np.repeat(np.arange(3), n)
    return centers[y] + rng.normal(scale=0.3, size=(y.size, 2)), y


class TestClusteringScores:
    def test_purity_hand_worked(self):
        y = np.array([0, 0, 1, 1, 1, 2])
        clusters = np.array([0, 0, 0, 1, 1, 1])  # cluster 0: two 0s + one 1; cluster 1: two 1s + one 2
        assert C.purity(y, clusters) == pytest.approx(4 / 6)

    def test_purity_rejects_mismatch(self):
        with pytest.raises(ValueError):
            C.purity(np.array([0, 1]), np.array([0]))

    def test_majority_map_and_fallback(self):
        mapping = C.cluster_majority_map(np.array([2, 2, 1, 0]), np.array([5, 5, 5, 7]))
        assert mapping == {5: 2, 7: 0}
        assert C.apply_majority_map(np.array([5, 7, 9]), mapping, fallback=1).tolist() == [2, 0, 1]

    def test_perfect_clustering_scores_one(self):
        X, y = blobs()
        scores = C.cluster_scores(X, y, y + 10)  # cluster ids differ from class ids: scores must not care
        assert scores["ari"] == pytest.approx(1.0)
        assert scores["purity"] == pytest.approx(1.0)
        assert scores["silhouette"] > 0.8
        assert scores["noise_fraction"] == 0.0

    def test_noise_is_excluded_from_internal_scores(self):
        X, y = blobs()
        clusters = y.copy()
        clusters[:5] = C.NOISE
        scores = C.cluster_scores(X, y, clusters)
        assert scores["n_clusters"] == 3
        assert scores["noise_fraction"] == pytest.approx(5 / y.size)

    def test_kmeans_sweep_finds_the_blobs(self):
        X, y = blobs()
        X2, y2 = blobs(seed=1)
        sweep, models = C.kmeans_sweep(X, y, X2, y2, k_values=(2, 3), n_classes=3)
        row = sweep.set_index("k").loc[3]
        assert row["ari"] == pytest.approx(1.0)
        assert row["holdout_macro_f1"] == pytest.approx(1.0)
        assert set(models) == {2, 3}

    def test_dbscan_and_composition(self):
        X, y = blobs()
        table, curve, labels = C.dbscan_grid(X, y, min_samples=5, quantiles=(0.9,))
        assert len(table) == 1 and np.all(np.diff(curve) >= 0)
        comp = C.cluster_composition(np.array(["a", "b", "c"], dtype=object)[y], next(iter(labels.values())))
        assert comp["size"].sum() == y.size
        assert (comp["majority_share"] <= 1).all()

    def test_agglomerative(self):
        X, y = blobs()
        labels, scores = C.agglomerative(X, y, 3)
        assert scores["ari"] == pytest.approx(1.0) and labels.shape == y.shape


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
class TestAnomaly:
    def test_thresholds_are_calibration_quantiles(self):
        t = A.thresholds_from_calibration(np.arange(1000), (0.01, 0.1))
        assert t[0.01] == pytest.approx(np.quantile(np.arange(1000), 0.99))
        assert t[0.1] < t[0.01]

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_bad_target_rejected(self, bad):
        with pytest.raises(ValueError):
            A.thresholds_from_calibration(np.arange(10), (bad,))

    def test_operating_point_counts(self):
        scores = np.array([0.1, 0.9, 0.8, 0.2, 0.95])
        is_attack = np.array([False, True, False, True, True])
        op = A.operating_point(scores, is_attack, 0.5)
        assert op["detection_rate"] == pytest.approx(2 / 3)
        assert op["false_alarm_rate"] == pytest.approx(1 / 2)
        assert op["precision"] == pytest.approx(2 / 3)
        assert op["alarms"] == 3

    @pytest.mark.parametrize("detector", [A.IsolationForestDetector(n_estimators=50), A.LOFDetector(n_neighbors=10), A.AutoencoderDetector(hidden=(4, 2, 4), max_iter=30)])
    def test_detectors_rank_outliers_higher(self, detector):
        rng = np.random.default_rng(0)
        normal = rng.normal(size=(600, 4))
        test = np.vstack([rng.normal(size=(200, 4)), rng.normal(loc=8.0, size=(50, 4))])
        names = np.array([settings.BENIGN_LABEL] * 200 + ["Attack"] * 50, dtype=object)
        result, scores = A.evaluate_detector(detector, normal[:400], normal[400:], test, names, (0.05,))
        assert result.roc_auc > 0.95
        assert scores.shape == (250,)
        per_class = result.per_class.set_index("class")
        assert per_class.loc["Attack", "flagged_at_far_0.05"] > 0.9
        assert per_class.loc[settings.BENIGN_LABEL, "flagged_at_far_0.05"] < 0.2


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------
def flows(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for label, port, duration in (("BENIGN", 443, 1e6), ("FTP-Patator", 21, 10.0), ("DoS Hulk", 80, 1e3)):
        part = {c: rng.uniform(1, 100, n_per_class) for c in R.RULE_FEATURES}
        part["Flow Duration"] = rng.uniform(0.9, 1.1, n_per_class) * duration
        part["Down/Up Ratio"] = rng.integers(0, 3, n_per_class)
        part.update({c: rng.integers(0, 2, n_per_class) for c in R.FLAG_FEATURES})
        part[R.PORT_COLUMN] = port
        part[settings.LABEL_COLUMN] = label
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)


class TestAssociation:
    def test_port_items(self):
        assert R.port_item(21) == "port=FTP"
        assert R.port_item(1000) == "port=other well-known"
        assert R.port_item(5000) == "port=registered"
        assert R.port_item(60000) == "port=ephemeral"

    def test_encoder_bins_zero_and_exact_values(self):
        df = flows()
        df.loc[0, "Flow Bytes/s"] = 0
        items = R.ItemEncoder().fit(df).transform(df)
        assert items.dtypes.eq(bool).all()
        assert items.loc[0, "Flow Bytes/s=0"]
        assert {"Flow Duration=low", "Flow Duration=mid", "Flow Duration=high"} <= set(items.columns)
        assert "Down/Up Ratio=2" in items.columns  # few distinct values -> exact items
        # every flow has exactly one item per binned feature
        assert (items[[c for c in items.columns if c.startswith("Flow Duration=")]].sum(axis=1) == 1).all()
        assert "class=FTP-Patator" in items.columns

    def test_encoder_requires_fit_and_columns(self):
        with pytest.raises(RuntimeError):
            R.ItemEncoder().transform(flows())
        with pytest.raises(KeyError):
            R.ItemEncoder().fit(flows().drop(columns=[R.PORT_COLUMN]))

    def test_class_rules_and_classifier(self):
        train, test = flows(), flows(seed=1)
        encoder = R.ItemEncoder().fit(train)
        rules = R.class_rules(encoder.transform(train), within_class_support=0.5, min_confidence=0.9)
        assert {"FTP-Patator", "DoS Hulk"} <= set(rules["class"])
        assert (rules["confidence"] >= 0.9).all()
        predicted = R.RuleClassifier(rules).predict(encoder.transform(test, with_class=False))
        assert np.mean(predicted == test[settings.LABEL_COLUMN].to_numpy()) > 0.95

    def test_fpgrowth_rules_and_class_filter(self):
        items = R.ItemEncoder().fit(flows()).transform(flows())
        itemsets, rules = R.mine_rules(items, min_support=0.2, min_confidence=0.9, max_len=2)
        assert len(itemsets) > 0 and (rules["confidence"] >= 0.9).all()
        to_class = R.rules_with_class_consequent(rules)
        assert to_class["consequents"].map(lambda s: next(iter(s)).startswith(R.CLASS_PREFIX)).all()
        assert "port=FTP" in R.describe_rules(to_class)["antecedents"].str.cat()


# ---------------------------------------------------------------------------
# Importance and reporting
# ---------------------------------------------------------------------------
class TestImportanceAndReporting:
    def test_permutation_finds_the_useful_feature(self):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 3))
        y = (X[:, 1] > 0).astype(int)
        model = DecisionTreeClassifier(random_state=0).fit(X, y)
        table = I.permutation_table(model, X, y, ["a", "b", "c"], n_repeats=3)
        assert table.iloc[0]["feature"] == "b"
        builtin = I.builtin_importance(model, ["a", "b", "c"])
        assert builtin.iloc[0]["feature"] == "b" and builtin["importance"].sum() == pytest.approx(1.0)
        agreement = I.rank_agreement(builtin, table, "importance", "f1_drop_mean", top=1)
        assert agreement["n_shared"] == 1

    def test_builtin_importance_absent(self):
        assert I.builtin_importance(object(), ["a"]) is None

    def test_markdown_table(self):
        text = markdown_table(pd.DataFrame({"name": ["a|b"], "value": [0.123456], "n": [3]}), digits=2)
        lines = text.splitlines()
        assert lines[0] == "| name | value | n |"
        assert lines[1] == "|---|---:|---:|"
        assert lines[2] == "| a\\|b | 0.12 | 3 |"
