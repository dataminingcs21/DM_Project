"""Tests of the experiment helpers (weighted votes, stacking input, day split, scoring) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from nids.experiments import day_split as D
from nids.experiments import ensembles as E
from nids.experiments.common import per_class_f1, score
from nids.voting import majority_voting as V


class TestWeightedVote:
    def test_equal_weights_match_the_project_majority_vote(self):
        rng = np.random.default_rng(0)
        predictions = {f"m{i}": rng.integers(0, 4, 500) for i in range(11)}
        ours = E.weighted_vote(predictions, 4, None, benign_index=0)
        reference = V.vote_labels(predictions, 4, benign_index=0, tie_break="prefer_non_benign").labels
        assert np.array_equal(ours, reference)

    def test_weights_let_a_strong_model_win(self):
        predictions = {"strong": np.array([2]), "weak1": np.array([1]), "weak2": np.array([1])}
        assert E.weighted_vote(predictions, 3, None)[0] == 1
        weights = {"strong": 0.9, "weak1": 0.5, "weak2": 0.5}
        assert E.weighted_vote(predictions, 3, weights)[0] == 1  # 0.9 < 0.5 + 0.5
        assert E.weighted_vote(predictions, 3, weights, power=4)[0] == 2  # 0.656 > 0.0625 + 0.0625

    def test_per_class_weights(self):
        predictions = {"a": np.array([1]), "b": np.array([2])}
        weights = {"a": np.array([1.0, 0.1, 1.0]), "b": np.array([1.0, 1.0, 0.5])}
        assert E.weighted_vote(predictions, 3, weights)[0] == 2  # a's vote for class 1 is worth 0.1

    def test_tie_prefers_an_attack_then_lowest_code(self):
        assert E.weighted_vote({"a": np.array([0]), "b": np.array([3])}, 4, None, benign_index=0)[0] == 3
        assert E.weighted_vote({"a": np.array([2]), "b": np.array([1])}, 4, None, benign_index=0)[0] == 1

    def test_errors(self):
        with pytest.raises(ValueError):
            E.weighted_vote({}, 3)
        with pytest.raises(ValueError):
            E.weighted_vote({"a": np.array([0, 1]), "b": np.array([0])}, 3)
        with pytest.raises(ValueError):
            E.weighted_vote({"a": np.array([5])}, 3)

    def test_oof_weights_and_subset(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        oof = {"perfect": y.copy(), "bad": np.zeros(6, dtype=int)}
        w = E.oof_model_weights(oof, y, 3)
        assert w["perfect"] == pytest.approx(1.0) and w["bad"] < 0.5
        assert E.best_subset(w, 1) == ["perfect"]
        assert E.oof_class_weights(oof, y, 3)["perfect"].tolist() == [1.0, 1.0, 1.0]

    def test_one_hot_votes_and_stacker(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        predictions = {"a": y.copy(), "b": rng.integers(0, 3, 300)}
        Z = E.one_hot_votes(predictions, ["a", "b"], 3)
        assert Z.shape == (300, 6) and np.all(Z.sum(axis=1) == 2)
        model = E.fit_stacker("logistic", Z, y, 3)
        assert np.mean(model.predict(Z) == y) > 0.95
        with pytest.raises(ValueError):
            E.fit_stacker("unknown", Z, y, 3)


class TestScoring:
    def test_score_and_reliable_classes(self):
        y_true = np.array([0] * 50 + [1] * 40 + [2] * 2)
        y_pred = y_true.copy()
        y_pred[-1] = 0  # one of the two class-2 rows missed
        s = score(y_true, y_pred, 3, benign_index=0)
        # class 2 (2 rows) is left out of the reliable average, where only BENIGN's precision suffers
        assert s["macro_f1_reliable"] > 0.99
        assert s["macro_f1"] < s["macro_f1_reliable"] - 0.1
        assert s["detection_rate"] == pytest.approx(41 / 42)
        assert s["false_alarm_rate"] == 0.0
        table = per_class_f1(y_true, y_pred, ["B", "X", "Y"])
        assert table["support"].tolist() == [50, 40, 2]


class TestDaySplit:
    def test_short_name(self):
        assert D.short_name("Wednesday-workingHours.pcap_ISCX.csv") == "Wednesday"
        assert D.short_name("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") == "Friday-Afternoon-DDos"

    def test_cap_benign_keeps_every_attack(self):
        df = pd.DataFrame({settings.LABEL_COLUMN: [settings.BENIGN_LABEL] * 100 + ["DoS"] * 10})
        capped = D.cap_benign(df, 20, seed=0)
        assert (capped[settings.LABEL_COLUMN] == "DoS").sum() == 10
        assert (capped[settings.LABEL_COLUMN] == settings.BENIGN_LABEL).sum() == 20
        assert capped.index.is_monotonic_increasing
        assert len(D.cap_benign(df, None, seed=0)) == 110

    def test_evaluate_binary(self):
        labels = np.array([settings.BENIGN_LABEL] * 4 + ["DoS"] * 2 + ["Bot"] * 2)
        scores = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.7, 0.3, 0.6])
        metrics, per_class = D.evaluate_binary(scores, labels)
        assert metrics["detection_rate"] == pytest.approx(3 / 4)
        assert metrics["false_alarm_rate"] == pytest.approx(1 / 4)
        assert per_class.set_index("class").loc["Bot", "flagged_share"] == pytest.approx(0.5)

    def test_leave_one_file_out_on_synthetic_days(self):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        parts = []
        for day, attack, shift in (("Mon.pcap_ISCX.csv", None, 0), ("Tue.pcap_ISCX.csv", "DoS", 6.0), ("Wed.pcap_ISCX.csv", "DDoS", 5.0)):
            n_benign = 150
            part = pd.DataFrame(rng.normal(size=(n_benign, 3)), columns=["Feature A", "Feature B", "Feature C"])
            part[settings.LABEL_COLUMN] = settings.BENIGN_LABEL
            if attack:
                bad = pd.DataFrame(rng.normal(loc=shift, size=(40, 3)), columns=part.columns[:3])
                bad[settings.LABEL_COLUMN] = attack
                part = pd.concat([part, bad], ignore_index=True)
            part[settings.SOURCE_FILE_COLUMN] = day
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        folds = D.leave_one_file_out(df, lambda: RandomForestClassifier(n_estimators=20, random_state=0), benign_cap=None)
        assert [f.held_out for f in folds] == ["Tue", "Wed"]
        assert folds[0].unseen_classes == ["DoS"]
        # DoS and DDoS are both "far from normal": a detector trained on one catches the other
        assert all(f.metrics["detection_rate"] > 0.9 for f in folds)
        metrics, _ = D.random_split_reference(df, lambda: RandomForestClassifier(n_estimators=20, random_state=0), benign_cap=None)
        assert metrics["detection_rate"] > 0.9
