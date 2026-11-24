"""Tests for the majority vote (File 53).

Synthetic data only.  The main test compares the vectorised vote with a plain Python
reference implementation of the written rule on thousands of random flows, so the two
cannot agree by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import model_params, settings
from nids.preprocessing.target_builder import LabelCodec
from nids.voting import majority_voting as V

KEYS = list(model_params.MODEL_KEYS)
CODEC = LabelCodec(("BENIGN", "A", "B", "C"))
K = CODEC.n_classes


def reference_vote(columns: list[int], rule: str, benign: int | None = 0) -> int:
    """The written rule, as simply as possible: count, find the leaders, break ties."""
    counts = [columns.count(c) for c in range(K)]
    leaders = [c for c in range(K) if counts[c] == max(counts)]
    if rule == "prefer_non_benign" and benign is not None:
        attacks = [c for c in leaders if c != benign]
        if attacks:
            leaders = attacks
    return min(leaders)


class FakeModel:
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype="int64")

    def predict(self, X):
        return self.labels[: len(X)]


class BrokenModel:
    def predict(self, X):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
class TestRule:
    def test_clear_majority(self):
        r = V.vote_labels({"a": [0, 1], "b": [0, 1], "c": [1, 1]}, K, expected_keys=["a", "b", "c"])
        assert r.labels.tolist() == [0, 1]
        assert not r.tied.any()
        assert r.top_votes.tolist() == [2, 3]

    def test_one_equal_vote_per_model(self):
        # 3 weak models say A, 2 strong ones say B: the majority wins, nobody counts double
        p = {f"m{i}": [1] for i in range(3)} | {f"s{i}": [2] for i in range(2)}
        assert V.vote_labels(p, K, expected_keys=list(p)).labels.tolist() == [1]

    def test_tie_prefers_attack_over_benign(self):
        r = V.vote_labels({"a": [0], "b": [2]}, K, expected_keys=["a", "b"])
        assert r.labels.tolist() == [2] and r.tied.tolist() == [True]

    def test_tie_between_attacks_takes_lowest_code(self):
        r = V.vote_labels({"a": [3], "b": [1], "c": [2]}, K, expected_keys=["a", "b", "c"])
        assert r.labels.tolist() == [1]

    def test_three_way_tie_including_benign(self):
        r = V.vote_labels({"a": [0], "b": [2], "c": [3]}, K, expected_keys=["a", "b", "c"])
        assert r.labels.tolist() == [2]

    def test_lowest_index_rule(self):
        r = V.vote_labels({"a": [0], "b": [2]}, K, tie_break="lowest_index", expected_keys=["a", "b"])
        assert r.labels.tolist() == [0] and r.tie_break == "lowest_index"

    def test_no_benign_index_means_lowest_code(self):
        r = V.vote_labels({"a": [0], "b": [2]}, K, benign_index=None, expected_keys=["a", "b"])
        assert r.labels.tolist() == [0]

    def test_benign_wins_only_when_it_has_most_votes(self):
        r = V.vote_labels({"a": [0], "b": [0], "c": [2]}, K, expected_keys=["a", "b", "c"])
        assert r.labels.tolist() == [0]

    @pytest.mark.parametrize("rule", settings.ALLOWED_TIE_BREAK_RULES)
    @pytest.mark.parametrize("n_models", [1, 2, 3, 6, 11])
    def test_matches_reference_on_random_flows(self, rule, n_models):
        rng = np.random.default_rng(n_models)
        keys = KEYS[:n_models]
        preds = {k: rng.integers(0, K, size=2000) for k in keys}
        result = V.vote_labels(preds, K, 0, tie_break=rule, expected_keys=keys)
        expected = [reference_vote([int(preds[k][i]) for k in keys], rule) for i in range(2000)]
        assert result.labels.tolist() == expected

    def test_order_of_models_does_not_matter(self):
        rng = np.random.default_rng(0)
        preds = {k: rng.integers(0, K, size=500) for k in KEYS}
        a = V.vote_labels(preds, K)
        b = V.vote_labels(dict(reversed(list(preds.items()))), K)
        assert (a.labels == b.labels).all()

    def test_repeatable(self):
        rng = np.random.default_rng(1)
        preds = {k: rng.integers(0, K, size=300) for k in KEYS}
        assert (V.vote_labels(preds, K).labels == V.vote_labels(preds, K).labels).all()

    def test_counts_add_up_to_number_of_voters(self):
        rng = np.random.default_rng(2)
        preds = {k: rng.integers(0, K, size=100) for k in KEYS[:7]}
        r = V.vote_labels(preds, K, expected_keys=list(preds))
        assert (r.counts.sum(axis=1) == 7).all()
        assert r.labels.dtype == np.int64

    def test_unanimous_models_give_their_label(self):
        y = np.array([0, 1, 2, 3, 1])
        r = V.vote_labels({k: y for k in KEYS}, K)
        assert (r.labels == y).all() and (r.top_votes == len(KEYS)).all() and not r.tied.any()

    def test_probabilities_are_never_used(self):
        # 6 models with 0.51 confidence for BENIGN beat 5 models with 0.99 confidence for A:
        # only labels are given, so there is nothing else to average
        preds = {f"m{i}": [0] for i in range(6)} | {f"n{i}": [1] for i in range(5)}
        assert V.vote_labels(preds, K, expected_keys=list(preds)).labels.tolist() == [0]


# ---------------------------------------------------------------------------
class TestMissingModels:
    def test_all_present_reports_nothing_missing(self):
        preds = {k: [0] for k in KEYS}
        assert V.vote_labels(preds, K).missing == {}

    def test_missing_models_are_reported_with_default_reason(self):
        preds = {k: [0] for k in KEYS[:9]}
        r = V.vote_labels(preds, K)
        assert set(r.missing) == set(KEYS[9:])
        assert all(reason == "no predictions supplied" for reason in r.missing.values())
        assert r.n_voters == 9

    def test_given_reason_is_kept(self):
        preds = {k: [0] for k in KEYS[1:]}
        r = V.vote_labels(preds, K, missing={KEYS[0]: "not trained"})
        assert r.missing == {KEYS[0]: "not trained"}

    def test_require_all_refuses(self):
        with pytest.raises(V.VotingError, match="missing from the vote"):
            V.vote_labels({KEYS[0]: [0]}, K, require_all=True)

    def test_require_all_passes_when_complete(self):
        assert V.vote_labels({k: [0] for k in KEYS}, K, require_all=True).n_voters == 11

    def test_custom_expected_keys(self):
        r = V.vote_labels({"a": [0]}, K, expected_keys=["a", "b"])
        assert list(r.missing) == ["b"]

    def test_summary_is_json_safe(self):
        import json

        r = V.vote_labels({k: [0, 1] for k in KEYS[:5]}, K)
        s = r.summary()
        json.dumps(s)
        assert s["n_voters"] == 5 and s["unanimous_fraction"] == 1.0 and set(s["missing"]) == set(KEYS[5:])


# ---------------------------------------------------------------------------
class TestInputChecks:
    def test_no_predictions(self):
        with pytest.raises(V.VotingError, match="nothing to vote on"):
            V.vote_labels({}, K)

    def test_float_labels_refused(self):
        with pytest.raises(V.VotingError, match="integer class codes"):
            V.vote_labels({"a": np.array([0.0, 1.0])}, K)

    def test_out_of_range_refused(self):
        with pytest.raises(V.VotingError, match="outside"):
            V.vote_labels({"a": [0, 4]}, K)
        with pytest.raises(V.VotingError, match="outside"):
            V.vote_labels({"a": [-1]}, K)

    def test_different_lengths_refused(self):
        with pytest.raises(V.VotingError, match="predicted 1 rows"):
            V.vote_labels({"a": [0, 1], "b": [0]}, K)

    def test_two_dimensional_refused(self):
        with pytest.raises(V.VotingError, match="1-D"):
            V.vote_labels({"a": np.zeros((2, 2), dtype=int)}, K)

    def test_unknown_rule_refused(self):
        with pytest.raises(V.VotingError, match="tie-break"):
            V.vote_labels({"a": [0]}, K, tie_break="coin_flip")

    def test_bad_benign_index_and_class_count(self):
        with pytest.raises(V.VotingError, match="benign_index"):
            V.vote_labels({"a": [0]}, K, benign_index=9)
        with pytest.raises(V.VotingError, match="2 classes"):
            V.vote_labels({"a": [0]}, 1)

    def test_zero_rows(self):
        r = V.vote_labels({"a": np.array([], dtype="int64")}, K, expected_keys=["a"])
        assert r.n_rows == 0 and r.summary()["tied_fraction"] == 0.0

    def test_default_rule_comes_from_settings(self):
        assert V.vote_labels({"a": [0]}, K, expected_keys=["a"]).tie_break == settings.TIE_BREAK_RULE


# ---------------------------------------------------------------------------
class TestWithModels:
    X = np.zeros((3, 2))

    def test_vote_models(self):
        models = {"knn": FakeModel([0, 1, 2]), "sgd": FakeModel([0, 1, 3]), "svm": FakeModel([0, 2, 3])}
        r = V.vote_models(models, self.X, CODEC, expected_keys=list(models))
        assert r.labels.tolist() == [0, 1, 3]

    def test_failing_model_is_isolated_and_reported(self):
        models = {"knn": FakeModel([0, 1, 2]), "sgd": BrokenModel()}
        r = V.vote_models(models, self.X, CODEC, expected_keys=["knn", "sgd"])
        assert r.voters == ["knn"] and "RuntimeError: boom" in r.missing["sgd"]
        assert r.labels.tolist() == [0, 1, 2]

    def test_not_loaded_reasons_are_merged(self):
        r = V.vote_models(
            {"knn": FakeModel([0, 1, 2])}, self.X, CODEC, expected_keys=["knn", "svm"], not_loaded={"svm": "file missing"}
        )
        assert r.missing == {"svm": "file missing"}

    def test_every_model_failing_raises(self):
        with pytest.raises(V.VotingError):
            V.vote_models({"a": BrokenModel()}, self.X, CODEC, expected_keys=["a"])

    def test_collect_predictions_can_stop_on_error(self):
        with pytest.raises(RuntimeError):
            V.collect_predictions({"a": BrokenModel()}, self.X, continue_on_error=False)


# ---------------------------------------------------------------------------
class TestExplain:
    def setup_method(self):
        self.preds = {"knn": np.array([0, 3]), "sgd": np.array([2, 3]), "svm": np.array([0, 1])}
        self.result = V.vote_labels(self.preds, K, expected_keys=list(self.preds))

    def test_voters_and_tally(self):
        voters, tally = V.explain_vote(self.preds, self.result, CODEC, row=0)
        assert isinstance(voters, pd.DataFrame) and voters["vote"].tolist() == ["BENIGN", "B", "BENIGN"]
        assert tally.to_dict("records") == [
            {"class": "BENIGN", "votes": 2, "winner": True},
            {"class": "B", "votes": 1, "winner": False},
        ]

    def test_display_names_used(self):
        voters, _ = V.explain_vote({"knn": [0], "svm": [0]}, V.vote_labels({"knn": [0], "svm": [0]}, K), CODEC)
        assert voters["model"].tolist() == [model_params.DISPLAY_NAMES["knn"], model_params.DISPLAY_NAMES["svm"]]

    def test_clear_winner_row(self):
        _, tally = V.explain_vote(self.preds, self.result, CODEC, row=1)
        assert tally.loc[tally["winner"], "class"].tolist() == ["C"]  # C has 2 votes, A has 1

    def test_tied_row_marks_the_tie_break_winner(self):
        preds = {"a": np.array([0]), "b": np.array([2])}
        result = V.vote_labels(preds, K, expected_keys=list(preds))
        _, tally = V.explain_vote(preds, result, CODEC, row=0)
        assert tally.loc[tally["winner"], "class"].tolist() == ["B"]  # tie BENIGN/B: attack wins
        assert tally["votes"].tolist() == [1, 1]

    def test_row_out_of_range(self):
        with pytest.raises(V.VotingError, match="outside"):
            V.explain_vote(self.preds, self.result, CODEC, row=5)