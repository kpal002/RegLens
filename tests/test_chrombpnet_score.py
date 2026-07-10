"""Unit tests for the ChromBPNet scoring interface, using an offline stub model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reglens.genome import Variant, build_sequence_windows
from reglens.tools.chrombpnet_score import (
    ChromBPNetScorer,
    StubBackend,
    VariantScore,
    aggregate_predictions,
    discover_fold_models,
    jensen_shannon_distance,
    load_backend,
    one_hot_encode,
    reverse_complement_onehot,
)


class TestOneHotEncode:
    def test_encodes_acgt(self):
        oh = one_hot_encode("ACGT")
        assert oh.shape == (4, 4)
        assert np.array_equal(oh, np.eye(4, dtype=np.float32))

    def test_n_is_all_zero(self):
        oh = one_hot_encode("N")
        assert oh.sum() == 0.0

    def test_case_insensitive(self):
        assert np.array_equal(one_hot_encode("acgt"), one_hot_encode("ACGT"))


class TestStubBackend:
    def test_deterministic(self):
        b1, b2 = StubBackend(seed=0), StubBackend(seed=0)
        batch = np.stack([one_hot_encode("ACGT" * 10)])
        assert np.allclose(b1.predict(batch).log_counts, b2.predict(batch).log_counts)

    def test_single_base_change_moves_counts_and_profile(self):
        backend = StubBackend(seed=0)
        ref = one_hot_encode("ACGT" * 10)
        alt = ref.copy()
        alt[0] = one_hot_encode("T")[0]  # flip the first base A->T
        pred = backend.predict(np.stack([ref, alt]))
        # Both heads must respond to a single-base change.
        assert pred.log_counts[0] != pred.log_counts[1]
        assert pred.profile_logits is not None
        assert not np.array_equal(pred.profile_logits[0], pred.profile_logits[1])

    def test_profile_head_shape(self):
        pred = StubBackend(seed=0, profile_length=1000).predict(
            np.stack([one_hot_encode("ACGT" * 10)])
        )
        assert pred.profile_logits.shape == (1, 1000)

    def test_satisfies_backend_protocol(self):
        from reglens.tools.chrombpnet_score import ModelBackend

        assert isinstance(StubBackend(), ModelBackend)


class TestJensenShannonDistance:
    def test_identical_profiles_zero(self):
        logits = np.array([0.1, 2.0, -1.0, 0.5])
        assert jensen_shannon_distance(logits, logits) == pytest.approx(0.0, abs=1e-9)

    def test_bounded_in_unit_interval(self):
        a = np.array([10.0, -10.0, -10.0, -10.0])
        b = np.array([-10.0, -10.0, -10.0, 10.0])
        d = jensen_shannon_distance(a, b)
        assert 0.0 <= d <= 1.0
        assert d > 0.5  # disjoint distributions → large distance


class TestScorer:
    def _scorer(self) -> ChromBPNetScorer:
        return ChromBPNetScorer(load_backend(None), window_length=40, model_name="stub")

    def test_score_variant_end_to_end(self, test_genome: Path, known_locus: dict):
        variant = Variant(
            chrom=known_locus["chrom"],
            pos=known_locus["pos"],
            ref=known_locus["ref"],
            alt=known_locus["alt"],
        )
        result = self._scorer().score_variant(
            variant, genome_path=test_genome, celltype="test-cell"
        )
        assert isinstance(result, VariantScore)
        # Δ is exactly alt - ref, and effect_size is its magnitude.
        assert result.delta_log_counts == pytest.approx(
            result.alt_log_counts - result.ref_log_counts
        )
        assert result.effect_size == pytest.approx(abs(result.delta_log_counts))
        assert result.direction in {"increase", "decrease", "neutral"}
        assert result.celltype == "test-cell"
        # The stub exposes a profile head, so a footprint-shape JSD is reported.
        assert result.profile_jsd is not None
        assert 0.0 <= result.profile_jsd <= 1.0

    def test_direction_matches_sign(self, test_genome: Path, known_locus: dict):
        variant = Variant(
            chrom=known_locus["chrom"],
            pos=known_locus["pos"],
            ref=known_locus["ref"],
            alt=known_locus["alt"],
        )
        result = self._scorer().score_variant(variant, genome_path=test_genome)
        if result.delta_log_counts > 1e-3:
            assert result.direction == "increase"
        elif result.delta_log_counts < -1e-3:
            assert result.direction == "decrease"
        else:
            assert result.direction == "neutral"

    def test_score_window_reproducible(self, test_genome: Path, known_locus: dict):
        variant = Variant(
            chrom=known_locus["chrom"],
            pos=known_locus["pos"],
            ref=known_locus["ref"],
            alt=known_locus["alt"],
        )
        window = build_sequence_windows(variant, genome_path=test_genome, window_length=40)
        s1 = self._scorer().score_window(window, variant)
        s2 = self._scorer().score_window(window, variant)
        assert s1.delta_log_counts == s2.delta_log_counts


class TestLoadBackend:
    def test_returns_stub_without_path(self):
        assert isinstance(load_backend(None), StubBackend)


class TestReverseComplementOnehot:
    def test_matches_revcomp_sequence(self):
        # AAC → revcomp GTT
        rc = reverse_complement_onehot(one_hot_encode("AAC")[None])[0]
        assert np.array_equal(rc, one_hot_encode("GTT"))

    def test_palindrome_unchanged(self):
        # ACGT is its own reverse complement.
        rc = reverse_complement_onehot(one_hot_encode("ACGT")[None])[0]
        assert np.array_equal(rc, one_hot_encode("ACGT"))

    def test_double_rc_is_identity(self):
        oh = one_hot_encode("ACGTACG")[None]
        assert np.array_equal(reverse_complement_onehot(reverse_complement_onehot(oh)), oh)


class TestAggregatePredictions:
    def test_averages_over_folds(self):
        oh = one_hot_encode("ACGT" * 10)[None]
        preds = [
            lambda x: (np.array([2.0]), np.array([[1.0, 3.0]])),
            lambda x: (np.array([4.0]), np.array([[3.0, 1.0]])),
        ]
        out = aggregate_predictions(oh, preds, average_rc=False)
        assert out.log_counts[0] == pytest.approx(3.0)  # (2+4)/2
        assert np.allclose(out.profile_logits[0], [2.0, 2.0])  # ([1,3]+[3,1])/2

    def test_rc_profile_realigned_before_averaging(self):
        oh = one_hot_encode("ACGTAC")[None]
        # Predictor ignores input, always returns the same profile; RC pass must be
        # position-flipped back to forward coords before averaging.
        def pred(x):
            return np.array([5.0]), np.array([[0.0, 1.0, 2.0]])

        out = aggregate_predictions(oh, [pred], average_rc=True)
        assert out.log_counts[0] == pytest.approx(5.0)  # 5 and 5 averaged
        # forward [0,1,2] + realigned RC [2,1,0] → /2 = [1,1,1]
        assert np.allclose(out.profile_logits[0], [1.0, 1.0, 1.0])

    def test_none_profile_propagates(self):
        oh = one_hot_encode("ACGT")[None]
        out = aggregate_predictions(oh, [lambda x: (np.array([1.0]), None)], average_rc=False)
        assert out.profile_logits is None


class TestDiscoverFoldModels:
    def test_finds_nobias_folds_sorted(self, tmp_path):
        for fold in (1, 0):
            d = tmp_path / f"fold_{fold}"
            d.mkdir()
            (d / f"model.chrombpnet_nobias.fold_{fold}.ENCSR868FGK.h5").touch()
            (d / f"model.chrombpnet.fold_{fold}.ENCSR868FGK.h5").touch()  # decoy
            (d / f"model.bias_scaled.fold_{fold}.ENCSR868FGK.h5").touch()  # decoy
        found = discover_fold_models(tmp_path)
        assert len(found) == 2  # only the two nobias models
        assert all("chrombpnet_nobias" in p for p in found)
        assert found == sorted(found)  # fold_0 before fold_1

    def test_empty_when_none(self, tmp_path):
        assert discover_fold_models(tmp_path) == []
