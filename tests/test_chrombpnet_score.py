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
    load_backend,
    one_hot_encode,
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
        assert np.allclose(b1.predict_log_counts(batch), b2.predict_log_counts(batch))

    def test_single_base_change_moves_score(self):
        backend = StubBackend(seed=0)
        ref = one_hot_encode("ACGT" * 10)
        alt = ref.copy()
        alt[0] = one_hot_encode("T")[0]  # flip the first base A->T
        scores = backend.predict_log_counts(np.stack([ref, alt]))
        assert scores[0] != scores[1]

    def test_satisfies_backend_protocol(self):
        from reglens.tools.chrombpnet_score import ModelBackend

        assert isinstance(StubBackend(), ModelBackend)


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
