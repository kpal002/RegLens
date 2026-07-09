"""Unit tests for variant parsing and sequence-window construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.genome import (
    GENOME_ENV_VAR,
    Variant,
    build_sequence_windows,
    resolve_genome_path,
)


class TestVariantParse:
    def test_parses_colon_form(self):
        v = Variant.parse("chr7:5530601:C>T")
        assert v == Variant(chrom="chr7", pos=5530601, ref="C", alt="T")

    def test_parses_space_form_and_uppercases(self):
        v = Variant.parse("chr7:5530601 c>t")
        assert v.ref == "C" and v.alt == "T"

    def test_is_snv(self):
        assert Variant.parse("chr1:100:A>G").is_snv
        assert not Variant.parse("chr1:100:AT>G").is_snv

    def test_str_roundtrip(self):
        assert str(Variant.parse("chr1:100:A>G")) == "chr1:100:A>G"

    @pytest.mark.parametrize("bad", ["nonsense", "chr1:100", "chr1:100:A", "chr1:x:A>G"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            Variant.parse(bad)


class TestResolveGenomePath:
    def test_uses_explicit_path(self, test_genome: Path):
        assert resolve_genome_path(test_genome) == str(test_genome)

    def test_falls_back_to_env(self, test_genome: Path, monkeypatch):
        monkeypatch.setenv(GENOME_ENV_VAR, str(test_genome))
        assert resolve_genome_path(None) == str(test_genome)

    def test_errors_without_path_or_env(self, monkeypatch):
        monkeypatch.delenv(GENOME_ENV_VAR, raising=False)
        with pytest.raises(ValueError):
            resolve_genome_path(None)

    def test_errors_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            resolve_genome_path("/no/such/genome.fa")


class TestBuildSequenceWindows:
    def test_known_locus_ref_and_alt(self, test_genome: Path, known_locus: dict):
        variant = Variant(
            chrom=known_locus["chrom"],
            pos=known_locus["pos"],
            ref=known_locus["ref"],
            alt=known_locus["alt"],
        )
        window = build_sequence_windows(variant, genome_path=test_genome, window_length=40)

        # Window is the requested length, variant centered at window_length // 2.
        assert window.length == 40
        assert window.variant_offset == 20

        # Ref window carries the declared reference base at the variant offset.
        assert window.ref_seq[window.variant_offset] == known_locus["ref"]
        # Alt window differs by exactly one base — the substituted allele.
        assert window.alt_seq[window.variant_offset] == known_locus["alt"]
        differing = [
            i
            for i, (a, b) in enumerate(zip(window.ref_seq, window.alt_seq, strict=True))
            if a != b
        ]
        assert differing == [window.variant_offset]

    def test_reference_mismatch_raises(self, test_genome: Path, known_locus: dict):
        # Declare the wrong reference base at the known position.
        wrong = Variant(chrom=known_locus["chrom"], pos=known_locus["pos"], ref="A", alt="G")
        with pytest.raises(ValueError, match="Reference mismatch"):
            build_sequence_windows(wrong, genome_path=test_genome, window_length=40)

    def test_window_off_contig_raises(self, test_genome: Path, known_locus: dict):
        variant = Variant(chrom=known_locus["chrom"], pos=5, ref="C", alt="T")
        with pytest.raises(ValueError, match="runs off contig"):
            build_sequence_windows(variant, genome_path=test_genome, window_length=200)

    def test_unknown_contig_raises(self, test_genome: Path):
        variant = Variant(chrom="chr_missing", pos=50, ref="C", alt="T")
        with pytest.raises(ValueError, match="not found in genome"):
            build_sequence_windows(variant, genome_path=test_genome, window_length=40)

    def test_indel_not_implemented(self, test_genome: Path, known_locus: dict):
        indel = Variant(chrom=known_locus["chrom"], pos=known_locus["pos"], ref="C", alt="CA")
        with pytest.raises(NotImplementedError):
            build_sequence_windows(indel, genome_path=test_genome, window_length=40)
