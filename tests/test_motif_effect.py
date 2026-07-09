"""Unit tests for the JASPAR-match motif-effect tool (offline, synthetic sequences)."""

from __future__ import annotations

from reglens.genome import SequenceWindow, Variant
from reglens.tools.motif_effect import (
    DEFAULT_MOTIF_DB,
    Motif,
    load_motifs,
    motif_effect,
    reverse_complement,
)


def _consensus(motif: Motif) -> str:
    """Highest-scoring base at each position of a motif."""
    return "".join(max(col, key=col.get) for col in motif.pwm)


def _worst_base(motif: Motif, pos: int) -> str:
    """Lowest-scoring base at a given motif position (max disruption)."""
    return min(motif.pwm[pos], key=motif.pwm[pos].get)


def _by_name(motifs: list[Motif], name: str) -> Motif:
    return next(m for m in motifs if m.tf_name == name)


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAC") == "GTTT"

    def test_preserves_n(self):
        assert reverse_complement("ACN") == "NGT"


class TestLoadMotifs:
    def test_parses_bundled_db(self):
        motifs = load_motifs()
        names = {m.tf_name for m in motifs}
        assert {"GATA1", "CTCF", "GATA1::TAL1"} <= names

    def test_widths(self):
        by = {m.tf_name: m.width for m in load_motifs()}
        assert by["GATA1"] == 11
        assert by["CTCF"] == 19
        assert by["GATA1::TAL1"] == 18

    def test_default_db_path_exists(self):
        assert DEFAULT_MOTIF_DB.exists()

    def test_consensus_scores_above_random(self):
        ctcf = _by_name(load_motifs(), "CTCF")
        cons = _consensus(ctcf)
        assert ctcf.score(cons) > ctcf.score("A" * ctcf.width)


def _window_with_motif(
    motif: Motif, upstream: str, downstream: str, mutate_pos: int, alt_base: str
) -> tuple[SequenceWindow, int]:
    """Embed a motif consensus in flanks and mutate one motif position.

    Returns the (ref/alt) window and the absolute variant offset.
    """
    cons = _consensus(motif)
    ref_motif = cons
    var_offset = len(upstream) + mutate_pos
    ref_seq = upstream + ref_motif + downstream
    alt_seq = ref_seq[:var_offset] + alt_base + ref_seq[var_offset + 1 :]
    window = SequenceWindow(
        chrom="chr_test",
        start=0,
        end=len(ref_seq),
        variant_offset=var_offset,
        ref_seq=ref_seq,
        alt_seq=alt_seq,
    )
    return window, var_offset


class TestMotifEffect:
    def test_finds_and_disrupts_embedded_ctcf(self):
        motifs = load_motifs()
        ctcf = _by_name(motifs, "CTCF")
        # Mutate a high-information central position to its worst base.
        pos = ctcf.width // 2
        alt = _worst_base(ctcf, pos)
        flank = "GTACGTACGTACGTACGTACGTACGTAC"  # neutral-ish, no strong sites
        window, _ = _window_with_motif(ctcf, flank, flank, pos, alt)

        result = motif_effect(window, Variant("chr_test", 100, "N", alt), motifs=motifs)
        assert result.top is not None
        assert result.top.tf_name == "CTCF"
        assert result.top.effect == "disrupted"
        assert result.top.delta_score < 0

    def test_created_when_alt_builds_site(self):
        # Reverse the disruption: start from a disrupted motif as "ref", and let the
        # alt restore the consensus base -> the alt *creates* the site.
        motifs = load_motifs()
        ctcf = _by_name(motifs, "CTCF")
        pos = ctcf.width // 2
        cons = _consensus(ctcf)
        good_base = cons[pos]
        bad_base = _worst_base(ctcf, pos)
        flank = "GTACGTACGTACGTACGTACGTACGTAC"
        # ref carries the bad base; alt restores the consensus base.
        ref_motif = cons[:pos] + bad_base + cons[pos + 1 :]
        var_offset = len(flank) + pos
        ref_seq = flank + ref_motif + flank
        alt_seq = ref_seq[:var_offset] + good_base + ref_seq[var_offset + 1 :]
        window = SequenceWindow(
            chrom="chr_test", start=0, end=len(ref_seq), variant_offset=var_offset,
            ref_seq=ref_seq, alt_seq=alt_seq,
        )
        result = motif_effect(window, Variant("chr_test", 100, bad_base, good_base), motifs=motifs)
        assert result.top is not None
        assert result.top.tf_name == "CTCF"
        assert result.top.effect == "created"
        assert result.top.delta_score > 0

    def test_no_hit_in_random_sequence(self):
        motifs = load_motifs()
        seq = "GTACGTAC" * 10  # 80 bp, no strong motif site
        var_offset = 40
        alt = "A" if seq[var_offset] != "A" else "C"
        alt_seq = seq[:var_offset] + alt + seq[var_offset + 1 :]
        window = SequenceWindow(
            chrom="chr_test", start=0, end=len(seq), variant_offset=var_offset,
            ref_seq=seq, alt_seq=alt_seq,
        )
        # A high binding bar should reject weak spurious hits.
        result = motif_effect(window, Variant("chr_test", 40, seq[var_offset], alt),
                              motifs=motifs, min_binding_bits=8.0)
        assert result.top is None

    def test_hits_ranked_by_disruption(self):
        motifs = load_motifs()
        ctcf = _by_name(motifs, "CTCF")
        pos = ctcf.width // 2
        alt = _worst_base(ctcf, pos)
        flank = "GTACGTACGTACGTACGTACGTACGTAC"
        window, _ = _window_with_motif(ctcf, flank, flank, pos, alt)
        result = motif_effect(window, Variant("chr_test", 100, "N", alt), motifs=motifs)
        deltas = [abs(h.delta_score) for h in result.hits]
        assert deltas == sorted(deltas, reverse=True)
