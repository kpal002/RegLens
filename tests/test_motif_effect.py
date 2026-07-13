"""Unit tests for the JASPAR-match motif-effect tool (offline, synthetic sequences)."""

from __future__ import annotations

from reglens.genome import SequenceWindow, Variant
from reglens.tools.motif_effect import (
    DEFAULT_MOTIF_DB,
    SUBSET_MOTIF_DB,
    Motif,
    calibrate_binding_null,
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
        motifs = load_motifs(SUBSET_MOTIF_DB)
        names = {m.tf_name for m in motifs}
        assert {"GATA1", "CTCF", "GATA1::TAL1"} <= names

    def test_widths(self):
        by = {m.tf_name: m.width for m in load_motifs(SUBSET_MOTIF_DB)}
        assert by["GATA1"] == 11
        assert by["CTCF"] == 19
        assert by["GATA1::TAL1"] == 18

    def test_default_db_path_exists(self):
        assert DEFAULT_MOTIF_DB.exists()

    def test_consensus_scores_above_random(self):
        ctcf = _by_name(load_motifs(SUBSET_MOTIF_DB), "CTCF")
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
        motifs = load_motifs(SUBSET_MOTIF_DB)
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
        motifs = load_motifs(SUBSET_MOTIF_DB)
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
        motifs = load_motifs(SUBSET_MOTIF_DB)
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

    def test_reconciles_created_to_ref_disrupts(self):
        # When the alt allele CREATES a site, the ref allele is the disrupting one —
        # this is what reconciles our "created" call with literature "disruption".
        motifs = load_motifs(SUBSET_MOTIF_DB)
        ctcf = _by_name(motifs, "CTCF")
        pos = ctcf.width // 2
        cons = _consensus(ctcf)
        bad = _worst_base(ctcf, pos)
        flank = "GTACGTACGTACGTACGTACGTACGTAC"
        ref_motif = cons[:pos] + bad + cons[pos + 1 :]  # ref weak, alt restores
        off = len(flank) + pos
        ref_seq = flank + ref_motif + flank
        alt_seq = ref_seq[:off] + cons[pos] + ref_seq[off + 1 :]
        window = SequenceWindow("chr_test", 0, len(ref_seq), off, ref_seq, alt_seq)
        result = motif_effect(window, Variant("chr_test", 100, bad, cons[pos]), motifs=motifs)
        assert result.top.effect == "created"
        assert result.disrupting_allele == bad  # the reference allele breaks the site
        note = result.reconciling_note()
        assert bad in note and cons[pos] in note
        assert "the " + bad + " allele is the one that breaks the site" in result.summary()

    def test_disrupting_allele_none_when_unchanged(self):
        motifs = load_motifs(SUBSET_MOTIF_DB)
        seq = "GTACGTAC" * 10
        window = SequenceWindow("chr_test", 0, len(seq), 40, seq, seq[:40] + "A" + seq[41:])
        result = motif_effect(window, Variant("chr_test", 40, seq[40], "A"),
                              motifs=motifs, min_binding_bits=8.0)
        assert result.disrupting_allele is None  # no credible hit
        assert result.reconciling_note() is None

    def test_hits_ranked_by_disruption(self):
        motifs = load_motifs(SUBSET_MOTIF_DB)
        ctcf = _by_name(motifs, "CTCF")
        pos = ctcf.width // 2
        alt = _worst_base(ctcf, pos)
        flank = "GTACGTACGTACGTACGTACGTACGTAC"
        window, _ = _window_with_motif(ctcf, flank, flank, pos, alt)
        result = motif_effect(window, Variant("chr_test", 100, "N", alt), motifs=motifs)
        deltas = [abs(h.delta_score) for h in result.hits]
        assert deltas == sorted(deltas, reverse=True)


class TestLargeLibraryFalsePositiveGate:
    """The empirical family-wise gate is what makes the full ~880-motif JASPAR library
    safe: without it, scanning that many short PWMs manufactures a 'mechanism' on
    random sequence. These tests lock in both halves — it stays quiet on noise, and it
    still recovers a real embedded site.
    """

    @staticmethod
    def _random_window(rng, n=81):
        import numpy as np

        bg = np.array([0.295, 0.205, 0.205, 0.295])
        bg /= bg.sum()
        bases = np.array(list("ACGT"))
        seq = "".join(rng.choice(bases, size=n, p=bg))
        vp = n // 2
        alt = rng.choice([b for b in "ACGT" if b != seq[vp]])
        window = SequenceWindow("chr_test", 0, n, vp, seq, seq[:vp] + alt + seq[vp + 1 :])
        return window, Variant("chr_test", vp + 1, seq[vp], alt)

    def test_full_library_stays_quiet_on_random_sequence(self):
        """On random genomic-background sequence the full library must almost always
        return no motif call — the confabulation guard. (Without the gate this is
        100% false positives; the NFIX/HOXA2 spurious hit is the failure mode.)"""
        import numpy as np

        full = load_motifs(DEFAULT_MOTIF_DB)
        rng = np.random.default_rng(7)
        n_trials = 60
        called = 0
        for _ in range(n_trials):
            window, variant = self._random_window(rng)
            if motif_effect(window, variant, motifs=full, null_panel_size=120).top is not None:
                called += 1
        # alpha=0.05 gate; allow generous slack for a small sample.
        assert called <= 0.20 * n_trials, f"{called}/{n_trials} false positives — gate too loose"

    def test_full_library_recovers_embedded_real_site(self):
        """A real CTCF consensus embedded in flanks must survive the gate against the
        full library — the gate must not be so strict it kills true positives."""
        full = load_motifs(DEFAULT_MOTIF_DB)
        ctcf = _by_name(load_motifs(SUBSET_MOTIF_DB), "CTCF")
        cons = _consensus(ctcf)
        pos = ctcf.width // 2
        worst = _worst_base(ctcf, pos)
        flank = "ACGTGACTGACTGATCAGTCAGTC" * 4
        seq = flank + cons + flank
        if len(seq) % 2 == 0:
            seq = seq[:-1]
        voff = len(flank) + pos
        window = SequenceWindow("chr_test", 0, len(seq), voff, seq,
                                seq[:voff] + worst + seq[voff + 1 :])
        result = motif_effect(window, Variant("chr_test", voff + 1, cons[pos], worst),
                              motifs=full, null_panel_size=120)
        assert result.top is not None, "real CTCF site was gated out (false negative)"
        assert any(h.tf_name == "CTCF" for h in result.hits)

    def test_gate_can_be_disabled(self):
        """alpha=1.0 disables the statistical gate (prefilter-only, old behavior)."""
        full = load_motifs(DEFAULT_MOTIF_DB)
        import numpy as np

        rng = np.random.default_rng(1)
        window, variant = self._random_window(rng)
        ungated = motif_effect(window, variant, motifs=full, alpha=1.0)
        gated = motif_effect(window, variant, motifs=full, alpha=0.05, null_panel_size=120)
        # Disabling the gate can only keep >= as many hits.
        assert len(ungated.hits) >= len(gated.hits)


class TestBindingNullCalibration:
    def test_threshold_is_positive_and_cached(self):
        full = load_motifs(DEFAULT_MOTIF_DB)
        thr1, null1 = calibrate_binding_null(full, panel_size=80, seed=0)
        thr2, null2 = calibrate_binding_null(full, panel_size=80, seed=0)
        assert thr1 > 0
        assert thr1 == thr2  # cached, deterministic
        assert len(null1) == 80
