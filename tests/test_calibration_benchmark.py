"""Offline tests for the corroboration-ladder calibration benchmark (agent_eval).

The benchmark validates the HIGH-confidence regime the MPRA strata can't reach: it
*measures* evidence completeness (five channels) from each real hematopoietic variant's
bundle and checks the agent's confidence tracks it. These tests pin the measurement and
the two calibration checks with fakes — no network, no model.
"""

from __future__ import annotations

import types

from reglens.genome import Variant
from reglens.validation import agent_eval as ae


def _bundle(*, delta=0.8, motif_delta=-6.0, motif_p=0.01, eqtls=1, assocs=1, lit=1):
    """A minimal fake evidence bundle exposing just what evidence_limbs reads."""
    top = None
    if motif_delta is not None:
        top = types.SimpleNamespace(delta_score=motif_delta, p_value=motif_p)
    return types.SimpleNamespace(
        variant=Variant("chr1", 1, "A", "G"),
        chrombpnet=None if delta is None else types.SimpleNamespace(delta_log_counts=delta),
        motif=None if top is None else types.SimpleNamespace(top=top),
        gene=types.SimpleNamespace(eqtls=list(range(eqtls))),
        trait=types.SimpleNamespace(associations=list(range(assocs))),
        literature=types.SimpleNamespace(hit_count=lit),
    )


class TestEvidenceLimbs:
    def test_all_five_present(self):
        limbs = ae.evidence_limbs(_bundle())
        assert limbs == {k: True for k in ae.EVIDENCE_CHANNELS}

    def test_weak_delta_drops_chrombpnet(self):
        limbs = ae.evidence_limbs(_bundle(delta=0.05), min_delta=0.30)
        assert limbs["chrombpnet"] is False
        assert limbs["motif"] is True  # other limbs unaffected

    def test_insignificant_motif_drops_motif(self):
        limbs = ae.evidence_limbs(_bundle(motif_p=0.5), sig_alpha=0.05)
        assert limbs["motif"] is False

    def test_missing_pvalue_counts_as_present(self):
        # A library with no empirical p-value: a top hit alone is the motif limb.
        limbs = ae.evidence_limbs(_bundle(motif_p=None))
        assert limbs["motif"] is True

    def test_empty_channels_are_absent(self):
        limbs = ae.evidence_limbs(_bundle(eqtls=0, assocs=0, lit=0))
        assert limbs["eqtl"] is False
        assert limbs["gwas"] is False
        assert limbs["literature"] is False
        assert limbs["chrombpnet"] is True and limbs["motif"] is True

    def test_no_motif_no_chrombpnet(self):
        limbs = ae.evidence_limbs(_bundle(delta=None, motif_delta=None))
        assert limbs["chrombpnet"] is False and limbs["motif"] is False

    def test_concordance_sign(self):
        assert ae._concordant(_bundle(delta=-0.8, motif_delta=-6.0)) is True   # same sign
        assert ae._concordant(_bundle(delta=-0.8, motif_delta=+6.0)) is False  # opposite
        assert ae._concordant(_bundle(delta=None)) is False                    # no signal


class _Interp:
    def __init__(self, confidence):
        self.confidence = confidence


class _Agent:
    """Fake single-agent interpreter: confidence keyed by the bundle's completeness."""

    def __init__(self, by_completeness):
        self._map = by_completeness

    def interpret(self, bundle):
        n = sum(ae.evidence_limbs(bundle).values())
        return _Interp(self._map[n])


class TestRunBenchmark:
    def test_end_to_end_with_fakes(self, monkeypatch):
        variants = [
            ae.CalibrationVariant("rsFULL", "G1"),   # 5/5 -> high
            ae.CalibrationVariant("rsMID", "G2"),    # 3/5 -> medium
            ae.CalibrationVariant("rsLOW", "G3"),    # 1/5 -> low
        ]
        bundles = {
            "rsFULL": _bundle(),                                   # 5 limbs
            "rsMID": _bundle(delta=0.05, lit=0),                   # motif+eqtl+gwas = 3
            "rsLOW": _bundle(delta=0.05, motif_p=0.9, eqtls=0, assocs=0),  # lit only = 1
        }
        monkeypatch.setattr(ae, "resolve_variant",
                            lambda rsid, *a, **k: Variant("chr1", 1, "A", "G"))
        monkeypatch.setattr(ae, "analyze_variant",
                            lambda v, rsid=None, **k: bundles[rsid])
        agent = _Agent({5: "high", 3: "medium", 1: "low"})
        outcomes = ae.run_calibration_benchmark(agent, variants, scorer=object(),
                                                genome_path="x")
        by_rsid = {o.rsid: o for o in outcomes}
        assert by_rsid["rsFULL"].completeness == 5 and by_rsid["rsFULL"].confidence == "high"
        assert by_rsid["rsMID"].completeness == 3 and by_rsid["rsMID"].confidence == "medium"
        assert by_rsid["rsLOW"].completeness == 1 and by_rsid["rsLOW"].confidence == "low"

        summary = ae.calibration_benchmark_summary(outcomes)
        assert summary["monotone"] is True           # 5 >= 3 >= 1
        assert summary["high_at_full_only"] is True   # the only high is the 5/5
        assert summary["max_completeness"] == 5
        rendered = ae.render_calibration_benchmark(outcomes)
        assert "monotone (high ≥ medium ≥ low): ✓" in rendered
        assert "high only at full corroboration" in rendered


class TestSummaryChecks:
    def _outcome(self, completeness, confidence, rsid="rs"):
        return ae.CalibrationOutcome(
            rsid=rsid, gene="G", variant=Variant("chr1", 1, "A", "G"),
            limbs={k: False for k in ae.EVIDENCE_CHANNELS},
            completeness=completeness, concordant=False,
            confidence=confidence, interpretation=None,
        )

    def test_non_monotone_flagged(self):
        # A high call on a *thin* bundle (2/5) below a medium at 4/5 breaks monotonicity.
        outs = [self._outcome(2, "high"), self._outcome(4, "medium"), self._outcome(1, "low")]
        s = ae.calibration_benchmark_summary(outs)
        assert s["monotone"] is False

    def test_high_overcall_flagged(self):
        # max completeness is 5, but a high call sits at 3 -> over-call.
        outs = [self._outcome(5, "medium"), self._outcome(3, "high")]
        s = ae.calibration_benchmark_summary(outs)
        assert s["high_at_full_only"] is False

    def test_no_high_is_na(self):
        outs = [self._outcome(4, "medium"), self._outcome(1, "low")]
        s = ae.calibration_benchmark_summary(outs)
        assert s["high_at_full_only"] is None
        assert "n/a (no high calls)" in ae.render_calibration_benchmark(outs)


def test_curated_set_well_formed():
    assert len(ae.HEMATOPOIETIC_CALIBRATION) >= 10
    for cv in ae.HEMATOPOIETIC_CALIBRATION:
        assert cv.rsid.startswith("rs")
        assert cv.celltype == "K562"   # in-lineage for these blood loci
