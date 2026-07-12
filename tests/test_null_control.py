"""Offline tests for the agent null-control harness (reglens.validation.null_control).

Covers the decline-vs-confabulate rubric, deterministic negative selection, the run
loop (with analyze_variant monkeypatched so no network/genome is touched), and the
summary aggregation.
"""

from __future__ import annotations

from reglens.agents.interpreter import MechanisticInterpretation
from reglens.genome import Variant
from reglens.validation import null_control as nc
from reglens.validation.dataset import LabeledVariant


def _interp(mechanism: str, confidence: str, tf: str | None = None) -> MechanisticInterpretation:
    return MechanisticInterpretation(
        mechanism=mechanism, direction="unclear", tf=tf, confidence=confidence
    )


class TestRubric:
    def test_declines_when_no_mechanism_asserted(self):
        i = _interp("No coherent mechanism; the signals do not converge.", "low")
        assert nc.classify_decline(i) == "declined"

    def test_declines_low_conf_even_if_motif_named(self):
        # Named a motif but explicitly concluded no mechanism, at low confidence.
        i = _interp("A weak GATA1 motif is present but there is no clear mechanism.",
                    "low", tf="GATA1")
        assert nc.classify_decline(i) == "declined"

    def test_confabulates_confident_story(self):
        i = _interp("The alt allele disrupts a GATA1 motif, abolishing the enhancer.",
                    "high", tf="GATA1")
        assert nc.classify_decline(i) == "confabulated"

    def test_medium_confidence_assertion_is_confabulation(self):
        i = _interp("The variant creates a TAL1 binding site that activates the element.",
                    "medium", tf="TAL1")
        assert nc.classify_decline(i) == "confabulated"

    def test_borderline_low_conf_but_asserts(self):
        i = _interp("The alt allele disrupts a GATA1 site (speculative).", "low", tf="GATA1")
        assert nc.classify_decline(i) == "borderline"

    def test_empty_tf_is_decline(self):
        i = _interp("The alt allele disrupts something, unclear what.", "medium", tf="")
        assert nc.classify_decline(i) == "declined"


class TestSelection:
    def _variants(self) -> list[LabeledVariant]:
        out = []
        for elem in ("BCL11A", "HBB", "PKLR-48h"):
            for k in range(10):
                out.append(
                    LabeledVariant(
                        variant=Variant(chrom="chr2", pos=1000 + k, ref="C", alt="T"),
                        label=0, source=elem,
                    )
                )
        # some positives that must never be selected
        for k in range(5):
            out.append(
                LabeledVariant(
                    variant=Variant(chrom="chr2", pos=9000 + k, ref="A", alt="G"),
                    label=1, source="BCL11A",
                )
            )
        return out

    def test_only_negatives_selected(self):
        picked = nc.select_negatives(self._variants(), n=9, seed=0)
        assert len(picked) == 9
        assert all(v.label == 0 for v in picked)

    def test_deterministic(self):
        a = nc.select_negatives(self._variants(), n=6, seed=3)
        b = nc.select_negatives(self._variants(), n=6, seed=3)
        assert [str(v.variant) for v in a] == [str(v.variant) for v in b]

    def test_spread_across_elements(self):
        # Round-robin should touch all three elements within the first 6 picks.
        picked = nc.select_negatives(self._variants(), n=6, seed=0)
        assert len({v.source for v in picked}) == 3

    def test_element_filter(self):
        picked = nc.select_negatives(self._variants(), n=5, seed=0, elements=["HBB"])
        assert all(v.source == "HBB" for v in picked)


class _FakeInterpreter:
    """Returns a canned interpretation per call (no API)."""

    def __init__(self, interps):
        self._interps = list(interps)
        self.calls = 0

    def interpret(self, bundle):
        i = self._interps[self.calls % len(self._interps)]
        self.calls += 1
        return i


class _StubBundle:
    chrombpnet = None  # annotation-only

    def to_dict(self):
        return {}


class TestRunLoop:
    def test_run_and_summarize(self, monkeypatch):
        variants = [
            LabeledVariant(variant=Variant(chrom="chr2", pos=p, ref="C", alt="T"),
                           label=0, source="BCL11A")
            for p in range(1000, 1010)
        ]
        # Two decline, one confabulation — cycled by the fake interpreter.
        interps = [
            _interp("No coherent mechanism.", "low"),
            _interp("Insufficient evidence for a mechanism.", "low"),
            _interp("Disrupts a GATA1 motif, ablating the enhancer.", "high", tf="GATA1"),
        ]
        monkeypatch.setattr(nc, "analyze_variant", lambda *a, **k: _StubBundle())

        outcomes = nc.run_null_control(
            variants, _FakeInterpreter(interps), n=6, seed=1
        )
        assert len(outcomes) == 6
        assert all(o.sequence_signal is False for o in outcomes)

        report = nc.summarize(outcomes)
        assert report.n == 6
        assert sum(report.verdicts.values()) == 6
        assert report.verdicts["confabulated"] >= 1
        assert "declined" in nc.render_summary(outcomes)

    def test_report_flags_missing_sequence_signal(self, monkeypatch):
        variants = [
            LabeledVariant(variant=Variant(chrom="chr2", pos=1000, ref="C", alt="T"),
                           label=0, source="HBB")
        ]
        monkeypatch.setattr(nc, "analyze_variant", lambda *a, **k: _StubBundle())
        outcomes = nc.run_null_control(
            variants, _FakeInterpreter([_interp("No mechanism.", "low")]), n=1
        )
        assert nc.summarize(outcomes).sequence_signal is False
