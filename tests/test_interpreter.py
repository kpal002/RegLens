"""Offline tests for the single-agent interpreter (stub + citation guard)."""

from __future__ import annotations

from reglens.agents.interpreter import (
    Interpreter,
    MechanisticInterpretation,
    StubInterpreter,
    _from_payload,
    _validate_citations,
    build_interpreter,
)
from reglens.genome import SequenceWindow, Variant
from reglens.report.schema import EvidenceBundle
from reglens.tools.chrombpnet_score import VariantScore
from reglens.tools.gene_target import GeneOverlap, GeneTargetResult
from reglens.tools.literature import Citation, LiteratureResult
from reglens.tools.motif_effect import MotifEffectResult, MotifHit
from reglens.tools.trait_link import TraitAssociation, TraitLinkResult

_VARIANT = Variant("chr2", 60490908, "T", "G")


def _rich_bundle() -> EvidenceBundle:
    window = SequenceWindow("chr2", 0, 4, 2, "TTGT", "TTGT")
    score = VariantScore(_VARIANT, "K562", 6.0, 6.2, 0.2, "increase", 0.02, window, "stub")
    motif = MotifEffectResult(
        _VARIANT,
        hits=[MotifHit("MA0140.2", "GATA1::TAL1", "-", 28, 3.2, 8.66, 5.45, "created")],
        top=MotifHit("MA0140.2", "GATA1::TAL1", "-", 28, 3.2, 8.66, 5.45, "created"),
    )
    gene = GeneTargetResult(
        _VARIANT,
        nearest_gene=GeneOverlap("ENSG1", "BCL11A", "protein_coding", 40, 100, -1, 0),
        overlapping_genes=[GeneOverlap("ENSG1", "BCL11A", "protein_coding", 40, 100, -1, 0)],
    )
    trait = TraitLinkResult(
        "rs1427407",
        associations=[TraitAssociation(["fetal hemoglobin measurement"], 4e-53, beta=0.3)],
    )
    cite = Citation("1", "MED", "26375006", None, "BCL11A enhancer", "Canver", "Nature", "2015")
    lit = LiteratureResult("q", 5, citations=[cite])
    return EvidenceBundle(
        variant=_VARIANT, rsid="rs1427407", celltype="K562",
        chrombpnet=score, motif=motif, gene=gene, trait=trait, literature=lit,
    )


class TestStubInterpreter:
    def test_satisfies_protocol(self):
        assert isinstance(StubInterpreter(), Interpreter)

    def test_composes_from_bundle(self):
        interp = StubInterpreter().interpret(_rich_bundle())
        assert interp.direction == "increases_accessibility"
        assert interp.tf == "GATA1::TAL1"
        assert interp.gene == "BCL11A"
        assert interp.trait == "fetal hemoglobin measurement"
        assert "GATA1::TAL1" in interp.mechanism
        # Only real bundle PMIDs are cited.
        assert interp.citations == ["26375006"]

    def test_handles_empty_bundle(self):
        interp = StubInterpreter().interpret(EvidenceBundle(variant=_VARIANT))
        assert interp.direction == "unclear"
        assert interp.citations == []
        assert interp.mechanism  # non-empty fallback text

    def test_format_smoke(self):
        text = StubInterpreter().interpret(_rich_bundle()).format()
        assert "mechanism" in text and "confidence" in text


class TestCitationGuard:
    def test_drops_invented_pmids(self):
        bundle = _rich_bundle()
        # 26375006 is real; 99999999 is invented and must be dropped.
        kept = _validate_citations(["26375006", "99999999"], bundle)
        assert kept == ["26375006"]

    def test_from_payload_validates_citations(self):
        bundle = _rich_bundle()
        payload = {
            "mechanism": "m", "direction": "increases_accessibility",
            "tf": "GATA1::TAL1", "gene": "BCL11A", "trait": "fetal hemoglobin",
            "celltype": None, "confidence": "medium",
            "caveats": ["c"], "citations": ["26375006", "00000000"],
        }
        interp = _from_payload(payload, bundle, model="claude-opus-4-8")
        assert interp.citations == ["26375006"]  # invented PMID removed
        assert interp.celltype == "K562"  # falls back to bundle celltype
        assert interp.model == "claude-opus-4-8"


class TestBuildInterpreter:
    def test_offline_returns_stub(self):
        assert isinstance(build_interpreter(use_claude=False), StubInterpreter)


def test_interpretation_to_dict_roundtrips():
    interp = MechanisticInterpretation(mechanism="m", direction="unclear", citations=["1"])
    d = interp.to_dict()
    assert d["mechanism"] == "m" and d["citations"] == ["1"]
