"""Offline tests for the single-agent interpreter (stub + citation guard)."""

from __future__ import annotations

import pytest

from reglens.agents.interpreter import (
    ClaudeInterpreter,
    Interpreter,
    MechanisticInterpretation,
    StubInterpreter,
    _extract_json,
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


# --- Live-path logic, exercised with a fake Anthropic client -----------------

class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Resp:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class FakeAnthropic:
    """Mimics anthropic.Anthropic().messages.create; optionally 400s on output_config."""

    def __init__(self, reply: str, fail_on_output_config: bool = False):
        self.reply = reply
        self.fail_on_output_config = fail_on_output_config
        self.calls: list[dict] = []
        self.messages = self  # so `.messages.create` resolves to self.create

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_output_config and "output_config" in kwargs:
            raise TypeError("unexpected keyword argument 'output_config'")
        return _Resp(self.reply)


_VALID_JSON = (
    '{"mechanism": "alt G restores a GATA1::TAL1 motif", '
    '"direction": "increases_accessibility", "tf": "GATA1::TAL1", "gene": "BCL11A", '
    '"trait": "fetal hemoglobin", "celltype": "", "confidence": "medium", '
    '"caveats": ["hypothesis"], "citations": ["26375006", "99999999"]}'
)


class TestExtractJson:
    def test_plain(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_preamble(self):
        assert _extract_json('Here you go:\n{"a": 1}\nDone') == {"a": 1}

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")


class TestClaudeInterpreterLogic:
    def test_structured_path(self):
        client = FakeAnthropic(_VALID_JSON)
        interp = ClaudeInterpreter(client=client).interpret(_rich_bundle())
        # Used schema-constrained output on the first (only) call.
        assert "output_config" in client.calls[0]
        assert client.calls[0]["thinking"] == {"type": "adaptive"}
        assert interp.tf == "GATA1::TAL1"
        assert interp.celltype == "K562"  # "" cleaned, falls back to bundle
        # Invented PMID stripped by the guard even on the live path.
        assert interp.citations == ["26375006"]

    def test_falls_back_to_prompted_json(self):
        client = FakeAnthropic(_VALID_JSON, fail_on_output_config=True)
        ci = ClaudeInterpreter(client=client)
        interp = ci.interpret(_rich_bundle())
        # First call tried output_config (raised), second omitted it (prompted).
        assert "output_config" in client.calls[0]
        assert "output_config" not in client.calls[1]
        assert ci.use_structured is False  # downgrade remembered
        assert interp.gene == "BCL11A" and interp.citations == ["26375006"]

    def test_non_format_error_propagates(self):
        class Boom(FakeAnthropic):
            def create(self, **kwargs):
                raise RuntimeError("rate limited")

        with pytest.raises(RuntimeError, match="rate limited"):
            ClaudeInterpreter(client=Boom(_VALID_JSON)).interpret(_rich_bundle())
