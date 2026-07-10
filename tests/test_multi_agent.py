"""Offline tests for the multi-agent reasoning layer (routing fake Anthropic client)."""

from __future__ import annotations

import json

from reglens.agents.interpreter import Interpreter, MechanisticInterpretation
from reglens.agents.multi_agent import MultiAgentInterpreter, SpecialistOpinion
from reglens.genome import Variant
from reglens.report.schema import EvidenceBundle
from reglens.tools.literature import Citation, LiteratureResult


def _bundle() -> EvidenceBundle:
    lit = LiteratureResult(
        query="rs1427407 AND BCL11A",
        hit_count=134,
        citations=[Citation("26375006", "MED", "26375006", None,
                            "BCL11A enhancer dissection", "Canver MC", "Nature", "2015")],
    )
    return EvidenceBundle(
        variant=Variant("chr2", 60490908, "T", "G"), rsid="rs1427407",
        celltype="K562", literature=lit,
    )


_SPECIALIST_JSON = json.dumps({
    "assessment": "The variant creates a GATA1::TAL1 motif and modestly raises accessibility.",
    "key_signals": ["motif Δ=+5.45", "ChromBPNet Δ=+0.0185"],
    "confidence": "medium",
    "concerns": ["small accessibility effect"],
})
_REDTEAM_JSON = json.dumps({
    "overall_risk": "medium",
    "challenges": [
        {"claim": "BCL11A is the target", "concern": "no BCL11A eQTL", "severity": "medium"},
        {"claim": "causal for HbF", "concern": "possible LD hitchhiker", "severity": "low"},
    ],
})
# Adjudicator returns one valid + one invented PMID to exercise the citation guard.
_ADJUDICATOR_JSON = json.dumps({
    "mechanism": "Alt G creates a GATA1::TAL1 motif in the BCL11A enhancer.",
    "direction": "increases_accessibility", "tf": "GATA1::TAL1", "gene": "BCL11A",
    "trait": "fetal hemoglobin", "celltype": "", "confidence": "medium",
    "caveats": ["small effect", "no BCL11A eQTL"], "citations": ["26375006", "00000000"],
})


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Resp:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class RoutingFakeAnthropic:
    """Routes messages.create by the agent marker in the system prompt."""

    def __init__(self):
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["system"]
        if "ADJUDICATOR" in system:
            return _Resp(_ADJUDICATOR_JSON)
        if "RED-TEAM" in system:
            return _Resp(_REDTEAM_JSON)
        return _Resp(_SPECIALIST_JSON)  # any of the four specialists


class TestMultiAgentInterpreter:
    def test_is_an_interpreter(self):
        assert isinstance(MultiAgentInterpreter(client=RoutingFakeAnthropic()), Interpreter)

    def test_deliberate_full_transcript(self):
        result = MultiAgentInterpreter(client=RoutingFakeAnthropic()).deliberate(_bundle())
        # Four specialists, in canonical order.
        assert [o.agent for o in result.opinions] == [
            "regulatory-effect", "celltype-context", "gene-target", "trait-link"
        ]
        assert all(isinstance(o, SpecialistOpinion) for o in result.opinions)
        # Red-team produced challenges.
        assert result.critique.overall_risk == "medium"
        assert len(result.critique.challenges) == 2
        # Final adjudicated interpretation.
        assert isinstance(result.interpretation, MechanisticInterpretation)
        assert result.interpretation.tf == "GATA1::TAL1"
        assert result.interpretation.model == "multi-agent/claude-opus-4-8"

    def test_citation_guard_applies_to_adjudicator(self):
        result = MultiAgentInterpreter(client=RoutingFakeAnthropic()).deliberate(_bundle())
        # Invented PMID stripped; only the bundle's real PMID survives.
        assert result.interpretation.citations == ["26375006"]

    def test_celltype_falls_back_to_bundle(self):
        interp = MultiAgentInterpreter(client=RoutingFakeAnthropic()).interpret(_bundle())
        assert interp.celltype == "K562"  # "" from model, filled from bundle

    def test_makes_six_calls(self):
        client = RoutingFakeAnthropic()
        MultiAgentInterpreter(client=client).deliberate(_bundle())
        # 4 specialists + 1 red-team + 1 adjudicator.
        assert len(client.calls) == 6
        assert sum("ADJUDICATOR" in c["system"] for c in client.calls) == 1
        assert sum("RED-TEAM" in c["system"] for c in client.calls) == 1

    def test_redteam_can_be_disabled(self):
        client = RoutingFakeAnthropic()
        result = MultiAgentInterpreter(client=client, redteam=False).deliberate(_bundle())
        assert result.critique.overall_risk == "n/a"
        assert result.critique.challenges == []
        assert not any("RED-TEAM" in c["system"] for c in client.calls)
        assert len(client.calls) == 5  # 4 specialists + adjudicator

    def test_to_dict_is_json_able(self):
        result = MultiAgentInterpreter(client=RoutingFakeAnthropic()).deliberate(_bundle())
        s = json.dumps(result.to_dict())
        assert "opinions" in s and "critique" in s and "interpretation" in s
