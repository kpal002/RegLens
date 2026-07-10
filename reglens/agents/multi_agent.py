"""Multi-agent reasoning: specialists → red-team → adjudicator (spec §3).

Refactors the single-agent interpreter into a deliberation over the same evidence
bundle:

1. Four **specialists** each assess one facet — regulatory effect, cell-type
   context, gene target, trait link — running concurrently (fan-out).
2. An optional **red-team** challenges the emerging story (model artifact? LD
   hitchhiker? cell-type mismatch? eQTL absence?).
3. An **adjudicator** synthesizes the final cited :class:`MechanisticInterpretation`,
   weighing the specialists and folding the red-team's concerns into the caveats and
   the calibrated confidence.

Every call is a schema-constrained Claude call via the shared
:class:`~reglens.agents._llm.StructuredCaller` (with the prompted-JSON fallback), and
the same citation-validation guard runs on the adjudicator's output — so the golden
rules (cite only retrieved PMIDs, never invent a number, hedge) hold here too.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from reglens.agents._llm import StructuredCaller
from reglens.agents.interpreter import (
    _OUTPUT_SCHEMA,
    _PROMPTED_JSON_SUFFIX,
    DEFAULT_MODEL,
    MechanisticInterpretation,
    _from_payload,
)
from reglens.report.schema import EvidenceBundle


@dataclass(frozen=True)
class Specialist:
    """A specialist agent: a name, a facet label, and its system prompt."""

    name: str
    focus: str
    system: str


# The four specialists (spec §3). Each sees the whole bundle but is told to reason
# about its facet; naming the facet in the prompt lets a test fake route by marker.
_RULES = (
    "Reason ONLY over numbers present in the bundle; never invent a value. Be concise. "
    "This is a hypothesis, not proof."
)
SPECIALISTS: tuple[Specialist, ...] = (
    Specialist(
        "regulatory-effect", "regulatory / chromatin effect",
        "You are RegLens's REGULATORY-EFFECT specialist. From the evidence bundle, assess "
        "whether and how the variant changes chromatin accessibility and which TF motif is "
        "disrupted or created (ChromBPNet Δ log-counts + profile JSD, the JASPAR motif effect, "
        "and the ENCODE cCRE context). " + _RULES,
    ),
    Specialist(
        "celltype-context", "cell-type relevance",
        "You are RegLens's CELL-TYPE-CONTEXT specialist. Assess whether the predicted effect is "
        "relevant to the stated cell-type context: does the ChromBPNet model's cell type and the "
        "regulatory annotation support activity in that lineage? Flag cell-type mismatch. "
        + _RULES,
    ),
    Specialist(
        "gene-target", "target gene",
        "You are RegLens's GENE-TARGET specialist. From the nearest/overlapping gene and the GTEx "
        "eQTLs, assess which gene the variant most plausibly regulates, and how strong that link "
        "is (overlap vs eQTL; note if the eQTL is in an irrelevant tissue or absent). " + _RULES,
    ),
    Specialist(
        "trait-link", "trait association",
        "You are RegLens's TRAIT-LINK specialist. From the GWAS associations and the retrieved "
        "literature, assess which trait the variant is associated with and how well supported that "
        "is (p-values, effect direction, and whether PMIDs corroborate it). " + _RULES,
    ),
)

_SPECIALIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "description": "2-3 sentence facet assessment."},
        "key_signals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "concerns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["assessment", "key_signals", "confidence", "concerns"],
    "additionalProperties": False,
}
_SPECIALIST_SUFFIX = (
    '\n\nReturn ONLY a JSON object with keys: "assessment" (string), "key_signals" '
    '(array of strings), "confidence" ("high"/"medium"/"low"), "concerns" (array of strings).'
)

REDTEAM_SYSTEM = (
    "You are RegLens's RED-TEAM. You are given the evidence bundle and the specialists' opinions. "
    "Challenge the emerging mechanistic story hard: is the ChromBPNet effect a model artifact "
    "(small magnitude, single fold)? Is the GWAS signal an LD hitchhiker rather than causal? Is "
    "there a cell-type mismatch? Does the eQTL evidence actually implicate the proposed gene, "
    "or is it absent / in an irrelevant tissue? Only raise concerns grounded in the bundle."
)
_REDTEAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_risk": {"type": "string", "enum": ["high", "medium", "low"]},
        "challenges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim being challenged."},
                    "concern": {"type": "string", "description": "Why it might be wrong."},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["claim", "concern", "severity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall_risk", "challenges"],
    "additionalProperties": False,
}
_REDTEAM_SUFFIX = (
    '\n\nReturn ONLY a JSON object with keys: "overall_risk" ("high"/"medium"/"low") and '
    '"challenges" (array of objects each with "claim", "concern", "severity").'
)

ADJUDICATOR_SYSTEM = (
    "You are RegLens's ADJUDICATOR. You are given the evidence bundle, the four specialists' "
    "opinions, and the red-team's challenges. Synthesize ONE cited mechanistic hypothesis: which "
    "TF motif is disrupted/created, in which regulatory element, plausibly affecting which gene, "
    "linked to which trait, in the cell-type context. Weigh the specialists; explicitly fold the "
    "red-team's valid challenges into your caveats and let them lower your confidence. Rules: "
    "reason only over bundle numbers (never invent one); cite ONLY PMIDs present in the bundle's "
    "literature; frame as a hypothesis with calibrated confidence (high/medium/low) and concrete "
    "caveats. Respond ONLY with the requested JSON object."
)


@dataclass
class SpecialistOpinion:
    """One specialist's assessment of its facet."""

    agent: str
    focus: str
    assessment: str
    key_signals: list[str] = field(default_factory=list)
    confidence: str = "low"
    concerns: list[str] = field(default_factory=list)


@dataclass
class Challenge:
    """A single red-team challenge to the emerging story."""

    claim: str
    concern: str
    severity: str


@dataclass
class RedTeamCritique:
    """The red-team's challenges and overall risk assessment."""

    overall_risk: str
    challenges: list[Challenge] = field(default_factory=list)


@dataclass
class MultiAgentResult:
    """The full deliberation: specialist opinions, red-team critique, final call."""

    opinions: list[SpecialistOpinion]
    critique: RedTeamCritique
    interpretation: MechanisticInterpretation

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole deliberation to a JSON-able dict."""
        return {
            "opinions": [asdict(o) for o in self.opinions],
            "critique": asdict(self.critique),
            "interpretation": self.interpretation.to_dict(),
        }


class MultiAgentInterpreter:
    """Specialists → red-team → adjudicator over one evidence bundle.

    Implements the :class:`~reglens.agents.interpreter.Interpreter` protocol
    (``interpret`` returns the adjudicated :class:`MechanisticInterpretation`), and
    exposes :meth:`deliberate` for the full transcript used by the cited report.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8000,
        client: Any | None = None,
        use_structured: bool = True,
        redteam: bool = True,
        max_workers: int = 4,
    ) -> None:
        """Initialize the multi-agent interpreter.

        Args:
            model: Anthropic model id.
            max_tokens: Per-call output token ceiling.
            client: Anthropic-like client; constructed if ``None``.
            use_structured: Prefer schema-constrained output.
            redteam: Include the red-team stage.
            max_workers: Thread-pool size for the specialist fan-out.
        """
        self.model = model
        self.redteam = redteam
        self.max_workers = max_workers
        # One caller shared across all stages so a structured→prompted downgrade is
        # remembered once for the whole deliberation.
        self._caller = StructuredCaller(
            client=client, model=model, max_tokens=max_tokens, use_structured=use_structured
        )

    def interpret(self, bundle: EvidenceBundle) -> MechanisticInterpretation:
        """See :meth:`Interpreter.interpret` — returns the adjudicated call."""
        return self.deliberate(bundle).interpretation

    def deliberate(self, bundle: EvidenceBundle) -> MultiAgentResult:
        """Run the full specialists → red-team → adjudicator deliberation."""
        user = f"EVIDENCE BUNDLE:\n{json.dumps(bundle.to_dict(), indent=2)}"
        opinions = self._run_specialists(user)
        critique = self._run_redteam(user, opinions) if self.redteam else RedTeamCritique("n/a")
        interpretation = self._adjudicate(bundle, user, opinions, critique)
        return MultiAgentResult(opinions=opinions, critique=critique, interpretation=interpretation)

    def _run_specialists(self, user: str) -> list[SpecialistOpinion]:
        """Fan out the four specialists concurrently, preserving their order."""

        def run(spec: Specialist) -> SpecialistOpinion:
            data = self._caller.call(spec.system, user, _SPECIALIST_SCHEMA, _SPECIALIST_SUFFIX)
            return SpecialistOpinion(
                agent=spec.name,
                focus=spec.focus,
                assessment=str(data.get("assessment", "")),
                key_signals=list(data.get("key_signals", [])),
                confidence=str(data.get("confidence", "low")),
                concerns=list(data.get("concerns", [])),
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # `map` preserves input order regardless of completion order.
            return list(pool.map(run, SPECIALISTS))

    def _run_redteam(self, user: str, opinions: list[SpecialistOpinion]) -> RedTeamCritique:
        """Run the red-team over the bundle and the specialist opinions."""
        content = user + "\n\nSPECIALIST OPINIONS:\n" + json.dumps(
            [asdict(o) for o in opinions], indent=2
        )
        data = self._caller.call(REDTEAM_SYSTEM, content, _REDTEAM_SCHEMA, _REDTEAM_SUFFIX)
        challenges = [
            Challenge(
                claim=str(c.get("claim", "")),
                concern=str(c.get("concern", "")),
                severity=str(c.get("severity", "low")),
            )
            for c in data.get("challenges", [])
        ]
        return RedTeamCritique(
            overall_risk=str(data.get("overall_risk", "low")), challenges=challenges
        )

    def _adjudicate(
        self,
        bundle: EvidenceBundle,
        user: str,
        opinions: list[SpecialistOpinion],
        critique: RedTeamCritique,
    ) -> MechanisticInterpretation:
        """Synthesize the final cited interpretation from opinions + critique."""
        content = (
            user
            + "\n\nSPECIALIST OPINIONS:\n"
            + json.dumps([asdict(o) for o in opinions], indent=2)
            + "\n\nRED-TEAM CRITIQUE:\n"
            + json.dumps(asdict(critique), indent=2)
        )
        data = self._caller.call(
            ADJUDICATOR_SYSTEM, content, _OUTPUT_SCHEMA, _PROMPTED_JSON_SUFFIX
        )
        return _from_payload(data, bundle, model=f"multi-agent/{self.model}")
