"""Single-agent mechanistic interpretation over the deterministic evidence bundle.

This is RegLens's reasoning layer (single-agent milestone). It does **not** compute
any numbers — it reasons over the :class:`~reglens.report.schema.EvidenceBundle`
that the deterministic tools already produced and returns a cited, hedged
mechanistic hypothesis.

Design: the interpreter is a single *structured* Claude call, not a tool-calling
agent — the tools have already run, so the model only reasons over provided data.
It sits behind the :class:`Interpreter` protocol so tests (and the offline demo)
use a deterministic stub, while the real :class:`ClaudeInterpreter` calls the
Anthropic Messages API with a JSON-schema output format. **Golden rules enforced
here:** cite only PMIDs present in the bundle (validated post-hoc), never invent a
number, and frame the output as a hypothesis with confidence + caveats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from reglens.agents._llm import DEFAULT_MODEL, StructuredCaller
from reglens.report.schema import EvidenceBundle

# The system prompt encodes RegLens's golden rules for the reasoning layer.
SYSTEM_PROMPT = """\
You are RegLens's mechanistic-interpretation agent for noncoding regulatory variants.

You are given a JSON EVIDENCE BUNDLE that a deterministic tool layer already computed
for one variant: a ChromBPNet chromatin-accessibility effect (Δ log-counts + a
profile Jensen–Shannon distance), the most affected TF motif (JASPAR PWM match), the
ENCODE regulatory-element (cCRE) context, the nearest/target gene and GTEx eQTLs, GWAS
trait associations, and retrieved literature (with PMIDs).

The motif object, when present, carries a "p_value": this is the EMPIRICAL EXCEEDANCE
of the disrupted/created site against a family-wise binding null — the fraction of
random genomic-background variants whose top motif match (across the whole ~880-motif
JASPAR library) binds at least as strongly. It is the tool layer's own guard against
naming a spurious motif when scanning a large library. A very small p_value (e.g. 0.00
– 0.01) means the site binds far more strongly than chance and the TF call is credible;
a p_value near the gate's threshold means the call is marginal. A present motif object
has ALREADY passed the significance gate — the tool never reports a motif that failed
it — so treat the absence of a motif as "no credible TF disruption found," NOT as a
tool error.

Your job: synthesize these signals into ONE mechanistic hypothesis — which TF motif is
disrupted or created, in which regulatory element, plausibly affecting which gene, and
linked to which trait, in the given cell-type context.

Hard rules:
1. Reason ONLY over numbers present in the bundle. NEVER invent or alter a score,
   p-value, distance, or effect size. Refer to the actual values.
2. Cite ONLY PMIDs that appear in the bundle's literature list. Never fabricate a
   citation. If no literature is present, cite nothing.
3. This is a HYPOTHESIS, not a verified mechanism. State a calibrated confidence
   (high/medium/low) and list concrete caveats (e.g. cell-type mismatch of the model,
   variant not inside a cCRE, GTEx lacking the relevant tissue, small effect size, LD
   confounding, model artifact risk). Let the motif p_value inform confidence: a
   marginal motif p_value is a reason to temper a TF-centric claim and name it as a
   caveat; do not assert a specific TF mechanism with high confidence off a borderline
   site. If no motif object is present, do not name a disrupted TF at all.
4. Be honest about direction: the ChromBPNet Δ is defined as alt minus ref. Explain
   what the sign means for accessibility, and reconcile it with the motif effect.
5. Do not overclaim causality; a nearby gene or an eQTL in an unrelated tissue is
   suggestive, not proof.

Respond ONLY with the requested JSON object."""

# Appended to the system prompt on the prompted-JSON fallback path (used when the
# schema-constrained output format isn't accepted by the model/API).
_PROMPTED_JSON_SUFFIX = """

Return ONLY a single JSON object (no prose, no markdown fences) with exactly these
keys: "mechanism" (string), "direction" (one of "increases_accessibility",
"decreases_accessibility", "unclear"), "tf" (string, "" if none), "gene" (string, ""
if none), "trait" (string, "" if none), "celltype" (string, "" if none), "confidence"
(one of "high", "medium", "low"), "caveats" (array of strings), "citations" (array of
PMID strings drawn only from the bundle)."""

# JSON schema constraining the model's output (structured outputs).
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mechanism": {
            "type": "string",
            "description": "2-5 sentence cited hypothesis referencing the actual numbers.",
        },
        "direction": {
            "type": "string",
            "enum": ["increases_accessibility", "decreases_accessibility", "unclear"],
        },
        # Plain strings (empty = unknown) rather than nullable unions — some
        # json_schema validators reject type arrays; "" is mapped to None on parse.
        "tf": {"type": "string", "description": "Implicated TF; empty string if none."},
        "gene": {"type": "string", "description": "Likely target gene; empty if none."},
        "trait": {"type": "string", "description": "Associated trait; empty if none."},
        "celltype": {"type": "string", "description": "Cell-type context; empty if none."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supporting PMIDs; MUST be a subset of the bundle's PMIDs.",
        },
    },
    "required": [
        "mechanism", "direction", "tf", "gene", "trait", "celltype",
        "confidence", "caveats", "citations",
    ],
    "additionalProperties": False,
}


@dataclass
class MechanisticInterpretation:
    """A cited, hedged mechanistic hypothesis for a variant.

    Attributes:
        mechanism: The prose hypothesis (references the bundle's numbers).
        direction: Predicted accessibility direction.
        tf: Implicated transcription factor, if any.
        gene: Likely target gene, if any.
        trait: Associated trait, if any.
        celltype: Cell-type context, if any.
        confidence: ``"high"`` / ``"medium"`` / ``"low"``.
        caveats: Explicit limitations of the hypothesis.
        citations: Supporting PMIDs (validated to exist in the bundle).
        model: Identifier of the interpreter that produced this.
    """

    mechanism: str
    direction: str
    tf: str | None = None
    gene: str | None = None
    trait: str | None = None
    celltype: str | None = None
    confidence: str = "low"
    caveats: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    model: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-able dict."""
        return asdict(self)

    def format(self) -> str:
        """A readable multi-line rendering of the interpretation."""
        lines = [
            f"  mechanism  : {self.mechanism}",
            f"  direction  : {self.direction}",
            f"  TF/gene    : {self.tf or '—'} / {self.gene or '—'}",
            f"  trait      : {self.trait or '—'}  [context: {self.celltype or '—'}]",
            f"  confidence : {self.confidence}  (model: {self.model})",
        ]
        if self.caveats:
            lines.append("  caveats    :")
            lines.extend(f"    - {c}" for c in self.caveats)
        if self.citations:
            lines.append(f"  citations  : {', '.join('PMID:' + p for p in self.citations)}")
        return "\n".join(lines)


@runtime_checkable
class Interpreter(Protocol):
    """Produces a :class:`MechanisticInterpretation` from an evidence bundle."""

    def interpret(self, bundle: EvidenceBundle) -> MechanisticInterpretation:
        """Reason over ``bundle`` and return a cited mechanistic hypothesis."""
        ...


def _validate_citations(citations: list[str], bundle: EvidenceBundle) -> list[str]:
    """Drop any cited PMID not present in the bundle (enforces golden rule #4).

    Args:
        citations: PMIDs the interpreter claims to cite.
        bundle: The evidence bundle whose literature is the only allowed source.

    Returns:
        The subset of ``citations`` that actually appear in the bundle.
    """
    allowed = set(bundle.literature.pmids()) if bundle.literature is not None else set()
    return [c for c in citations if c in allowed]


class ClaudeInterpreter:
    """Single-agent interpreter backed by the Anthropic Messages API.

    Sends the evidence bundle as JSON to Claude and constrains the reply to
    :data:`_OUTPUT_SCHEMA` (with a prompted-JSON fallback via
    :class:`~reglens.agents._llm.StructuredCaller`). The citation-validation guard
    runs on the parsed result. A client may be injected for testing.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8000,
        client: Any | None = None,
        use_structured: bool = True,
    ) -> None:
        """Initialize the interpreter.

        Args:
            model: Anthropic model id (default :data:`DEFAULT_MODEL`).
            max_tokens: Output token ceiling (covers thinking + JSON).
            client: An Anthropic client to use; if ``None`` one is constructed.
            use_structured: Try schema-constrained output first (falls back if not).
        """
        self.model = model
        self._caller = StructuredCaller(
            client=client, model=model, max_tokens=max_tokens, use_structured=use_structured
        )

    @property
    def use_structured(self) -> bool:
        """Whether the schema-constrained path is still in use (False after downgrade)."""
        return self._caller.use_structured

    def interpret(self, bundle: EvidenceBundle) -> MechanisticInterpretation:
        """See :meth:`Interpreter.interpret` (calls the Anthropic API)."""
        payload = json.dumps(bundle.to_dict(), indent=2)
        data = self._caller.call(
            SYSTEM_PROMPT, f"EVIDENCE BUNDLE:\n{payload}", _OUTPUT_SCHEMA, _PROMPTED_JSON_SUFFIX
        )
        return _from_payload(data, bundle, model=self.model)


class StubInterpreter:
    """Deterministic, offline interpreter that composes a hypothesis from the bundle.

    No LLM: it stitches the deterministic tool summaries into a plain-language
    hypothesis so the end-to-end path (and the CLI demo) runs with no API key. Not a
    substitute for the model's reasoning — clearly labeled as such.
    """

    model = "stub(offline)"

    def interpret(self, bundle: EvidenceBundle) -> MechanisticInterpretation:
        """Build a templated interpretation from the bundle's signals."""
        cb, motif = bundle.chrombpnet, (bundle.motif.top if bundle.motif else None)
        gene = bundle.gene.nearest_gene if bundle.gene else None
        traits = bundle.trait.unique_traits() if bundle.trait else []

        direction = "unclear"
        parts: list[str] = []
        if cb is not None:
            direction = (
                "increases_accessibility" if cb.direction == "increase"
                else "decreases_accessibility" if cb.direction == "decrease"
                else "unclear"
            )
            parts.append(f"ChromBPNet predicts the alt allele {cb.direction}s accessibility "
                         f"(Δ log-counts={cb.delta_log_counts:+.4f}).")
        if motif is not None:
            parts.append(f"The variant {motif.effect} a {motif.tf_name} motif "
                         f"(Δ={motif.delta_score:+.2f} bits).")
        if gene is not None:
            loc = "inside" if gene.overlaps else f"{gene.distance:,} bp from"
            parts.append(f"It lies {loc} {gene.symbol or gene.gene_id}.")
        if traits:
            parts.append(f"GWAS links the variant to {traits[0]}.")

        caveats = ["Automated stub synthesis — not model reasoning; for offline demo only."]
        if bundle.regulatory is not None and not bundle.regulatory.in_ccre:
            caveats.append("Variant is near, not inside, an ENCODE cCRE.")
        if bundle.gene is not None and not bundle.gene.eqtls:
            caveats.append("No significant GTEx eQTL — relevant cell type may be absent from GTEx.")

        return MechanisticInterpretation(
            mechanism=" ".join(parts) or "Insufficient signals to form a hypothesis.",
            direction=direction,
            tf=motif.tf_name if motif else None,
            gene=(gene.symbol or gene.gene_id) if gene else None,
            trait=traits[0] if traits else None,
            celltype=bundle.celltype,
            confidence="low",
            caveats=caveats,
            citations=bundle.literature.pmids()[:3] if bundle.literature else [],
            model=self.model,
        )


def _from_payload(
    data: dict[str, Any], bundle: EvidenceBundle, model: str
) -> MechanisticInterpretation:
    """Build a :class:`MechanisticInterpretation` from a model JSON payload.

    Citations are validated against the bundle so an invented PMID cannot survive.
    """
    def _clean(value: Any) -> str | None:
        """Empty strings / falsy → None (the schema uses "" for 'unknown')."""
        return value or None

    return MechanisticInterpretation(
        mechanism=str(data.get("mechanism", "")),
        direction=str(data.get("direction", "unclear")),
        tf=_clean(data.get("tf")),
        gene=_clean(data.get("gene")),
        trait=_clean(data.get("trait")),
        celltype=_clean(data.get("celltype")) or bundle.celltype,
        confidence=str(data.get("confidence", "low")),
        caveats=list(data.get("caveats", [])),
        citations=_validate_citations(list(data.get("citations", [])), bundle),
        model=model,
    )


def build_interpreter(
    use_claude: bool = True, model: str = DEFAULT_MODEL, multi_agent: bool = False
) -> Interpreter:
    """Return the appropriate interpreter, falling back to the offline stub.

    Args:
        use_claude: If True, try to build a Claude-backed interpreter; on any failure
            (missing SDK/credentials) fall back to the offline stub.
        model: Model id for the Claude interpreter.
        multi_agent: If True (and ``use_claude``), use the specialists → red-team →
            adjudicator :class:`~reglens.agents.multi_agent.MultiAgentInterpreter`.

    Returns:
        An :class:`Interpreter` implementation.
    """
    if use_claude:
        try:
            if multi_agent:
                # Lazy import: multi_agent imports from this module.
                from reglens.agents.multi_agent import MultiAgentInterpreter

                return MultiAgentInterpreter(model=model)
            return ClaudeInterpreter(model=model)
        except Exception:  # noqa: BLE001 - graceful offline fallback
            return StubInterpreter()
    return StubInterpreter()
