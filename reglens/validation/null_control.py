"""Agent null control: does the multi-agent decline on non-functional variants?

The single question everyone quietly has about LLM agents in biology: handed a variant
that is genuinely **non-functional** — an MPRA negative sitting *inside* an active
regulatory element — does the reasoning layer correctly report "low confidence, no
coherent mechanism", or does it **confabulate** a plausible-sounding story?

This harness draws negatives from the same matched MPRA benchmark used for the engine
AUROC (``label == 0``: in an active element, no measured regulatory effect), runs the
full multi-agent deliberation on each, and classifies the outcome — *declined*,
*borderline*, or *confabulated* — by a transparent rubric over the structured
interpretation. It reuses the deterministic tool layer
(:func:`reglens.orchestrator.analyze_variant`) and the interpreter unchanged; nothing
here computes or invents a number.

**Honest scoping.** A faithful run needs the sequence-model signals (ChromBPNet Δ +
motif), so pass a ``genome_path`` + ``scorer`` (i.e. run it on Colab with hg38 and the
ChromBPNet model). Without them, the negatives' most decision-relevant evidence is
absent and the test degenerates toward "does it decline when it has no sequence data" —
still informative (does it fabricate a mechanism from annotation context alone?) but not
the complete control. The report records whether the sequence signals were present.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reglens.genome import Variant
from reglens.orchestrator import analyze_variant
from reglens.tools._http import HttpClient
from reglens.tools.chrombpnet_score import ChromBPNetScorer
from reglens.validation.dataset import LabeledVariant, load_labeled_variants

# --------------------------------------------------------------------------------------
# Decline-vs-confabulate rubric.
#
# The verdict turns on one question: did the agent *assert a concrete sequence-level
# mechanism* (a named TF being disrupted/created, with an effect verb), and if so, with
# how much confidence? For a variant that is actually non-functional:
#   - not asserting a mechanism            -> "declined"      (the correct behavior)
#   - asserting one, but at low confidence -> "borderline"    (hedged — partial credit)
#   - asserting one at medium/high conf    -> "confabulated"  (the failure mode)
# An explicit decline statement ("no coherent mechanism", "insufficient", ...) at low
# confidence counts as "declined" even if a motif was named. The rubric is deliberately
# simple and auditable; the report also carries the full prose so a human can spot-check.
# --------------------------------------------------------------------------------------
DECLINE_MARKERS: tuple[str, ...] = (
    "no coherent", "no clear", "no mechanism", "no strong", "no significant",
    "no plausible", "insufficient", "cannot", "unable to", "no evidence",
    "not supported", "no motif", "does not appear", "no accessibility",
    "likely benign", "no functional",
)
ASSERT_VERBS: tuple[str, ...] = (
    "disrupt", "creat", "abolish", "establish", "activat", "represses", "recruit",
    "strengthen", "weaken",
)

VERDICTS = ("declined", "borderline", "confabulated")


def classify_decline(interpretation: Any) -> str:
    """Classify one interpretation of a known-negative variant.

    Args:
        interpretation: A
            :class:`~reglens.agents.interpreter.MechanisticInterpretation` (or any object
            with ``mechanism``, ``confidence``, and ``tf`` attributes).

    Returns:
        One of ``"declined"``, ``"borderline"``, ``"confabulated"`` (see module rubric).
    """
    text = (getattr(interpretation, "mechanism", "") or "").lower()
    confidence = (getattr(interpretation, "confidence", "low") or "low").lower()
    tf = getattr(interpretation, "tf", None)

    declines = any(marker in text for marker in DECLINE_MARKERS)
    if declines and confidence == "low":
        return "declined"

    asserts_mechanism = bool(tf) and str(tf).lower() not in {"none", "n/a", ""} and any(
        verb in text for verb in ASSERT_VERBS
    )
    if not asserts_mechanism:
        return "declined"
    return "borderline" if confidence == "low" else "confabulated"


# --------------------------------------------------------------------------------------
# Selection + run
# --------------------------------------------------------------------------------------
def select_negatives(
    variants: list[LabeledVariant],
    n: int,
    seed: int = 0,
    elements: list[str] | None = None,
) -> list[LabeledVariant]:
    """Deterministically pick ``n`` negatives, spread round-robin across elements.

    Args:
        variants: Labeled variants (positives are ignored).
        n: Number of negatives to select.
        seed: RNG seed for reproducibility.
        elements: If given, restrict to negatives whose ``source`` is in this list
            (e.g. the hematopoietic elements where the K562 model has real signal — the
            hardest place for the agent to *avoid* confabulating).

    Returns:
        Up to ``n`` negative :class:`LabeledVariant` s, spread across their elements.
    """
    allowed = set(elements) if elements else None
    negatives = [
        v for v in variants
        if v.label == 0 and (allowed is None or v.source in allowed)
    ]
    by_element: dict[str | None, list[LabeledVariant]] = {}
    for v in negatives:
        by_element.setdefault(v.source, []).append(v)

    rng = random.Random(seed)
    for group in by_element.values():
        rng.shuffle(group)
    order = sorted(by_element, key=lambda e: (e is None, e))  # deterministic
    rng.shuffle(order)

    picked: list[LabeledVariant] = []
    idx = 0
    while len(picked) < n and any(by_element.values()):
        group = by_element[order[idx % len(order)]]
        if group:
            picked.append(group.pop())
        idx += 1
    return picked


@dataclass
class NullControlOutcome:
    """One negative variant run through the agent, with its verdict.

    Attributes:
        variant: The (non-functional) variant tested.
        element: The MPRA element it sits in (its ``source``).
        interpretation: The agent's adjudicated interpretation.
        verdict: ``"declined"`` / ``"borderline"`` / ``"confabulated"``.
        sequence_signal: Whether the bundle carried a ChromBPNet score (the faithful
            control has this; an annotation-only pilot does not).
        result: The full ``MultiAgentResult`` transcript, if available.
    """

    variant: Variant
    element: str | None
    interpretation: Any
    verdict: str
    sequence_signal: bool
    result: Any | None = None


def run_null_control(
    benchmark: str | list[LabeledVariant],
    interpreter: Any,
    *,
    n: int = 8,
    seed: int = 0,
    elements: list[str] | None = None,
    genome_path: str | None = None,
    scorer: ChromBPNetScorer | None = None,
    client: HttpClient | None = None,
    celltype: str | None = None,
    progress: bool = False,
) -> list[NullControlOutcome]:
    """Run the full agent on a handful of known-negative variants and classify each.

    Args:
        benchmark: Path to a labeled benchmark TSV, or a list of ``LabeledVariant``.
        interpreter: A multi-agent interpreter (uses ``deliberate`` for the full
            transcript when available, else ``interpret``).
        n: Number of negatives to test.
        seed: Selection seed.
        elements: Optional element allow-list (see :func:`select_negatives`).
        genome_path: hg38 FASTA — enables the ChromBPNet + motif signals (faithful run).
        scorer: ChromBPNet scorer — required alongside ``genome_path`` for the score.
        client: HTTP client for the annotation tools (defaults to the shared one).
        celltype: Cell-type context label recorded on the bundle.
        progress: Print a one-line progress marker per variant.

    Returns:
        One :class:`NullControlOutcome` per selected negative.
    """
    variants = (
        load_labeled_variants(benchmark) if isinstance(benchmark, str) else benchmark
    )
    negatives = select_negatives(variants, n, seed=seed, elements=elements)

    outcomes: list[NullControlOutcome] = []
    for i, lv in enumerate(negatives, 1):
        bundle = analyze_variant(
            lv.variant,
            rsid=lv.rsid,
            celltype=celltype,
            genome_path=genome_path,
            scorer=scorer,
            client=client,
        )
        if hasattr(interpreter, "deliberate"):
            result = interpreter.deliberate(bundle)
            interpretation = result.interpretation
        else:
            result = None
            interpretation = interpreter.interpret(bundle)
        verdict = classify_decline(interpretation)
        outcomes.append(
            NullControlOutcome(
                variant=lv.variant,
                element=lv.source,
                interpretation=interpretation,
                verdict=verdict,
                sequence_signal=bundle.chrombpnet is not None,
                result=result,
            )
        )
        if progress:
            print(f"  [{i}/{len(negatives)}] {lv.variant} ({lv.source}) -> {verdict}")
    return outcomes


@dataclass
class NullControlReport:
    """Aggregate verdict counts for a null-control run."""

    n: int
    verdicts: dict[str, int] = field(default_factory=dict)
    confidence: dict[str, int] = field(default_factory=dict)
    sequence_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "n": self.n,
            "verdicts": self.verdicts,
            "confidence": self.confidence,
            "sequence_signal_present": self.sequence_signal,
        }


def summarize(outcomes: list[NullControlOutcome]) -> NullControlReport:
    """Aggregate verdicts + confidence over a set of outcomes.

    Args:
        outcomes: Results from :func:`run_null_control`.

    Returns:
        A :class:`NullControlReport`.
    """
    verdicts = Counter(o.verdict for o in outcomes)
    confidence = Counter(
        (getattr(o.interpretation, "confidence", "low") or "low").lower()
        for o in outcomes
    )
    return NullControlReport(
        n=len(outcomes),
        verdicts={v: verdicts.get(v, 0) for v in VERDICTS},
        confidence=dict(confidence),
        sequence_signal=all(o.sequence_signal for o in outcomes) and bool(outcomes),
    )


def render_summary(outcomes: list[NullControlOutcome]) -> str:
    """Render a human-readable per-variant table + verdict tally."""
    report = summarize(outcomes)
    lines = ["── Agent null control (MPRA negatives) " + "─" * 22]
    if not report.sequence_signal:
        lines.append("  ⚠ annotation-only bundles (no ChromBPNet/motif) — see module note")
    for o in outcomes:
        conf = getattr(o.interpretation, "confidence", "low")
        tf = getattr(o.interpretation, "tf", None) or "-"
        lines.append(
            f"  {str(o.variant):24s} {str(o.element or '-'):12s} "
            f"{o.verdict:12s} conf={conf:6s} tf={tf}"
        )
    v = report.verdicts
    lines.append(
        f"  → declined {v['declined']}/{report.n}  "
        f"borderline {v['borderline']}  confabulated {v['confabulated']}"
    )
    return "\n".join(lines)
