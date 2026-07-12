"""Agent null control: does the multi-agent decline on non-functional variants?

The single question everyone quietly has about LLM agents in biology: handed a variant
that is genuinely **non-functional** — an MPRA negative sitting *inside* an active
regulatory element — does the reasoning layer correctly report "low confidence, no
coherent mechanism", or does it **confabulate** a plausible-sounding story?

This harness draws variants from the same matched MPRA benchmark used for the engine
AUROC, runs the full multi-agent deliberation on each, and scores it against ground
truth. The core is a **label-neutral behavior** — did the agent *decline*, *hedge*, or
*assert* a mechanism? — mapped to a **label-aware verdict**:

- negatives (``label == 0``, should decline): declined ✓ · borderline · **confabulated** ✗
- positives (``label == 1``, should assert):  **recovered** ✓ · borderline · **missed** ✗

Run the **null control** alone (:func:`run_null_control`), or the **paired control**
(:func:`run_paired_control`) for the full decline-vs-assert contrast in one call — the
discrimination story: does the agent decline on non-functional variants *and* recover a
confident mechanism on functional ones? It reuses the deterministic tool layer
(:func:`reglens.orchestrator.analyze_variant`) and the interpreter unchanged; nothing
here computes or invents a number.

**Honest scoping.** A faithful run needs the sequence-model signals (ChromBPNet Δ +
motif), so pass a ``genome_path`` + ``scorer`` (i.e. run it on Colab with hg38 and the
ChromBPNet model). Without them the paired contrast is meaningless — the positives simply
*miss* for lack of data, not lack of signal. The null arm alone is still informative on
annotation-only bundles (does it fabricate a mechanism from context alone?). Every report
records whether the sequence signals were present.
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
# Rubric: label-neutral *behavior* + label-aware *verdict*.
#
# Behavior asks one question: did the agent *assert a concrete sequence-level mechanism*
# (a named TF disrupted/created, with an effect verb), and at what confidence?
#   - not asserting a mechanism            -> "declined"
#   - asserting one, but at low confidence -> "hedged"
#   - asserting one at medium/high conf    -> "asserted"
# An explicit decline statement ("no coherent mechanism", ...) at low confidence counts as
# "declined" even if a motif was named. The verdict then maps behavior against ground truth:
#   negative (should decline): declined=correct, hedged=borderline, asserted=CONFABULATED
#   positive (should assert):  asserted=RECOVERED, hedged=borderline, declined=MISSED
# The rubric is deliberately simple and auditable; the report carries the full prose so a
# human can spot-check every call.
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

#: Behaviors (label-neutral) and the two per-label verdict vocabularies.
BEHAVIORS = ("declined", "hedged", "asserted")
ALL_VERDICTS = ("declined", "borderline", "confabulated", "recovered", "missed")

# behavior -> verdict, keyed by ground-truth label (0 = negative, 1 = positive).
_VERDICT_MAP: dict[int, dict[str, str]] = {
    0: {"declined": "declined", "hedged": "borderline", "asserted": "confabulated"},
    1: {"declined": "missed", "hedged": "borderline", "asserted": "recovered"},
}
#: Verdicts that count as the agent behaving correctly for its label.
CORRECT_VERDICTS = frozenset({"declined", "recovered"})


def assertiveness(interpretation: Any) -> str:
    """Classify what the agent *did*, independent of ground truth.

    Args:
        interpretation: A
            :class:`~reglens.agents.interpreter.MechanisticInterpretation` (or any object
            with ``mechanism``, ``confidence``, and ``tf`` attributes).

    Returns:
        One of :data:`BEHAVIORS`: ``"declined"``, ``"hedged"``, or ``"asserted"``.
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
    return "hedged" if confidence == "low" else "asserted"


def verdict_for(interpretation: Any, label: int) -> str:
    """Map the agent's behavior to a correctness verdict given the true label.

    Args:
        interpretation: The agent's interpretation.
        label: Ground-truth label — ``0`` (non-functional, should decline) or ``1``
            (functional, should assert a mechanism).

    Returns:
        A verdict from :data:`ALL_VERDICTS`.
    """
    return _VERDICT_MAP[int(label)][assertiveness(interpretation)]


def classify_decline(interpretation: Any) -> str:
    """Verdict for a **known-negative** variant (back-compat wrapper for ``verdict_for``).

    Returns:
        ``"declined"`` / ``"borderline"`` / ``"confabulated"``.
    """
    return verdict_for(interpretation, 0)


# --------------------------------------------------------------------------------------
# Selection + run
# --------------------------------------------------------------------------------------
def select_by_label(
    variants: list[LabeledVariant],
    n: int,
    label: int,
    seed: int = 0,
    elements: list[str] | None = None,
) -> list[LabeledVariant]:
    """Deterministically pick ``n`` variants of a given label, spread across elements.

    Args:
        variants: Labeled variants.
        n: Number to select.
        label: ``0`` (negatives) or ``1`` (positives).
        seed: RNG seed for reproducibility.
        elements: If given, restrict to variants whose ``source`` is in this list
            (e.g. the hematopoietic elements where the K562 model has real signal).

    Returns:
        Up to ``n`` :class:`LabeledVariant` s of that label, spread across their elements.
    """
    allowed = set(elements) if elements else None
    chosen = [
        v for v in variants
        if v.label == label and (allowed is None or v.source in allowed)
    ]
    by_element: dict[str | None, list[LabeledVariant]] = {}
    for v in chosen:
        by_element.setdefault(v.source, []).append(v)

    rng = random.Random(f"{seed}:{label}")
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


def select_negatives(
    variants: list[LabeledVariant], n: int, seed: int = 0,
    elements: list[str] | None = None,
) -> list[LabeledVariant]:
    """Deterministically pick ``n`` negatives (``label == 0``); see :func:`select_by_label`."""
    return select_by_label(variants, n, 0, seed=seed, elements=elements)


def select_positives(
    variants: list[LabeledVariant], n: int, seed: int = 0,
    elements: list[str] | None = None,
) -> list[LabeledVariant]:
    """Deterministically pick ``n`` positives (``label == 1``); see :func:`select_by_label`."""
    return select_by_label(variants, n, 1, seed=seed, elements=elements)


@dataclass
class NullControlOutcome:
    """One variant run through the agent, with its correctness verdict.

    Attributes:
        variant: The variant tested.
        element: The MPRA element it sits in (its ``source``).
        label: Ground-truth label — ``0`` (should decline) or ``1`` (should assert).
        interpretation: The agent's adjudicated interpretation.
        verdict: A verdict from :data:`ALL_VERDICTS` (label-aware).
        sequence_signal: Whether the bundle carried a ChromBPNet score (the faithful
            control has this; an annotation-only pilot does not).
        result: The full ``MultiAgentResult`` transcript, if available.
    """

    variant: Variant
    element: str | None
    label: int
    interpretation: Any
    verdict: str
    sequence_signal: bool
    result: Any | None = None


def run_control(
    benchmark: str | list[LabeledVariant],
    interpreter: Any,
    label: int,
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
    """Run the full agent on ``n`` variants of a given label and score each verdict.

    Args:
        benchmark: Path to a labeled benchmark TSV, or a list of ``LabeledVariant``.
        interpreter: A multi-agent interpreter (uses ``deliberate`` for the full
            transcript when available, else ``interpret``).
        label: ``0`` for the null (negative) control, ``1`` for the positive control.
        n: Number of variants to test.
        seed: Selection seed.
        elements: Optional element allow-list (see :func:`select_by_label`).
        genome_path: hg38 FASTA — enables the ChromBPNet + motif signals (faithful run).
        scorer: ChromBPNet scorer — required alongside ``genome_path`` for the score.
        client: HTTP client for the annotation tools (defaults to the shared one).
        celltype: Cell-type context label recorded on the bundle.
        progress: Print a one-line progress marker per variant.

    Returns:
        One :class:`NullControlOutcome` per selected variant.
    """
    variants = (
        load_labeled_variants(benchmark) if isinstance(benchmark, str) else benchmark
    )
    selected = select_by_label(variants, n, label, seed=seed, elements=elements)

    outcomes: list[NullControlOutcome] = []
    for i, lv in enumerate(selected, 1):
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
        verdict = verdict_for(interpretation, label)
        outcomes.append(
            NullControlOutcome(
                variant=lv.variant,
                element=lv.source,
                label=label,
                interpretation=interpretation,
                verdict=verdict,
                sequence_signal=bundle.chrombpnet is not None,
                result=result,
            )
        )
        if progress:
            print(f"  [{i}/{len(selected)}] {lv.variant} ({lv.source}) -> {verdict}")
    return outcomes


def run_null_control(
    benchmark: str | list[LabeledVariant], interpreter: Any, **kwargs: Any
) -> list[NullControlOutcome]:
    """Run the null (negative) control — the agent *should decline*. See :func:`run_control`."""
    return run_control(benchmark, interpreter, 0, **kwargs)


def run_positive_control(
    benchmark: str | list[LabeledVariant], interpreter: Any, **kwargs: Any
) -> list[NullControlOutcome]:
    """Run the positive control — the agent *should assert a mechanism*. See :func:`run_control`."""
    return run_control(benchmark, interpreter, 1, **kwargs)


def run_paired_control(
    benchmark: str | list[LabeledVariant],
    interpreter: Any,
    *,
    n_neg: int = 6,
    n_pos: int = 6,
    **kwargs: Any,
) -> tuple[list[NullControlOutcome], list[NullControlOutcome]]:
    """Run both controls in one call — the full decline-vs-assert contrast.

    A meaningful contrast needs the sequence signals present (``genome_path`` + ``scorer``):
    on annotation-only bundles the positives simply *miss* for lack of data, not lack of
    signal, so run this with hg38 + the ChromBPNet model (i.e. on Colab).

    Args:
        benchmark: Path to a labeled benchmark TSV, or a list of ``LabeledVariant``.
        interpreter: A multi-agent interpreter.
        n_neg: Number of negatives (should decline).
        n_pos: Number of positives (should assert).
        **kwargs: Forwarded to :func:`run_control` (``seed``, ``elements``,
            ``genome_path``, ``scorer``, ``client``, ``celltype``, ``progress``).

    Returns:
        ``(negative_outcomes, positive_outcomes)``.
    """
    variants = (
        load_labeled_variants(benchmark) if isinstance(benchmark, str) else benchmark
    )
    negatives = run_control(variants, interpreter, 0, n=n_neg, **kwargs)
    positives = run_control(variants, interpreter, 1, n=n_pos, **kwargs)
    return negatives, positives


@dataclass
class NullControlReport:
    """Aggregate verdict counts for one control arm."""

    n: int
    verdicts: dict[str, int] = field(default_factory=dict)
    confidence: dict[str, int] = field(default_factory=dict)
    n_correct: int = 0
    sequence_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "n": self.n,
            "verdicts": self.verdicts,
            "confidence": self.confidence,
            "n_correct": self.n_correct,
            "sequence_signal_present": self.sequence_signal,
        }


def summarize(outcomes: list[NullControlOutcome]) -> NullControlReport:
    """Aggregate verdicts, confidence, and the correct-behavior rate over outcomes.

    Args:
        outcomes: Results from :func:`run_control` (or the null/positive wrappers).

    Returns:
        A :class:`NullControlReport`.
    """
    verdicts = Counter(o.verdict for o in outcomes)
    confidence = Counter(
        (getattr(o.interpretation, "confidence", "low") or "low").lower()
        for o in outcomes
    )
    n_correct = sum(1 for o in outcomes if o.verdict in CORRECT_VERDICTS)
    return NullControlReport(
        n=len(outcomes),
        verdicts={v: verdicts.get(v, 0) for v in ALL_VERDICTS},
        confidence=dict(confidence),
        n_correct=n_correct,
        sequence_signal=all(o.sequence_signal for o in outcomes) and bool(outcomes),
    )


def _tally(verdicts: dict[str, int]) -> str:
    """One-line tally of the non-zero verdicts."""
    return "  ".join(f"{v} {verdicts[v]}" for v in ALL_VERDICTS if verdicts.get(v))


def render_summary(outcomes: list[NullControlOutcome], title: str | None = None) -> str:
    """Render a human-readable per-variant table + verdict tally for one control arm."""
    report = summarize(outcomes)
    lines = [f"── {title or 'Agent control'} " + "─" * 22]
    if outcomes and not report.sequence_signal:
        lines.append("  ⚠ annotation-only bundles (no ChromBPNet/motif) — see module note")
    for o in outcomes:
        conf = getattr(o.interpretation, "confidence", "low")
        tf = getattr(o.interpretation, "tf", None) or "-"
        lines.append(
            f"  {str(o.variant):24s} {str(o.element or '-'):12s} "
            f"{o.verdict:12s} conf={conf:6s} tf={tf}"
        )
    lines.append(f"  → {_tally(report.verdicts)}   ({report.n_correct}/{report.n} correct)")
    return "\n".join(lines)


def render_paired(
    negatives: list[NullControlOutcome], positives: list[NullControlOutcome]
) -> str:
    """Render the full decline-vs-assert contrast from a paired run."""
    neg, pos = summarize(negatives), summarize(positives)
    declined = neg.verdicts.get("declined", 0)
    recovered = pos.verdicts.get("recovered", 0)
    return "\n".join([
        render_summary(negatives, "Null control — negatives (should DECLINE)"),
        "",
        render_summary(positives, "Positive control — positives (should ASSERT)"),
        "",
        "── Discrimination " + "─" * 40,
        f"  declined on negatives : {declined}/{neg.n}",
        f"  recovered on positives: {recovered}/{pos.n}",
        f"  → the agent separates non-functional from functional variants "
        f"{'cleanly' if declined == neg.n and recovered == pos.n else 'partially'}.",
    ])
