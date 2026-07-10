"""Render RegLens results for humans: evidence bundle, interpretation, deliberation.

Text renderers for the CLI. The JSON report is ``bundle.to_dict()`` /
``MultiAgentResult.to_dict()``; an HTML renderer can build on the same objects later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reglens.report.schema import EvidenceBundle

if TYPE_CHECKING:  # avoid importing the agents layer (and its deps) at import time
    from reglens.agents.interpreter import MechanisticInterpretation
    from reglens.agents.multi_agent import MultiAgentResult

_RULE = "─" * 66


def render_text(bundle: EvidenceBundle) -> str:
    """Render an evidence bundle as a plain-text report block.

    Args:
        bundle: The evidence bundle to render.

    Returns:
        A multi-line string suitable for terminal output.
    """
    lines: list[str] = [_RULE, f"  RegLens evidence · {bundle.variant}"]
    if bundle.rsid:
        lines.append(f"  rsID: {bundle.rsid}")
    if bundle.celltype:
        lines.append(f"  context: {bundle.celltype}")
    lines.append(_RULE)

    if bundle.chrombpnet is not None:
        s = bundle.chrombpnet
        jsd = f", profile-JSD={s.profile_jsd:.4f}" if s.profile_jsd is not None else ""
        lines.append("  ChromBPNet accessibility:")
        lines.append(
            f"    Δlog-counts={s.delta_log_counts:+.4f} ({s.direction}{jsd}) [{s.model_name}]"
        )
    if bundle.motif is not None:
        lines.append("  TF motif effect:")
        lines.append(f"    {bundle.motif.summary()}")
    if bundle.regulatory is not None:
        lines.append("  Regulatory context:")
        lines.append(f"    {bundle.regulatory.summary()}")
    if bundle.gene is not None:
        lines.append("  Gene target:")
        lines.append(f"    {bundle.gene.summary()}")
    if bundle.trait is not None:
        lines.append("  Trait link:")
        lines.append(f"    {bundle.trait.summary()}")
    if bundle.literature is not None and bundle.literature.citations:
        lines.append(f"  Literature ({bundle.literature.hit_count} hits):")
        for c in bundle.literature.citations[:3]:
            lines.append(f"    - {c.format()}")

    if bundle.errors:
        lines.append("  Tool errors:")
        for tool, msg in bundle.errors.items():
            lines.append(f"    ! {tool}: {msg}")
    lines.append(_RULE)
    return "\n".join(lines)


def render_interpretation(interp: MechanisticInterpretation) -> str:
    """Render a mechanistic interpretation as a titled text block."""
    return "\n".join(["── RegLens interpretation " + "─" * 40, interp.format(), _RULE])


def render_deliberation(result: MultiAgentResult) -> str:
    """Render the full multi-agent deliberation (specialists, red-team, adjudication)."""
    lines = ["── RegLens multi-agent deliberation " + "─" * 30, "  Specialists:"]
    for o in result.opinions:
        lines.append(f"    [{o.agent}] ({o.confidence}) {o.assessment}")
        for c in o.concerns:
            lines.append(f"        ⚠ {c}")
    crit = result.critique
    lines.append(f"  Red-team (overall risk: {crit.overall_risk}):")
    for ch in crit.challenges:
        lines.append(f"    [{ch.severity}] {ch.claim} — {ch.concern}")
    lines.append(_RULE)
    lines.append(render_interpretation(result.interpretation))
    return "\n".join(lines)
