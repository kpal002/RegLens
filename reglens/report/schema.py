"""Structured evidence bundle: the deterministic signals for one variant.

The :class:`EvidenceBundle` is the hand-off between the deterministic tool layer and
the reasoning layer. It aggregates each tool's result (any of which may be missing if
that tool wasn't run or errored) plus a per-tool error log, and serializes to a plain
JSON-able dict for the report / the agent prompt. It holds *only retrieved numbers* —
no interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reglens.genome import Variant
from reglens.tools.chrombpnet_score import VariantScore
from reglens.tools.gene_target import GeneTargetResult
from reglens.tools.literature import LiteratureResult
from reglens.tools.motif_effect import MotifEffectResult
from reglens.tools.regulatory_context import RegulatoryContextResult
from reglens.tools.trait_link import TraitLinkResult


@dataclass
class EvidenceBundle:
    """All deterministic signals gathered for a single variant.

    Attributes:
        variant: The analyzed variant.
        rsid: dbSNP rsID used for rsID-keyed tools, if any.
        celltype: Cell-type / model context label.
        chrombpnet: ChromBPNet Δ-accessibility score, if computed.
        motif: Motif-effect result, if computed.
        regulatory: Regulatory-context result, if computed.
        gene: Gene-target result, if computed.
        trait: GWAS trait-link result, if computed.
        literature: Europe PMC literature result, if computed.
        errors: Per-tool error messages for tools that failed (name → message).
    """

    variant: Variant
    rsid: str | None = None
    celltype: str | None = None
    chrombpnet: VariantScore | None = None
    motif: MotifEffectResult | None = None
    regulatory: RegulatoryContextResult | None = None
    gene: GeneTargetResult | None = None
    trait: TraitLinkResult | None = None
    literature: LiteratureResult | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def headline_signals(self) -> list[str]:
        """One-line ``summary()`` from each tool that produced a result."""
        lines: list[str] = []
        if self.chrombpnet is not None:
            lines.append(self.chrombpnet.summary())
        for result in (self.motif, self.regulatory, self.gene, self.trait):
            if result is not None:
                lines.append(result.summary())
        if self.literature is not None and self.literature.citations:
            top = self.literature.citations[0]
            lines.append(f"literature: {len(self.literature.citations)} refs, e.g. {top.format()}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-able dict (large sequences are omitted)."""
        d: dict[str, Any] = {
            "variant": str(self.variant),
            "rsid": self.rsid,
            "celltype": self.celltype,
            "errors": dict(self.errors),
        }
        if self.chrombpnet is not None:
            s = self.chrombpnet
            d["chrombpnet"] = {
                "delta_log_counts": s.delta_log_counts,
                "direction": s.direction,
                "ref_log_counts": s.ref_log_counts,
                "alt_log_counts": s.alt_log_counts,
                "profile_jsd": s.profile_jsd,
                "model": s.model_name,
            }
        if self.motif is not None and self.motif.top is not None:
            t = self.motif.top
            d["motif"] = {
                "tf": t.tf_name, "motif_id": t.motif_id, "effect": t.effect,
                "strand": t.strand, "ref_score": t.ref_score, "alt_score": t.alt_score,
                "delta_score": t.delta_score,
            }
        if self.regulatory is not None:
            n = self.regulatory.nearest
            d["regulatory"] = {
                "in_ccre": self.regulatory.in_ccre,
                "nearest": None if n is None else {
                    "id": n.element_id, "type": n.element_type,
                    "type_label": n.type_label, "distance": n.distance,
                    "source": n.source,
                },
            }
        if self.gene is not None:
            g = self.gene.nearest_gene
            d["gene"] = {
                "nearest": None if g is None else {
                    "symbol": g.symbol, "gene_id": g.gene_id,
                    "distance": g.distance, "overlaps": g.overlaps,
                },
                "overlapping": [x.symbol or x.gene_id for x in self.gene.overlapping_genes],
                "eqtls": [
                    {"gene": e.gene_symbol, "tissue": e.tissue, "nes": e.nes, "p": e.p_value}
                    for e in self.gene.eqtls
                ],
            }
        if self.trait is not None:
            d["trait"] = {
                "top_traits": self.trait.unique_traits()[:5],
                "associations": [
                    {"traits": a.traits, "p_value": a.p_value, "effect": a.effect_str()}
                    for a in self.trait.associations[:5]
                ],
            }
        if self.literature is not None:
            d["literature"] = {
                "query": self.literature.query,
                "hit_count": self.literature.hit_count,
                "citations": [
                    {"pmid": c.pmid, "title": c.title, "year": c.year,
                     "journal": c.journal, "url": c.url}
                    for c in self.literature.citations
                ],
            }
        return d
