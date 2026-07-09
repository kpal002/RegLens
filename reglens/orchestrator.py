"""Run the deterministic tool layer end-to-end for one variant.

:func:`analyze_variant` builds the variant's sequence windows once, then runs every
deterministic tool over them, assembling an :class:`~reglens.report.schema.EvidenceBundle`.
Each tool is isolated in its own try/except so one failing external API never sinks
the whole report — failures are recorded in ``bundle.errors``. Downstream tool outputs
(gene symbol, TF, trait) are chained into the Europe PMC literature query so the
citations match the entities actually implicated.

This is the deterministic backbone. The single-agent reasoning layer (next) consumes
the bundle; it does not replace any of this computation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from reglens.genome import DEFAULT_WINDOW_LENGTH, Variant, build_sequence_windows
from reglens.report.schema import EvidenceBundle
from reglens.tools._http import HttpClient
from reglens.tools.chrombpnet_score import ChromBPNetScorer
from reglens.tools.gene_target import gene_target
from reglens.tools.literature import build_query, search_literature
from reglens.tools.motif_effect import Motif, motif_effect
from reglens.tools.regulatory_context import regulatory_context
from reglens.tools.trait_link import trait_link

T = TypeVar("T")


def _safe(errors: dict[str, str], tool: str, fn: Callable[[], T]) -> T | None:
    """Run ``fn``; on any exception, record it under ``tool`` and return ``None``."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - deliberately resilient per-tool boundary
        errors[tool] = f"{type(exc).__name__}: {exc}"
        return None


def analyze_variant(
    variant: Variant,
    rsid: str | None = None,
    celltype: str | None = None,
    genome_path: str | os.PathLike[str] | None = None,
    scorer: ChromBPNetScorer | None = None,
    client: HttpClient | None = None,
    motifs: list[Motif] | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    trait: str | None = None,
) -> EvidenceBundle:
    """Gather all deterministic evidence for a variant into one bundle.

    Args:
        variant: The variant to analyze.
        rsid: dbSNP rsID; enables the GTEx eQTL, GWAS trait, and rsID-anchored
            literature lookups.
        celltype: Cell-type / model context label (recorded, not computed on).
        genome_path: hg38 FASTA path; required for the ChromBPNet score and the
            motif-effect scan (they need the sequence windows).
        scorer: A configured :class:`ChromBPNetScorer`; if omitted the ChromBPNet
            score is skipped (the motif scan still runs if a genome is given).
        client: HTTP client for the API-backed tools; defaults to the shared one.
        motifs: Motif library for the motif scan; defaults to the bundled JASPAR set.
        window_length: Sequence window width for the scan/score.
        trait: An override trait term for the literature query (else taken from the
            GWAS result).

    Returns:
        A populated :class:`EvidenceBundle`. Tools that error are omitted and noted
        in ``bundle.errors``.
    """
    errors: dict[str, str] = {}
    bundle = EvidenceBundle(variant=variant, rsid=rsid, celltype=celltype, errors=errors)

    # Sequence-window-dependent tools (need the genome FASTA).
    window = None
    if genome_path is not None:
        window = _safe(
            errors,
            "genome",
            lambda: build_sequence_windows(
                variant, genome_path=genome_path, window_length=window_length
            ),
        )
    if window is not None:
        if scorer is not None:
            bundle.chrombpnet = _safe(
                errors, "chrombpnet", lambda: scorer.score_window(window, variant, celltype)
            )
        bundle.motif = _safe(errors, "motif", lambda: motif_effect(window, variant, motifs=motifs))

    # Coordinate/rsID-based API tools.
    bundle.regulatory = _safe(errors, "regulatory", lambda: regulatory_context(variant, client))
    bundle.gene = _safe(errors, "gene", lambda: gene_target(variant, rsid=rsid, client=client))
    if rsid:
        bundle.trait = _safe(errors, "trait", lambda: trait_link(rsid, client=client))

    # Literature: anchor on the rsID + target gene — the two most reliable, specific
    # terms. ANDing in the TF or the top trait as well tends to over-constrain the
    # query to zero hits (e.g. a pQTL trait name that no variant paper uses verbatim);
    # the TF/trait are already captured by their own tools.
    gene_symbol = (
        bundle.gene.nearest_gene.symbol
        if bundle.gene is not None and bundle.gene.nearest_gene is not None
        else None
    )
    if rsid or gene_symbol:
        query = _safe(
            errors,
            "literature_query",
            lambda: build_query(variant=variant, rsid=rsid, gene=gene_symbol, trait=trait),
        )
        if query is not None:
            bundle.literature = _safe(
                errors, "literature", lambda: search_literature(query, client=client)
            )
    return bundle
