"""Prospective screen: surface *interpretable, uncharacterized* regulatory variants.

Every other experiment shows RegLens *recovering* known mechanisms (trustworthiness). This
module does the tool's actual job — screening noncoding variants for a **prospective,
falsifiable hypothesis** — under strict discipline so the integrity earned elsewhere is not
spent here:

1. **Stay in-domain.** Screen only in a lineage where the engine is *validated* (erythroid
   / hematopoietic, K562 AUROC 0.716). A prospective claim where the model runs at chance
   would violate our own calibration finding.
2. **Screen with the pipeline, don't cherry-pick.** Rank blood-trait GWAS variants by the
   quadrant that matters — large ``|ChromBPNet Δ|`` + a *concordant* motif + a real GWAS
   trait + *sparse* literature — and take what surfaces. That is the tool working, not a
   hand-pick.
3. **Literature sparsity is a *screening flag*, not a novelty claim.** The literature tool
   is unreliable for proving absence; a low hit-count only *prioritizes a candidate for
   manual verification*. Novelty must be confirmed by a real manual check and phrased "to
   our knowledge, no published mechanism".
4. **Make it falsifiable.** The write-up must name the experiment that would kill it (CRISPRi
   of the element; allele-specific ATAC/MPRA at the site).
5. **Let calibration work.** Run the multi-agent on the surfaced candidate and report its
   confidence + red-team caveats verbatim — engine+motif+GWAS but no eQTL/literature limb
   should land it at ~*medium*. If the agent hedges, show the hedge.

This module ranks; it never asserts novelty. The prospective claim is written up by hand
from the screen output plus a manual literature check (see ``render_screen`` guidance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reglens.genome import Variant
from reglens.orchestrator import analyze_variant
from reglens.tools._http import HttpClient
from reglens.validation.agent_eval import resolve_variant

#: Trait substrings that count as a blood / hematopoietic GWAS phenotype (in-domain).
BLOOD_TRAIT_TERMS: tuple[str, ...] = (
    "hemoglobin", "haemoglobin", "erythrocyte", "red blood", "red cell", "rbc",
    "reticulocyte", "mean corpuscular", "hematocrit", "haematocrit", "platelet",
    "monocyte", "neutrophil", "basophil", "eosinophil", "lymphocyte", "leukocyte",
    "white blood", "blood cell", "myeloid", "megakaryocyte",
)


@dataclass(frozen=True)
class DiscoveryCandidate:
    """A blood-trait GWAS noncoding variant to screen.

    Attributes:
        rsid: dbSNP rsID (resolved to hg38 at run time).
        note: Locus / trait context (for the operator — *not* a mechanism claim).
        variant: Optional explicit hg38 variant (else resolved by rsID).
        alt: Optional functional alt to pin for a multi-allelic rsID.
    """

    rsid: str
    note: str = ""
    variant: Variant | None = None
    alt: str | None = None


#: A **starter** pool of real blood-trait GWAS noncoding variants (traits verified at run
#: time via the GWAS tool). It intentionally mixes solved loci — BCL11A, HBS1L-MYB, Duffy —
#: which act as screen controls (they should show literature and be filtered *out* of the
#: novel quadrant) with less-characterized ones. For a real discovery run, EXPAND this from
#: a GWAS-Catalog blood-trait query; do not treat this list as exhaustive.
BLOOD_TRAIT_CANDIDATES: list[DiscoveryCandidate] = [
    DiscoveryCandidate("rs1427407", "BCL11A +58 enhancer; HbF (solved — control)", alt="G"),
    DiscoveryCandidate("rs2814778", "ACKR1/Duffy promoter; neutrophil (solved — control)"),
    DiscoveryCandidate("rs4895441", "HBS1L-MYB intergenic; HbF / RBC"),
    DiscoveryCandidate("rs9399137", "HBS1L-MYB region; HbF / RBC"),
    DiscoveryCandidate("rs7776054", "HBS1L-MYB region; RBC"),
    DiscoveryCandidate("rs737092", "RBM38 region; RBC indices"),
    DiscoveryCandidate("rs1175550", "SMIM1; RBC count"),
    DiscoveryCandidate("rs342293", "PIK3CG intergenic; platelet count / volume"),
    DiscoveryCandidate("rs1354034", "ARHGEF3; platelet count"),
]


@dataclass
class DiscoveryScore:
    """The deterministic screen signals for one candidate (no LLM)."""

    rsid: str
    variant: Variant | None
    delta: float                 # |ChromBPNet Δ log-counts|
    motif_tf: str | None
    motif_concordant: bool       # motif effect sign matches accessibility change sign
    gene: str | None
    blood_gwas: bool
    traits: list[str]
    literature_hits: int
    in_quadrant: bool            # engine fires + concordant motif + blood GWAS + sparse lit
    errors: dict[str, str] = field(default_factory=dict)


def screen_bundle(
    bundle: Any,
    *,
    min_delta: float = 0.30,
    max_literature: int = 2,
    blood_terms: tuple[str, ...] = BLOOD_TRAIT_TERMS,
) -> DiscoveryScore:
    """Compute the discovery-quadrant signals from a completed evidence bundle.

    The quadrant is **large ``|Δ|`` + a concordant motif + a blood GWAS trait + sparse
    literature**. Concordance assumes an *activator* motif: the motif-effect sign
    (create/strengthen = +, disrupt/weaken = −) should match the ChromBPNet accessibility
    change sign. Sparse literature only *flags* a candidate for manual novelty checking — it
    is **not** a novelty claim.

    Args:
        bundle: A completed :class:`~reglens.report.schema.EvidenceBundle`.
        min_delta: Minimum ``|Δ log-counts|`` for "the engine fires".
        max_literature: Literature hit-count at or below which a candidate is flagged for
            manual novelty verification.
        blood_terms: Trait substrings that count as in-domain (blood/hematopoietic).

    Returns:
        A :class:`DiscoveryScore`.
    """
    cp = bundle.chrombpnet
    delta = abs(cp.delta_log_counts) if cp is not None else 0.0

    motif_top = bundle.motif.top if (bundle.motif is not None and bundle.motif.top) else None
    motif_tf = motif_top.tf_name if motif_top is not None else None
    concordant = bool(
        motif_top is not None and cp is not None
        and motif_top.delta_score * cp.delta_log_counts > 0
    )

    traits = bundle.trait.unique_traits() if bundle.trait is not None else []
    blood_gwas = any(any(bt in t.lower() for bt in blood_terms) for t in traits)

    lit = bundle.literature.hit_count if bundle.literature is not None else 0
    gene = (
        bundle.gene.nearest_gene.symbol
        if bundle.gene is not None and bundle.gene.nearest_gene is not None
        else None
    )

    in_quadrant = (
        delta >= min_delta and motif_tf is not None and concordant
        and blood_gwas and lit <= max_literature
    )
    return DiscoveryScore(
        rsid=bundle.rsid or "", variant=bundle.variant, delta=delta, motif_tf=motif_tf,
        motif_concordant=concordant, gene=gene, blood_gwas=blood_gwas, traits=traits[:5],
        literature_hits=lit, in_quadrant=in_quadrant, errors=dict(bundle.errors),
    )


def run_discovery_screen(
    candidates: list[DiscoveryCandidate] | None = None,
    *,
    scorer: Any = None,
    genome_path: str | None = None,
    client: HttpClient | None = None,
    resolve: bool = True,
    min_delta: float = 0.30,
    max_literature: int = 2,
    celltype: str = "K562",
    progress: bool = False,
) -> list[tuple[DiscoveryScore, Any]]:
    """Screen blood-trait candidates through the deterministic pipeline and rank them.

    Ranking: quadrant members first, then by descending ``|Δ|``. The top quadrant member
    with sparse literature is the candidate to **manually verify** and (if genuinely
    uncharacterized) write up as a prospective, falsifiable hypothesis.

    Args:
        candidates: Candidates to screen (defaults to :data:`BLOOD_TRAIT_CANDIDATES`).
        scorer: ChromBPNet scorer (with ``genome_path``) — required; screening without the
            engine is meaningless.
        genome_path: hg38 FASTA path.
        client: HTTP client for the annotation tools + rsID resolution.
        resolve: Resolve rsID → hg38 via Ensembl (recommended).
        min_delta: "engine fires" threshold.
        max_literature: sparse-literature flag threshold.
        celltype: Cell-type context label (should be an in-domain erythroid model).
        progress: Print a per-candidate marker.

    Returns:
        ``[(score, bundle)]`` sorted best-first. Keep the bundle so the surfaced candidate
        can be handed straight to the multi-agent for the hypothesis.
    """
    candidates = candidates if candidates is not None else BLOOD_TRAIT_CANDIDATES
    out: list[tuple[DiscoveryScore, Any]] = []
    for i, cand in enumerate(candidates, 1):
        variant = cand.variant or (
            resolve_variant(cand.rsid, client, prefer_alt=cand.alt) if resolve else None
        )
        if variant is None:
            raise ValueError(f"No coordinate for {cand.rsid}; set resolve=True or .variant")
        bundle = analyze_variant(
            variant, rsid=cand.rsid, celltype=celltype,
            genome_path=genome_path, scorer=scorer, client=client,
        )
        score = screen_bundle(
            bundle, min_delta=min_delta, max_literature=max_literature
        )
        out.append((score, bundle))
        if progress:
            flag = "★ QUADRANT" if score.in_quadrant else ""
            print(f"  [{i}/{len(candidates)}] {cand.rsid} |Δ|={score.delta:.2f} "
                  f"tf={score.motif_tf} conc={score.motif_concordant} "
                  f"blood={score.blood_gwas} lit={score.literature_hits} {flag}")
    out.sort(key=lambda t: (t[0].in_quadrant, t[0].delta), reverse=True)
    return out


def render_screen(scored: list[tuple[DiscoveryScore, Any]]) -> str:
    """Render the screen: per-candidate quadrant signals + what to do next."""
    lines = ["── Prospective discovery screen (blood-trait, in-domain K562) " + "─" * 8,
             f"  {'rsID':12s} {'gene':9s} {'|Δ|':>5s} {'TF':12s} conc blood lit  quadrant"]
    for score, _ in scored:
        lines.append(
            f"  {score.rsid:12s} {str(score.gene or '-'):9s} {score.delta:5.2f} "
            f"{str(score.motif_tf or '-'):12s} "
            f"{'  ✓' if score.motif_concordant else '  ·'}  "
            f"{'✓' if score.blood_gwas else '·'}   {score.literature_hits:>2d}   "
            f"{'★' if score.in_quadrant else ''}"
        )
    quadrant = [s for s, _ in scored if s.in_quadrant]
    lines.append(f"  → {len(quadrant)} candidate(s) in the discovery quadrant.")
    if quadrant:
        top = quadrant[0]
        lines.append(f"    NEXT: manually verify no published mechanism for {top.rsid} "
                     f"({top.gene}/{top.motif_tf}); if confirmed, interpret it and write up")
        lines.append("    a *falsifiable* hypothesis (see docs/prospective_hypothesis.md).")
    return "\n".join(lines)
