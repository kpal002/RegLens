"""Link a variant to its likely target gene: nearest/overlapping gene + GTEx eQTLs.

Two independent, deterministic signals for the "which gene" half of the mechanism:

* **Nearest / overlapping gene** (Ensembl REST) — the gene the variant sits in or
  closest to. For an intronic enhancer variant this is often the regulated gene
  itself.
* **GTEx eQTLs** — tissues/genes where the variant is a significant expression QTL.
  Note the honest caveat surfaced for cases like rs1427407: GTEx v8 lacks erythroid
  progenitors, so an *absent* eQTL for the true target is uninformative, and the
  only hit may be an incidental gene in an unrelated tissue.

The reasoning layer reconciles the two; this tool only retrieves and structures
them (no LLM, no invented numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reglens.genome import Variant
from reglens.tools._http import HttpClient, resolve_client

ENSEMBL_OVERLAP = "https://rest.ensembl.org/overlap/region/human"
GTEX_VARIANT = "https://gtexportal.org/api/v2/dataset/variant"
GTEX_EQTL = "https://gtexportal.org/api/v2/association/singleTissueEqtl"

# Biotypes we treat as "real" target genes when picking the nearest one.
_CODING_BIOTYPE = "protein_coding"


@dataclass
class GeneOverlap:
    """A gene near or overlapping the variant.

    Attributes:
        gene_id: Ensembl gene id.
        symbol: Gene symbol (may be empty for un-named lncRNAs).
        biotype: Ensembl biotype (e.g. ``"protein_coding"``).
        start: Gene start (1-based, GRCh38).
        end: Gene end (1-based, GRCh38).
        strand: ``+1`` or ``-1``.
        distance: bp from the variant to the gene body; ``0`` if the variant is
            inside the gene.
    """

    gene_id: str
    symbol: str
    biotype: str
    start: int
    end: int
    strand: int
    distance: int

    @property
    def overlaps(self) -> bool:
        """Whether the variant falls within the gene body."""
        return self.distance == 0


@dataclass
class Eqtl:
    """A significant GTEx expression QTL for the variant.

    Attributes:
        gene_symbol: eGene symbol.
        gencode_id: Versioned GENCODE gene id.
        tissue: GTEx tissue (``tissueSiteDetailId``).
        nes: Normalized effect size (slope) of the alt allele.
        p_value: Nominal eQTL p-value.
    """

    gene_symbol: str
    gencode_id: str
    tissue: str
    nes: float
    p_value: float


@dataclass
class GeneTargetResult:
    """Combined gene-target evidence for a variant.

    Attributes:
        variant: The variant analyzed.
        rsid: dbSNP rsID used for the GTEx lookup, if any.
        overlapping_genes: Genes whose body contains the variant.
        nearest_gene: Nearest protein-coding gene (or nearest overall if none code).
        eqtls: Significant GTEx eQTLs for the variant.
    """

    variant: Variant
    rsid: str | None = None
    overlapping_genes: list[GeneOverlap] = field(default_factory=list)
    nearest_gene: GeneOverlap | None = None
    eqtls: list[Eqtl] = field(default_factory=list)

    def summary(self) -> str:
        """A one-line human-readable summary of the gene-target evidence."""
        if self.nearest_gene is None:
            gene_part = "no gene found nearby"
        elif self.nearest_gene.overlaps:
            gene_part = f"inside {self.nearest_gene.symbol or self.nearest_gene.gene_id}"
        else:
            g = self.nearest_gene
            gene_part = f"nearest {g.symbol or g.gene_id} ({g.distance:,} bp)"
        if self.eqtls:
            genes = sorted({e.gene_symbol for e in self.eqtls})
            eqtl_part = f"GTEx eQTL for {', '.join(genes)}"
        else:
            eqtl_part = "no significant GTEx eQTL"
        return f"{self.variant}: {gene_part}; {eqtl_part}"


def _ensembl_chrom(chrom: str) -> str:
    """Convert a ``chrN`` contig name to Ensembl's bare form (``N``)."""
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def _distance(pos: int, start: int, end: int) -> int:
    """bp distance from a 1-based position to a [start, end] interval (0 if inside)."""
    if start <= pos <= end:
        return 0
    return min(abs(pos - start), abs(pos - end))


def nearest_genes(
    variant: Variant, client: HttpClient | None = None, window: int = 100_000
) -> list[GeneOverlap]:
    """Fetch genes within ``window`` bp of the variant, sorted by distance.

    Args:
        variant: The variant to search around.
        client: HTTP client; defaults to the shared urllib client.
        window: Half-width (bp) of the search region.

    Returns:
        Genes near the variant, closest first.
    """
    http = resolve_client(client)
    chrom = _ensembl_chrom(variant.chrom)
    start = max(1, variant.pos - window)
    end = variant.pos + window
    url = f"{ENSEMBL_OVERLAP}/{chrom}:{start}-{end}"
    records = http.get_json(url, {"feature": "gene", "content-type": "application/json"})

    genes = [
        GeneOverlap(
            gene_id=str(r.get("id", "")),
            symbol=str(r.get("external_name") or ""),
            biotype=str(r.get("biotype", "")),
            start=int(r.get("start", 0)),
            end=int(r.get("end", 0)),
            strand=int(r.get("strand", 0)),
            distance=_distance(variant.pos, int(r.get("start", 0)), int(r.get("end", 0))),
        )
        for r in (records or [])
    ]
    genes.sort(key=lambda g: g.distance)
    return genes


def _pick_nearest(genes: list[GeneOverlap]) -> GeneOverlap | None:
    """Nearest protein-coding gene, or nearest overall if none are coding."""
    coding = [g for g in genes if g.biotype == _CODING_BIOTYPE]
    if coding:
        return coding[0]
    return genes[0] if genes else None


def gtex_eqtls(
    rsid: str, client: HttpClient | None = None, dataset: str = "gtex_v8"
) -> list[Eqtl]:
    """Fetch significant GTEx eQTLs for a variant by rsID.

    Resolves the rsID to a GTEx variant id, then queries single-tissue eQTLs.

    Args:
        rsid: dbSNP rsID (e.g. ``"rs1427407"``).
        client: HTTP client; defaults to the shared urllib client.
        dataset: GTEx dataset id.

    Returns:
        Significant eQTLs (possibly empty). Empty does not imply "no effect" — the
        relevant tissue may simply be absent from GTEx.
    """
    http = resolve_client(client)
    lookup = http.get_json(GTEX_VARIANT, {"snpId": rsid, "datasetId": dataset})
    rows = (lookup or {}).get("data", [])
    if not rows:
        return []
    variant_id = rows[0].get("variantId")
    if not variant_id:
        return []

    payload = http.get_json(GTEX_EQTL, {"variantId": variant_id, "datasetId": dataset})
    return [
        Eqtl(
            gene_symbol=str(r.get("geneSymbol", "")),
            gencode_id=str(r.get("gencodeId", "")),
            tissue=str(r.get("tissueSiteDetailId", "")),
            nes=float(r.get("nes", 0.0)),
            p_value=float(r.get("pValue", 1.0)),
        )
        for r in (payload or {}).get("data", [])
    ]


def gene_target(
    variant: Variant,
    rsid: str | None = None,
    client: HttpClient | None = None,
    window: int = 100_000,
) -> GeneTargetResult:
    """Assemble nearest-gene and GTEx-eQTL evidence for a variant.

    Args:
        variant: The variant to analyze.
        rsid: dbSNP rsID; required for the GTEx eQTL lookup (skipped if absent).
        client: HTTP client; defaults to the shared urllib client.
        window: Half-width (bp) of the nearest-gene search region.

    Returns:
        A :class:`GeneTargetResult`.
    """
    genes = nearest_genes(variant, client=client, window=window)
    eqtls = gtex_eqtls(rsid, client=client) if rsid else []
    return GeneTargetResult(
        variant=variant,
        rsid=rsid,
        overlapping_genes=[g for g in genes if g.overlaps],
        nearest_gene=_pick_nearest(genes),
        eqtls=eqtls,
    )
