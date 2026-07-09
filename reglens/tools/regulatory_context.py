"""Place a variant in its regulatory-element context (ENCODE SCREEN cCREs).

Answers "is this variant in a regulatory element, and what kind?" by overlapping it
against the ENCODE SCREEN registry of candidate cis-regulatory elements (cCREs),
retrieved from the UCSC ``encodeCcreCombined`` track. Reports whether the variant is
inside a cCRE and, if not, the nearest one and its distance/type — an honest signal
even when a variant falls just outside the fixed-width registry boundaries (the
chromatin model can still predict an effect there).

Deterministic retrieval only; the reasoning layer interprets. Ensembl Regulatory
Build is queried as a secondary source when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reglens.genome import Variant
from reglens.tools._http import HttpClient, resolve_client

UCSC_TRACK = "https://api.genome.ucsc.edu/getData/track"
ENSEMBL_OVERLAP = "https://rest.ensembl.org/overlap/region/human"

# Human-readable names for the SCREEN cCRE class labels.
_CCRE_LABELS = {
    "prom": "promoter-like",
    "enhP": "proximal enhancer-like",
    "enhD": "distal enhancer-like",
    "K4m3": "DNase-H3K4me3",
    "CTCF": "CTCF-only",
}


@dataclass
class RegulatoryElement:
    """A regulatory element near or overlapping the variant.

    Attributes:
        source: Data source (``"ENCODE-SCREEN"`` or ``"Ensembl"``).
        element_id: Element accession/id.
        element_type: Class label (e.g. ``"enhD"`` or an Ensembl feature type).
        description: Human-readable description.
        start: 0-based start (UCSC convention).
        end: 0-based exclusive end.
        distance: bp from the variant to the element; ``0`` if inside.
    """

    source: str
    element_id: str
    element_type: str
    description: str
    start: int
    end: int
    distance: int

    @property
    def overlaps(self) -> bool:
        """Whether the variant falls within this element."""
        return self.distance == 0

    @property
    def type_label(self) -> str:
        """Friendly class label where known, else the raw type."""
        return _CCRE_LABELS.get(self.element_type, self.element_type)


@dataclass
class RegulatoryContextResult:
    """Regulatory-element context for a variant.

    Attributes:
        variant: The variant analyzed.
        elements: Nearby elements, sorted by distance (closest first).
        nearest: The closest element, or ``None`` if none found in the window.
    """

    variant: Variant
    elements: list[RegulatoryElement] = field(default_factory=list)
    nearest: RegulatoryElement | None = None

    @property
    def in_ccre(self) -> bool:
        """Whether the variant overlaps an ENCODE SCREEN cCRE."""
        return any(e.overlaps and e.source == "ENCODE-SCREEN" for e in self.elements)

    def summary(self) -> str:
        """A one-line human-readable regulatory summary."""
        if self.nearest is None:
            return f"{self.variant}: no regulatory element within the search window."
        n = self.nearest
        if n.overlaps:
            return f"{self.variant}: inside a {n.type_label} element ({n.element_id})."
        return (
            f"{self.variant}: not in a cCRE; nearest is a {n.type_label} element "
            f"({n.element_id}) {n.distance:,} bp away."
        )


def _ensembl_chrom(chrom: str) -> str:
    """Convert ``chrN`` to Ensembl's bare contig form (``N``)."""
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def _distance0(pos0: int, start: int, end: int) -> int:
    """Distance from a 0-based position to a [start, end) interval (0 if inside)."""
    if start <= pos0 < end:
        return 0
    return min(abs(pos0 - start), abs(pos0 - (end - 1)))


def encode_ccres(
    variant: Variant, client: HttpClient | None = None, window: int = 3000
) -> list[RegulatoryElement]:
    """Fetch ENCODE SCREEN cCREs near the variant from the UCSC track API.

    Args:
        variant: The variant to search around.
        client: HTTP client; defaults to the shared urllib client.
        window: Half-width (bp) of the search region.

    Returns:
        cCREs near the variant (unsorted).
    """
    http = resolve_client(client)
    pos0 = variant.pos - 1  # UCSC track coords are 0-based
    params = {
        "genome": "hg38",
        "track": "encodeCcreCombined",
        "chrom": variant.chrom,
        "start": max(0, pos0 - window),
        "end": pos0 + window,
    }
    payload = http.get_json(UCSC_TRACK, params)
    rows = payload.get("encodeCcreCombined", []) if isinstance(payload, dict) else []
    elements = []
    for r in rows:
        start, end = int(r.get("chromStart", 0)), int(r.get("chromEnd", 0))
        elements.append(
            RegulatoryElement(
                source="ENCODE-SCREEN",
                element_id=str(r.get("name", "")),
                element_type=str(r.get("ucscLabel", "")),
                description=str(r.get("description", "")),
                start=start,
                end=end,
                distance=_distance0(pos0, start, end),
            )
        )
    return elements


def ensembl_regulatory(
    variant: Variant, client: HttpClient | None = None, window: int = 3000
) -> list[RegulatoryElement]:
    """Fetch Ensembl Regulatory Build features near the variant (secondary source).

    Args:
        variant: The variant to search around.
        client: HTTP client; defaults to the shared urllib client.
        window: Half-width (bp) of the search region.

    Returns:
        Regulatory features near the variant (may be empty; the build does not
        annotate every locus).
    """
    http = resolve_client(client)
    chrom = _ensembl_chrom(variant.chrom)
    pos0 = variant.pos - 1
    start = max(1, variant.pos - window)
    url = f"{ENSEMBL_OVERLAP}/{chrom}:{start}-{variant.pos + window}"
    records = http.get_json(url, {"feature": "regulatory", "content-type": "application/json"})
    elements = []
    for r in records or []:
        # Ensembl overlap uses 1-based coords; normalize to 0-based like UCSC.
        s0, e0 = int(r.get("start", 0)) - 1, int(r.get("end", 0))
        elements.append(
            RegulatoryElement(
                source="Ensembl",
                element_id=str(r.get("id", "")),
                element_type=str(r.get("feature_type", "")),
                description=str(r.get("description", "")),
                start=s0,
                end=e0,
                distance=_distance0(pos0, s0, e0),
            )
        )
    return elements


def regulatory_context(
    variant: Variant,
    client: HttpClient | None = None,
    window: int = 3000,
    include_ensembl: bool = True,
) -> RegulatoryContextResult:
    """Assemble regulatory-element context for a variant.

    Args:
        variant: The variant to analyze.
        client: HTTP client; defaults to the shared urllib client.
        window: Half-width (bp) of the search region.
        include_ensembl: Also query the Ensembl Regulatory Build (secondary).

    Returns:
        A :class:`RegulatoryContextResult` with elements sorted by distance.
    """
    elements = encode_ccres(variant, client=client, window=window)
    if include_ensembl:
        elements += ensembl_regulatory(variant, client=client, window=window)
    elements.sort(key=lambda e: e.distance)
    return RegulatoryContextResult(
        variant=variant, elements=elements, nearest=elements[0] if elements else None
    )
