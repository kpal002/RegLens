"""Retrieve real, citable literature for a variant/gene/TF/trait via Europe PMC.

This tool backs RegLens golden rule #4 — *cite only retrieved evidence*. Every
:class:`Citation` it returns comes from a live Europe PMC query and carries a real
PMID (or other source id), so the reasoning layer can attach citations it never
invented. No LLM here: just a query and structured results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reglens.genome import Variant
from reglens.tools._http import HttpClient, resolve_client

# Europe PMC REST search endpoint (open, no key required).
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass
class Citation:
    """A single literature reference retrieved from Europe PMC.

    Attributes:
        europepmc_id: Europe PMC record id.
        source: Europe PMC source database (e.g. ``"MED"`` for PubMed/MEDLINE).
        pmid: PubMed id, if available.
        doi: DOI, if available.
        title: Article title.
        authors: Author string as returned by Europe PMC.
        journal: Journal title.
        year: Publication year.
        cited_by_count: Number of Europe PMC citations (rough impact proxy).
    """

    europepmc_id: str
    source: str
    pmid: str | None
    doi: str | None
    title: str
    authors: str
    journal: str
    year: str
    cited_by_count: int = 0

    @property
    def url(self) -> str:
        """A stable URL for the reference (PubMed if a PMID exists, else Europe PMC)."""
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return f"https://europepmc.org/article/{self.source}/{self.europepmc_id}"

    def format(self) -> str:
        """A compact one-line citation with a resolvable id."""
        ident = f"PMID:{self.pmid}" if self.pmid else f"{self.source}:{self.europepmc_id}"
        return f"{self.authors} ({self.year}). {self.title}. {self.journal}. {ident}"


@dataclass
class LiteratureResult:
    """The outcome of a Europe PMC search.

    Attributes:
        query: The exact query string sent to Europe PMC.
        hit_count: Total matches reported by Europe PMC (may exceed ``citations``).
        citations: The retrieved citations (up to the requested page size).
    """

    query: str
    hit_count: int
    citations: list[Citation] = field(default_factory=list)

    def pmids(self) -> list[str]:
        """PMIDs of the retrieved citations (skipping any without one)."""
        return [c.pmid for c in self.citations if c.pmid]


def _parse_result(record: dict) -> Citation:
    """Convert a Europe PMC result record into a :class:`Citation`."""
    return Citation(
        europepmc_id=str(record.get("id", "")),
        source=str(record.get("source", "")),
        pmid=str(record["pmid"]) if record.get("pmid") else None,
        doi=record.get("doi"),
        title=str(record.get("title", "")).rstrip("."),
        authors=str(record.get("authorString", "")),
        journal=str(record.get("journalTitle", "")),
        year=str(record.get("pubYear", "")),
        cited_by_count=int(record.get("citedByCount", 0) or 0),
    )


def search_literature(
    query: str,
    client: HttpClient | None = None,
    page_size: int = 5,
    sort_by_cited: bool = True,
) -> LiteratureResult:
    """Search Europe PMC and return structured citations.

    Args:
        query: A Europe PMC query string (e.g. ``'rs1427407 AND BCL11A'``).
        client: HTTP client; defaults to the shared urllib client.
        page_size: Maximum number of citations to return.
        sort_by_cited: If True, request most-cited first (helps surface key papers).

    Returns:
        A :class:`LiteratureResult`.
    """
    http = resolve_client(client)
    params: dict[str, object] = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "lite",
    }
    if sort_by_cited:
        params["sort"] = "CITED desc"
    payload = http.get_json(EUROPE_PMC_SEARCH, params)

    result_list = (payload or {}).get("resultList", {}).get("result", [])
    citations = [_parse_result(r) for r in result_list]
    return LiteratureResult(
        query=query,
        hit_count=int((payload or {}).get("hitCount", 0) or 0),
        citations=citations,
    )


def build_query(
    variant: Variant | None = None,
    rsid: str | None = None,
    gene: str | None = None,
    tf: str | None = None,
    trait: str | None = None,
    operator: str = "AND",
) -> str:
    """Build a Europe PMC query from the entities other tools have identified.

    Terms are joined with ``operator`` (default ``"AND"`` for specificity — an OR of
    common gene/TF names retrieves thousands of generic reviews and buries the
    variant-specific papers). Pass the 2–3 most specific entities the upstream tools
    found (e.g. rsID + gene) for the sharpest results.

    Args:
        variant: The variant (used only if no ``rsid`` is given, as a ``chr:pos`` term).
        rsid: dbSNP rsID, if known (the most specific term).
        gene: Target gene symbol (e.g. ``"BCL11A"``).
        tf: Transcription factor name (e.g. ``"GATA1"``).
        trait: Associated trait (e.g. ``"fetal hemoglobin"``).
        operator: How to join terms — ``"AND"`` (specific) or ``"OR"`` (broad recall).

    Returns:
        A Europe PMC query string.

    Raises:
        ValueError: If no search terms are supplied, or ``operator`` is invalid.
    """
    if operator not in {"AND", "OR"}:
        raise ValueError(f"operator must be 'AND' or 'OR', got {operator!r}")
    terms: list[str] = []
    if rsid:
        terms.append(rsid)
    elif variant is not None:
        terms.append(f'"{variant.chrom}:{variant.pos}"')
    for term in (gene, tf, trait):
        if term:
            # Quote multi-word terms so Europe PMC treats them as a phrase.
            terms.append(f'"{term}"' if " " in term else term)
    if not terms:
        raise ValueError("build_query needs at least one of rsid/variant/gene/tf/trait.")
    return f" {operator} ".join(terms)
