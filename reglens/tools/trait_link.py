"""Link a variant to GWAS traits via the GWAS Catalog REST API.

The "which trait" half of the mechanism: retrieve real, published trait
associations (EFO trait, p-value, effect size) for a variant's rsID from the
NHGRI-EBI GWAS Catalog. All numbers are retrieved, never invented; the reasoning
layer interprets direction/relevance.

Open Targets Genetics (L2G, fine-mapping) is a natural richer follow-up but needs a
GraphQL client; the GWAS Catalog REST API is the license-clean MVP that already
recovers the key associations (e.g. rs1427407 → fetal hemoglobin, p=4e-53).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reglens.tools._http import HttpClient, resolve_client

GWAS_CATALOG_SNP = "https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms"


@dataclass
class TraitAssociation:
    """A single GWAS Catalog association for a variant.

    Attributes:
        traits: EFO trait name(s) reported for this association.
        p_value: Association p-value.
        beta: Effect size (beta), if reported as a beta.
        beta_unit: Unit of the beta (e.g. ``"unit increase"``).
        beta_direction: ``"increase"`` / ``"decrease"``, if reported.
        or_per_copy: Odds ratio per copy, if reported instead of a beta.
        risk_allele: Reported risk/effect allele, if available.
    """

    traits: list[str]
    p_value: float
    beta: float | None = None
    beta_unit: str | None = None
    beta_direction: str | None = None
    or_per_copy: float | None = None
    risk_allele: str | None = None

    def effect_str(self) -> str:
        """Human-readable effect size (beta or OR), or an empty string."""
        if self.beta is not None:
            unit = f" {self.beta_unit}" if self.beta_unit else ""
            direction = f" ({self.beta_direction})" if self.beta_direction else ""
            return f"β={self.beta}{unit}{direction}"
        if self.or_per_copy is not None:
            return f"OR={self.or_per_copy}"
        return ""


@dataclass
class TraitLinkResult:
    """GWAS trait associations for a variant.

    Attributes:
        rsid: The queried rsID.
        associations: Associations, most significant (smallest p) first.
    """

    rsid: str
    associations: list[TraitAssociation] = field(default_factory=list)

    def unique_traits(self) -> list[str]:
        """Distinct trait names, preserving most-significant-first order."""
        seen: dict[str, None] = {}
        for assoc in self.associations:
            for trait in assoc.traits:
                seen.setdefault(trait, None)
        return list(seen)

    def summary(self) -> str:
        """A one-line summary: top trait, its p-value, and how many traits total."""
        if not self.associations:
            return f"{self.rsid}: no GWAS Catalog associations."
        top = self.associations[0]
        traits = "; ".join(top.traits) or "(unnamed trait)"
        n = len(self.unique_traits())
        more = f" (+{n - 1} more trait{'s' if n - 1 != 1 else ''})" if n > 1 else ""
        return f"{self.rsid}: {traits} (p={top.p_value:g}){more}"


def _parse_association(record: dict) -> TraitAssociation:
    """Convert a GWAS Catalog association record into a :class:`TraitAssociation`."""
    traits = [t.get("trait", "") for t in record.get("efoTraits", []) if t.get("trait")]
    # A risk allele may be nested under loci→strongestRiskAlleles.
    risk_allele = None
    for locus in record.get("loci", []):
        for allele in locus.get("strongestRiskAlleles", []):
            risk_allele = allele.get("riskAlleleName")
            break
        if risk_allele:
            break
    return TraitAssociation(
        traits=traits,
        p_value=float(record.get("pvalue", 1.0) or 1.0),
        beta=float(record["betaNum"]) if record.get("betaNum") is not None else None,
        beta_unit=record.get("betaUnit"),
        beta_direction=record.get("betaDirection"),
        or_per_copy=float(record["orPerCopyNum"]) if record.get("orPerCopyNum") else None,
        risk_allele=risk_allele,
    )


def trait_link(
    rsid: str, client: HttpClient | None = None, max_associations: int | None = None
) -> TraitLinkResult:
    """Retrieve GWAS Catalog trait associations for a variant by rsID.

    Args:
        rsid: dbSNP rsID (e.g. ``"rs1427407"``).
        client: HTTP client; defaults to the shared urllib client.
        max_associations: If set, keep only the top-N most significant associations.

    Returns:
        A :class:`TraitLinkResult` with associations sorted by ascending p-value.
    """
    http = resolve_client(client)
    url = f"{GWAS_CATALOG_SNP}/{rsid}/associations"
    payload = http.get_json(url, {"projection": "associationBySnp"})

    records = (payload or {}).get("_embedded", {}).get("associations", [])
    associations = [_parse_association(r) for r in records]
    associations.sort(key=lambda a: a.p_value)
    if max_associations is not None:
        associations = associations[:max_associations]
    return TraitLinkResult(rsid=rsid, associations=associations)
