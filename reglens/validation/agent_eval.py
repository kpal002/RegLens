"""Agent validation: known-mechanism recovery, architecture ablation, confidence calibration.

Three experiments that validate the **reasoning layer** (not the engine), complementing the
null control in :mod:`reglens.validation.null_control`:

- **Recovery** (:func:`run_recovery`) — on a curated set of *characterized* regulatory
  variants, does the agent recover the established **TF / gene / trait**? Turns two
  hand-picked demos into "recovered the correct TF in N/M".
- **Ablation** (:func:`run_ablation`) — single-agent vs multi-agent−redteam vs full
  multi-agent on the *same* bundles: what does the architecture buy? Does the red-team
  correctly lower confidence on weak/null cases?
- **Calibration** (:func:`calibration_table`) — does confidence (high/med/low) track
  evidence strength across strata (strong known mechanism / weak effect / null)?

Everything reuses the deterministic tools and the interpreters unchanged; nothing here
computes or invents a number. Variant coordinates are resolved from the rsID via Ensembl
(:func:`resolve_variant`) so the curated set carries no hand-typed hg38 positions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reglens.genome import Variant
from reglens.orchestrator import analyze_variant
from reglens.tools._http import HttpClient, resolve_client

ENSEMBL_VARIATION = "https://rest.ensembl.org/variation/human"


# --------------------------------------------------------------------------------------
# Curated known-mechanism set.
#
# Each is a well-characterized *noncoding regulatory* variant whose disrupted/created TF,
# target gene, and trait are established in the cited literature. The scorer only checks
# that the agent *names* the TF/gene/trait (substring, with aliases) — direction is a
# bonus, read from the transcript. Coordinates are resolved from the rsID at run time.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KnownMechanism:
    """A characterized regulatory variant and its established mechanism.

    Attributes:
        rsid: dbSNP rsID (used to resolve the hg38 coordinate and drive eQTL/GWAS/lit).
        gene: Established target gene.
        trait: Established trait/phenotype.
        pmid: A primary reference PMID.
        tf: Established disrupted/created TF (``None`` if the mechanism is not a single
            named TF — then TF recovery is not scored for this variant).
        celltype: The cell type in which the mechanism acts (for context).
        variant: Optional explicit hg38 variant (skips rsID resolution).
        alt: Functional alt allele to pin when the rsID is multi-allelic (else the
            resolver takes Ensembl's first alt).
        tf_aliases / gene_aliases / trait_terms: Extra strings that count as a recovery.
        note: One-line description of the mechanism.
    """

    rsid: str
    gene: str
    trait: str
    pmid: str
    tf: str | None = None
    celltype: str = ""
    variant: Variant | None = None
    alt: str | None = None
    tf_aliases: tuple[str, ...] = ()
    gene_aliases: tuple[str, ...] = ()
    trait_terms: tuple[str, ...] = ()
    note: str = ""


#: Curated characterized regulatory variants (established mechanisms; checkable by PMID).
KNOWN_MECHANISMS: list[KnownMechanism] = [
    KnownMechanism(
        "rs1427407", "BCL11A", "fetal hemoglobin", "24115442", tf="GATA1",
        celltype="K562", alt="G", tf_aliases=("TAL1", "GATA1::TAL1"),
        trait_terms=("fetal hemoglobin", "HbF", "F-cell", "hemoglobin"),
        note="Alt G disrupts a GATA1/TAL1 site in the BCL11A +58 erythroid enhancer.",
    ),
    KnownMechanism(
        "rs2814778", "ACKR1", "neutrophil count", "7663520", tf="GATA1",
        celltype="K562", gene_aliases=("DARC", "Duffy"),
        trait_terms=("Duffy", "neutrophil", "malaria", "blood group"),
        note="Alt abolishes a GATA1 site in the Duffy/ACKR1 erythroid promoter.",
    ),
    KnownMechanism(
        "rs12740374", "SORT1", "LDL cholesterol", "20686566", tf="CEBP",
        celltype="HepG2", tf_aliases=("C/EBP", "CEBPA", "CCAAT"),
        trait_terms=("LDL", "cholesterol", "low-density", "lipid"),
        note="Minor allele creates a C/EBP site, raising hepatic SORT1 expression.",
    ),
    KnownMechanism(
        "rs6983267", "MYC", "colorectal cancer", "19680224", tf="TCF7L2",
        gene_aliases=("MYC",), tf_aliases=("TCF4", "TCF-4", "Wnt"),
        trait_terms=("colorectal", "colon", "cancer", "prostate"),
        note="Allele alters a TCF7L2/Wnt-responsive enhancer 335 kb from MYC (8q24).",
    ),
    KnownMechanism(
        "rs4988235", "LCT", "lactase persistence", "14634648", tf="POU2F1",
        gene_aliases=("MCM6",), tf_aliases=("Oct-1", "Oct1", "OCT-1"),
        trait_terms=("lactase", "lactose", "milk"),
        note="-13910 T enhances an Oct-1 (POU2F1) enhancer driving LCT persistence.",
    ),
    KnownMechanism(
        "rs1421085", "IRX3", "body mass index", "26287746", tf="ARID5B",
        gene_aliases=("IRX5", "FTO"),
        trait_terms=("obesity", "BMI", "body mass", "adipocyte"),
        note="T>C disrupts an ARID5B repressor motif, derepressing IRX3/IRX5 (FTO locus).",
    ),
    KnownMechanism(
        "rs12821256", "KITLG", "hair color", "24647431", tf="LEF1",
        trait_terms=("hair", "blond", "pigment"),
        note="Allele alters a LEF1 site in a KITLG enhancer affecting hair color.",
    ),
    KnownMechanism(
        "rs2168101", "LMO1", "neuroblastoma", "26466567", tf="GATA3",
        trait_terms=("neuroblastoma", "tumor", "cancer"),
        note="G>T disrupts a GATA3 site in a LMO1 super-enhancer (neuroblastoma).",
    ),
    KnownMechanism(
        "rs339331", "RFX6", "prostate cancer", "24740154", tf="HOXB13",
        trait_terms=("prostate", "cancer"),
        note="T allele creates a HOXB13 site increasing RFX6 in prostate cancer.",
    ),
    KnownMechanism(
        "rs6801957", "SCN5A", "cardiac conduction", "22705117", tf="TBX5",
        gene_aliases=("SCN10A",), tf_aliases=("T-box", "TBX3"),
        trait_terms=("conduction", "QRS", "PR interval", "cardiac", "arrhythmia"),
        note="Allele weakens a TBX5 site in an SCN5A/SCN10A enhancer (conduction).",
    ),
    KnownMechanism(
        "rs4784227", "TOX3", "breast cancer", "21372275", tf="FOXA1",
        trait_terms=("breast", "cancer"),
        note="Risk allele strengthens FOXA1 binding at a TOX3 enhancer (breast cancer).",
    ),
]


def resolve_variant(
    rsid: str, client: HttpClient | None = None, prefer_alt: str | None = None
) -> Variant:
    """Resolve an rsID to its hg38 (GRCh38) :class:`Variant` via Ensembl REST.

    Args:
        rsid: dbSNP rsID (e.g. ``"rs1427407"``).
        client: HTTP client; defaults to the shared urllib client.
        prefer_alt: For a multi-allelic rsID, pick this alt allele if present; otherwise
            the first alt in Ensembl's ``allele_string`` is used.

    Returns:
        The variant on GRCh38 (chr-prefixed to match the benchmark/genome).

    Raises:
        ValueError: If no GRCh38 SNV mapping is found.
    """
    http = resolve_client(client)
    data = http.get_json(
        f"{ENSEMBL_VARIATION}/{rsid}", {"content-type": "application/json"}
    )
    for mapping in (data or {}).get("mappings", []):
        if mapping.get("assembly_name") != "GRCh38":
            continue
        alleles = str(mapping.get("allele_string", "")).split("/")
        if len(alleles) < 2:
            continue
        ref, alts = alleles[0], alleles[1:]
        alt = prefer_alt if (prefer_alt and prefer_alt in alts) else alts[0]
        chrom = str(mapping.get("seq_region_name", ""))
        chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        return Variant(chrom=chrom, pos=int(mapping["start"]), ref=ref, alt=alt)
    raise ValueError(f"No GRCh38 mapping found for {rsid}")


# --------------------------------------------------------------------------------------
# Recovery scoring
# --------------------------------------------------------------------------------------
def _named(target: str, aliases: tuple[str, ...], *texts: str) -> bool:
    """Whether ``target`` (or an alias) appears in any of ``texts`` (case-insensitive)."""
    hay = " ".join(t for t in texts if t).lower()
    return any(term and term.lower() in hay for term in (target, *aliases))


@dataclass
class RecoveryResult:
    """Whether the agent recovered the known TF / gene / trait for one variant."""

    known: KnownMechanism
    interpretation: Any
    tf_recovered: bool | None
    gene_recovered: bool
    trait_recovered: bool
    confidence: str = "low"
    result: Any = None


def score_recovery(interpretation: Any, known: KnownMechanism) -> RecoveryResult:
    """Score whether an interpretation recovers ``known``'s TF/gene/trait.

    Matching is substring-based over the structured field **and** the mechanism prose
    (a human should still read the transcript). TF recovery is ``None`` when the known
    mechanism has no single named TF.

    Args:
        interpretation: The agent's interpretation.
        known: The curated known mechanism.

    Returns:
        A :class:`RecoveryResult`.
    """
    mech = getattr(interpretation, "mechanism", "") or ""
    tf_field = getattr(interpretation, "tf", "") or ""
    gene_field = getattr(interpretation, "gene", "") or ""
    trait_field = getattr(interpretation, "trait", "") or ""

    tf_recovered: bool | None = None
    if known.tf is not None:
        tf_recovered = _named(known.tf, known.tf_aliases, tf_field, mech)
    return RecoveryResult(
        known=known,
        interpretation=interpretation,
        tf_recovered=tf_recovered,
        gene_recovered=_named(known.gene, known.gene_aliases, gene_field, mech),
        trait_recovered=_named(known.trait, known.trait_terms, trait_field, mech),
        confidence=(getattr(interpretation, "confidence", "low") or "low").lower(),
        result=getattr(interpretation, "_result", None),
    )


def run_recovery(
    interpreter: Any,
    knowns: list[KnownMechanism] | None = None,
    *,
    scorer: Any = None,
    genome_path: str | None = None,
    client: HttpClient | None = None,
    resolve: bool = True,
    progress: bool = False,
) -> list[RecoveryResult]:
    """Run the agent on the curated known-mechanism set and score TF/gene/trait recovery.

    Args:
        interpreter: A multi-agent interpreter (``deliberate`` used when available).
        knowns: Known mechanisms to test (defaults to :data:`KNOWN_MECHANISMS`).
        scorer: ChromBPNet scorer (with ``genome_path``) for the faithful run.
        genome_path: hg38 FASTA path.
        client: HTTP client for the annotation tools + rsID resolution.
        resolve: Resolve each rsID to its hg38 coordinate via Ensembl (recommended).
        progress: Print a per-variant marker.

    Returns:
        One :class:`RecoveryResult` per known mechanism.
    """
    knowns = knowns if knowns is not None else KNOWN_MECHANISMS
    results: list[RecoveryResult] = []
    for i, km in enumerate(knowns, 1):
        variant = km.variant or (
            resolve_variant(km.rsid, client, prefer_alt=km.alt) if resolve else None
        )
        if variant is None:
            raise ValueError(f"No coordinate for {km.rsid}; set resolve=True or km.variant")
        bundle = analyze_variant(
            variant, rsid=km.rsid, celltype=km.celltype or None,
            genome_path=genome_path, scorer=scorer, client=client,
        )
        if hasattr(interpreter, "deliberate"):
            result = interpreter.deliberate(bundle)
            interp = result.interpretation
        else:
            result = None
            interp = interpreter.interpret(bundle)
        rec = score_recovery(interp, km)
        rec.result = result
        results.append(rec)
        if progress:
            got = "".join(
                m for m, ok in (("T", rec.tf_recovered), ("G", rec.gene_recovered),
                                ("R", rec.trait_recovered)) if ok
            ) or "-"
            print(f"  [{i}/{len(knowns)}] {km.rsid} ({km.gene}) recovered={got}")
    return results


def recovery_rates(results: list[RecoveryResult]) -> dict[str, str]:
    """Fraction recovering TF / gene / trait (TF over records that have a known TF)."""
    tf_scored = [r for r in results if r.tf_recovered is not None]
    return {
        "tf": f"{sum(bool(r.tf_recovered) for r in tf_scored)}/{len(tf_scored)}",
        "gene": f"{sum(r.gene_recovered for r in results)}/{len(results)}",
        "trait": f"{sum(r.trait_recovered for r in results)}/{len(results)}",
    }


def render_recovery(results: list[RecoveryResult]) -> str:
    """Render a per-variant recovery table + aggregate rates."""
    lines = ["── Known-mechanism recovery " + "─" * 30,
             f"  {'rsID':12s} {'gene':8s} {'TF':10s}  tf gene trait  conf"]
    for r in results:
        km = r.known
        mark = lambda b: "  ✓" if b else ("  ·" if b is not None else "  –")  # noqa: E731
        lines.append(
            f"  {km.rsid:12s} {km.gene:8s} {str(km.tf or '-'):10s} "
            f"{mark(r.tf_recovered)}{mark(r.gene_recovered)}{mark(r.trait_recovered)}"
            f"   {r.confidence}"
        )
    rates = recovery_rates(results)
    lines.append(f"  → TF {rates['tf']}   gene {rates['gene']}   trait {rates['trait']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Architecture ablation: single vs multi−redteam vs full multi-agent
# --------------------------------------------------------------------------------------
@dataclass
class AblationRow:
    """One variant scored by all three interpreter configurations (same bundle)."""

    label: str
    stratum: str
    single: Any
    multi_no_redteam: Any
    multi_full: Any


def _conf(interp: Any) -> str:
    return (getattr(interp, "confidence", "low") or "low").lower()


def run_ablation(
    items: list[tuple[Variant, str | None, str, str]],
    *,
    single: Any,
    multi_no_redteam: Any,
    multi_full: Any,
    scorer: Any = None,
    genome_path: str | None = None,
    client: HttpClient | None = None,
    progress: bool = False,
) -> list[AblationRow]:
    """Score each variant with three interpreter configs over the **same** evidence bundle.

    Build the deterministic bundle once per variant, then run single-agent,
    multi-agent−redteam, and full multi-agent on it — a fair, cheap comparison isolating
    what the architecture adds.

    Args:
        items: ``(variant, rsid, label, stratum)`` tuples; ``stratum`` is one of
            ``"strong"`` / ``"weak"`` / ``"null"``.
        single: A single-agent interpreter (``ClaudeInterpreter``).
        multi_no_redteam: ``MultiAgentInterpreter(redteam=False)``.
        multi_full: ``MultiAgentInterpreter(redteam=True)``.
        scorer: ChromBPNet scorer; genome_path enables the sequence signals.
        genome_path: hg38 FASTA path.
        client: HTTP client for the annotation tools.
        progress: Print a per-variant marker.

    Returns:
        One :class:`AblationRow` per item.
    """
    rows: list[AblationRow] = []
    for i, (variant, rsid, label, stratum) in enumerate(items, 1):
        bundle = analyze_variant(
            variant, rsid=rsid, genome_path=genome_path, scorer=scorer, client=client,
        )
        row = AblationRow(
            label=label, stratum=stratum,
            single=single.interpret(bundle),
            multi_no_redteam=multi_no_redteam.interpret(bundle),
            multi_full=multi_full.interpret(bundle),
        )
        rows.append(row)
        if progress:
            print(f"  [{i}/{len(items)}] {label} ({stratum}): "
                  f"single={_conf(row.single)} noRT={_conf(row.multi_no_redteam)} "
                  f"full={_conf(row.multi_full)}")
    return rows


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def render_ablation(rows: list[AblationRow]) -> str:
    """Render the ablation: per-variant confidences + whether the red-team lowered it.

    The headline question — does the red-team correctly *lower* confidence on weak/null
    cases? — is summarized per stratum as the count where full < no-redteam confidence.
    """
    lines = ["── Architecture ablation (confidence by config) " + "─" * 12,
             f"  {'variant':22s} {'stratum':7s}  single  noRT   full   redteam-effect"]
    lowered: Counter[str] = Counter()
    n_by_stratum: Counter[str] = Counter()
    for r in rows:
        s, nrt, full = _conf(r.single), _conf(r.multi_no_redteam), _conf(r.multi_full)
        delta = _CONF_RANK[full] - _CONF_RANK[nrt]
        effect = "↓ lowered" if delta < 0 else ("↑ raised" if delta > 0 else "= same")
        n_by_stratum[r.stratum] += 1
        if delta < 0:
            lowered[r.stratum] += 1
        lines.append(f"  {r.label:22s} {r.stratum:7s}  {s:6s} {nrt:6s} {full:6s}  {effect}")
    lines.append("  ── red-team lowered confidence (full < no-redteam):")
    for stratum in ("strong", "weak", "null"):
        if n_by_stratum.get(stratum):
            lines.append(f"     {stratum:7s}: {lowered.get(stratum, 0)}/{n_by_stratum[stratum]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Confidence calibration table
# --------------------------------------------------------------------------------------
CONFIDENCE_LEVELS = ("high", "medium", "low")
STRATA = ("strong", "weak", "null")


@dataclass
class CalibrationTable:
    """Confidence counts per stratum (rows = strata, cols = high/medium/low)."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Serialize to a plain nested dict."""
        return {s: dict(self.counts.get(s, {})) for s in self.counts}


def calibration_table(strata: dict[str, list[Any]]) -> CalibrationTable:
    """Tabulate confidence (high/med/low) per evidence stratum.

    Good calibration = strong→high, null→low: the honest form of "the agent knows what it
    doesn't know."

    Args:
        strata: ``{stratum: [interpretations]}`` — e.g. ``{"strong": [...], "weak": [...],
            "null": [...]}``.

    Returns:
        A :class:`CalibrationTable`.
    """
    counts: dict[str, dict[str, int]] = {}
    for stratum, interps in strata.items():
        c = Counter(_conf(i) for i in interps)
        counts[stratum] = {lvl: c.get(lvl, 0) for lvl in CONFIDENCE_LEVELS}
    return CalibrationTable(counts=counts)


def render_calibration(table: CalibrationTable) -> str:
    """Render the calibration table (strata × confidence)."""
    lines = ["── Confidence calibration " + "─" * 22,
             f"  {'stratum':10s} {'high':>6s} {'medium':>7s} {'low':>6s}"]
    for stratum in (*STRATA, *[s for s in table.counts if s not in STRATA]):
        row = table.counts.get(stratum)
        if row is None:
            continue
        lines.append(f"  {stratum:10s} {row['high']:6d} {row['medium']:7d} {row['low']:6d}")
    return "\n".join(lines)
