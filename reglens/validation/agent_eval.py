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


def _transition(rows: list[AblationRow], a: str, b: str) -> str:
    """Per-stratum ``lowered/raised`` counts for a confidence transition ``a → b``."""
    getter = {"single": lambda r: r.single, "noRT": lambda r: r.multi_no_redteam,
              "full": lambda r: r.multi_full}
    lowered: Counter[str] = Counter()
    raised: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for r in rows:
        n[r.stratum] += 1
        delta = _CONF_RANK[_conf(getter[b](r))] - _CONF_RANK[_conf(getter[a](r))]
        if delta < 0:
            lowered[r.stratum] += 1
        elif delta > 0:
            raised[r.stratum] += 1
    parts = [f"{s} {lowered.get(s, 0)}↓/{raised.get(s, 0)}↑ of {n[s]}"
             for s in (*STRATA, *[s for s in n if s not in STRATA]) if n.get(s)]
    return "   ".join(parts)


def render_ablation(rows: list[AblationRow]) -> str:
    """Render the ablation: per-variant confidences + what each layer buys.

    Reports three transitions so the multi-agent structure's effect isn't hidden behind
    the red-team's marginal one: single→multi (specialists+adjudicator), multi→+red-team,
    and the net single→full. De-escalation (↓) on weak/null and preservation on strong is
    the calibration signal.
    """
    lines = ["── Architecture ablation (confidence by config) " + "─" * 12,
             f"  {'variant':22s} {'stratum':7s}  single  noRT   full   net"]
    for r in rows:
        s, nrt, full = _conf(r.single), _conf(r.multi_no_redteam), _conf(r.multi_full)
        delta = _CONF_RANK[full] - _CONF_RANK[s]
        net = "↓ lowered" if delta < 0 else ("↑ raised" if delta > 0 else "= same")
        lines.append(f"  {r.label:22s} {r.stratum:7s}  {s:6s} {nrt:6s} {full:6s}  {net}")
    lines.append("  ── confidence change by stratum (lowered↓ / raised↑):")
    lines.append(f"     multi-agent (single→noRT): {_transition(rows, 'single', 'noRT')}")
    lines.append(f"     red-team    (noRT→full)  : {_transition(rows, 'noRT', 'full')}")
    lines.append(f"     net         (single→full): {_transition(rows, 'single', 'full')}")
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


# --------------------------------------------------------------------------------------
# Calibration benchmark with corroborating evidence (the HIGH-confidence regime).
#
# The MPRA calibration strata (strong/weak/null) are *synthetic saturation-mutagenesis*
# variants: they carry no rsID, hence no eQTL / GWAS / literature by construction, so the
# agent can structurally never reach **high** confidence on them (missing limbs). Only
# rs2814778 ever reached high in the whole suite — the high regime is essentially
# untested (n=1).
#
# This benchmark closes that gap with a curated ladder of *real hematopoietic* variants
# (K562-lineage, so the ChromBPNet signal is in-cell-type) that carry the corroborating
# limbs. Crucially the "should be high" tier can't be labeled a priori — the matched,
# *strong* ChromBPNet Δ is only known after scoring (rs1427407 is a real, fully-cited
# erythroid BCL11A variant that still lands medium because its Δ is small). So we don't
# hard-code an expected tier: we *measure* evidence completeness from the actual bundle
# (five channels) and validate that confidence tracks it — rises monotonically, and that
# **high appears only at full corroboration**. Both outcomes are honest: high on several
# fully-corroborated variants validates high-confidence calibration; high staying rare
# confirms the agent correctly reserves it for the strong-matched-signal case.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationVariant:
    """A real hematopoietic (K562-lineage) variant curated to carry corroborating limbs.

    Attributes:
        rsid: dbSNP rsID (resolves the hg38 coordinate and drives eQTL/GWAS/lit).
        gene: Nearby/target gene (context only; not scored).
        trait: Blood trait it associates with (context only).
        alt: Functional alt allele to pin when the rsID is multi-allelic.
        celltype: Scoring cell type — K562 (erythroid) matches these blood loci.
        note: One-line description.
    """

    rsid: str
    gene: str = ""
    trait: str = ""
    alt: str | None = None
    celltype: str = "K562"
    note: str = ""


#: Hematopoietic variants spanning a corroboration ladder. All are real noncoding
#: blood-trait loci (erythroid/HbF/RBC + a couple of megakaryocyte/platelet), so K562
#: gives an in-lineage ChromBPNet read and the rsIDs carry eQTL/GWAS/literature limbs.
#: Where each lands on the ladder is *measured* (see :func:`evidence_limbs`), not assumed.
HEMATOPOIETIC_CALIBRATION: list[CalibrationVariant] = [
    CalibrationVariant("rs2814778", "ACKR1", "neutrophil count / Duffy", alt="C",
                       note="Duffy-null GATA1 promoter site; strong erythroid signal"),
    CalibrationVariant("rs1427407", "BCL11A", "fetal hemoglobin", alt="G",
                       note="BCL11A +58 erythroid enhancer, GATA1/half-E-box"),
    CalibrationVariant("rs766432", "BCL11A", "fetal hemoglobin",
                       note="BCL11A intron-2 HbF-associated enhancer variant"),
    CalibrationVariant("rs11886868", "BCL11A", "fetal hemoglobin",
                       note="BCL11A HbF-associated regulatory variant"),
    CalibrationVariant("rs4895441", "HBS1L-MYB", "fetal hemoglobin / RBC",
                       note="HBS1L-MYB intergenic erythroid enhancer"),
    CalibrationVariant("rs9399137", "HBS1L-MYB", "red blood cell / hematocrit",
                       note="HBS1L-MYB erythroid enhancer cluster"),
    CalibrationVariant("rs7776054", "HBS1L-MYB", "red blood cell count",
                       note="HBS1L-MYB erythroid regulatory variant"),
    CalibrationVariant("rs1175550", "SMIM1", "red blood cell count",
                       note="SMIM1 (Vel) erythroid enhancer, GATA1-regulated"),
    CalibrationVariant("rs737092", "RBM38", "red blood cell indices",
                       note="RBM38 regulatory variant, RBC size/count"),
    CalibrationVariant("rs342293", "PIK3CG", "platelet count",
                       note="7q22.3 megakaryocyte enhancer (off-erythroid lineage)"),
    CalibrationVariant("rs1354034", "ARHGEF3", "platelet count",
                       note="ARHGEF3 megakaryocyte regulatory variant"),
]

#: The five corroborating evidence channels the agent can weight into confidence.
EVIDENCE_CHANNELS = ("chrombpnet", "motif", "eqtl", "gwas", "literature")


def evidence_limbs(
    bundle: Any, *, min_delta: float = 0.30, sig_alpha: float = 0.05
) -> dict[str, bool]:
    """Which corroborating limbs a bundle actually carries (measured, not assumed).

    Five independent channels, each a boolean:

    - ``chrombpnet``: a *strong* matched-cell-type accessibility change
      (``|Δ log-counts| >= min_delta``) — the limb the MPRA strata can have but the one
      that gates ``high`` in practice.
    - ``motif``: a *significant* motif hit (a top hit whose empirical ``p_value`` clears
      ``sig_alpha``; if the library reports no p-value, presence of a top hit suffices).
    - ``eqtl`` / ``gwas`` / ``literature``: at least one GTEx eQTL / GWAS association /
      literature citation — the rsID-keyed limbs the synthetic MPRA variants lack.

    Args:
        bundle: An :class:`~reglens.report.schema.EvidenceBundle`.
        min_delta: Threshold on ``|Δ log-counts|`` for a "strong" ChromBPNet signal.
        sig_alpha: Significance threshold on the motif hit's empirical ``p_value``.

    Returns:
        ``{channel: present}`` over :data:`EVIDENCE_CHANNELS`.
    """
    cp = getattr(bundle, "chrombpnet", None)
    strong_cp = bool(cp is not None and abs(cp.delta_log_counts) >= min_delta)

    motif = getattr(bundle, "motif", None)
    top = getattr(motif, "top", None) if motif is not None else None
    if top is None:
        sig_motif = False
    else:
        p = getattr(top, "p_value", None)
        sig_motif = True if p is None else (p <= sig_alpha)

    gene = getattr(bundle, "gene", None)
    trait = getattr(bundle, "trait", None)
    lit = getattr(bundle, "literature", None)
    return {
        "chrombpnet": strong_cp,
        "motif": sig_motif,
        "eqtl": bool(gene is not None and getattr(gene, "eqtls", None)),
        "gwas": bool(trait is not None and getattr(trait, "associations", None)),
        "literature": bool(lit is not None and getattr(lit, "hit_count", 0)),
    }


def _concordant(bundle: Any) -> bool:
    """Whether the motif Δ and the ChromBPNet Δ point the same way (both signed)."""
    cp = getattr(bundle, "chrombpnet", None)
    motif = getattr(bundle, "motif", None)
    top = getattr(motif, "top", None) if motif is not None else None
    if cp is None or top is None:
        return False
    return top.delta_score * cp.delta_log_counts > 0


@dataclass
class CalibrationOutcome:
    """One benchmark variant: measured evidence completeness vs the agent's confidence."""

    rsid: str
    gene: str
    variant: Variant
    limbs: dict[str, bool]
    completeness: int
    concordant: bool
    confidence: str
    interpretation: Any
    result: Any = None


def run_calibration_benchmark(
    interpreter: Any,
    variants: list[CalibrationVariant] | None = None,
    *,
    scorer: Any = None,
    genome_path: str | None = None,
    client: HttpClient | None = None,
    min_delta: float = 0.30,
    sig_alpha: float = 0.05,
    progress: bool = False,
) -> list[CalibrationOutcome]:
    """Run the agent on the corroboration ladder and pair confidence with evidence limbs.

    For each variant: resolve the rsID, build the deterministic bundle (which fetches the
    eQTL/GWAS/literature limbs live), run the interpreter, and record the *measured*
    completeness (:func:`evidence_limbs`) alongside the agent's confidence.

    Args:
        interpreter: Multi-agent interpreter (``deliberate`` used when available).
        variants: Ladder to run (defaults to :data:`HEMATOPOIETIC_CALIBRATION`).
        scorer: ChromBPNet scorer (with ``genome_path``) for the matched signal.
        genome_path: hg38 FASTA path.
        client: HTTP client for the annotation tools + rsID resolution.
        min_delta: "Strong ChromBPNet" threshold passed to :func:`evidence_limbs`.
        sig_alpha: Motif significance threshold passed to :func:`evidence_limbs`.
        progress: Print a per-variant marker.

    Returns:
        One :class:`CalibrationOutcome` per variant.
    """
    variants = variants if variants is not None else HEMATOPOIETIC_CALIBRATION
    outcomes: list[CalibrationOutcome] = []
    for i, cv in enumerate(variants, 1):
        variant = resolve_variant(cv.rsid, client, prefer_alt=cv.alt)
        bundle = analyze_variant(
            variant, rsid=cv.rsid, celltype=cv.celltype or None,
            genome_path=genome_path, scorer=scorer, client=client,
        )
        if hasattr(interpreter, "deliberate"):
            result = interpreter.deliberate(bundle)
            interp = result.interpretation
        else:
            result = None
            interp = interpreter.interpret(bundle)
        limbs = evidence_limbs(bundle, min_delta=min_delta, sig_alpha=sig_alpha)
        outcomes.append(CalibrationOutcome(
            rsid=cv.rsid, gene=cv.gene, variant=variant,
            limbs=limbs, completeness=sum(limbs.values()),
            concordant=_concordant(bundle), confidence=_conf(interp),
            interpretation=interp, result=result,
        ))
        if progress:
            got = "".join(k[0].upper() for k in EVIDENCE_CHANNELS if limbs[k]) or "-"
            print(f"  [{i}/{len(variants)}] {cv.rsid} ({cv.gene}) "
                  f"limbs={got} ({sum(limbs.values())}/5) conf={_conf(interp)}")
    return outcomes


def calibration_benchmark_summary(outcomes: list[CalibrationOutcome]) -> dict[str, Any]:
    """Validate that confidence tracks measured evidence completeness.

    Two checks make up the calibration claim:

    - **monotone**: mean completeness is non-increasing down high → medium → low (more
      corroboration ⇒ higher confidence).
    - **high_at_full_only**: every ``high`` call sits at full corroboration (5/5) — the
      agent does not over-call ``high`` on a partial bundle. ``None`` if no ``high`` calls.

    Returns:
        A dict with per-tier mean completeness, the two boolean checks, and counts.
    """
    by_tier = {lvl: [o.completeness for o in outcomes if o.confidence == lvl]
               for lvl in CONFIDENCE_LEVELS}
    means = {lvl: (sum(v) / len(v) if v else None) for lvl, v in by_tier.items()}

    present = [means[lvl] for lvl in CONFIDENCE_LEVELS if means[lvl] is not None]
    monotone = all(a >= b for a, b in zip(present, present[1:], strict=False))

    highs = by_tier["high"]
    max_complete = max((o.completeness for o in outcomes), default=0)
    high_at_full_only = None if not highs else all(c == max_complete for c in highs)

    return {
        "n": len(outcomes),
        "mean_completeness": means,
        "counts": {lvl: len(by_tier[lvl]) for lvl in CONFIDENCE_LEVELS},
        "monotone": monotone,
        "high_at_full_only": high_at_full_only,
        "max_completeness": max_complete,
    }


def render_calibration_benchmark(outcomes: list[CalibrationOutcome]) -> str:
    """Render the corroboration ladder: per-variant limbs + the two calibration checks."""
    hdr = "".join(f"{c[:4]:>5s}" for c in EVIDENCE_CHANNELS)
    lines = ["── Calibration benchmark (hematopoietic corroboration ladder) " + "─" * 6,
             f"  {'rsID':12s} {'gene':10s}{hdr}  cc  limbs  conf"]
    for o in sorted(outcomes, key=lambda x: (-x.completeness, x.rsid)):
        cells = "".join(f"{'  ✓' if o.limbs[c] else '  ·':>5s}" for c in EVIDENCE_CHANNELS)
        cc = "✓" if o.concordant else "·"
        lines.append(f"  {o.rsid:12s} {o.gene:10s}{cells}   {cc}  {o.completeness}/5   "
                     f"{o.confidence}")

    s = calibration_benchmark_summary(outcomes)
    mc = s["mean_completeness"]
    fmt = lambda v: "  –  " if v is None else f"{v:4.1f}"  # noqa: E731
    lines.append("  ── confidence vs measured completeness (0–5):")
    lines.append(f"     mean completeness:  high {fmt(mc['high'])} ({s['counts']['high']})"
                 f"   medium {fmt(mc['medium'])} ({s['counts']['medium']})"
                 f"   low {fmt(mc['low'])} ({s['counts']['low']})")
    lines.append(f"     monotone (high ≥ medium ≥ low): {'✓' if s['monotone'] else '✗'}")
    hf = s["high_at_full_only"]
    hf_str = "n/a (no high calls)" if hf is None else ("✓" if hf else "✗")
    lines.append(f"     high only at full corroboration ({s['max_completeness']}/5): {hf_str}")
    return "\n".join(lines)
