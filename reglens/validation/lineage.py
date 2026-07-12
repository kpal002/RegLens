"""Cell-type-specificity analysis of per-element validation AUROC.

Curates the lineage of each Kircher satMutMPRA element (by its primary regulated gene /
disease context) so per-element AUROC can be stratified by whether the element is in
the model's lineage. For a **K562 (erythroid/hematopoietic)** ChromBPNet model, the
hypothesis is that it discriminates functional variants better in hematopoietic
elements than in non-hematopoietic ones — the measurable form of RegLens's
cell-type-specificity thesis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reglens.validation.harness import ValidationReport

# Element → lineage of its primary regulated gene / element context.
ELEMENT_LINEAGE: dict[str, str] = {
    # Hematopoietic (K562's lineage)
    "BCL11A": "erythroid", "HBB": "erythroid", "HBG1": "erythroid",
    "PKLR-24h": "erythroid", "PKLR-48h": "erythroid",
    "GP1BA": "megakaryocyte",
    # Hepatic
    "SORT1": "hepatic", "SORT1.2": "hepatic", "SORT1-flip": "hepatic",
    "LDLR": "hepatic", "LDLR.2": "hepatic", "HNF4A": "hepatic", "F9": "hepatic",
    # Tissue-specific, non-hematopoietic
    "FOXE1": "thyroid", "RET": "neural-crest", "IRF6": "craniofacial",
    "IRF4": "melanocyte", "TCF7L2": "gut/pancreas", "ZFAND3": "metabolic",
    "MSMB": "prostate", "UC88": "developmental",
    "ZRSh-13": "limb", "ZRSh-13h2": "limb",
    # Broadly active (telomerase / MYC amplicon)
    "MYCrs6983267": "broad", "MYCrs11986220": "broad",
    "TERT-GSc": "broad", "TERT-GBM": "broad", "TERT-GAa": "broad", "TERT-HEK": "broad",
}

# Lineages that belong to the hematopoietic (K562) compartment.
HEMATOPOIETIC = frozenset({"erythroid", "megakaryocyte"})
# Lineages that belong to the hepatic (HepG2) compartment.
HEPATIC = frozenset({"hepatic"})


def lineage(element: str) -> str:
    """Return the curated lineage for an element (``"unknown"`` if unmapped)."""
    return ELEMENT_LINEAGE.get(element, "unknown")


def is_hematopoietic(element: str) -> bool:
    """Whether an element's lineage is hematopoietic (K562's compartment)."""
    return lineage(element) in HEMATOPOIETIC


def is_hepatic(element: str) -> bool:
    """Whether an element's lineage is hepatic (HepG2's compartment)."""
    return lineage(element) in HEPATIC


# The two compartments used for the K562-vs-HepG2 crossover.
_LINEAGE_TESTS = {"hematopoietic": is_hematopoietic, "hepatic": is_hepatic}


def crossover_summary(
    per_model: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    """Mean AUROC per (lineage compartment × model) for the crossover experiment.

    Args:
        per_model: ``{model_name: {element: auroc}}`` — e.g. ``{"K562": {...},
            "HepG2": {...}}``.

    Returns:
        ``{compartment: {model: mean_auroc}}`` for ``hematopoietic`` and ``hepatic``.
    """
    out: dict[str, dict[str, float | None]] = {}
    for compartment, test in _LINEAGE_TESTS.items():
        out[compartment] = {}
        for model, aurocs in per_model.items():
            vals = [a for e, a in aurocs.items() if test(e) and a is not None]
            out[compartment][model] = sum(vals) / len(vals) if vals else None
    return out


def is_double_dissociation(
    summary: dict[str, dict[str, float | None]], hema_model: str, hep_model: str
) -> bool:
    """Whether a crossover shows a double dissociation.

    True iff the hematopoietic model wins on hematopoietic elements **and** the hepatic
    model wins on hepatic elements — the intervention that proves the signal is
    cell-type-driven, not a model artifact.

    Args:
        summary: Output of :func:`crossover_summary`.
        hema_model: Name of the hematopoietic (K562) model in ``summary``.
        hep_model: Name of the hepatic (HepG2) model in ``summary``.

    Returns:
        Whether the double dissociation holds.
    """
    h, p = summary["hematopoietic"], summary["hepatic"]
    if None in (h.get(hema_model), h.get(hep_model), p.get(hema_model), p.get(hep_model)):
        return False
    return h[hema_model] > h[hep_model] and p[hep_model] > p[hema_model]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_crossover_ci(
    per_model: dict[str, dict[str, float]],
    hema_model: str,
    hep_model: str,
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Cluster-bootstrap CI for the crossover deltas, resampling **elements**.

    The element is the correct resampling unit (variants within one element are
    correlated), so this quantifies how robust each compartment's advantage is to
    *which elements* are included. It does **not** capture within-element sampling
    noise — that would need per-variant scores, not the per-element AUROCs here.

    The delta is signed so positive = the compartment's own model wins:
    hematopoietic ``= hema_model - hep_model``; hepatic ``= hep_model - hema_model``.

    Args:
        per_model: ``{model_name: {element: auroc}}`` (K562 and HepG2).
        hema_model: Name of the hematopoietic (K562) model.
        hep_model: Name of the hepatic (HepG2) model.
        n_boot: Bootstrap resamples per compartment.
        seed: RNG seed (reproducible).
        ci: Central interval mass (0.95 → 2.5%/97.5% percentiles).

    Returns:
        ``{compartment: {"delta", "lo", "hi", "p_wrong_sign", "n"}}`` — ``delta`` is
        the point estimate, ``lo``/``hi`` the percentile CI, ``p_wrong_sign`` the
        bootstrap fraction where the compartment's own model did **not** win, and
        ``n`` the number of elements.
    """
    import random

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out: dict[str, dict[str, float]] = {}
    for compartment, test in _LINEAGE_TESTS.items():
        elements = [
            e for e in per_model[hema_model]
            if test(e) and per_model[hema_model].get(e) is not None
            and per_model[hep_model].get(e) is not None
        ]
        own, other = (hema_model, hep_model) if compartment == "hematopoietic" else (hep_model, hema_model)
        deltas_point = [per_model[own][e] - per_model[other][e] for e in elements]
        point = _mean(deltas_point) or 0.0
        rng = random.Random(f"{seed}:{compartment}")
        boot = []
        for _ in range(n_boot):
            sample = [deltas_point[rng.randrange(len(deltas_point))] for _ in deltas_point]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        wrong = sum(1 for b in boot if b <= 0) / len(boot)
        out[compartment] = {
            "delta": point,
            "lo": boot[int(lo_q * (len(boot) - 1))],
            "hi": boot[int(hi_q * (len(boot) - 1))],
            "p_wrong_sign": wrong,
            "n": float(len(elements)),
        }
    return out


def stratify(
    per_element: list[tuple[str, float | None, int, int]],
) -> dict[str, dict[str, object]]:
    """Split per-element AUROC into hematopoietic vs non-hematopoietic groups.

    Args:
        per_element: ``(element, auroc, n_pos, n_neg)`` rows (from
            :meth:`ValidationReport.per_source_auroc`).

    Returns:
        ``{"hematopoietic": {...}, "other": {...}}`` where each group carries its
        member elements and their mean AUROC.
    """
    hema = [(s, a) for s, a, _, _ in per_element if a is not None and is_hematopoietic(s)]
    other = [(s, a) for s, a, _, _ in per_element if a is not None and not is_hematopoietic(s)]
    return {
        "hematopoietic": {
            "elements": [s for s, _ in hema], "mean_auroc": _mean([a for _, a in hema]),
            "n": len(hema),
        },
        "other": {
            "elements": [s for s, _ in other], "mean_auroc": _mean([a for _, a in other]),
            "n": len(other),
        },
    }


def render_celltype_specificity(report: ValidationReport) -> str:
    """Render the hematopoietic-vs-other AUROC contrast for a report."""
    groups = stratify(report.per_source_auroc())
    lines = ["── Cell-type specificity (K562 = erythroid/hematopoietic) " + "─" * 8]
    for name, key in (("hematopoietic elements", "hematopoietic"), ("other elements", "other")):
        g = groups[key]
        mean = f"{g['mean_auroc']:.3f}" if g["mean_auroc"] is not None else "n/a"
        lines.append(f"  {name:24s} mean AUROC={mean}  (n={g['n']})")
    lines.append("  (K562 model should discriminate hematopoietic elements better.)")
    return "\n".join(lines)
