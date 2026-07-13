"""Name the TF motif a variant disrupts or creates (JASPAR-match MVP).

Given the reference/alternate sequence windows around a variant, this tool scans a
small local region across a library of JASPAR position weight matrices (PWMs) and
reports which transcription-factor motif is most affected by the single-base
change — the "which TF" half of the mechanistic story.

Approach (the spec's MVP, §4): for every motif and every frame that overlaps the
variant, score the reference and alternate local sequence with the motif's PWM
(both strands), then take the change in match score. A large drop = the alt allele
**disrupts** that motif; a large gain = it **creates** one. This is deliberately
simpler than full TF-MoDISco / model-based attribution but is enough to *name* the
TF (e.g. GATA1 for rs1427407), and it pairs with the ChromBPNet Δ log-counts /
profile-JSD signals from :mod:`reglens.tools.chrombpnet_score`.

All numbers here are deterministic PWM log-odds — no model, no LLM. Scores are in
bits relative to a uniform background.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reglens.genome import SequenceWindow, Variant

_MOTIF_DIR = Path(__file__).parent.parent / "data" / "motifs"

# Default motif library: the full JASPAR CORE 2024 vertebrates *non-redundant* PFM
# set (~880 motifs). "Non-redundant" collapses near-duplicate profiles for the same
# TF, but a TF with genuinely distinct binding modes still contributes several
# matrices (e.g. CTCF: MA0139.2 / MA1929.2 / MA1930.2) — so this is not one-per-TF,
# and the per-motif FDR gate below is what keeps the top pick honest, not the dedup.
DEFAULT_MOTIF_DB = _MOTIF_DIR / "jaspar_core_vertebrates.jaspar"

# A tiny hand-curated 3-motif subset (GATA1, CTCF, GATA1::TAL1), kept for fast,
# version-pinned unit tests. Real analysis uses DEFAULT_MOTIF_DB.
SUBSET_MOTIF_DB = _MOTIF_DIR / "jaspar_core_subset.jaspar"

# Uniform background — kept for reference / tests that assume it.
_UNIFORM_BG = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
# Human genomic background (hg38 autosomal): AT-rich (~0.29 A/T, ~0.21 C/G). JASPAR's
# own scoring uses a genomic background; uniform inflates GC-rich motif scores relative
# to AT-rich ones, which biases *which* motif wins once the library is large. This is
# the default background for the log-odds conversion.
_GENOMIC_BG = {"A": 0.295, "C": 0.205, "G": 0.205, "T": 0.295}
_BASES = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# How far either side of the variant to scan for overlapping motif hits (bp). Must
# be >= the widest motif so any motif overlapping the variant fits in the window. The
# full JASPAR set has matrices up to 33 bp (some CTCF variants), so 40 gives headroom.
DEFAULT_FLANK = 40

# Effect-call thresholds on the ref→alt score change (bits).
DISRUPT_THRESHOLD = 1.0  # alt weaker than ref by >= 1 bit → "disrupted"
# Below this best-binding score (bits) a hit is too weak to keep as a *candidate* — a
# cheap prefilter before the statistical gate, not the significance call itself.
MIN_BINDING_BITS = 3.0

# Family-wise false-positive rate for the significance gate. With ~880 short PWMs
# scanned per variant, |Δ| is the WRONG discriminator: a short weak motif swings
# wildly on one base, so random sequence produces large |Δ| (median top ≈ 15 bits)
# off sites that never bound. What actually separates a real motif occurrence from
# noise is BINDING STRENGTH — a genuine site binds far harder than chance (a real
# CTCF site ≈ 26 bits vs. a family-wise binding-noise p95 ≈ 15 bits). So the gate is:
# (1) the variant sits in a site whose best-binding score beats the empirical
# family-wise binding null at level ``alpha``, AND (2) the variant actually changes it
# (|Δ| ≥ DISRUPT_THRESHOLD). The null is calibrated once per library on random
# background windows, which holds the false-positive rate at ~``alpha`` for any
# library size.
DEFAULT_ALPHA = 0.05
# Number of random background windows used to calibrate the empirical binding null.
# More = a tighter quantile; 500 puts the 95th percentile on ~25 tail samples.
_NULL_PANEL_SIZE = 500


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence (N preserved)."""
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


@dataclass(frozen=True)
class Motif:
    """A transcription-factor motif as a log-odds position weight matrix.

    Attributes:
        motif_id: JASPAR matrix accession (e.g. ``"MA0035.4"``).
        tf_name: Transcription factor name (e.g. ``"GATA1"``).
        pwm: Per-position log2 odds, ``pwm[i][base]`` in bits vs. background.
        width: Motif length in bp.
    """

    motif_id: str
    tf_name: str
    pwm: tuple[dict[str, float], ...]
    width: int

    def score(self, seq: str) -> float:
        """Score a sequence of exactly ``width`` bp as summed log-odds (bits).

        Args:
            seq: An uppercase DNA string of length ``width``.

        Returns:
            The summed per-position log-odds. Any non-ACGT base contributes the
            column minimum (a strong penalty), so ``N`` never inflates a hit.
        """
        total = 0.0
        for i, base in enumerate(seq):
            column = self.pwm[i]
            total += column.get(base, min(column.values()))
        return total


def _pfm_to_motif(
    motif_id: str,
    tf_name: str,
    counts: dict[str, list[float]],
    background: dict[str, float],
    pseudocount: float = 0.8,
) -> Motif:
    """Convert a position frequency matrix to a log-odds :class:`Motif`.

    Args:
        motif_id: JASPAR accession.
        tf_name: TF name.
        counts: Per-base count arrays, each of equal length (the motif width).
        background: Background base frequencies.
        pseudocount: Added (split by background) to avoid ``log(0)``.

    Returns:
        The log-odds :class:`Motif`.
    """
    width = len(counts["A"])
    pwm: list[dict[str, float]] = []
    for i in range(width):
        col_total = sum(counts[b][i] for b in _BASES) + pseudocount
        column: dict[str, float] = {}
        for b in _BASES:
            # Laplace-style smoothing split across bases by their background prob.
            prob = (counts[b][i] + pseudocount * background[b]) / col_total
            column[b] = math.log2(prob / background[b])
        pwm.append(column)
    return Motif(motif_id=motif_id, tf_name=tf_name, pwm=tuple(pwm), width=width)


def load_motifs(
    path: str | os.PathLike[str] = DEFAULT_MOTIF_DB,
    background: dict[str, float] | None = None,
) -> list[Motif]:
    """Load motifs from a JASPAR-format PFM file into log-odds :class:`Motif`s.

    Args:
        path: Path to a JASPAR ``.jaspar`` PFM file (``>ID NAME`` headers followed
            by ``A [ ... ]`` / ``C`` / ``G`` / ``T`` count rows).
        background: Background base frequencies; defaults to uniform.

    Returns:
        The parsed motifs, in file order.

    Raises:
        ValueError: If the file is malformed.
    """
    background = background or _GENOMIC_BG
    motifs: list[Motif] = []
    motif_id = tf_name = None
    counts: dict[str, list[float]] = {}

    def _flush() -> None:
        if motif_id is not None:
            if set(counts) != set(_BASES):
                raise ValueError(f"Motif {motif_id} is missing rows; got {sorted(counts)}")
            motifs.append(_pfm_to_motif(motif_id, tf_name, counts, background))

    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                parts = line[1:].split(None, 1)
                motif_id = parts[0]
                tf_name = parts[1].strip() if len(parts) > 1 else parts[0]
                counts = {}
            else:
                # Row form: "A [ 1 2 3 ]" (brackets optional).
                base = line[0].upper()
                nums = line[1:].replace("[", " ").replace("]", " ").split()
                counts[base] = [float(x) for x in nums]
    _flush()
    if not motifs:
        raise ValueError(f"No motifs parsed from {path}")
    return motifs


@dataclass
class MotifHit:
    """A motif's best binding site overlapping the variant, ref vs alt.

    Attributes:
        motif_id: JASPAR accession of the motif.
        tf_name: TF name.
        strand: ``"+"`` or ``"-"`` — the orientation of the best site.
        offset: 0-based start of the site within the *local* scan window.
        ref_score: PWM score (bits) of the reference allele at this site.
        alt_score: PWM score (bits) of the alternate allele at this site.
        delta_score: ``alt_score - ref_score`` (negative = weakened by alt).
        effect: ``"disrupted"``, ``"created"`` or ``"unchanged"``.
        p_value: Right-tail p-value of the best-binding allele under the PWM null
            (probability a random background site binds this well). ``None`` until
            computed. Lower = a more credible genuine motif match.
    """

    motif_id: str
    tf_name: str
    strand: str
    offset: int
    ref_score: float
    alt_score: float
    delta_score: float
    effect: str
    p_value: float | None = None

    @property
    def best_binding(self) -> float:
        """The stronger of the ref/alt scores — how good a site this is at all."""
        return max(self.ref_score, self.alt_score)


@dataclass
class MotifEffectResult:
    """Ranked motif-disruption results for a variant.

    Attributes:
        variant: The variant analyzed.
        hits: Motif hits overlapping the variant, ranked by disruption magnitude.
        top: The single most-affected credible motif, or ``None`` if none bind.
    """

    variant: Variant
    hits: list[MotifHit] = field(default_factory=list)
    top: MotifHit | None = None

    @property
    def disrupting_allele(self) -> str | None:
        """Which allele base *weakens* the top motif (the lower-scoring allele).

        ``effect`` is stated relative to the alt allele (created/disrupted), but the
        literature usually frames the mechanism around whichever allele *breaks* the
        site. This exposes that framing explicitly so the two reconcile: for a motif
        the alt allele *creates* (``delta > 0``), the *reference* allele is the one
        that disrupts it; for one the alt allele *disrupts* (``delta < 0``), the alt
        allele is the disrupting one. ``None`` if the effect is neutral.
        """
        if self.top is None or self.top.effect == "unchanged":
            return None
        return self.variant.ref if self.top.delta_score > 0 else self.variant.alt

    def reconciling_note(self) -> str | None:
        """Plain-language note stating both allele framings (for the reasoning layer)."""
        if self.top is None or self.disrupting_allele is None:
            return None
        v = self.variant
        strong = v.alt if self.disrupting_allele == v.ref else v.ref
        return (
            f"The {self.disrupting_allele} allele weakens/abolishes the {self.top.tf_name} "
            f"site while the {strong} allele forms it; whether this reads as 'created' or "
            f"'disrupted' depends only on which allele is taken as reference."
        )

    def summary(self) -> str:
        """A one-line summary of the top motif effect, framing-reconciled."""
        if self.top is None:
            return f"{self.variant}: no credible TF motif overlaps the variant."
        t = self.top
        base = (
            f"{self.variant}: alt allele {t.effect} {t.tf_name} ({t.motif_id}, {t.strand} strand) "
            f"— score {t.ref_score:.2f}(ref)→{t.alt_score:.2f}(alt) bits (Δ={t.delta_score:+.2f})"
        )
        if self.disrupting_allele is not None:
            base += f"; the {self.disrupting_allele} allele is the one that breaks the site"
        return base


def _classify(delta: float) -> str:
    """Classify a ref→alt score change into a motif effect label."""
    if delta <= -DISRUPT_THRESHOLD:
        return "disrupted"
    if delta >= DISRUPT_THRESHOLD:
        return "created"
    return "unchanged"


# Empirical |Δ| null threshold, cached per (library identity, flank, background, alpha)
# so the calibration panel runs once per process, not once per variant scanned.
_DELTA_NULL_CACHE: dict[Any, tuple[float, np.ndarray]] = {}


def _library_key(motifs: list[Motif]) -> Any:
    """A cheap, stable identity for a motif library (for the null-threshold cache)."""
    return (len(motifs),
            tuple(m.motif_id for m in motifs[:8]),
            tuple(m.motif_id for m in motifs[-8:]))


def _random_top_binding(
    motifs: list[Motif], flank: int,
    background: dict[str, float], rng: np.random.Generator,
) -> float:
    """Top best-binding score a random background variant produces across the library.

    Draws a ``2*flank+1`` bp window iid from ``background``, mutates the centre base,
    scans every motif, and returns the largest ``best_binding`` seen at a site that
    actually changes (|Δ| ≥ DISRUPT_THRESHOLD) — one draw from the family-wise binding
    null the gate calibrates against. Restricting to changed sites makes the null the
    right comparison for the variant hits the gate is applied to.
    """
    n = 2 * flank + 1
    bases = np.array(_BASES)
    p = np.array([background[b] for b in _BASES], dtype=np.float64)
    p /= p.sum()
    seq = "".join(rng.choice(bases, size=n, p=p))
    vp = n // 2
    alt = rng.choice([b for b in _BASES if b != seq[vp]])
    ref_local, alt_local = seq, seq[:vp] + alt + seq[vp + 1:]
    best = 0.0
    for motif in motifs:
        hit = _scan_motif(motif, ref_local, alt_local, vp)
        if hit is not None and abs(hit.delta_score) >= DISRUPT_THRESHOLD:
            best = max(best, hit.best_binding)
    return best


# Precomputed default-config null shipped with the package (see the generator note in
# reglens/data/motifs/). Loading it lets a single-variant run skip ~40s of simulation.
_BUNDLED_NULL_PATH = _MOTIF_DIR / "binding_null.default.json"


def _load_bundled_null(
    library_key: Any, flank: int, alpha: float,
    background: dict[str, float], panel_size: int, seed: int,
) -> tuple[float, np.ndarray] | None:
    """Return the bundled null threshold+sample iff this call matches its config.

    The shipped null is valid only for the exact configuration it was computed under:
    the library identity (size + head/tail motif-id fingerprint), flank, alpha,
    background, panel size, and seed. Any deviation — including a different library
    that merely happens to have the same size — falls through to live calibration, so
    a custom setup is never silently mis-thresholded.
    """
    if not _BUNDLED_NULL_PATH.exists():
        return None
    try:
        import json
        with open(_BUNDLED_NULL_PATH) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    n_motifs, head, tail = library_key
    bg_match = all(
        abs(data["background"].get(b, -1) - background[b]) < 1e-6 for b in _BASES
    )
    lib_match = (data["n_motifs"] == n_motifs
                 and tuple(data.get("lib_head", ())) == head
                 and tuple(data.get("lib_tail", ())) == tail)
    if not (lib_match and data["flank"] == flank and abs(data["alpha"] - alpha) < 1e-12
            and data["panel_size"] == panel_size and data["seed"] == seed and bg_match):
        return None
    return float(data["threshold"]), np.array(data["null_sorted"], dtype=np.float64)


def calibrate_binding_null(
    motifs: list[Motif],
    flank: int = DEFAULT_FLANK,
    alpha: float = DEFAULT_ALPHA,
    background: dict[str, float] | None = None,
    panel_size: int = _NULL_PANEL_SIZE,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """The library's family-wise binding-strength threshold at level ``alpha``.

    Simulates ``panel_size`` random background variants (:func:`_random_top_binding`),
    collects the per-variant top best-binding score across the whole library at a
    changed site, and returns the ``(1 − alpha)`` quantile — the binding score a real
    variant's site must beat to be credible. By construction a random variant clears
    it only ``alpha`` of the time, so the false-positive rate is controlled at
    ``alpha`` regardless of library size. Cached per library/flank/background/alpha.

    Args:
        motifs: The motif library being scanned.
        flank: Half-width (bp) of the scan window (must match :func:`motif_effect`).
        alpha: Target family-wise false-positive rate.
        background: Background base frequencies (defaults to genomic).
        panel_size: Number of random background variants to simulate.
        seed: RNG seed (kept fixed so the threshold is reproducible).

    Returns:
        ``(threshold, null)``: the best-binding (bits) threshold, and the sorted array
        of per-variant top binding scores from the null panel (for empirical
        exceedance).
    """
    background = background or _GENOMIC_BG
    key = (_library_key(motifs), flank, alpha,
           tuple(round(background[b], 4) for b in _BASES), panel_size, seed)
    cached = _DELTA_NULL_CACHE.get(key)
    if cached is not None:
        return cached

    # Fast path: the default library+config is a fixed deterministic threshold, so a
    # single-variant run shouldn't pay ~40s of simulation. Load the value bundled with
    # the package if this call matches the defaults exactly.
    bundled = _load_bundled_null(
        _library_key(motifs), flank, alpha, background, panel_size, seed
    )
    if bundled is not None:
        _DELTA_NULL_CACHE[key] = bundled
        return bundled

    rng = np.random.default_rng(seed)
    null = np.sort(np.array([
        _random_top_binding(motifs, flank, background, rng)
        for _ in range(panel_size)
    ]))
    threshold = float(np.quantile(null, 1.0 - alpha))
    result = (threshold, null)
    _DELTA_NULL_CACHE[key] = result
    return result


def _scan_motif(
    motif: Motif, ref_local: str, alt_local: str, var_pos: int
) -> MotifHit | None:
    """Find a motif's best site overlapping ``var_pos`` and its ref→alt change.

    Enumerates every frame (start position) whose span covers the variant, scores
    both strands, and keeps the frame/strand with the strongest binding (best of
    ref/alt). Returns ``None`` if no full-width frame overlaps the variant.

    Args:
        motif: The motif to scan.
        ref_local: Local reference sequence window.
        alt_local: Local alternate sequence window (same length as ref).
        var_pos: 0-based index of the variant base within the local windows.

    Returns:
        The best :class:`MotifHit`, or ``None``.
    """
    width = motif.width
    # Frames [start, start+width) that both fit in the window and cover var_pos.
    first = max(0, var_pos - width + 1)
    last = min(len(ref_local) - width, var_pos)
    best: MotifHit | None = None
    for start in range(first, last + 1):
        ref_sub = ref_local[start : start + width]
        alt_sub = alt_local[start : start + width]
        # Evaluate forward (+) and reverse-complement (−) orientations; ref and alt
        # always share the same frame+strand so their delta is meaningful.
        for strand, transform in (("+", lambda s: s), ("-", reverse_complement)):
            ref_score = motif.score(transform(ref_sub))
            alt_score = motif.score(transform(alt_sub))
            binding = max(ref_score, alt_score)
            if best is None or binding > best.best_binding:
                delta = alt_score - ref_score
                best = MotifHit(
                    motif_id=motif.motif_id,
                    tf_name=motif.tf_name,
                    strand=strand,
                    offset=start,
                    ref_score=ref_score,
                    alt_score=alt_score,
                    delta_score=delta,
                    effect=_classify(delta),
                )
    return best


def motif_effect(
    window: SequenceWindow,
    variant: Variant,
    motifs: list[Motif] | None = None,
    flank: int = DEFAULT_FLANK,
    min_binding_bits: float = MIN_BINDING_BITS,
    alpha: float = DEFAULT_ALPHA,
    background: dict[str, float] | None = None,
    null_panel_size: int = _NULL_PANEL_SIZE,
) -> MotifEffectResult:
    """Identify the TF motif most disrupted or created by a variant.

    Two-stage credibility filter so scanning a large motif library (~880 JASPAR
    matrices) can't manufacture a mechanism on random sequence:

    1. **Binding prefilter** — keep only sites whose stronger allele clears
       ``min_binding_bits`` (a cheap magnitude gate that discards obvious noise).
    2. **Empirical family-wise gate** — a short PWM swings wildly on one base, so |Δ|
       alone is large even on noise; what separates a real occurrence from chance is
       BINDING STRENGTH. The library's family-wise binding null is calibrated once on
       random background windows (:func:`calibrate_binding_null`), and a hit is kept
       only if (a) its best-binding score beats the ``(1 − alpha)`` quantile of that
       null AND (b) the variant actually changes it (|Δ| ≥ ``DISRUPT_THRESHOLD``).
       This holds the false-positive rate near ``alpha`` whether the library has 3
       motifs or 880 — what makes the large library *honest*.

    Args:
        window: The ref/alt :class:`SequenceWindow` around the variant.
        variant: The variant being analyzed (for reporting).
        motifs: Motif library; defaults to the full JASPAR CORE vertebrate set.
        flank: Half-width (bp) of the local region scanned around the variant.
        min_binding_bits: Minimum best-binding score for the prefilter.
        alpha: Family-wise false-positive rate for the empirical |Δ| gate. Set to
            ``1.0`` to disable the gate (prefilter only — the old behavior, safe only
            for a tiny curated library).
        background: Background base frequencies for the null (defaults to genomic).
        null_panel_size: Random background windows used to calibrate the gate (cached
            per library, so the cost is paid once). Smaller = faster but a noisier
            threshold; the default is tuned for accuracy, tests override it for speed.

    Returns:
        A :class:`MotifEffectResult`. ``top`` is the most-disrupted hit that clears
        the empirical |Δ| threshold, or ``None`` if none survives. Each hit's
        ``p_value`` is its empirical exceedance — the fraction of random background
        variants whose top |Δ| reaches that hit's |Δ| — so a reader sees how unusual
        the call is, not just that it passed.
    """
    if motifs is None:
        motifs = load_motifs()
    background = background or _GENOMIC_BG

    # Carve out a local window centered on the variant so we only score motifs that
    # actually overlap it (and keep the scan cheap).
    lo = max(0, window.variant_offset - flank)
    hi = min(window.length, window.variant_offset + flank + 1)
    ref_local = window.ref_seq[lo:hi]
    alt_local = window.alt_seq[lo:hi]
    var_pos = window.variant_offset - lo

    # Stage 1: scan + binding prefilter + require the variant to actually change the
    # site (|Δ| ≥ DISRUPT_THRESHOLD). An unchanged site is not a mechanism.
    hits: list[MotifHit] = []
    for motif in motifs:
        hit = _scan_motif(motif, ref_local, alt_local, var_pos)
        if (hit is not None and hit.best_binding >= min_binding_bits
                and abs(hit.delta_score) >= DISRUPT_THRESHOLD):
            hits.append(hit)

    # Stage 2: empirical family-wise BINDING gate. Threshold calibrated (and cached)
    # once per library on random background windows; keep only hits whose site binds
    # more strongly than chance ever produces across the library.
    if hits and alpha < 1.0:
        threshold, null = calibrate_binding_null(
            motifs, flank=flank, alpha=alpha, background=background,
            panel_size=null_panel_size,
        )
        hits = [h for h in hits if h.best_binding >= threshold]
        # Empirical exceedance: fraction of null variants whose top binding reaches this.
        n = len(null)
        for h in hits:
            beaten = int(np.searchsorted(null, h.best_binding, side="left"))
            h.p_value = (n - beaten) / n

    # Rank by how much the variant changes the site, then by absolute binding.
    hits.sort(key=lambda h: (abs(h.delta_score), h.best_binding), reverse=True)
    top = hits[0] if hits else None
    return MotifEffectResult(variant=variant, hits=hits, top=top)
