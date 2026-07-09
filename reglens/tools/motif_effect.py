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

from reglens.genome import SequenceWindow, Variant

# Bundled JASPAR subset shipped with the package (see reglens/data/motifs/).
DEFAULT_MOTIF_DB = Path(__file__).parent.parent / "data" / "motifs" / "jaspar_core_subset.jaspar"

# Uniform background nucleotide frequencies for the log-odds conversion.
_UNIFORM_BG = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
_BASES = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# How far either side of the variant to scan for overlapping motif hits (bp). Must
# be >= the widest motif so any motif overlapping the variant fits in the window.
DEFAULT_FLANK = 30

# Effect-call thresholds on the ref→alt score change (bits).
DISRUPT_THRESHOLD = 1.0  # alt weaker than ref by >= 1 bit → "disrupted"
# Below this best-binding score (bits) a hit is too weak to call a motif at all.
MIN_BINDING_BITS = 3.0


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
    background = background or _UNIFORM_BG
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
    """

    motif_id: str
    tf_name: str
    strand: str
    offset: int
    ref_score: float
    alt_score: float
    delta_score: float
    effect: str

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

    def summary(self) -> str:
        """A one-line human-readable summary of the top motif effect."""
        if self.top is None:
            return f"{self.variant}: no credible TF motif overlaps the variant."
        t = self.top
        return (
            f"{self.variant}: {t.effect} {t.tf_name} ({t.motif_id}, {t.strand} strand) "
            f"— motif score {t.ref_score:.2f}→{t.alt_score:.2f} bits (Δ={t.delta_score:+.2f})"
        )


def _classify(delta: float) -> str:
    """Classify a ref→alt score change into a motif effect label."""
    if delta <= -DISRUPT_THRESHOLD:
        return "disrupted"
    if delta >= DISRUPT_THRESHOLD:
        return "created"
    return "unchanged"


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
) -> MotifEffectResult:
    """Identify the TF motif most disrupted or created by a variant.

    Args:
        window: The ref/alt :class:`SequenceWindow` around the variant.
        variant: The variant being analyzed (for reporting).
        motifs: Motif library; defaults to the bundled JASPAR subset.
        flank: Half-width (bp) of the local region scanned around the variant.
        min_binding_bits: Minimum best-binding score for a hit to be credible.

    Returns:
        A :class:`MotifEffectResult`. ``top`` is the credible hit with the largest
        ``|delta_score|`` (ties broken by binding strength), or ``None`` if no
        motif binds above ``min_binding_bits``.
    """
    if motifs is None:
        motifs = load_motifs()

    # Carve out a local window centered on the variant so we only score motifs that
    # actually overlap it (and keep the scan cheap).
    lo = max(0, window.variant_offset - flank)
    hi = min(window.length, window.variant_offset + flank + 1)
    ref_local = window.ref_seq[lo:hi]
    alt_local = window.alt_seq[lo:hi]
    var_pos = window.variant_offset - lo

    hits: list[MotifHit] = []
    for motif in motifs:
        hit = _scan_motif(motif, ref_local, alt_local, var_pos)
        if hit is not None and hit.best_binding >= min_binding_bits:
            hits.append(hit)

    # Rank by how much the variant changes the site, then by absolute binding.
    hits.sort(key=lambda h: (abs(h.delta_score), h.best_binding), reverse=True)
    top = hits[0] if hits else None
    return MotifEffectResult(variant=variant, hits=hits, top=top)
