"""Merge CADD baseline scores into a benchmark TSV.

Workflow: submit the benchmark's variants (as VCF) to the CADD web service
(https://cadd.gs.washington.edu/, GRCh38), download the scored TSV, then merge its
PHRED-scaled CADD onto the benchmark's ``cadd`` column so the harness uses it as the
naive baseline. Matching is by ``(chrom, pos, ref, alt)``.
"""

from __future__ import annotations

import csv
import os


def _norm_chrom(chrom: str) -> str:
    """Normalize a contig name (CADD output has no ``chr`` prefix)."""
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def load_cadd_scores(cadd_tsv: str | os.PathLike[str]) -> dict[tuple[str, str, str, str], float]:
    """Load a CADD-scored TSV into a ``(chrom,pos,ref,alt) → PHRED`` lookup.

    Handles the CADD web-service output header (``#Chrom/Chrom, Pos, Ref, Alt,
    RawScore, PHRED``), comment lines, and the missing ``chr`` prefix.

    Args:
        cadd_tsv: Path to the CADD output (optionally gzipped is *not* handled here —
            decompress first).

    Returns:
        Mapping from normalized ``(chrom, pos, ref, alt)`` to PHRED CADD score.
    """
    lookup: dict[tuple[str, str, str, str], float] = {}
    with open(cadd_tsv, newline="") as handle:
        # Skip leading '##' comment lines; the header line may start with '#Chrom'.
        lines = [ln for ln in handle if not ln.startswith("##")]
    if lines and lines[0].startswith("#"):
        lines[0] = lines[0][1:]
    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        try:
            key = (_norm_chrom(row["Chrom"]), row["Pos"], row["Ref"].upper(), row["Alt"].upper())
            lookup[key] = float(row["PHRED"])
        except (KeyError, ValueError):
            continue
    return lookup


def annotate_benchmark(
    benchmark_tsv: str | os.PathLike[str],
    cadd_tsv: str | os.PathLike[str],
    out_tsv: str | os.PathLike[str],
) -> tuple[int, int]:
    """Add a ``cadd`` column to a benchmark TSV from a CADD-scored TSV.

    Args:
        benchmark_tsv: The harness benchmark (chrom,pos,ref,alt,label,...).
        cadd_tsv: CADD web-service output for those variants.
        out_tsv: Output path for the annotated benchmark.

    Returns:
        ``(n_annotated, n_total)`` — how many rows got a CADD score.
    """
    scores = load_cadd_scores(cadd_tsv)
    with open(benchmark_tsv, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "cadd" not in fields:
        fields.append("cadd")

    annotated = 0
    for row in rows:
        key = (_norm_chrom(row["chrom"]), row["pos"], row["ref"].upper(), row["alt"].upper())
        if key in scores:
            row["cadd"] = str(scores[key])
            annotated += 1
        else:
            row.setdefault("cadd", "")

    with open(out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return annotated, len(rows)
