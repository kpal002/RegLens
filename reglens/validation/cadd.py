"""CADD baseline for the validation benchmark: export a VCF, then merge scores back.

End-to-end workflow (the harness reads the resulting ``cadd`` column automatically):

1. ``python -m reglens.validation.cadd export -b <benchmark.tsv> -o variants.vcf``
2. Upload ``variants.vcf`` to the **CADD web service**
   (https://cadd.gs.washington.edu/score), GRCh38 / GISCADD v1.7, "include annotations"
   off, PHRED on; download the scored ``.tsv.gz`` and ``gunzip`` it.
3. ``python -m reglens.validation.cadd merge -b <bench.tsv> -c cadd.tsv -o <bench.cadd.tsv>``
4. Validate on ``<bench.cadd.tsv>`` — ``baseline(CADD)`` is now populated.

Matching is by ``(chrom, pos, ref, alt)``; CADD uses no ``chr`` prefix, handled here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
from typing import TextIO


def _open_maybe_gzip(path: str | os.PathLike[str]) -> TextIO:
    """Open a text file, transparently decompressing if it is gzip (by magic bytes).

    CADD downloads are ``.tsv.gz``, but browsers sometimes rename or pre-decompress
    them — so detect gzip from the content, not the extension.
    """
    with open(path, "rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    if is_gzip:
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8", newline="")


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
    with _open_maybe_gzip(cadd_tsv) as handle:
        # Skip leading '##' comment lines; the header line may start with '#Chrom'.
        lines = [ln for ln in handle if not ln.startswith("##")]
    if lines and lines[0].startswith("#"):
        lines[0] = lines[0][1:]
    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        chrom, pos, ref, alt, phred = (
            row.get("Chrom"), row.get("Pos"), row.get("Ref"), row.get("Alt"), row.get("PHRED")
        )
        if not all((chrom, pos, ref, alt, phred)):  # skip malformed / short rows
            continue
        try:
            lookup[(_norm_chrom(chrom), pos, ref.upper(), alt.upper())] = float(phred)
        except (ValueError, TypeError):
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


def export_vcf(benchmark_tsv: str | os.PathLike[str], out_vcf: str | os.PathLike[str]) -> int:
    """Write the benchmark's variants as a minimal VCF for CADD scoring.

    Emits GRCh38 records with the ``chr`` prefix stripped (CADD convention). Duplicate
    ``(chrom,pos,ref,alt)`` variants are de-duplicated.

    Args:
        benchmark_tsv: The harness benchmark TSV.
        out_vcf: Output VCF path.

    Returns:
        Number of unique variants written.
    """
    seen: set[tuple[str, str, str, str]] = set()
    with open(benchmark_tsv, newline="") as handle, open(out_vcf, "w") as out:
        out.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\n")
        for row in csv.DictReader(handle, delimiter="\t"):
            chrom = _norm_chrom(row["chrom"])
            key = (chrom, row["pos"], row["ref"].upper(), row["alt"].upper())
            if key in seen:
                continue
            seen.add(key)
            out.write(f"{chrom}\t{row['pos']}\t.\t{row['ref'].upper()}\t{row['alt'].upper()}\n")
    return len(seen)


CADD_SNV_URL = "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz"


def annotate_from_remote_cadd(
    benchmark_tsv: str | os.PathLike[str],
    out_tsv: str | os.PathLike[str],
    url: str = CADD_SNV_URL,
) -> tuple[int, int]:  # pragma: no cover - network + pysam
    """Annotate the benchmark's ``cadd`` column directly from CADD's pre-scored file.

    Bypasses the CADD web form: reads the remote tabix-indexed whole-genome SNV file
    over HTTP range requests, fetching one region per element (fast), and fills the
    ``cadd`` (PHRED) column. Requires ``pysam`` and network access.

    Args:
        benchmark_tsv: The harness benchmark TSV.
        out_tsv: Output path for the CADD-annotated benchmark.
        url: The remote CADD whole-genome SNV file (GRCh38, tabix-indexed).

    Returns:
        ``(n_annotated, n_total)``.
    """
    import collections

    import pysam

    with open(benchmark_tsv, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for r in rows:
        groups[(_norm_chrom(r["chrom"]), r["source"])].append(int(r["pos"]))

    tb = pysam.TabixFile(url)
    lookup: dict[tuple[str, str, str, str], str] = {}
    for (chrom, _elem), positions in groups.items():
        for line in tb.fetch(chrom, min(positions) - 1, max(positions)):
            c, pos, ref, alt, _raw, phred = line.split("\t")
            lookup[(c, pos, ref, alt)] = phred

    fields = list(rows[0]) + (["cadd"] if "cadd" not in rows[0] else [])
    annotated = 0
    for r in rows:
        r["cadd"] = lookup.get((_norm_chrom(r["chrom"]), r["pos"], r["ref"].upper(),
                                r["alt"].upper()), "")
        annotated += bool(r["cadd"])
    with open(out_tsv, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return annotated, len(rows)


def main() -> None:  # pragma: no cover - CLI entry point
    """CLI: export benchmark → VCF, or merge CADD scores → annotated benchmark."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="benchmark TSV → VCF for CADD")
    e.add_argument("-b", "--benchmark", required=True)
    e.add_argument("-o", "--out", required=True)
    m = sub.add_parser("merge", help="merge CADD-scored TSV → benchmark cadd column")
    m.add_argument("-b", "--benchmark", required=True)
    m.add_argument("-c", "--cadd", required=True)
    m.add_argument("-o", "--out", required=True)
    rp = sub.add_parser("remote", help="annotate cadd directly from CADD's remote file (pysam)")
    rp.add_argument("-b", "--benchmark", required=True)
    rp.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        n = export_vcf(args.benchmark, args.out)
        print(f"wrote {n} unique variants to {args.out} — upload to cadd.gs.washington.edu")
    elif args.cmd == "remote":
        annotated, total = annotate_from_remote_cadd(args.benchmark, args.out)
        print(f"annotated {annotated}/{total} rows from remote CADD → {args.out}")
    else:
        annotated, total = annotate_benchmark(args.benchmark, args.cadd, args.out)
        print(f"annotated {annotated}/{total} rows with CADD → {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
