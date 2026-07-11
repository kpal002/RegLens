"""Build a MATCHED regulatory-variant benchmark from the Kircher satMutMPRA data.

Turns the Kircher et al. 2019 saturation-mutagenesis MPRA table (>30k SNVs in 20
disease-associated regulatory elements; Nature Commun., GSE126550) into a labeled TSV
for :mod:`reglens.validation`. Crucially, positives and negatives come from the **same
regulatory elements** — so the negatives are *matched by design*, not random genomic
controls. Random negatives make the task trivially easy and the AUROC a mirage; this
is the honest, hard comparison.

Labeling (all configurable):
* **positive** (label 1): significant regulatory effect — ``pValue < pos_alpha``
  (optionally ``|Coefficient| >= min_abs_effect``).
* **negative** (label 0): tested in the same element with no effect —
  ``pValue >= neg_alpha``.
* variants in the ambiguous middle are dropped so labels are clean.

SNVs only (our scorer handles substitutions, not the assay's deletions). The element
name is written to the ``source`` column so the matched design stays auditable. Add a
``cadd`` column afterwards (CADD web service / precomputed scores) for the baseline.

Run: ``python -m reglens.validation.build_mpra_benchmark -o data/benchmarks/mpra.tsv``
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# The per-variant Kircher satMutMPRA table (Chrom Pos Ref Alt ... Coefficient pValue
# Element Release), hosted with the lab's Shiny app.
RAW_URL = "https://raw.githubusercontent.com/kircherlab/MPRA_SaturationMutagenesis/master/data/elements.tsv.gz"

_BASES = frozenset("ACGT")


@dataclass
class CurationStats:
    """Summary of a curation run."""

    n_pos: int
    n_neg: int
    dropped_ambiguous: int
    dropped_nonsnv: int
    dropped_wrong_release: int
    per_element: dict[str, tuple[int, int]]  # element → (n_pos, n_neg)


def download_rows(url: str = RAW_URL) -> Iterator[dict[str, str]]:
    """Yield rows of the Kircher table from a URL or local path (gzip-aware)."""
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    else:
        with open(url, "rb") as handle:
            data = handle.read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    yield from csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")


def curate(
    rows: Iterable[dict[str, str]],
    release: str = "GRCh38",
    pos_alpha: float = 0.01,
    neg_alpha: float = 0.10,
    min_abs_effect: float = 0.0,
) -> tuple[list[dict[str, str]], CurationStats]:
    """Label Kircher rows into matched positives/negatives.

    Args:
        rows: Kircher table rows (dicts with Chrom/Pos/Ref/Alt/Coefficient/pValue/
            Element/Release).
        release: Keep only this genome release (``"GRCh38"``).
        pos_alpha: A variant is positive if ``pValue < pos_alpha``.
        neg_alpha: A variant is negative if ``pValue >= neg_alpha``.
        min_abs_effect: Also require ``|Coefficient| >= min_abs_effect`` for positives.

    Returns:
        ``(out_rows, stats)`` where each out row has the harness columns
        ``chrom,pos,ref,alt,label,rsid,source``.
    """
    out: list[dict[str, str]] = []
    dropped_ambiguous = dropped_nonsnv = dropped_wrong_release = 0
    per_element: dict[str, list[int]] = {}

    for r in rows:
        if r.get("Release") != release:
            dropped_wrong_release += 1
            continue
        ref, alt = r["Ref"].upper(), r["Alt"].upper()
        if ref not in _BASES or alt not in _BASES:  # SNVs only (skip deletions "-")
            dropped_nonsnv += 1
            continue
        try:
            pval = float(r["pValue"])
            coef = float(r["Coefficient"])
        except (ValueError, KeyError):
            dropped_nonsnv += 1
            continue

        if pval < pos_alpha and abs(coef) >= min_abs_effect:
            label = 1
        elif pval >= neg_alpha:
            label = 0
        else:
            dropped_ambiguous += 1
            continue

        element = r.get("Element", "")
        chrom = r["Chrom"] if r["Chrom"].startswith("chr") else f"chr{r['Chrom']}"
        out.append({
            "chrom": chrom, "pos": r["Pos"], "ref": ref, "alt": alt,
            "label": str(label), "rsid": "", "source": element,
        })
        per_element.setdefault(element, [0, 0])[label] += 1

    stats = CurationStats(
        n_pos=sum(1 for o in out if o["label"] == "1"),
        n_neg=sum(1 for o in out if o["label"] == "0"),
        dropped_ambiguous=dropped_ambiguous,
        dropped_nonsnv=dropped_nonsnv,
        dropped_wrong_release=dropped_wrong_release,
        per_element={k: (v[1], v[0]) for k, v in per_element.items()},
    )
    return out, stats


def write_tsv(rows: list[dict[str, str]], path: str) -> None:
    """Write curated rows to a harness-compatible TSV."""
    cols = ["chrom", "pos", "ref", "alt", "label", "rsid", "source"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - CLI entry point
    """Download the Kircher table, curate a matched benchmark, and write a TSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", required=True, help="Output TSV path.")
    ap.add_argument("-i", "--input", default=RAW_URL, help="Kircher table URL or local path.")
    ap.add_argument("--pos-alpha", type=float, default=0.01)
    ap.add_argument("--neg-alpha", type=float, default=0.10)
    ap.add_argument("--min-abs-effect", type=float, default=0.0)
    ap.add_argument("--element", default=None, help="Restrict to a single element (e.g. BCL11A).")
    args = ap.parse_args()

    rows = download_rows(args.input)
    if args.element:
        rows = (r for r in rows if r.get("Element") == args.element)
    out, stats = curate(
        rows, pos_alpha=args.pos_alpha, neg_alpha=args.neg_alpha,
        min_abs_effect=args.min_abs_effect,
    )
    write_tsv(out, args.out)
    print(f"wrote {len(out)} variants to {args.out}: {stats.n_pos} pos / {stats.n_neg} neg "
          f"(dropped {stats.dropped_ambiguous} ambiguous, {stats.dropped_nonsnv} non-SNV)")
    print("per-element (pos/neg):")
    for el, (p, n) in sorted(stats.per_element.items()):
        print(f"  {el:12s} {p:5d} / {n:5d}")
    print("\nNext: annotate a `cadd` column for the baseline, then `reglens validate`.")
    print("NOTE: negatives are matched (same elements) — do NOT swap in random controls.")


if __name__ == "__main__":  # pragma: no cover
    main()
