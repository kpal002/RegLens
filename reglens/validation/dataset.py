"""Labeled variant sets for validation (regulatory/causal vs benign).

A tiny TSV format so any benchmark (MPRA saturation-mutagenesis hits, CAGI regulatory
variants, fine-mapped causal variants) can be scored: one variant per row with a binary
label and optional baseline-score columns (e.g. CADD, phyloP) for the naive comparator.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from reglens.genome import Variant


@dataclass
class LabeledVariant:
    """A variant with a ground-truth regulatory label and optional annotations.

    Attributes:
        variant: The variant.
        label: 1 = regulatory / causal (positive), 0 = benign (negative).
        rsid: dbSNP rsID, if known.
        source: Provenance of the label (e.g. ``"MPRA"``, ``"CAGI"``, ``"fine-mapped"``).
        annotations: Extra numeric columns (e.g. ``{"cadd": 22.1, "phylop": 3.4}``)
            usable as baseline scores.
    """

    variant: Variant
    label: int
    rsid: str | None = None
    source: str | None = None
    annotations: dict[str, float] = field(default_factory=dict)


# Required TSV columns; everything else numeric is loaded into `annotations`.
_REQUIRED = ("chrom", "pos", "ref", "alt", "label")


def load_labeled_variants(path: str | os.PathLike[str]) -> list[LabeledVariant]:
    """Load a labeled variant set from a TSV file.

    The file must have a header with at least ``chrom, pos, ref, alt, label``. Optional
    columns ``rsid`` and ``source`` are read as strings; any other column whose value
    parses as a float is stored in ``annotations`` (for baseline scores like ``cadd``).

    Args:
        path: Path to the TSV file.

    Returns:
        The parsed labeled variants.

    Raises:
        ValueError: If required columns are missing or a label is not 0/1.
    """
    out: list[LabeledVariant] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"TSV {path} missing required columns: {missing}")
        known = set(_REQUIRED) | {"rsid", "source"}
        for row in reader:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"label must be 0 or 1, got {label!r}")
            annotations: dict[str, float] = {}
            for key, value in row.items():
                if key in known or value in (None, ""):
                    continue
                try:
                    annotations[key] = float(value)
                except ValueError:
                    continue  # non-numeric extra column → ignore
            out.append(
                LabeledVariant(
                    variant=Variant(
                        chrom=row["chrom"], pos=int(row["pos"]),
                        ref=row["ref"].upper(), alt=row["alt"].upper(),
                    ),
                    label=label,
                    rsid=row.get("rsid") or None,
                    source=row.get("source") or None,
                    annotations=annotations,
                )
            )
    return out
