"""Genome plumbing: turn a noncoding variant into ChromBPNet-ready sequence windows.

The deep-learning chromatin model (ChromBPNet) scores a variant by comparing its
prediction on the *reference* sequence window against its prediction on the
*alternate* window — the two are identical except for the single substituted base
at the variant position. This module is the deterministic front door that:

1. Parses a variant string (``chr:pos:ref>alt`` or ``chr:pos ref>alt``, hg38).
2. Extracts the reference sequence window centered on the variant from a genome
   FASTA via :mod:`pyfaidx` (genome path is configurable / env-overridable).
3. Verifies the FASTA base at the variant position matches the declared reference
   allele, then builds the alternate window by substituting the alt allele.

No model, no scoring, no randomness — pure, testable coordinate arithmetic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ChromBPNet's default receptive field / input length. The model consumes a
# fixed-width one-hot window; 2114 bp is the standard ChromBPNet input size.
DEFAULT_WINDOW_LENGTH = 2114

# Environment variable used to locate the genome FASTA when a path is not passed
# explicitly. Keeping the path configurable avoids hard-coding a multi-GB file.
GENOME_ENV_VAR = "REGLENS_GENOME"

# A variant string looks like "chr7:5530601:C>T" or "chr7:5530601 C>T". We accept
# ':' or whitespace between position and alleles, and tolerate an optional 'chr'.
_VARIANT_RE = re.compile(
    r"""^\s*
        (?P<chrom>[\w.]+)      # contig, e.g. chr7 / 7 / chrX
        [:\-]
        (?P<pos>\d+)           # 1-based position
        [:\s]+
        (?P<ref>[ACGTNacgtn]+) # reference allele
        >
        (?P<alt>[ACGTNacgtn]+) # alternate allele
        \s*$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Variant:
    """A single-nucleotide (or short) genomic variant on the hg38 assembly.

    Attributes:
        chrom: Contig name as it appears in the genome FASTA (e.g. ``"chr7"``).
        pos: 1-based genomic position of the (first) reference base.
        ref: Reference allele (uppercase A/C/G/T/N).
        alt: Alternate allele (uppercase A/C/G/T/N).
    """

    chrom: str
    pos: int
    ref: str
    alt: str

    @classmethod
    def parse(cls, spec: str) -> Variant:
        """Parse a variant specification string into a :class:`Variant`.

        Accepts ``chr:pos:ref>alt`` and ``chr:pos ref>alt`` forms, with an
        optional ``chr`` prefix on the contig. Alleles are upper-cased.

        Args:
            spec: The variant string, e.g. ``"chr7:5530601:C>T"``.

        Returns:
            The parsed :class:`Variant`.

        Raises:
            ValueError: If ``spec`` does not match the expected grammar.
        """
        match = _VARIANT_RE.match(spec)
        if match is None:
            raise ValueError(
                f"Could not parse variant {spec!r}; expected 'chr:pos:ref>alt' "
                "(hg38), e.g. 'chr7:5530601:C>T'."
            )
        return cls(
            chrom=match.group("chrom"),
            pos=int(match.group("pos")),
            ref=match.group("ref").upper(),
            alt=match.group("alt").upper(),
        )

    @property
    def is_snv(self) -> bool:
        """Whether this is a single-nucleotide variant (1bp ref and alt)."""
        return len(self.ref) == 1 and len(self.alt) == 1

    def __str__(self) -> str:  # noqa: D105 - trivial repr
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"


@dataclass(frozen=True)
class SequenceWindow:
    """A pair of reference/alternate sequence windows centered on a variant.

    The two sequences have identical length and differ only at ``variant_offset``.

    Attributes:
        chrom: Contig the window was extracted from.
        start: 0-based, inclusive start of the window in genome coordinates.
        end: 0-based, exclusive end of the window in genome coordinates.
        variant_offset: 0-based index of the variant base *within* the window.
        ref_seq: Reference-allele sequence window (uppercase).
        alt_seq: Alternate-allele sequence window (uppercase).
    """

    chrom: str
    start: int
    end: int
    variant_offset: int
    ref_seq: str
    alt_seq: str

    @property
    def length(self) -> int:
        """Length of each window in base pairs."""
        return len(self.ref_seq)


def resolve_genome_path(genome_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the genome FASTA path from an argument or the environment.

    Args:
        genome_path: Explicit path to the genome FASTA. If ``None``, falls back
            to the ``REGLENS_GENOME`` environment variable.

    Returns:
        The resolved filesystem path as a string.

    Raises:
        ValueError: If no path is provided and the environment variable is unset.
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = str(genome_path) if genome_path is not None else os.environ.get(GENOME_ENV_VAR)
    if not resolved:
        raise ValueError(
            "No genome FASTA supplied. Pass genome_path or set the "
            f"{GENOME_ENV_VAR} environment variable to an hg38 FASTA."
        )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Genome FASTA not found: {resolved}")
    return resolved


def build_sequence_windows(
    variant: Variant,
    genome_path: str | os.PathLike[str] | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> SequenceWindow:
    """Build reference/alternate sequence windows for a variant, centered on it.

    The window spans ``window_length`` bases with the variant placed at offset
    ``window_length // 2``. The reference base extracted from the FASTA is checked
    against ``variant.ref``; a mismatch raises (guards against wrong-assembly or
    wrong-strand inputs). The alternate window is the reference window with the
    variant base substituted.

    Args:
        variant: The variant to build windows for. Must be an SNV for the MVP.
        genome_path: Path to the genome FASTA (see :func:`resolve_genome_path`).
        window_length: Total window width in bp. Defaults to ChromBPNet's 2114.

    Returns:
        A :class:`SequenceWindow` holding the ref and alt sequences.

    Raises:
        NotImplementedError: If ``variant`` is not a single-nucleotide variant.
        ValueError: If the window would run off the contig, or if the FASTA
            reference base does not match ``variant.ref``.
    """
    if not variant.is_snv:
        # Indels shift downstream coordinates and change window length handling;
        # deferred until after the SNV path is validated (see spec risks/scope).
        raise NotImplementedError(
            "Only single-nucleotide variants are supported in the MVP; "
            f"got ref={variant.ref!r} alt={variant.alt!r}."
        )

    resolved = resolve_genome_path(genome_path)

    # Imported lazily so the module (and the rest of the package) imports even if
    # pyfaidx is absent; it's only needed when actually reading a FASTA.
    from pyfaidx import Fasta

    genome = Fasta(resolved, sequence_always_upper=True)
    if variant.chrom not in genome:
        raise ValueError(
            f"Contig {variant.chrom!r} not found in genome {resolved}. "
            f"Available (first few): {list(genome.keys())[:5]}"
        )

    contig = genome[variant.chrom]
    contig_len = len(contig)

    # Center the window on the variant. `pos` is 1-based; convert to a 0-based
    # index, then place it at `half` within the window.
    variant_index0 = variant.pos - 1
    half = window_length // 2
    start = variant_index0 - half
    end = start + window_length  # 0-based exclusive

    if start < 0 or end > contig_len:
        raise ValueError(
            f"Window [{start}, {end}) for {variant} runs off contig "
            f"{variant.chrom} (length {contig_len}). Choose a smaller window "
            "or a variant farther from the contig edge."
        )

    # pyfaidx slicing is 0-based, half-open — matches [start, end).
    ref_seq = str(contig[start:end])
    offset = variant_index0 - start  # == half

    observed_ref = ref_seq[offset]
    if observed_ref != variant.ref:
        raise ValueError(
            f"Reference mismatch for {variant}: FASTA has {observed_ref!r} at "
            f"{variant.chrom}:{variant.pos}, but variant declares "
            f"ref={variant.ref!r}. Check the assembly/strand."
        )

    # Build the alternate window by substituting the single variant base.
    alt_seq = ref_seq[:offset] + variant.alt + ref_seq[offset + 1 :]

    return SequenceWindow(
        chrom=variant.chrom,
        start=start,
        end=end,
        variant_offset=offset,
        ref_seq=ref_seq,
        alt_seq=alt_seq,
    )
