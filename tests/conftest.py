"""Shared pytest fixtures for RegLens tests.

Provides a tiny synthetic genome FASTA so genome-plumbing and scoring tests run
fully offline — no hg38 download required. The 'known locus' used across tests is
``chr_test:50:C>T`` on this controlled contig.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# A deterministic 200 bp test contig. Position 50 (1-based) is pinned to 'C' so we
# have a stable, known reference base to assert against.
_CONTIG_LEN = 200
_KNOWN_POS = 50  # 1-based
_KNOWN_REF = "C"


def _make_test_sequence() -> str:
    """Build a deterministic ACGT sequence with a known base at ``_KNOWN_POS``."""
    # Simple fixed cycle so the sequence is reproducible without any RNG.
    cycle = "ACGTACGTGGCCAATT"
    seq = list((cycle * ((_CONTIG_LEN // len(cycle)) + 1))[:_CONTIG_LEN])
    seq[_KNOWN_POS - 1] = _KNOWN_REF
    return "".join(seq)


@pytest.fixture(scope="session")
def known_locus() -> dict:
    """Coordinates/alleles of the known test locus."""
    return {"chrom": "chr_test", "pos": _KNOWN_POS, "ref": _KNOWN_REF, "alt": "T"}


@pytest.fixture(scope="session")
def test_genome(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a tiny synthetic genome FASTA and return its path.

    pyfaidx will create the ``.fai`` index alongside it on first access.
    """
    seq = _make_test_sequence()
    fasta = tmp_path_factory.mktemp("genome") / "test.fa"
    with fasta.open("w") as handle:
        handle.write(">chr_test synthetic test contig\n")
        for line in textwrap.wrap(seq, 60):
            handle.write(line + "\n")
    return fasta
