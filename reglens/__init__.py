"""RegLens: an agentic mechanistic interpreter for noncoding regulatory variants.

RegLens bridges *what* a deep-learning chromatin model predicts about a noncoding
variant and *why* it matters biologically. The package is organized in two layers:

* A **deterministic tool layer** (this package's ``tools`` subpackage plus the
  genome plumbing here) that computes every number — Δ chromatin accessibility,
  motif effects, regulatory overlaps, eQTL and trait links.
* A **multi-agent reasoning layer** (added later) that reasons *over* those
  numbers to produce a cited mechanistic interpretation.

Only the Wednesday-milestone pieces (genome plumbing + pretrained ChromBPNet
scoring + CLI) are implemented so far. See ``RegLens_spec.md`` for the full design.
"""

__version__ = "0.0.1"
