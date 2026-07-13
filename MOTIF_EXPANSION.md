# Motif-library expansion + statistical gate

Expands RegLens's motif library from a 3-motif curated subset to the full JASPAR CORE
2024 vertebrates non-redundant set (879 matrices), and adds the significance gate that
makes a library that size *honest* rather than a confabulation source.

## Why the gate is the real work

The library swap alone is a one-line data change — the loader already parsed standard
JASPAR PFM format. But scanning ~880 short PWMs per variant and picking the max is an
extreme-order-statistic problem: on random genomic-background sequence the naive
`max |Δ|` selection fired a confident (wrong) motif call **100% of the time** (measured).
Expanding the library without a gate would move a false-positive problem out of the agent
layer — where RegLens measures it at ~0% — and into the deterministic tool.

The wrong fix and why: a per-site binding p-value doesn't separate signal from noise here,
because short PWMs genuinely match random sequence (median top |Δ| ≈ 15 bits on noise).
The discriminator that *does* separate them is **binding strength** — a real occurrence
binds far harder (a real CTCF site ≈ 26 bits) than chance ever produces across the library
(family-wise binding-noise p95 ≈ 15 bits).

## What the gate does

A hit is kept only if **(a)** its site's best-binding score beats the `(1 − alpha)`
quantile of an **empirical family-wise binding null** — calibrated once per library on
random background windows — **and (b)** the variant actually changes the site
(`|Δ| ≥ DISRUPT_THRESHOLD`). This controls the family-wise false-positive rate at ~`alpha`
regardless of library size.

Measured after the change (`alpha = 0.05`):
- False-positive rate on random sequence: **5.0%** (was 100%).
- Real embedded CTCF site recovered: **4/4** contexts, empirical exceedance 0.000.

## Changes

- `DEFAULT_MOTIF_DB` → full JASPAR CORE vertebrates (879 motifs); 3-motif subset kept as
  `SUBSET_MOTIF_DB` for fast, version-pinned unit tests.
- Background switched uniform → genomic (hg38 ~29% A/T) for the log-odds conversion.
- `DEFAULT_FLANK` 30 → 40 (the full set has matrices up to 33 bp).
- New `calibrate_binding_null()` + empirical gate wired into `motif_effect()`; new `alpha`
  and `null_panel_size` params (`alpha=1.0` disables the gate = old prefilter-only
  behavior). Each hit now carries an empirical-exceedance `p_value`, surfaced in the report
  schema.
- Precomputed default null bundled (`binding_null.default.json`) so a single-variant run
  loads the threshold instantly (~0.08 s) instead of paying ~40 s of simulation; validated
  against its exact config on load, else falls through to live calibration.
- Tests: 4 new regression tests locking in the false-positive gate, real-site recovery, the
  disable switch, and null determinism. Full suite **207 passed, 1 skipped**, ruff clean.

## Regenerating the bundled null

Whenever the default library or any default gate parameter changes, regenerate
`reglens/data/motifs/binding_null.default.json` — see the snippet in that directory's
`README.md`.