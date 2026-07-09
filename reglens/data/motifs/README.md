# Bundled TF motifs

`jaspar_core_subset.jaspar` — a small, hand-curated subset of the JASPAR CORE 2024
vertebrate collection in JASPAR PFM format, bundled so `motif_effect` runs offline.

| Matrix ID | TF | Width | Why it's here |
|---|---|---|---|
| MA0035.4 | GATA1 | 11 | Erythroid master TF; the money-shot motif for rs1427407 |
| MA0140.2 | GATA1::TAL1 | 18 | The erythroid GATA1–TAL1 composite element |
| MA0139.1 | CTCF | 19 | Distinctive insulator motif — a negative-control / sanity check |

JASPAR matrices are released under **CC0** (public domain), so they are safe to
redistribute. Expand this file with more of the CORE collection as needed, or point
`motif_effect` at a full JASPAR download.

Source: JASPAR 2024 REST API, e.g. `https://jaspar.elixir.no/api/v1/matrix/MA0035.4/`.
