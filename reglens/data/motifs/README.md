# Bundled TF motifs

`jaspar_core_subset.jaspar` — a small, hand-curated subset of the JASPAR CORE 2024
vertebrate collection in JASPAR PFM format, bundled so `motif_effect` runs offline.

| Matrix ID | TF | Width | Why it's here |
|---|---|---|---|
| MA0035.4 | GATA1 | 11 | Erythroid master TF; the money-shot motif for rs1427407 |
| MA0140.2 | GATA1::TAL1 | 18 | The erythroid GATA1–TAL1 composite element |
| MA0139.1 | CTCF | 19 | Distinctive insulator motif — a negative-control / sanity check |

`jaspar_core_vertebrates.jaspar` — the **full** JASPAR CORE 2024 vertebrates
*non-redundant* PFM set (~880 matrices), the default library `motif_effect` loads.
This is what lets RegLens name a TF beyond the three curated motifs above. Scanning
~880 short PWMs per variant means random sequence produces spurious high-|Δ| hits, so
the tool gates every call against an empirical family-wise **binding** null (see
`motif_effect.calibrate_binding_null`).

`binding_null.default.json` — the precomputed binding-null threshold + sample for the
default configuration (this library, `flank=40`, `alpha=0.05`, genomic background,
`panel_size=500`, `seed=0`). Bundled so a single-variant run loads the threshold
(~14.9 bits) instantly instead of paying ~40 s of simulation. It is validated against
the exact config on load — any change to library/flank/alpha/background/panel/seed
falls through to live re-calibration. **Regenerate** it whenever the default library
or any default gate parameter changes:

```python
import json
from reglens.tools.motif_effect import (
    load_motifs, calibrate_binding_null, DEFAULT_MOTIF_DB, DEFAULT_FLANK,
    DEFAULT_ALPHA, _GENOMIC_BG, _NULL_PANEL_SIZE, _library_key,
)
full = load_motifs(DEFAULT_MOTIF_DB)
thr, null = calibrate_binding_null(full)  # uses the defaults
lk = _library_key(full)
json.dump({
    "n_motifs": len(full), "flank": DEFAULT_FLANK, "alpha": DEFAULT_ALPHA,
    "background": _GENOMIC_BG, "panel_size": _NULL_PANEL_SIZE, "seed": 0,
    "threshold": thr, "null_sorted": [round(float(x), 4) for x in null],
    "lib_head": list(lk[1]), "lib_tail": list(lk[2]),
}, open("reglens/data/motifs/binding_null.default.json", "w"))
```

JASPAR matrices are released under **CC0** (public domain), so they are safe to
redistribute.

Sources: JASPAR 2024 REST API (`https://jaspar.elixir.no/api/v1/matrix/MA0035.4/`);
full set from the JASPAR 2024 CORE vertebrates non-redundant PFM download.
