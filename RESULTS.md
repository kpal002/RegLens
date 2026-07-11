# RegLens — validation results

Validates the **engine** (the ChromBPNet variant score, `|Δ log-counts|`) at ranking
functional regulatory variants above non-functional ones. This is a *separate* claim
from the **agent** (multi-agent mechanistic interpretation), which is validated by
recovering known mechanisms (rs1427407 BCL11A/GATA1, rs2814778 ACKR1/Duffy) and by the
red-team catching artifacts.

## Benchmark

**Kircher et al. 2019 saturation-mutagenesis MPRA** (GSE126550) — the matched, hard
comparison. Positives = variants with a significant regulatory effect (`pValue < 0.01`);
negatives = variants **in the same regulatory elements** with no effect (`pValue ≥ 0.10`).
Negatives are **matched by design**, not random genomic controls — the honest test.

- **33,359 SNVs · 29 element assays · 11,973 positive / 21,386 negative**, GRCh38.
- Model: **ENCODE K562 ATAC ChromBPNet** (ENCFF984RAF), 5-fold + reverse-complement
  averaged. Scored in ~26 min (batched); 0 scoring errors.

## Headline

| | AUROC |
|---|---|
| **Overall (matched, within-element)** | **0.622** |
| Baseline (CADD) | *pending annotation* |

0.622 on a within-element matched comparison is a modest-but-real, above-chance
signal — exactly the range expected on this hard task. Not inflated.

## Cell-type specificity (the real story)

The K562 model is **erythroid/hematopoietic**, and per-element AUROC shows it:

| Group | Mean AUROC | n |
|---|---|---|
| **Hematopoietic elements** (BCL11A, HBB, HBG1, PKLR-24h/48h, GP1BA) | **0.716** | 6 |
| Other elements | 0.601 | 23 |
| — of which tissue-specific developmental (FOXE1, RET, IRF6, IRF4, TCF7L2, ZRS×2, MYC) | **0.514** (≈ chance) | 8 |

**The erythroid model discriminates blood-regulatory variants (~0.72) but is at chance
on thyroid/gut/limb enhancers (~0.51).** That is RegLens's cell-type-specificity thesis,
measured — the right cell-type model matters.

**Honest caveats:** it's a tendency, not a clean dichotomy. A few broadly-active elements
(TERT ~0.67, LDLR ~0.69, UC88 0.71) also score moderately in K562. The best element is
PKLR (0.79–0.81, red-cell pyruvate kinase — a lovely result); the worst is FOXE1 (0.43,
thyroid). All numbers are on the matched benchmark; none are cherry-picked negatives.

## Full per-element AUROC

| Element | AUROC | pos / neg | | Element | AUROC | pos / neg |
|---|---|---|---|---|---|---|
| PKLR-48h | 0.805 | 464 / 717 | | LDLR.2 | 0.679 | 499 / 324 |
| PKLR-24h | 0.794 | 409 / 796 | | HBG1 | 0.663 | 285 / 403 |
| GP1BA | 0.729 | 313 / 668 | | TERT-GAa | 0.649 | 302 / 346 |
| UC88 | 0.709 | 184 / 1310 | | F9 | 0.642 | 280 / 490 |
| TERT-HEK | 0.705 | 238 / 399 | | SORT1-flip | 0.638 | 938 / 635 |
| LDLR | 0.691 | 406 / 405 | | ZFAND3 | 0.637 | 542 / 961 |
| HBB | 0.684 | 186 / 276 | | MYCrs6983267 | 0.634 | 219 / 1282 |
| TERT-GSc | 0.672 | 335 / 333 | | BCL11A | 0.620 | 199 / 1324 |
| TERT-GBM | 0.672 | 390 / 275 | | HNF4A | 0.611 | 263 / 451 |
| MSMB | 0.606 | 640 / 853 | | SORT1 | 0.586 | 1014 / 554 |
| SORT1.2 | 0.584 | 916 / 665 | | ZRSh-13h2 | 0.564 | 334 / 864 |
| IRF4 | 0.548 | 847 / 356 | | RET | 0.546 | 405 / 1081 |
| ZRSh-13 | 0.538 | 336 / 881 | | IRF6 | 0.533 | 620 / 862 |
| TCF7L2 | 0.487 | 281 / 1212 | | MYCrs11986220 | 0.463 | 63 / 1152 |
| FOXE1 | 0.430 | 65 / 1511 | | | | |

## Reproduce

```bash
# build the benchmark (matched negatives):
python -m reglens.validation.build_mpra_benchmark -o data/benchmarks/kircher_mpra_grch38.tsv
# run on a GPU box (hg38 + ENCODE model), see reglens/validation/run_validation.ipynb
```

## Remaining

- **CADD baseline** — annotate a `cadd` column (CADD web service → `validation/cadd.py`)
  to report the model-vs-CADD comparison and the "beats CADD" claim.
