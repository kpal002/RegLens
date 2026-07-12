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

## Headline — the model beats CADD, most on its own cell type

| | Model (`\|Δ log-counts\|`) | CADD (baseline) | Δ |
|---|---|---|---|
| **Overall (matched, within-element)** | **0.622** | 0.556 | **+0.066** |
| **Hematopoietic elements** | **0.716** | 0.587 | **+0.129** |
| Other elements | 0.601 | 0.586 | +0.015 |

CADD PHRED v1.7 baseline, computed on the **same** 33,359 variants (pulled from CADD's
pre-scored whole-genome file). The model **beats CADD overall (+0.066)** and wins in
**18 / 29 elements** — but the story is in *where* it wins.

## Cell-type specificity (the real story)

The K562 model is **erythroid/hematopoietic**, and both the absolute AUROC *and* the
margin over CADD track that:

- **On hematopoietic elements** (BCL11A, HBB, HBG1, PKLR-24h/48h, GP1BA): model **0.716**
  vs CADD **0.587** — a **+0.13** margin. The cell-type-appropriate deep model adds large
  signal beyond generic conservation.
- **On non-hematopoietic elements**: model 0.601 vs CADD 0.586 — **≈ tied** (+0.015). With
  the wrong cell type, the model has no edge over CADD.

**A cell-type-matched chromatin model beats a cell-type-agnostic conservation score — but
only in the matching cell type.** That is RegLens's whole thesis, measured. The biggest
per-element wins are all hematopoietic (PKLR-48h +0.175, PKLR-24h +0.200, GP1BA +0.178);
CADD wins on non-erythroid elements (LDLR −0.10 hepatic, IRF4/IRF6 −0.12, ZFAND3 −0.12).

**Honest caveats:** overall +0.066 is a modest margin; the strength is the *stratified*
result. It's a tendency, not perfect (a few broad elements like TERT are close; CADD wins
11/29). All numbers are on the matched benchmark; none are cherry-picked negatives.

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
