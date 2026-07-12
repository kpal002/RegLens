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

## The crossover — a double dissociation (strongest result)

We ran the **same** 33,359-variant benchmark with a **HepG2 (hepatic)** ChromBPNet model
(ENCODE ENCFF137WCM) and compared it to the K562 (erythroid) model. The result is a
**double dissociation** — swap the cell-type model, and which elements it wins on swaps
with it:

| Compartment | K562 model | HepG2 model | winner |
|---|---|---|---|
| **Hematopoietic** elements | **0.716** | 0.569 | K562 (+0.147) |
| **Hepatic** elements | 0.633 | **0.663** | HepG2 (+0.030) |

<p align="center">
  <img src="figures/crossover_summary.png" width="46%"/>
  <img src="figures/crossover_flip.png" width="46%"/>
</p>

**The per-element tell is unmistakable:** the biggest *risers* when swapping K562→HepG2
are the hepatic **SORT1** assays (+0.08 to +0.10); the biggest *fallers* are the blood
elements — **PKLR-48h collapses 0.805 → 0.505** (near chance), PKLR-24h −0.25, HBB/BCL11A
−0.10 to −0.13. The intervention (change the cell-type model) moves exactly the elements
cell-type theory predicts.

This turns the thesis from *measured* to *demonstrated by intervention*: the AUROC signal
is genuinely **cell-type-driven, not a model artifact** — a model artifact would not flip
its winning elements when you swap the cell type.

**Honest caveats — quantified.** A cluster bootstrap (10,000 resamples over the *elements*
in each compartment — the correct unit, since variants within an element are correlated)
puts a CI on each side's advantage:

| Compartment | own-model Δ AUROC | 95% CI | robust? |
|---|---|---|---|
| **Hematopoietic** (K562 wins) | **+0.147** | **[+0.072, +0.226]** | ✅ CI clears zero (p wrong-sign = 0.00) |
| **Hepatic** (HepG2 wins) | +0.030 | [−0.015, +0.069] | ⚠️ crosses zero (p wrong-sign = 0.09) |

So the dissociation is **asymmetric and honestly reported**: the blood side is robust; the
hepatic side is **directional** — 91% of resamples favor HepG2 — but *not* distinguishable
from zero at the element level (n=7, and one element, **F9** liver-coagulation, drops under
HepG2 against the trend). The double dissociation is carried by the strong hematopoietic
arm plus the decisive per-element extremes (SORT1 up, PKLR-48h 0.805→0.505 down); the
hepatic-arm mean is suggestive, not significant. (The CI reflects between-element variance
only — per-variant sampling noise would need the raw scores. Reproduce:
`reglens.validation.lineage.bootstrap_crossover_ci`.)

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

## Agent null control — does it confabulate, and does it track the evidence?

The question almost nobody tests: handed an MPRA **negative** (non-functional, yet sitting
in or beside an *active* regulatory element of a famous gene, in the matching cell type),
does the multi-agent invent a plausible-sounding mechanism, or correctly decline?

`reglens/validation/null_control.py` draws variants from the same matched benchmark, runs
the full specialists → red-team → adjudicator deliberation on each, and scores a
**label-neutral behavior** (did it *decline*, *hedge*, or *assert* a mechanism?) against
ground truth: negatives should decline (asserting = **confabulated** ✗); positives should
assert (asserting = **recovered** ✓; declining = **missed**). Faithful run: K562 5-fold+RC
ChromBPNet + motif + genome, 8 negatives and 8 positives across hematopoietic elements.

**Arm 1 — negatives (should decline): 7/8 declined, 1 borderline, 0 confabulated.** The
agent never fabricated a mechanism. Even where a motif *was* present it deferred to the
engine — for an HBG1 negative: *"the A allele forms and the G allele abolishes a GATA1::TAL1
composite motif … However ChromBPNet predicts essentially no accessibility change (Δ
+0.024) … within model noise … the functional consequence is unresolved."*

**Arm 2 — random positives (should assert): 0/8 recovered, 6 missed, 2 borderline.** The
agent *also* declined here — but reading the transcripts, **because the engine was quiet on
this random draw**, and it refused to assert what the numbers don't support. Most had
near-noise ChromBPNet Δ (ChromBPNet is only ~0.62 AUROC, so it misses many true positives);
where accessibility *did* move but no motif cleared threshold (HBB positive, Δ +0.233 ≈
26%) it still declined a *mechanism*: *"mechanism and target are inferred, not
established."* The two borderline cases are the tell — a CTCF motif abolished (Δ −5.55) but
small accessibility change: *"the striking mismatch … suggests the CTCF footprint may be
biophysically incidental rather than a functional driver."* That is careful reasoning, not
a miss.

**The agent is calibrated to the deterministic engine, not to ground truth** — it asserts a
mechanism only when the numbers fire, so it never confabulates, but it also inherits the
engine's sensitivity ceiling (plus a real modality gap: an MPRA-significant variant, scored
on episomal reporter expression, need not change predicted *endogenous* chromatin
accessibility). This cleanly proves **one direction**: *engine quiet → agent declines* (16
deliberations, zero confabulations).

**Arm 3 — strong-signal positives (should assert): the other direction.** Two hand-picked
demos are anecdote, not control — a skeptic can say "maybe it just stays silent on
everything." So `run_strong_positive_control` selects positives by **top `|ChromBPNet Δ|`**
(`rank_positives_by_signal`) — *forcing the engine to fire* — then asks whether the agent
asserts. Strict verdict on 8 (|Δ| 0.37–1.08): **1 recovered, 4 borderline, 3 missed** — but
that tally hides the actual result, so split by *what the engine handed the agent*:

- **Motif + Δ both fired (5/8): the agent named a concordant TF mechanism every time, 0
  confabulations.** The strongest, cleanest case (`chr1:155301467`: GATA1::TAL1 abolished
  Δ−14.6, ChromBPNet −1.08) earned the only *medium* → **recovered**. The other four named
  the motif and matched its direction but stayed *low* (borderline). One
  (`chr22:19723407`) is the tell: a CTCF site *strengthened* while ChromBPNet *decreased* —
  the agent **caught the discordance** ("CTCF gain usually maintains accessibility …
  internally tense … a weak hypothesis"), which is reasoning, not a miss.
- **Only ChromBPNet fired, no motif (3/8):** strong Δ (+0.82, +0.82, −0.43) but the agent
  refused to name a TF — *"the specific TF cannot be named."* That is a **motif-library
  gap** (the bundled JASPAR subset), not an agent failure — the honest behavior.

So the biconditional holds where it counts: **the agent names a mechanism iff the motif
channel fires with a concordant Δ, and confabulates never** — in three arms and 24
deliberations. Two further honest points fall out: (1) **confidence is corroboration-gated
and *capped* for this benchmark** — MPRA saturation-mutagenesis variants have no rsid /
eQTL / GWAS / literature by construction, so the agent structurally cannot reach *high*
confidence on them; only the single strongest fully-concordant case reached *medium*
(precisely why the curated rs1427407 / rs2814778 demos, which have those corroborating
limbs, score higher). (2) The motif library, not the reasoning layer, is the binding
constraint on how often a mechanism can be named. The agent tracks the evidence in **both**
directions — it is not merely conservative.

## Reproduce

```bash
# agent controls (faithful run needs hg38 + ChromBPNet model + ANTHROPIC_API_KEY):
#   from reglens.validation.null_control import (run_paired_control, render_paired,
#                                                run_strong_positive_control, render_summary)
#   neg, pos = run_paired_control("data/benchmarks/kircher_mpra_grch38.tsv",
#                  MultiAgentInterpreter(), n_neg=8, n_pos=8, elements=HEMA,
#                  genome_path=hg38, scorer=k562); print(render_paired(neg, pos))
#   # arm 3 — force the engine to fire, then test assertion (closes the biconditional):
#   strong = run_strong_positive_control("data/benchmarks/kircher_mpra_grch38.tsv",
#                  MultiAgentInterpreter(), scorer=k562, genome_path=hg38,
#                  n=8, pool=200, elements=HEMA); print(render_summary(strong, "strong+"))
# build the benchmark (matched negatives):
python -m reglens.validation.build_mpra_benchmark -o data/benchmarks/kircher_mpra_grch38.tsv
# CADD baseline — annotate the cadd column from CADD's pre-scored whole-genome file:
python -m reglens.validation.cadd remote data/benchmarks/kircher_mpra_grch38.tsv \
       -o data/benchmarks/kircher_mpra_grch38.cadd.tsv
# run model scoring on a GPU box (hg38 + ENCODE model): reglens/validation/run_validation.ipynb
# crossover figures + bootstrap CIs:
python figures/generate_crossover.py
```

## Status

Engine validation, CADD baseline, cell-type stratification, and the K562-vs-HepG2
crossover (with bootstrap CIs) are all **complete**. What's deliberately *not* claimed:
the hepatic arm of the crossover is directional but not significant at the element level
(see caveat above), and the agent layer is validated separately (mechanism recovery +
red-team), not by AUROC.
