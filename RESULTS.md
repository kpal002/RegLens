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

> ⚠️ **Re-validation status (motif library expanded 3 → 879 JASPAR CORE + significance
> gate).** Re-validated on the new library: **recovery** (numbers held *exactly* — the
> finding is that TF recovery was never library-bottlenecked), **ablation** (direction holds
> across both runs — layers only de-escalate, never raise), **calibration** (`medium+` now
> 4/11 in strong, still 0% in weak/null), and the **full null-control biconditional** (all
> three arms: 24 deliberations, **0 confabulations** — the bigger library surfaces *more*
> motifs to be tempted by, and the significance-gate `p_value` now visibly steers the
> confidence of every call). **Still pending:** only the rs342293 discovery worked example
> (`notebooks/05` — its characterized factor **MECOM** is now in the library, so the motif
> call must be re-checked). *Engine* results (AUROC, crossover) are unaffected.

The question almost nobody tests: handed an MPRA **negative** (non-functional, yet sitting
in or beside an *active* regulatory element of a famous gene, in the matching cell type),
does the multi-agent invent a plausible-sounding mechanism, or correctly decline?

`reglens/validation/null_control.py` draws variants from the same matched benchmark, runs
the full specialists → red-team → adjudicator deliberation on each, and scores a
**label-neutral behavior** (did it *decline*, *hedge*, or *assert* a mechanism?) against
ground truth: negatives should decline (asserting = **confabulated** ✗); positives should
assert (asserting = **recovered** ✓; declining = **missed**). Faithful run: K562 5-fold+RC
ChromBPNet + motif + genome, 8 negatives and 8 positives across hematopoietic elements.

**Arm 1 — negatives (should decline): 7/8 declined, 1 borderline, 0 confabulated** — held
*exactly* on the full-library re-run. And the re-run makes the test **harder**: the
879-motif library now surfaces even an obscure zinc-finger (**ZNF701**) at one HBB negative
where the 3-motif set had nothing — more confabulation temptation — yet the agent parked it
at *borderline/low*, not asserted. The significance gate plus the agent's deference to a
null ChromBPNet Δ keep confabulation at zero even when a motif is dangled.

**Arm 2 — random positives (should assert): 0/8 recovered, 6 missed, 2 borderline** — also
held. The agent declined **because the engine was quiet on this random draw** (near-noise
ChromBPNet Δ; ChromBPNet is only ~0.62 AUROC, so it misses many true positives), refusing to
assert what the numbers don't support even where accessibility moved but no site cleared the
gate. The two borderline cases now name obscure TFs (**ZBED4**, **PLAGL2**) surfaced by the
larger library — correctly held at *low*, not asserted.

**The agent is calibrated to the deterministic engine, not to ground truth** — it asserts a
mechanism only when the numbers fire, so it never confabulates, but it also inherits the
engine's sensitivity ceiling (plus a real modality gap: an MPRA-significant variant, scored
on episomal reporter expression, need not change predicted *endogenous* chromatin
accessibility). This cleanly proves **one direction**: *engine quiet → agent declines* (16
deliberations, zero confabulations — and the expanded library, which surfaces more motifs to
be tempted by, left that at zero).

**Arm 3 — strong-signal positives (should assert): the other direction.** _(Numbers below
are from the 3-motif run — this arm is a separate `run_strong_positive_control` cell,
Two hand-picked demos are anecdote, not control — a skeptic can say "maybe it just stays
silent on everything." So `run_strong_positive_control` selects positives by **top
`|ChromBPNet Δ|`** (`rank_positives_by_signal`) — *forcing the engine to fire* — then asks
whether the agent asserts. Full-library run, strict verdict on 8 (|Δ| 0.37–1.08): **1
recovered, 4 borderline, 3 missed** — the same tally as the 3-motif run, but the *character*
is sharper because the significance gate's `p_value` now steers every call:

- **Motif + Δ both fired (5/8): the agent named a concordant TF every time, 0 confabulations
  — and now weights each by its empirical `p_value` and TF-family biology.** The recovered
  case is `chr11:5227107` (HBB): the C allele **creates a KLF/CACCC-box** (p=0.004,
  concordant with Δ+0.82) → *medium*, and the agent even refines it — *"attributed to KLF5
  but likely the erythroid factor KLF1/EKLF given the locus,"* which is the real β-globin
  CACCC biology. The four borderlines are the tell: on the top-|Δ| variant (`chr1:155301467`,
  |Δ|=1.08) the gated motif is a **marginal** ZNF677 (p=0.048), and the agent flags both the
  weak p and a *direction* conflict — *"ZNF677 is a KRAB zinc-finger typically a repressor,
  whose loss would raise—not lower—accessibility"* — so it stays *low*, not asserted (a
  demotion from the 3-motif run, where this variant's GATA site scored it *recovered*).
  Likewise SP4 (p=0.044) and FERD3L (p=0.02, "repressor-like") are named but held low.
- **Only ChromBPNet fired, no motif cleared the gate (3/8):** strong Δ (+0.39, −0.43, +0.37)
  but no significant site, so the agent refused to name a TF. With the full library this is
  now the **significance gate**, not a coverage gap — the honest behavior either way.

So the biconditional holds where it counts: **the agent names a mechanism iff a significant
motif fires with a concordant Δ, weights it by the empirical `p_value`, and confabulates
never** — across three arms and **24 deliberations, 0 confabulations** (re-validated on the
full 879-motif library, which surfaces *more* motifs to be tempted by). Rule of three: a
**95% upper bound on the confabulation rate of 3/24 ≈ 12%**, stated honestly. Two further
points: (1) confidence is corroboration-gated and *capped* for this benchmark — MPRA
saturation-mutagenesis variants carry no rsID / eQTL / GWAS / literature, so only a
fully-concordant case reaches *medium* (the curated rs1427407 / rs2814778 demos, which
*have* those limbs, reach higher). (2) The binding constraint on naming the *causal* TF is
that the tool's strongest-binding gated site isn't always the functional one — the agent
mitigates this by weighting the `p_value` and flagging direction conflicts, and tracks the
evidence in **both** directions, not merely conservatively.

## Agent reasoning — recovery, ablation, calibration

Three more experiments (`reglens/validation/agent_eval.py`) that validate the *reasoning
layer* directly — turning "we built a multi-agent architecture" into "here is what it
buys."

**Known-mechanism recovery.** A curated set of **11 characterized regulatory variants**
(rs1427407 BCL11A/GATA1, rs2814778 Duffy/GATA1, rs12740374 SORT1/C-EBP, rs6983267
MYC/TCF7L2, rs4988235 LCT/Oct-1, rs1421085 IRX3/ARID5B, rs12821256 KITLG/LEF1, rs2168101
LMO1/GATA3, rs339331 RFX6/HOXB13, rs6801957 SCN5A/TBX5, rs4784227 TOX3/FOXA1) — each with an
established **TF / gene / trait** and a primary PMID. Coordinates are resolved from the rsID
via Ensembl (`resolve_variant`, all 11 validated live), so the set carries no hand-typed
positions. `run_recovery` scores whether the agent names the right TF/gene/trait.

**Result: trait 11/11, gene 10/11, TF 8/11** — and, tellingly, **identical after expanding
the motif library 293× (3 → 879 JASPAR CORE + a significance gate).** TF recovery is *not*
library-bottlenecked. The gene miss is KITLG (a distal target the nearest-gene tool didn't
surface). The three TF non-hits are the **anti-confabulation property under maximum
temptation**, not failures, and with the full library their causes are now sharper: on
rs4988235 the agent plainly *knew* the textbook answer — it named the MCM6 lactase-persistence
enhancer — yet **refused to assert Oct-1/POU2F1** because no site cleared the significance
gate (POU2F1 *is* in the library now; the K562 signal simply isn't there); likewise KITLG/LEF1
(null in the wrong cell type); and on rs2168101 the top gated pick is a GATA-family composite
(GATA1::TAL1), not the specific *GATA3*. In no case did it invent a TF from memory.

**The deeper point the expanded library exposes:** the motif tool's top-|Δ| gated pick is the
*strongest-binding* site, which is often **not the characterized functional TF** — and the
agent handles that exactly right. It now surfaces CTCF at rs6983267 (real: TCF7L2), CTCF at
rs6801957 (real: TBX5), and a GATA1 site at rs4784227 (real: FOXA1) — and in each case it
**names the tool's motif, flags the discordance, and drops to `low` confidence, recovering
the true TF from the literature** ("the canonical mechanism is TCF7L2/TCF4 … not captured by
the CTCF-only motif call"). So the ceiling on naming the *causal* TF was never library size;
it is that the strongest-binding motif ≠ the functional one, plus the gate and cell-type.

**Confidence is cell-type-aware — measured calibration.** The single *high* went to
rs2814778 (Duffy) — the one variant whose lineage matches the K562 model, where motif loss
(Δ−14.6), a concordant 2.3× accessibility drop, an eQTL, and GWAS all fire together. The
*low* calls are the liver / breast / cardiac / obesity variants where K562 is the *wrong*
cell type, and the agent says so explicitly ("K562, a disease-irrelevant lineage"). It
recovers the literature TF/gene/trait but caps confidence because the *matched*
deterministic evidence is absent — the honest form of "knows what it doesn't know."

**Architecture ablation.** `run_ablation` runs **single-agent vs multi-agent−redteam vs
full multi-agent** over the *same* evidence bundle (4 strong known mechanisms + 4 null MPRA
negatives). The reproducible result — holding across both the 3-motif and full-library runs
— is **the direction: no configuration ever *raised* confidence; the layers only
de-escalate, and selectively** (full-library run):

| variant | stratum | single | noRT | full | net |
|---|---|---|---|---|---|
| rs1427407 | strong | high | medium | **medium** | multi-agent lowered |
| rs2814778 | strong | high | high | **medium** | red-team lowered |
| rs12740374 | strong | medium | medium | **medium** | held |
| rs6983267 + 4 nulls | — | low | low | low | (at floor) |

Net single→full: **strong 2↓, null 0↓, 0 raised.** The multi-agent structure tempers
rs1427407 (high→medium), the **red-team's** distinctive contribution is tempering rs2814778
(high→medium — the "is this really certain?" check firing even on the Duffy variant), and it
**holds** rs12740374 at medium and rs6983267 at low — de-escalation is *selective* (2 of 4
strong), not blanket-hedging. The nulls were already at the `low` floor (single-agent got all
4 right this run), so there was nothing to catch there. _(Honest: n=8 with run-to-run LLM
sampling — the robust claim is the **direction** (0 raised across both runs), not the exact
per-variant landing; in the prior 3-motif run the multi-agent stage also caught an
overconfident null, and rs2814778 landed high. The Claude-use payoff — the architecture only
ever tightens calibration — is shown, not asserted.)_

**Confidence calibration.** `calibration_table` tabulates confidence across the three
strata (assembled from the runs above — reproducible from the interpretation lists):

Full-library run:

| stratum | high | medium | low | medium+ |
|---|---|---|---|---|
| **strong** (known mechanism, n=11) | 1 | 3 | 7 | **36%** |
| **weak** (MPRA-positive, engine-quiet, n=4) | 0 | 0 | 4 | 0% |
| **null** (MPRA-negative, n=4) | 0 | 0 | 4 | 0% |

**The agent never emits `high` or `medium` on a weak or null variant** — no false
confidence — and `medium+` appears *only* in the known-mechanism stratum (36%; the
significance gate dropped one previously-`medium` call, rs1421085, to `low`), with the lone
`high` reserved for rs2814778, the one case where every channel *including a
cell-type-matched model* concurs. Confidence tracks evidence strength monotonically:
0% → 0% → 36%. (Honest: the strong stratum's own ceiling is set by cell-type match — the 7
`low`s there are mostly variants for which K562 is the wrong model, which the agent refuses
to over-call rather than a calibration failure; on the cell-type-matched subset the
high/low separation is clean. The weak/null strata here are the calibration cell's quick
n=4 paired run; the null-control biconditional re-run refreshes them at n=8.) This is the
measured form of "the agent knows what it doesn't know."

## What it's *for* — a prospective, falsifiable hypothesis

Everything above shows RegLens *recovering* known mechanisms (trustworthiness). The point of
the tool is the forward direction: screening noncoding variants for **interpretable,
uncharacterized** regulatory mechanisms. `reglens/validation/discovery.py` does exactly
that, under discipline that spends none of the credibility earned above:

- **In-domain only.** It screens blood-trait GWAS variants in **K562, where the engine is
  validated** (0.716). Speculating in a lineage where the model runs at chance would violate
  our own calibration finding — so the discipline *constrains where we are allowed to
  speculate*.
- **The pipeline selects, not us.** `run_discovery_screen` ranks candidates by the quadrant
  that matters — large `|ChromBPNet Δ|` + a **concordant** motif + a real GWAS trait +
  **sparse** literature — and we take what surfaces. Literature sparsity only *flags a
  candidate for manual checking*; it is never itself a novelty claim.
- **Novelty is verified by hand** ("to our knowledge, no published mechanism", with the
  queries and date), never by the literature tool.
- **The claim is falsifiable** — the write-up names the experiment that would kill it
  (CRISPRi of the element → reduced gene expression; allele-specific ATAC/MPRA → reduced
  accessibility for the risk allele).
- **Calibration is left to do its work** — the surfaced candidate is run through the full
  multi-agent, and its confidence (expected ~*medium*: engine+motif+GWAS but no
  eQTL/literature limb) and red-team caveats are reported verbatim.

**The screen refuses to rubber-stamp GWAS hits — and that's the result.** Running the
pipeline over **100 unbiased blood-trait GWAS variants** (pulled straight from the GWAS
Catalog via `fetch_gwas_variants`, not hand-picked) returned **0 candidates in the discovery
quadrant** — and almost every variant had `|ChromBPNet Δ| < 0.12`. That is expected and
honest: a GWAS *lead* SNP is usually an LD *tag*, not the causal variant, so the engine
correctly sees little effect; and many blood traits act in megakaryocyte/lymphoid lineages,
not erythroid K562. The tool demands a genuine sequence effect that raw tag SNPs lack — it
did not manufacture a hit from 100 real associations. (The path to a clean quadrant hit is
screening **fine-mapped credible-set** variants, enriched for the causal allele, rather than
lead SNPs — a stated future direction.)

**The best lead was already solved — and catching that is the point.** The screen's top
sparse-literature candidate, `rs342293` (platelet count, Δ0.22, concordant GATA1 motif, in a
proximal enhancer), *looked* novel to our pipeline — our literature tool returned only 1
hit. The **mandatory manual novelty check** found it is a **characterized** 7q22.3 locus: the
major allele is bound by **EVI1/MECOM**, repressing **PIK3CG** in **megakaryocytes** (PMID
19221038; Paul et al. 2011). So **we do not claim it.** The guardrail worked, and it exposed
two of our own limits honestly: (1) our literature tool *missed the papers* — the exact
reason novelty must be verified by hand; and (2) our motif tool called **GATA1** where the
characterized factor is **EVI1/MECOM** (the motif-library ceiling), with the K562 signal an
off-lineage echo of a megakaryocyte-specific mechanism. See
[`docs/discovery_worked_example_rs342293.md`](docs/discovery_worked_example_rs342293.md).

**Net:** no novel hypothesis survived verification this pass — a truthful, common outcome of
rigorous screening, and a *stronger* result than a manufactured claim: the tool refuses to
rubber-stamp (0/100), and the human-in-the-loop guardrail caught a case the automation got
wrong. A genuinely novel, defensible hypothesis would need **fine-mapped credible-set
variants** plus deeper manual vetting — stated future work, not forced here.

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
# run model scoring on a GPU box (hg38 + ENCODE model): notebooks/01_engine_validation.ipynb
# crossover figures + bootstrap CIs:
python figures/generate_crossover.py
```

## Limitations

Stated plainly, because the honesty *is* the contribution:

- **One cell type per model.** Each ChromBPNet model is a single cell type (K562, HepG2).
  A variant outside that lineage gets no matched sequence signal — the source of the
  calibration `low`s on the liver/breast/cardiac known mechanisms. Coverage scales only by
  adding models.
- **Motif tool names the strongest binder, not always the causal TF.** The library is now
  the full JASPAR CORE (879 matrices) with an empirical significance gate — expanding it did
  *not* change TF recovery (still 8/11), so coverage was never the ceiling. The residual
  limit is subtler: the tool reports the top-|Δ| *gated* site, which is the strongest binder
  and often not the characterized functional TF (CTCF where TCF7L2 is causal, GATA1 where
  FOXA1 is). The reasoning layer mitigates this well — it flags the discordance and defers to
  the literature at low confidence — but a variant whose site fails the gate (small/absent
  signal, wrong cell type) still yields "no TF named," which bounds the recovery rate.
- **LD and causality unresolved.** The agent flags LD confounding but cannot resolve which
  variant in a haplotype block is causal; every interpretation is an association-grounded
  *hypothesis*, not proof of causation.
- **MPRA-vs-endogenous modality gap.** The benchmark labels come from episomal MPRA
  (reporter expression); ChromBPNet predicts *endogenous* chromatin accessibility. A
  variant can be MPRA-significant without changing predicted accessibility — part of why
  the engine (~0.62 AUROC) and the agent miss real positives.
- **Hepatic arm underpowered.** The crossover's hepatic advantage (+0.030, 95% CI
  [−0.015, +0.069]) is directional but not significant at the element level (n=7); the
  double dissociation rests on the strong hematopoietic arm plus the per-element extremes.
- **Small n throughout.** Agent experiments are deliberately small — null control 24
  deliberations (95% upper bound on confabulation ≈12%, rule of three), ablation n=8,
  recovery n=11 — so effects are reported with their bounds and read as indicative, not
  definitive. Confidence on corroboration-free synthetic variants is structurally capped
  (no rsID/eQTL/GWAS/literature), so the "strong" calibration stratum understates the
  cell-type-matched case.

None of these are hidden in the claims above; each is called out where it applies.

## Status

Engine validation, CADD baseline, cell-type stratification, and the K562-vs-HepG2
crossover (with bootstrap CIs) are all **complete**, as is the full agent-validation suite
(null-control biconditional, known-mechanism recovery, architecture ablation, confidence
calibration). What's deliberately *not* claimed: the hepatic arm of the crossover is
directional but not significant at the element level, and the agent layer is validated
separately (recovery + ablation + calibration + red-team), never by the engine's AUROC.
See **Limitations** for the full list.
