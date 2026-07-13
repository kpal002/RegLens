# Discovery worked example: rs342293 — the guardrail working

The prospective screen flagged **rs342293** as its top sparse-literature lead. The mandatory
**manual novelty check found it is already characterized** — so **we do not claim it**. This
is the integrity system functioning exactly as designed, and it exposed two limits of our
own pipeline. We keep it as a worked example precisely because the honest negative is more
informative than a manufactured positive.

## What the screen saw (and what the agent would have proposed)

- `chr7:106731773 C>A`, in a proximal-enhancer cCRE (EH38E2579596, distance 0).
- ChromBPNet (K562) `|Δ log-counts| = 0.22`; a **concordant GATA1 motif**; robust
  **platelet-count** GWAS; **no** GTEx eQTL; our literature tool returned **1 hit**.
- On those signals the agent would have proposed: *a GATA1 site in a megakaryocyte enhancer
  modulating platelet count* — plausible, coherent, and **wrong in its specifics**.

## What the manual check found (real literature, not our tool)

rs342293 at **7q22.3** is an established platelet-trait locus with a **published regulatory
mechanism**: the major (C) allele is bound by the transcription factor **EVI1/MECOM**, which
**represses PIK3CG** transcription; the effect is **megakaryocyte-specific** (explicitly not
associated with red-blood-cell traits).

- Association: *A novel variant on chromosome 7q22.3 associated with mean platelet volume,
  counts, and function* — PMID **19221038**.
- Functional follow-up: *Maps of open chromatin guide the functional follow-up of GWAS
  signals: application to hematological traits* — Paul et al. 2011 (PMC3128100).

## Why this is the system working, not a failure

1. **Manual verification caught it.** Our own literature tool returned only 1 hit — it
   **missed the primary papers**. Had we trusted the tool, we would have made a false
   novelty claim. The rule "*verify by hand; never trust the literature tool*" did exactly
   its job. This is the single best argument for the discipline.
2. **It exposed a motif-call divergence.** Our motif tool named **GATA1**; the characterized
   factor is **EVI1/MECOM**. GATA1 was the closest match in our bundled JASPAR subset, not
   the operative TF — an honest instance of the **motif-library ceiling** (see Limitations).
3. **Cell-type was off, as our own caveats predict.** K562 is erythroid; the real mechanism
   is **megakaryocyte-specific**. The 0.22 K562 signal is a partial off-lineage echo, not
   the operative effect — consistent with the calibration finding that the engine's edge is
   lineage-bound.

## Outcome

**No novel prospective hypothesis survived verification this pass.** That is a truthful and
common result of rigorous screening — and it is *stronger* than a manufactured discovery: it
shows the tool refuses to rubber-stamp (0 quadrant hits across 100 unbiased GWAS variants),
and it shows the human-in-the-loop guardrail catching a case our automation got wrong.
Surfacing a genuinely novel, defensible hypothesis would require **fine-mapped credible-set
variants** (enriched for the causal allele) plus deeper manual vetting — a stated future
direction, not a hackathon deliverable. We report this honestly rather than force a claim.

_(The generic write-up scaffold for a future genuine hit remains at
[`prospective_hypothesis.md`](prospective_hypothesis.md).)_
