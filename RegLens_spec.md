# RegLens — an agentic mechanistic interpreter for noncoding regulatory variants

*Builder-track spec, Built with Claude: Life Sciences. Submit Mon 9:00 PM ET. (Name is a placeholder.)*

---

## 0. One-liner

Annotation tools tell you *where* a noncoding variant is (VEP/ANNOVAR: "intergenic, modifier"); sequence models score *that* it has an effect (ChromBPNet/Enformer) but don't explain it. RegLens is the **bridge**: give it a noncoding variant + a cell-type context, and a deep-learning chromatin model + a multi-agent reasoning layer produce a **mechanistic, cell-type-specific, cited interpretation** — which TF motif is disrupted, in which regulatory element, affecting which gene, linked to which trait, and why.

## 1. Thesis & wedge

- The **model** (ChromBPNet variant scoring) is not novel — Enformer, Borzoi, Sei, DeepSEA, ChromBPNet all score regulatory variants.
- The **wedge** is the *interpretation bridge*: nobody has built an agentic layer that turns a raw chromatin-effect score into a defensible, cited biological mechanism a researcher can act on. That's the Claude-native, demoable contribution.
- Honest framing: we don't claim to out-predict the sequence models; we claim to **explain** their output, grounded in real annotations + literature.

## 2. User

A researcher (statistical geneticist, disease-genomics lab, variant-curation scientist) staring at a noncoding GWAS/eQTL hit that annotation tools call "intergenic, unknown significance." They need a mechanistic hypothesis: what does this variant actually do, in which cell type, to which gene?

## 3. Architecture (two layers — ports from fold-critic)

```
 variant (chr:pos ref>alt, hg38) + cell-type context
        │
        ▼
 DETERMINISTIC TOOL LAYER (no LLM; exposed as MCP)
   chrombpnet_score   → Δ accessibility (ref vs alt), effect size + direction
   motif_effect       → in-silico mutagenesis around the site → disrupted/created TF motif (JASPAR/HOCOMOCO)
   regulatory_context → ENCODE cCRE / Ensembl Regulatory overlap; in-peak?
   gene_target        → nearest gene + GTEx eQTL (does it regulate a gene's expression?)
   trait_link         → GWAS Catalog / Open Targets Genetics association (+ LD proxies)
   literature         → Europe PMC (variant / gene / TF / trait), real PMIDs
        │ structured signals (JSON)
        ▼
 MULTI-AGENT REASONING LAYER
   regulatory-effect agent · cell-type-context agent · gene-target agent · trait-link agent
        │
        ▼  (red-team optional: "is this just a model artifact / LD hitchhiker?")
   ADJUDICATOR → cited mechanistic interpretation + confidence + caveats
        │
        ▼
   mechanistic variant report (JSON + HTML)
```

- **Deterministic layer** computes every number (Δ-accessibility, motif scores, overlaps). Agents reason over outputs; never invent scores.
- **Agentic layer** interprets, (optionally) red-teams, and adjudicates a cited mechanism.

## 4. The ChromBPNet engine (hybrid: pretrained first, train your own second)

- **Pretrained first (de-risks the product):** use a pretrained ChromBPNet model (Corces brain models / ENCODE cell types) to score variants immediately, so the pipeline works day one. Variant scoring = inference; runs on Colab/Kaggle GPU or even CPU for single variants.
- **Train your own (your ML component + validation):** fine-tune/train one ChromBPNet model on an ENCODE ATAC-seq experiment on **Colab Pro** (overnight, ~12 h) or **Kaggle** (free GPU-hours). This is the real deep-learning training + becomes your validation model.
- **Motif effect:** for MVP use **in-silico mutagenesis** (score ref vs all alts at the position + flanking bases) and match the disrupted window to a JASPAR/HOCOMOCO motif — simpler than full TF-MoDISco, good enough to name the TF.

## 5. Data (all open)

| Need | Source |
|---|---|
| Chromatin model + training data | ChromBPNet (kundajelab), ENCODE ATAC-seq, pretrained Corces/ENCODE models |
| Genome | hg38 FASTA (pyfaidx / genomepy) |
| TF motifs | JASPAR, HOCOMOCO |
| Regulatory elements | ENCODE SCREEN cCREs, Ensembl Regulatory Build |
| Gene target / eQTL | GTEx API |
| Trait link | GWAS Catalog, Open Targets Genetics |
| Literature | Europe PMC |
| Validation benchmark | MPRA / CAGI regulatory saturation-mutagenesis datasets; fine-mapped causal variants |

## 6. Validation (two levels)

- **Model level (your trained ChromBPNet):** does its variant Δ-scores discriminate **known causal/regulatory variants** (MPRA-validated or fine-mapped) from benign? Report **AUROC** vs. a naive baseline (e.g., conservation/CADD). Reuse the fold-critic AUROC harness verbatim.
- **End-to-end (the demo):** on 1–2 well-characterized noncoding variants with a *known* mechanism, show RegLens recovers it (correct TF, gene, cell type) with citations — where VEP/ANNOVAR said "intergenic, modifier."

## 7. Demo money-shot

A noncoding GWAS variant annotation tools shrug at → RegLens returns: "reduces chromatin accessibility (Δ = …) by disrupting a **GATA** motif in an enhancer active in **[cell type]**; GTEx eQTL links it to **[gene]**; GWAS-associated with **[trait]** (PMID …)." The mechanistic story annotation can't give.

## 8. Stack

Python. ChromBPNet (TF/Keras) on Colab/Kaggle for the model; agents via Claude Agent SDK; genome/seq handling (pyfaidx, pyBigWig, pysam); tools exposed as an MCP server. CLI (`typer`) + minimal Streamlit report. Type hints + Google-style docstrings; pytest.

## 9. Repo layout

```
reglens/
  tools/    # chrombpnet_score, motif_effect, regulatory_context, gene_target, trait_link, literature
  mcp_server.py
  model/    # training + inference wrappers for ChromBPNet (Colab/Kaggle notebooks + loaders)
  agents/   # regulatory_effect, celltype_context, gene_target, trait_link, redteam, adjudicator
  orchestrator.py
  report/   # schema + renderer
  validation/  # MPRA/CAGI harness, AUROC vs baseline
  cli.py ; app.py ; data/cases/
tests/ ; README.md ; LICENSE (MIT/Apache-2.0)
```

## 10. Day-by-day (Wed → Mon)

- **Wed (rest of today):** scaffold + LICENSE. Genome plumbing (variant → sequence windows). **Pretrained** ChromBPNet scoring one example variant end-to-end (ref vs alt Δ). Ship the core signal.
- **Thu:** add tools — motif effect (ISM), cCRE/regulatory overlap, nearest gene + GTEx eQTL, GWAS/Open Targets, Europe PMC. Single-agent interpretation working end-to-end on the example variant.
- **Fri:** refactor to multi-agent (specialists + adjudicator) + cited report. **In parallel, kick off your own ChromBPNet training run on Colab Pro (overnight).**
- **Sat:** validation harness — AUROC of your model on MPRA/CAGI known-regulatory vs benign; curate 1–2 demo variants with known mechanism; Streamlit/CLI demo.
- **Sun:** polish, README, dry-run demo, freeze.
- **Mon:** record 3-min demo (annotation-shrugs-vs-RegLens-explains + your trained-model AUROC), 100–200-word summary, license/rules check, **submit by 9 PM ET.**

## 11. Judging alignment

- **Impact (25%):** noncoding variants are the majority of GWAS hits and the hardest to interpret; a mechanistic-hypothesis generator is high-value.
- **Claude use (25%):** multi-agent reasoning bridging a deep model's output to cited biological mechanism — creative, non-trivial.
- **Depth (20%):** you trained a real deep-learning model + validated it (AUROC) + grounded interpretation in real annotations/literature.
- **Demo (30%):** the "annotation shrugs, RegLens explains" reveal + validation plot.

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ChromBPNet setup/data pipeline fiddly | Pretrained + single variant first; expand only after it works |
| Full motif interpretation (TF-MoDISco) too heavy | Use in-silico mutagenesis + JASPAR match for MVP |
| Training run eats time | It's *parallel* on Colab; product works on pretrained regardless |
| Multi-agent coordination | Single-agent end-to-end first (Thu), refactor Fri |
| Benchmark access | Confirm MPRA/CAGI dataset download Wed/Thu |
| Scope creep | One or two cell types, not genome-wide |

## 13. Honest positioning

Sequence-model variant scoring is established; RegLens's contribution is the **agentic mechanistic interpretation + integration + citation** on top, plus a validated model you trained. Frame it as "explains what the model sees," never as "beats the model."

## 14. Rules checklist

- [ ] MIT/Apache-2.0 committed before submission
- [ ] Built from scratch during the event
- [ ] Public data/tools only (all sources above qualify)
- [ ] Team ≤ 2
