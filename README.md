# RegLens

**An agentic mechanistic interpreter for noncoding regulatory variants.**

Annotation tools (VEP/ANNOVAR) tell you *where* a noncoding variant is — "intergenic,
modifier." Sequence models (ChromBPNet/Enformer) score *that* it has an effect but don't
explain it. **RegLens is the bridge:** give it a noncoding variant + a cell-type context,
and a deep-learning chromatin model plus a multi-agent reasoning layer produce a
**cited, cell-type-specific, mechanistic interpretation** — which TF motif is disrupted,
in which regulatory element, plausibly affecting which gene, linked to which trait, and
why — with a calibrated confidence and explicit caveats.

> We *explain what the sequence models see* — we do **not** claim to out-predict them.
> Every interpretation is a hypothesis with confidence + caveats, and every number is
> computed deterministically; the agents reason but never invent a score.

## The demo in one screen

**Recover** — `rs1427407` (chr2:60,490,908 T>G), which VEP calls "intergenic":
> Alt G creates a **GATA1::TAL1** composite motif (3.21→8.66 bits) — i.e. the ref **T
> allele disrupts** the erythroid element; ChromBPNet predicts a concordant K562
> accessibility gain; the variant is **inside BCL11A**'s +58 enhancer; GWAS-linked to
> **fetal hemoglobin** (p=4e-53). *Confidence: medium* — the red-team flagged the small
> ChromBPNet effect, the eQTL pointing to C2orf74 (not BCL11A), and LD confounding.
> Cited: PMID 24115442, 26375006. ✅ matches the textbook mechanism (Bauer/Canver).

**Discover** — `rs2814778` (chr1:159,204,893 T>C), fed in cold:
> Alt C **abolishes a GATA1::TAL1 motif** (16.78→2.18 bits) at the **ACKR1/Duffy**
> erythroid promoter; linked to **neutrophil count**. ✅ RegLens recovered the Duffy-null
> mechanism from scratch — nothing hardcoded.

## Architecture (two layers)

```
 variant (chr:pos ref>alt, hg38) + cell-type
        │
   DETERMINISTIC TOOL LAYER  (no LLM — computes every number)
     chrombpnet_score · motif_effect · regulatory_context · gene_target · trait_link · literature
        │  evidence bundle (JSON)
        ▼
   MULTI-AGENT REASONING LAYER  (Anthropic Messages API, structured output)
     4 specialists  →  red-team  →  adjudicator
        ▼
   cited mechanistic interpretation (confidence + caveats)
```

- **Deterministic layer** — 6 tools compute Δ accessibility (ChromBPNet, fold + RC
  averaged), the disrupted/created TF motif (JASPAR PWM), ENCODE cCRE overlap, nearest
  gene + GTEx eQTL, GWAS trait links, and real Europe PMC citations. No LLM; nothing
  invented.
- **Reasoning layer** — four specialists each assess one facet, an optional **red-team**
  challenges the story (model artifact? LD hitchhiker? cell-type mismatch?), and an
  **adjudicator** synthesizes the final cited hypothesis. A citation guard drops any PMID
  not in the bundle. (Runs on the Messages API — the tools are pre-computed, so the
  agents reason over data rather than calling tools.)

## Validation — the model beats CADD, most on its own cell type

We validate the **engine** (the variant score) on the **Kircher saturation-mutagenesis
MPRA** (33,359 SNVs, 29 disease-element assays), with **matched within-element negatives**
— the honest, hard comparison — against a **CADD** baseline. Full numbers in
[`RESULTS.md`](RESULTS.md).

| | Model | CADD | Δ |
|---|---|---|---|
| **Overall** (matched) | **0.622** | 0.556 | +0.066 |
| **Hematopoietic elements** (K562 lineage) | **0.716** | 0.587 | **+0.129** |
| Other lineages | 0.601 | 0.586 | +0.015 (tied) |

<p align="center">
  <img src="figures/model_vs_cadd_scatter.png" width="46%"/>
  <img src="figures/celltype_summary.png" width="46%"/>
</p>

**A cell-type-matched chromatin model beats a cell-type-agnostic conservation score by
+0.13 AUROC on elements in its lineage — and only ties it elsewhere.** CADD is flat
(~0.586) across both groups; the model's edge is lineage-specific. That is RegLens's whole
premise — *the right cell-type model matters* — measured with numbers, matched negatives,
and no cherry-picking (CADD still wins 11/29 elements).

### The crossover — a double dissociation

Run the *same* benchmark with a **HepG2 (hepatic)** model and the pattern **flips**:

| Compartment | K562 model | HepG2 model |
|---|---|---|
| **Hematopoietic** elements | **0.716** | 0.569 |
| **Hepatic** elements | 0.633 | **0.663** |

<p align="center"><img src="figures/crossover_summary.png" width="52%"/></p>

Swap the cell-type model and the winning elements swap with it — the hepatic **SORT1**
assays rise +0.08–0.10 under HepG2 while blood **PKLR-48h collapses 0.805→0.505**. That is
intervention-level proof the signal is **cell-type-driven, not a model artifact**.

*Honest, with CIs (cluster bootstrap over elements):* the blood side is robust —
**+0.147, 95% CI [+0.072, +0.226]** — while the hepatic side is directional but not
significant — **+0.030, 95% CI [−0.015, +0.069]**. The double dissociation is carried by the
strong hematopoietic arm and the per-element extremes; the hepatic mean is suggestive only.

> **Two separate claims, kept distinct:** the AUROC validates the **engine** (variant
> score). The **agent** is validated separately — by recovering rs1427407 / rs2814778
> mechanisms and by the red-team catching real artifacts.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests (offline stub backend, no TensorFlow)
# pip install -e ".[chrombpnet]"   # + TensorFlow for real pretrained inference
# pip install -e ".[agents]"       # + anthropic SDK for the reasoning layer (needs an API key)
```

## Quickstart

```bash
reglens demo                        # offline synthetic demo — no downloads, no API key

# Full analysis (deterministic evidence + optional interpretation):
reglens analyze 'chr2:60490908:T>G' --rsid rs1427407 --celltype K562 \
        --interpret --multi-agent   # add --genome hg38.fa --model <dir> for ChromBPNet+motif

# Validate the engine (AUROC vs CADD) on a labeled benchmark:
reglens validate data/benchmarks/kircher_mpra_grch38.cadd.tsv --genome hg38.fa --model <fold_dir>
```

Colab notebooks (`reglens/model/`, `reglens/validation/`) run the pretrained-model
verification, the training/extensibility demo, and the full validation on a GPU.

## MCP server

The deterministic tool layer is also exposed as an **MCP stdio server**, so any MCP host
(Claude Desktop, etc.) can call it directly. It serves seven tools —
`get_evidence_bundle` (the primary interface: all signals for a variant in one call),
plus `score_variant`, `motif_effect`, `regulatory_context`, `gene_target`, `trait_link`,
and `literature`. These are thin wrappers over `reglens.tools.*` — they compute no new
numbers and do no interpretation.

```bash
pip install -e ".[mcp]"     # installs the MCP SDK
reglens-mcp                 # run the stdio server (or: python -m reglens.mcp_server)
```

The server always starts and always serves the annotation tools (`regulatory_context`,
`gene_target`, `trait_link`, `literature`) — they need only network access. The
sequence-model tools read two environment variables:

- `REGLENS_GENOME` — path to an hg38 FASTA. Required for `score_variant` and
  `motif_effect`; when unset they return a clear, actionable error (the annotation tools
  are unaffected).
- `REGLENS_MODEL` — path to a pretrained ChromBPNet model file or fold directory. If
  unset, `score_variant` uses the offline stub backend and labels its `model` field
  `stub(offline)` so a stub score is never mistaken for a real one.

Register it in `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "reglens": {
      "command": "reglens-mcp",
      "env": {
        "REGLENS_GENOME": "/path/to/hg38.fa",
        "REGLENS_MODEL": "/path/to/chrombpnet_fold_dir"
      }
    }
  }
}
```

Use the absolute path to the `reglens-mcp` entry point from your environment (e.g.
`/path/to/.venv/bin/reglens-mcp`) if it isn't on Claude Desktop's `PATH`.

## Repo layout

```
reglens/
  tools/        chrombpnet_score · motif_effect · regulatory_context · gene_target · trait_link · literature
  agents/       interpreter (single) · multi_agent (specialists → red-team → adjudicator)
  validation/   metrics · harness · dataset · build_mpra_benchmark · cadd · lineage · run_validation.ipynb
  report/       schema · render · plot
  orchestrator.py · cli.py · genome.py · mcp_server.py (MCP stdio server)
  model/        ChromBPNet wrappers + notebooks
data/benchmarks/  Kircher MPRA benchmark (+ CADD)   figures/  validation money-shots
tests/  (164, all offline)   RESULTS.md  RegLens_spec.md
```

## Test

```bash
pytest        # 164 tests, fully offline (no network, no GPU, no API key)
ruff check reglens tests
```

## License

[Apache-2.0](LICENSE). Built entirely from open data/tools: ChromBPNet, ENCODE, UCSC,
JASPAR (CC0), GTEx, GWAS Catalog, Europe PMC, Kircher satMutMPRA, CADD.
