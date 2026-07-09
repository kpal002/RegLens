# CLAUDE.md — RegLens

Agentic mechanistic interpreter for noncoding regulatory variants. Full design in `RegLens_spec.md` — read it before building.

## Goal
Given a noncoding variant (hg38, chr:pos ref>alt) + a cell-type context, produce a **cited, mechanistic interpretation**: the chromatin-accessibility effect (ChromBPNet), the disrupted TF motif, the regulatory element, the likely target gene (eQTL), and the trait link — reasoned by a multi-agent layer over a deterministic tool layer. We explain what sequence models see; we do NOT claim to out-predict them.

## Golden rules (do not violate)
1. **Pretrained first, train second.** Get variant scoring working on a *pretrained* ChromBPNet model end-to-end before anything else. Train your own model in parallel on Colab Pro / Kaggle — it's the ML + validation component, never the critical path.
2. **Deterministic tools compute every number.** Agents reason over tool outputs (Δ-accessibility, motif scores, overlaps, eQTL, GWAS); they must NEVER invent a score.
3. **Single-agent before multi-agent.** One agent producing a good interpretation end-to-end is the fallback-safe milestone; refactor to specialists + adjudicator only after it works.
4. **Cite only retrieved evidence.** Literature claims come from a real Europe PMC query with the PMID attached. No invented citations.
5. **Honest framing.** "Explains what the model sees," not "beats the model." Interpretation is a hypothesis with confidence + caveats, not a verified mechanism.
6. **License-clean, open data only.** ChromBPNet, ENCODE, JASPAR, GTEx, GWAS Catalog, Europe PMC. Ship MIT/Apache-2.0.

## Stack & conventions
- Python 3.11+, type hints, **Google-style docstrings + detailed comments** (user pref), PEP 8, `ruff`, `pytest`.
- ChromBPNet (TF/Keras) trained/served via Colab Pro or Kaggle; inference wrappers in `model/`.
- Agents via Claude Agent SDK; deterministic tools exposed as an MCP server (`mcp_server.py`).
- CLI via `typer`; demo via Streamlit (`app.py`). Genome handling via pyfaidx/pyBigWig.

## Repo layout
```
reglens/
  tools/    # chrombpnet_score, motif_effect, regulatory_context, gene_target, trait_link, literature
  mcp_server.py
  model/    # ChromBPNet train + inference wrappers (+ Colab/Kaggle notebooks)
  agents/   # regulatory_effect, celltype_context, gene_target, trait_link, redteam, adjudicator
  orchestrator.py
  report/   # schema + renderer
  validation/  # MPRA/CAGI harness, AUROC vs baseline
  cli.py ; app.py ; data/cases/
tests/
```

## Build order (see spec §10)
- Wed: scaffold + LICENSE + genome plumbing + pretrained ChromBPNet scoring one variant end-to-end.
- Thu: add tools (motif/ISM, cCRE overlap, gene+eQTL, GWAS, Europe PMC) → single-agent interpretation. **Fallback-safe milestone.**
- Fri: multi-agent refactor (specialists + adjudicator) + cited report; kick off own ChromBPNet training on Colab (overnight).
- Sat: MPRA/CAGI validation + AUROC; curate demo variants; Streamlit/CLI.
- Sun: harden, README, dry-run, freeze.
- Mon: demo video + 100–200 word summary; submit by 9 PM ET.

## Working style
- Small, tested increments; run tests before calling a step done.
- After each milestone, summarize what changed for review.
- Ask before adding a heavyweight dependency.

## Hackathon rules
Built from scratch during the event; open-source; only public data/tools we have rights to; team ≤ 2.
