# RegLens

**An agentic mechanistic interpreter for noncoding regulatory variants.**

Annotation tools (VEP/ANNOVAR) tell you *where* a noncoding variant is
("intergenic, modifier"). Sequence models (ChromBPNet/Enformer) score *that* it
has an effect but don't explain it. **RegLens is the bridge:** give it a noncoding
variant + a cell-type context, and a deep-learning chromatin model plus a
multi-agent reasoning layer produce a **mechanistic, cell-type-specific, cited
interpretation** — which TF motif is disrupted, in which regulatory element,
affecting which gene, linked to which trait, and why.

> We *explain what the sequence models see* — we do **not** claim to out-predict
> them. Interpretations are hypotheses with confidence and caveats.

See [`RegLens_spec.md`](RegLens_spec.md) for the full design and
[`CLAUDE.md`](RegLens_CLAUDE.md) for the golden rules.

## Architecture (two layers)

1. **Deterministic tool layer** — computes every number (Δ accessibility, motif
   effects, regulatory overlaps, eQTL, trait links, literature). No LLM.
2. **Multi-agent reasoning layer** — reasons *over* those numbers to produce a
   cited mechanistic interpretation. Agents never invent scores.

## Status — Wednesday milestone

Implemented so far (the core deterministic scoring path):

- **Genome plumbing** (`reglens/genome.py`) — parse `chr:pos:ref>alt` (hg38),
  build reference/alternate ChromBPNet input windows via `pyfaidx`, with a
  reference-allele sanity check.
- **Variant scoring** (`reglens/tools/chrombpnet_score.py`) — load a *pretrained*
  ChromBPNet model (swappable backend) and compute ref-vs-alt Δ log-counts +
  direction. Runs offline with a stub backend (no TensorFlow required).
- **CLI** (`reglens/cli.py`) — `reglens score` / `reglens demo`.

Not yet built: the remaining deterministic tools (Thursday), the multi-agent
layer + cited report (Friday), our own trained model + validation (Fri–Sat). See
[`reglens/model/README.md`](reglens/model/README.md) for the training plan.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # core + test tooling (offline stub backend)
# pip install -e ".[chrombpnet]"  # add TensorFlow for real pretrained inference
```

## Quickstart

Run the bundled offline demo (synthetic contig + stub model — no downloads):

```bash
reglens demo
```

Score a real variant against an hg38 FASTA:

```bash
export REGLENS_GENOME=/path/to/hg38.fa
reglens score chr7:5530601:C>T --celltype K562
# with a real pretrained model:
reglens score chr7:5530601:C>T -c K562 -m /path/to/chrombpnet.keras
```

## Test

```bash
pytest
```

## License

[Apache-2.0](LICENSE). Built with open data/tools only (ChromBPNet, ENCODE,
JASPAR, GTEx, GWAS Catalog, Europe PMC).
