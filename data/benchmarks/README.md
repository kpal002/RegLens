# Validation benchmarks

## `kircher_mpra_grch38.tsv` — matched MPRA regulatory-variant benchmark

Curated from the **Kircher et al. 2019** saturation-mutagenesis MPRA (Nature
Communications, GEO **GSE126550**; data via
[kircherlab/MPRA_SaturationMutagenesis](https://github.com/kircherlab/MPRA_SaturationMutagenesis)),
built by [`reglens/validation/build_mpra_benchmark.py`](../../reglens/validation/build_mpra_benchmark.py).

**33,359 SNVs across 29 element assays — 11,973 positive / 21,386 negative.**

- **positive** (`label 1`): significant regulatory effect (`pValue < 0.01`).
- **negative** (`label 0`): tested in the **same element** with no effect (`pValue ≥ 0.10`).
- ambiguous middle dropped; deletions dropped (SNVs only); GRCh38 coordinates.

### Why the negatives are matched (and why that matters)

Positives and negatives come from the **same regulatory elements** (the `source`
column is the element, e.g. `BCL11A`, `SORT1`, `TERT-*`). This is the *hard, honest*
comparison: the model must distinguish functional from non-functional variants **within
the same active regions**. Random genomic controls far from peaks would make the task
trivially easy and the AUROC a mirage — do **not** substitute them.

### Regenerate / customize

```bash
python -m reglens.validation.build_mpra_benchmark -o data/benchmarks/kircher_mpra_grch38.tsv
python -m reglens.validation.build_mpra_benchmark -o bcl11a.tsv --element BCL11A   # one element
```

### Remaining step: the CADD baseline

This TSV has no `cadd` column yet. To enable the AUROC-vs-CADD comparison, annotate a
`cadd` (or `phylop`) column per variant (CADD web service / precomputed scores); the
harness reads it automatically. The **model AUROC is reported regardless**.

### Run

```bash
reglens validate data/benchmarks/kircher_mpra_grch38.tsv \
  --genome hg38.fa --model encode_models --json
```

(Needs an hg38 FASTA + a pretrained model — run on Colab / a GPU box. Validates the
*pretrained* model; the trained model is an extensibility demo, not under test.)
