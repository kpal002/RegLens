# `reglens/model/` — ChromBPNet training & inference wrappers

This directory holds the deep-learning **model** component of RegLens: wrappers
for loading a pretrained ChromBPNet model for inference (used today) and, later,
notebooks + loaders for training **our own** ChromBPNet model.

## Status (Wednesday milestone)

**Inference-only, pretrained-first.** Variant scoring currently runs through
`reglens/tools/chrombpnet_score.py`, which loads a *pretrained* ChromBPNet Keras
model (or an offline stub) and computes ref-vs-alt Δ log-counts **and** the
profile-shape change (Jensen–Shannon distance). **No training has been started,
and training is never on the critical path** (Golden Rule #1).

## Parallel-track de-risk (do this on Colab today)

`colab_verify_chrombpnet.ipynb` is a self-contained notebook that loads a **real**
pretrained checkpoint and scores one variant on a Colab GPU, to confirm the real
model behaves as our `KerasChromBPNetBackend` assumes — *before* Thursday's tools
depend on it. It fetches the 2114 bp window from the UCSC API (no genome download)
and cross-checks RegLens's wrapper against a manual scoring. Example variant:
`rs1427407` (chr2:60,495,250 G>T), which disrupts a GATA1 motif in the BCL11A
erythroid enhancer — the spec's money-shot.

### The money-shot model (notebook default)

The notebook downloads the **K562 ATAC ChromBPNet model from the ChromBPNet
manuscript** — the same model family used to score red-blood-cell-trait variants,
so `rs1427407` should show an accessibility drop:

- ENCODE annotation **[ENCSR467RSV](https://www.encodeproject.org/annotations/ENCSR467RSV/)**
  ("ChromBPNet models trained on ATAC-seq in K562 (ENCSR868FGK)").
- Models tar **`ENCFF984RAF`** (727 MB, GRCh38, 5 folds):
  `https://www.encodeproject.org/files/ENCFF984RAF/@@download/ENCFF984RAF.tar.gz`
- Extract → use one fold's **`chrombpnet_nobias.h5`** (the notebook globs for it).

Manuscript models are also mirrored on Synapse
([syn59449898](https://www.synapse.org/Synapse:syn59449898/files/)). For a quick
I/O-contract check with no ENCODE download, any `*_nobias.h5` from
[Zenodo](https://zenodo.org/records/16295014) works.

### Confirmed ChromBPNet model contract (kundajelab chrombpnet / variant-scorer)

| Aspect | Value | Why it matters |
|---|---|---|
| Model file | **`chrombpnet_nobias.h5`** (Tn5 bias-corrected) | The raw `chrombpnet.h5` / `bias_model_scaled.h5` give a garbage Δ. |
| Input | one-hot **`(N, 2114, 4)`** | Must match `window_length`. |
| Output heads | **`[profile_logits (N, ~1000), logcount (N, 1)]`** | Order assumed by `KerasChromBPNetBackend`; swap indices if a model differs. |
| Primary signal | `logfc` = Δ log-counts (alt − ref) | Total-accessibility change. |
| Secondary signal | `jsd` = JS distance of softmax'd profiles | Footprint-shape change → motif story. |
| Note | variant-scorer averages fwd + reverse-complement by default | Our MVP is forward-only; add RC-averaging if AUROC needs it. |

Once you've run the notebook, paste the printed `input_shape` / `output_shape`
here so the confirmed contract is on record. If anything differs, the notebook's
final markdown cell lists the exact `KerasChromBPNetBackend` knob to turn.

## Plan: train our own model (Friday, in parallel)

On Friday we fine-tune / train one ChromBPNet model as the project's real ML +
validation component. This runs **in parallel** on managed GPUs and does not block
the product — RegLens works on the pretrained model regardless of the outcome.

- **Data:** one ENCODE ATAC-seq experiment (a single, well-characterized cell
  type — no genome-wide scope creep), plus the hg38 FASTA. All open data.
- **Where:** Colab Pro (overnight, ~12 h) or Kaggle free GPU-hours. TF/Keras.
- **Output:** a saved Keras checkpoint that drops into `KerasChromBPNetBackend`
  via `load_backend(model_path=...)` with no code changes to the scorer.
- **Validation (Saturday):** does our model's variant Δ-scores discriminate
  known causal/regulatory variants (MPRA-validated or fine-mapped) from benign?
  Report AUROC vs. a naive baseline (conservation / CADD). See
  `reglens/validation/`.

## Honest framing

We do **not** claim to out-predict Enformer/Borzoi/ChromBPNet. Training our own
model demonstrates the ML depth and gives us a validated model we control; the
product's contribution is the *agentic mechanistic interpretation* on top.

## Planned contents (added Friday)

- `train_chrombpnet.ipynb` — Colab/Kaggle training notebook (ENCODE ATAC-seq).
- `loaders.py` — checkpoint discovery / download + `load_backend` helpers.
- `pretrained/` — notes on pretrained checkpoints (Corces brain / ENCODE cell types).
