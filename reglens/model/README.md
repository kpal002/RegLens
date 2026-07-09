# `reglens/model/` — ChromBPNet training & inference wrappers

This directory holds the deep-learning **model** component of RegLens: wrappers
for loading a pretrained ChromBPNet model for inference (used today) and, later,
notebooks + loaders for training **our own** ChromBPNet model.

## Status (Wednesday milestone)

**Inference-only, pretrained-first.** Variant scoring currently runs through
`reglens/tools/chrombpnet_score.py`, which loads a *pretrained* ChromBPNet Keras
model (or an offline stub) and computes ref-vs-alt Δ log-counts. **No training has
been started, and training is never on the critical path** (Golden Rule #1).

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
