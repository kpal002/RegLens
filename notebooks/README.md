# RegLens notebooks (Colab)

Task-based notebooks that reproduce every experiment in `RESULTS.md` on a GPU. Each is
**self-contained** — it installs RegLens, downloads the reference genome + model, and runs
one task top to bottom.

| Notebook | Reproduces | Needs |
|---|---|---|
| `01_engine_validation.ipynb` | AUROC vs CADD (33k MPRA) + cell-type specificity | GPU |
| `02_crossover.ipynb` | HepG2-vs-K562 double dissociation + bootstrap CI | GPU |
| `03_agent_null_control.ipynb` | Null / paired / strong-signal controls (biconditional) | GPU + API key |
| `04_agent_reasoning.ipynb` | Known-mechanism recovery, ablation, calibration | GPU + API key |
| `05_discovery_screen.ipynb` | Prospective GWAS screen (in-domain, falsifiable) | GPU + API key |

## Setup (once per Colab session)

1. **Runtime → Change runtime type → GPU.**
2. Build the repo snapshot locally and upload it:
   ```bash
   git archive --format=zip -o reglens_for_colab.zip HEAD   # run in the repo
   ```
   Upload `reglens_for_colab.zip` via the Colab **Files** panel.
3. Run the notebook's setup cells (install → genome + model). The agent notebooks (03–05)
   also install the Anthropic SDK and need your `ANTHROPIC_API_KEY` pasted into the key cell.

## Notes

- The model notebooks in [`../reglens/model/`](../reglens/model) cover pretrained-model
  verification and the training/extensibility demo (separate from validation).
- Runtimes: engine passes ~25–30 min (full 33k on GPU); each agent experiment ~15–25 min
  (LLM calls). The discovery screen scales with pool size.
- Everything is offline-testable without a GPU/API via `pytest` (204 tests); the notebooks
  are the on-GPU faithful runs.
