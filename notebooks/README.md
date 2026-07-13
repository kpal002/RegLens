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

Each notebook is **self-contained** — no zip upload. Just:

1. **Runtime → Change runtime type → GPU.**
2. Run the first cell: it **clones the repo** (`git clone …/kpal002/RegLens`) and installs
   it. Re-running that cell fast-forwards to the latest `main`.
3. For the agent notebooks (03–05), add your **`ANTHROPIC_API_KEY` as a Colab secret**
   (🔑 panel in the left sidebar → add it → toggle *Notebook access* on). The key cell reads
   it from there — it never appears in the notebook. (A one-time hidden prompt is the
   fallback if the secret isn't set.)

## Notes

- The model notebooks in [`../reglens/model/`](../reglens/model) cover pretrained-model
  verification and the training/extensibility demo (separate from validation).
- Runtimes: engine passes ~25–30 min (full 33k on GPU); each agent experiment ~15–25 min
  (LLM calls). The discovery screen scales with pool size.
- Everything is offline-testable without a GPU/API via `pytest` (204 tests); the notebooks
  are the on-GPU faithful runs.
