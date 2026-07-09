# Claude Code kickoff prompt — RegLens

Copy `RegLens_spec.md` and `RegLens_CLAUDE.md` (rename to `CLAUDE.md`) into your repo, run `claude`, and paste this:

---

Read `CLAUDE.md` and `RegLens_spec.md` fully before writing code — they define the project, architecture, and golden rules.

We're building RegLens, an agentic mechanistic interpreter for noncoding regulatory variants. Today is the Wednesday milestone. Do ONLY this, in order, then stop for my review:

1. Scaffold the repo per the spec layout, with an Apache-2.0 LICENSE, `pyproject.toml` (ruff + pytest), and a README stub.
2. Genome plumbing: given a variant `chr:pos ref>alt` (hg38), build the reference and alternate sequence windows needed for ChromBPNet input. Use pyfaidx; make the genome path configurable. Type hints + Google-style docstrings, with a unit test on a known locus.
3. `tools/chrombpnet_score.py`: a function that loads a **pretrained** ChromBPNet model and returns the predicted chromatin-accessibility effect of the variant (ref vs alt Δ log-counts + direction). Make the model backend swappable (local path / downloaded pretrained). Do NOT train anything yet — pretrained inference only. Unit-test the scoring interface with a small stub model so it runs offline.
4. A minimal SINGLE-tool end-to-end path: `cli.py` (`typer`) so `reglens score chr:pos:ref>alt --celltype X` prints the Δ-accessibility result on one example variant.
5. Write a short `model/README.md` noting the plan to fine-tune our own ChromBPNet on ENCODE ATAC-seq via Colab Pro later (Friday) — but do not start training now.

Rules from CLAUDE.md: deterministic tools compute all numbers; agents (later) only reason; pretrained-first, training is parallel and never the critical path; license-clean/open data only; docstrings + comments on everything.

When done, summarize what you built, show test output, and list what's next (Thursday's tool set + single-agent interpretation). Do not build the multi-agent layer or start model training without me.

---

**Parallel track (you, not Claude Code):** confirm you can load a pretrained ChromBPNet model on Colab Pro / Kaggle and run inference on one variant. That de-risks the whole engine on day one.
