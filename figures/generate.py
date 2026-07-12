"""Generate the RegLens validation figures (model vs CADD, cell-type stratified).

Reads the CADD-annotated benchmark for the CADD AUROCs and uses the model's per-element
AUROCs from the full 33k run (RESULTS.md). Run: `python figures/generate.py`.
"""

import collections
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reglens.validation.lineage import is_hematopoietic
from reglens.validation.metrics import roc_auc

# Model per-element AUROC from the full 33,359-variant run (K562 5-fold + RC).
MODEL = {
    "SORT1.2": 0.584, "FOXE1": 0.430, "SORT1-flip": 0.638, "SORT1": 0.586, "BCL11A": 0.620,
    "ZFAND3": 0.637, "MYCrs6983267": 0.634, "UC88": 0.709, "MSMB": 0.606, "TCF7L2": 0.487,
    "RET": 0.546, "IRF6": 0.533, "ZRSh-13": 0.538, "MYCrs11986220": 0.463, "PKLR-24h": 0.794,
    "IRF4": 0.548, "ZRSh-13h2": 0.564, "PKLR-48h": 0.805, "GP1BA": 0.729, "LDLR.2": 0.679,
    "LDLR": 0.691, "F9": 0.642, "HNF4A": 0.611, "HBG1": 0.663, "TERT-GSc": 0.672,
    "TERT-GBM": 0.672, "TERT-GAa": 0.649, "TERT-HEK": 0.705, "HBB": 0.684,
}
BENCH = "data/benchmarks/kircher_mpra_grch38.cadd.tsv"
RED, GREY = "#c0392b", "#7f8c8d"


def cadd_per_element():
    rows = [r for r in csv.DictReader(open(BENCH), delimiter="\t") if r["cadd"]]
    by = collections.defaultdict(lambda: ([], []))
    for r in rows:
        by[r["source"]][0].append(float(r["cadd"]))
        by[r["source"]][1].append(int(r["label"]))
    return {e: roc_auc(c, lab) for e, (c, lab) in by.items() if len(set(lab)) > 1}


def main():
    cadd = cadd_per_element()

    # Figure 1 — scatter, points colored by lineage, y = x tie line.
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0.4, 0.85], [0.4, 0.85], "--", color="grey", lw=1, label="y = x (tie)")
    for e in MODEL:
        if e in cadd:
            ax.scatter(cadd[e], MODEL[e], s=70, color=RED if is_hematopoietic(e) else GREY,
                       edgecolor="black", lw=0.4, zorder=3, alpha=0.9)
    for e in ["PKLR-48h", "PKLR-24h", "GP1BA", "BCL11A", "HBB", "FOXE1", "LDLR", "IRF4"]:
        if e in cadd:
            ax.annotate(e, (cadd[e], MODEL[e]), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
    ax.scatter([], [], color=RED, edgecolor="black", lw=0.4, label="hematopoietic (K562 lineage)")
    ax.scatter([], [], color=GREY, edgecolor="black", lw=0.4, label="other lineage")
    ax.set_xlabel("CADD AUROC (baseline)")
    ax.set_ylabel("ChromBPNet |Δ| AUROC (model)")
    ax.set_title("Model vs CADD per element — above the line = model wins\n"
                 "hematopoietic elements sit high and above (cell-type match)")
    ax.set_xlim(0.4, 0.85)
    ax.set_ylim(0.4, 0.85)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/model_vs_cadd_scatter.png", dpi=150)
    plt.close(fig)

    # Figure 2 — stratified summary bars.
    hema = [e for e in MODEL if is_hematopoietic(e) and e in cadd]
    other = [e for e in MODEL if not is_hematopoietic(e) and e in cadd]
    mean = lambda g, d: sum(d[e] for e in g) / len(g)  # noqa: E731
    groups = [f"Hematopoietic\n(K562 lineage, n={len(hema)})", f"Other lineages\n(n={len(other)})"]
    mvals = [mean(hema, MODEL), mean(other, MODEL)]
    cvals = [mean(hema, cadd), mean(other, cadd)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x, w = range(len(groups)), 0.36
    ax.bar([i - w / 2 for i in x], mvals, w, label="ChromBPNet model", color="#1f6feb")
    ax.bar([i + w / 2 for i in x], cvals, w, label="CADD baseline", color="#d1782f")
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    for i, (m, c) in enumerate(zip(mvals, cvals, strict=True)):
        ax.text(i - w / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, c + 0.005, f"{c:.3f}", ha="center", fontsize=9)
        ax.annotate(f"+{m - c:.3f}", (i, max(m, c) + 0.02), ha="center", fontsize=10,
                    weight="bold", color="#1f6feb" if m > c else "#d1782f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0.45, 0.78)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Cell-type match drives the model's edge over CADD")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/celltype_summary.png", dpi=150)
    plt.close(fig)
    print("wrote figures/model_vs_cadd_scatter.png and figures/celltype_summary.png")


if __name__ == "__main__":
    main()
