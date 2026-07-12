"""Generate the crossover (double dissociation) figures: K562 vs HepG2 per element.

Uses the per-element AUROCs from the two full 33k runs. Run: `python figures/generate_crossover.py`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from reglens.report.plot import plot_crossover
from reglens.validation.lineage import is_hematopoietic, is_hepatic

K562 = {"SORT1.2":0.584,"FOXE1":0.430,"SORT1-flip":0.638,"SORT1":0.586,"BCL11A":0.620,
"ZFAND3":0.637,"MYCrs6983267":0.634,"UC88":0.709,"MSMB":0.606,"TCF7L2":0.487,"RET":0.546,
"IRF6":0.533,"ZRSh-13":0.538,"MYCrs11986220":0.463,"PKLR-24h":0.794,"IRF4":0.548,
"ZRSh-13h2":0.564,"PKLR-48h":0.805,"GP1BA":0.729,"LDLR.2":0.679,"LDLR":0.691,"F9":0.642,
"HNF4A":0.611,"HBG1":0.663,"TERT-GSc":0.672,"TERT-GBM":0.672,"TERT-GAa":0.649,
"TERT-HEK":0.705,"HBB":0.684}
HEPG2 = {"SORT1.2":0.669,"FOXE1":0.482,"SORT1-flip":0.716,"SORT1":0.687,"BCL11A":0.516,
"ZFAND3":0.552,"MYCrs6983267":0.578,"UC88":0.702,"MSMB":0.631,"TCF7L2":0.516,"RET":0.511,
"IRF6":0.494,"ZRSh-13":0.577,"MYCrs11986220":0.501,"PKLR-24h":0.541,"IRF4":0.507,
"ZRSh-13h2":0.554,"PKLR-48h":0.505,"GP1BA":0.707,"LDLR.2":0.683,"LDLR":0.694,"F9":0.571,
"HNF4A":0.619,"HBG1":0.586,"TERT-GSc":0.665,"TERT-GBM":0.689,"TERT-GAa":0.658,
"TERT-HEK":0.704,"HBB":0.556}
RED, BLUE, GREY = "#c0392b", "#2c7fb8", "#bdc3c7"


def main():
    """Generate the crossover summary + per-element flip figures."""
    plot_crossover({"K562": K562, "HepG2": HEPG2}, "figures/crossover_summary.png")

    # Per-element ΔAUROC (HepG2 − K562): hematopoietic should drop, hepatic should rise.
    def col(e):
        return RED if is_hematopoietic(e) else BLUE if is_hepatic(e) else GREY
    rows = sorted(((e, HEPG2[e] - K562[e]) for e in K562 if e in HEPG2), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 8))
    ax.barh(range(len(rows)), [d for _, d in rows], color=[col(e) for e, _ in rows])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([e for e, _ in rows], fontsize=8)
    ax.set_xlabel("ΔAUROC  (HepG2 − K562)")
    ax.set_title("Swap K562 → HepG2: hematopoietic elements drop (red),\n"
                 "hepatic elements rise (blue) — the double dissociation")
    ax.legend(handles=[Patch(color=RED, label="hematopoietic (blood)"),
                       Patch(color=BLUE, label="hepatic (liver)"),
                       Patch(color=GREY, label="other lineage")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/crossover_flip.png", dpi=150)
    plt.close(fig)
    print("wrote figures/crossover_summary.png and figures/crossover_flip.png")
    for grp, pred in (("hematopoietic", is_hematopoietic), ("hepatic", is_hepatic)):
        els = [e for e in K562 if pred(e)]
        print(f"{grp}: K562 {sum(K562[e] for e in els)/len(els):.3f} "
              f"HepG2 {sum(HEPG2[e] for e in els)/len(els):.3f}  ({', '.join(els)})")


if __name__ == "__main__":
    main()
