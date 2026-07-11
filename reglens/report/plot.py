"""ROC-curve plot for the validation money-shot figure (model vs CADD baseline).

matplotlib is imported lazily (it's not a core dependency) so this only loads when a
plot is actually requested — e.g. in the validation notebook on Colab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reglens.validation.harness import ValidationReport


def plot_roc(report: ValidationReport, path: str, title: str | None = None) -> str:
    """Save a ROC-curve figure (model, and CADD baseline if available) to ``path``.

    Args:
        report: A computed :class:`~reglens.validation.harness.ValidationReport`.
        path: Output image path (e.g. ``"roc.png"``).
        title: Optional plot title; defaults to a summary with the AUROCs.

    Returns:
        The output path.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If the model ROC is not computable (no scores / one class).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError("matplotlib is required for plot_roc (pip install matplotlib)") from exc

    model_pts = report.roc_points(use_baseline=False)
    if model_pts is None:
        raise ValueError("Model ROC not computable (need scores for both classes).")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance (0.500)")
    m = report.model_auroc
    ax.plot(model_pts[0], model_pts[1], color="#1f6feb", lw=2,
            label=f"ChromBPNet |Δ| (AUROC={m:.3f})")
    base_pts = report.roc_points(use_baseline=True)
    if base_pts is not None and report.baseline_auroc is not None:
        ax.plot(base_pts[0], base_pts[1], color="#d1782f", lw=2,
                label=f"CADD baseline (AUROC={report.baseline_auroc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or f"Regulatory vs benign — matched MPRA "
                          f"({report.n_pos} pos / {report.n_neg} neg)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_element(report: ValidationReport, path: str, title: str | None = None) -> str:
    """Save a per-element AUROC bar chart, colored by hematopoietic lineage.

    Highlights the cell-type-specificity signal: for a K562 (erythroid) model, the
    hematopoietic elements should cluster above the non-hematopoietic ones.

    Args:
        report: A computed :class:`~reglens.validation.harness.ValidationReport`.
        path: Output image path.
        title: Optional title.

    Returns:
        The output path.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError("matplotlib is required for plot_per_element") from exc

    from reglens.validation.lineage import is_hematopoietic

    rows = [(s, a) for s, a, _, _ in report.per_source_auroc() if a is not None]
    rows.sort(key=lambda r: r[1])  # ascending AUROC
    labels = [s for s, _ in rows]
    values = [a for _, a in rows]
    colors = ["#c0392b" if is_hematopoietic(s) else "#95a5a6" for s, _ in rows]

    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * len(rows))))
    ax.barh(range(len(rows)), values, color=colors)
    ax.axvline(0.5, ls="--", color="grey", lw=1)  # chance
    if report.model_auroc is not None:
        ax.axvline(report.model_auroc, ls=":", color="#1f6feb", lw=1.5,
                   label=f"overall {report.model_auroc:.3f}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC (regulatory vs benign, within element)")
    ax.set_xlim(0.3, 0.9)
    ax.set_title(title or "Per-element AUROC — red = hematopoietic (K562 lineage)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
