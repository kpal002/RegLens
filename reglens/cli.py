"""RegLens command-line interface (Wednesday milestone: single-tool scoring).

Exposes ``reglens score`` — the minimal end-to-end path that turns a variant
string into a predicted chromatin-accessibility effect — and ``reglens demo``,
which runs the same path on a bundled synthetic example so the pipeline is
demonstrable offline with no genome download or TensorFlow install.

Later milestones layer the remaining deterministic tools, the multi-agent
reasoning layer, and the cited report on top of this entry point.
"""

from __future__ import annotations

from pathlib import Path

import typer

from reglens.genome import DEFAULT_WINDOW_LENGTH, Variant
from reglens.tools.chrombpnet_score import ChromBPNetScorer, VariantScore, load_backend

app = typer.Typer(
    add_completion=False,
    help="RegLens — mechanistic interpreter for noncoding regulatory variants.",
)

# Path to the bundled synthetic demo contig shipped with the package.
DEMO_GENOME = Path(__file__).parent / "data" / "demo" / "demo.fa"
# A known variant on that contig (see reglens/data/demo/demo.fa; ref base checked).
DEMO_VARIANT = "chr_demo:1500:C>T"
DEMO_WINDOW = 2048  # < demo contig length (3000 bp) so the window fits with margin


def _render(score: VariantScore) -> None:
    """Print a variant score as a readable block to stdout."""
    typer.echo("── RegLens variant score " + "─" * 40)
    typer.echo(f"  variant        : {score.variant}")
    typer.echo(f"  cell-type      : {score.celltype or '(unspecified)'}")
    typer.echo(f"  model          : {score.model_name}")
    typer.echo(
        f"  window         : {score.window.chrom}:{score.window.start}-"
        f"{score.window.end} ({score.window.length} bp, "
        f"variant@offset {score.window.variant_offset})"
    )
    typer.echo(f"  ref log-counts : {score.ref_log_counts:+.4f}")
    typer.echo(f"  alt log-counts : {score.alt_log_counts:+.4f}")
    typer.echo(f"  Δ log-counts   : {score.delta_log_counts:+.4f}")
    typer.echo(f"  direction      : {score.direction} accessibility")
    typer.echo(f"  effect size    : {score.effect_size:.4f}")
    jsd = f"{score.profile_jsd:.4f}" if score.profile_jsd is not None else "n/a"
    typer.echo(f"  profile JSD    : {jsd}  (footprint-shape change)")
    typer.echo("─" * 64)


@app.command()
def score(
    variant: str = typer.Argument(
        ..., help="Variant as chr:pos:ref>alt (hg38), e.g. chr7:5530601:C>T"
    ),
    celltype: str = typer.Option(
        None, "--celltype", "-c", help="Cell-type / model context label (for reporting)."
    ),
    genome: Path = typer.Option(
        None,
        "--genome",
        "-g",
        help="Path to genome FASTA (hg38). Defaults to $REGLENS_GENOME.",
    ),
    model: Path = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to a pretrained ChromBPNet Keras model. Omit to use the "
        "offline stub backend.",
    ),
    window: int = typer.Option(
        DEFAULT_WINDOW_LENGTH, "--window", "-w", help="Sequence window length (bp)."
    ),
) -> None:
    """Score a variant's chromatin-accessibility effect (ref vs alt Δ log-counts)."""
    parsed = Variant.parse(variant)
    backend = load_backend(str(model) if model else None)
    model_name = model.name if model else "stub(offline)"
    scorer = ChromBPNetScorer(backend, window_length=window, model_name=model_name)
    result = scorer.score_variant(
        parsed, genome_path=str(genome) if genome else None, celltype=celltype
    )
    _render(result)


@app.command()
def demo() -> None:
    """Run the end-to-end scoring path on the bundled synthetic demo variant.

    Uses the packaged demo contig and the offline stub backend, so it works with
    no downloads and no TensorFlow — proving the ref→alt→Δ pipeline is wired up.
    """
    typer.echo(
        "Running RegLens on a bundled SYNTHETIC demo (stub model — not a real "
        "biological result).\n"
    )
    parsed = Variant.parse(DEMO_VARIANT)
    scorer = ChromBPNetScorer(
        load_backend(None), window_length=DEMO_WINDOW, model_name="stub(offline)"
    )
    result = scorer.score_variant(parsed, genome_path=DEMO_GENOME, celltype="demo-cell")
    _render(result)


if __name__ == "__main__":
    app()
