"""Smoke tests for the RegLens CLI (offline, stub backend)."""

from __future__ import annotations

from typer.testing import CliRunner

from reglens.cli import app

runner = CliRunner()


def test_demo_runs_offline():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Δ log-counts" in result.stdout
    assert "chr_demo:1500:C>T" in result.stdout


def test_score_with_bundled_demo_genome():
    from reglens.cli import DEMO_GENOME

    result = runner.invoke(
        app,
        [
            "score",
            "chr_demo:1500:C>T",
            "--celltype",
            "K562",
            "--genome",
            str(DEMO_GENOME),
            "--window",
            "2048",
        ],
    )
    assert result.exit_code == 0
    assert "K562" in result.stdout
    assert "direction" in result.stdout


def test_score_rejects_bad_variant():
    result = runner.invoke(app, ["score", "not-a-variant", "--genome", "x"])
    assert result.exit_code != 0
