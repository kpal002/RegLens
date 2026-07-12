"""Tests for the RegLens MCP server (reglens.mcp_server).

Covers tool registration (the declarative registry + SDK wiring) and one offline
wrapper roundtrip through score_variant, plus the actionable-error path when no genome
is configured. All offline — the wrapper roundtrip injects a stub-backed scorer + the
synthetic test genome, so no network, no TensorFlow, no real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.genome import GENOME_ENV_VAR
from reglens.mcp_server import (
    MODEL_ENV_VAR,
    TOOL_SPECS,
    build_server,
    dispatch,
    score_variant,
    trait_link_tool,
)
from reglens.tools.chrombpnet_score import ChromBPNetScorer, load_backend


class _FakeClient:
    """Minimal HTTP client returning a canned JSON payload (offline)."""

    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params=None):
        return self.payload


# A trimmed GWAS Catalog associationBySnp payload (fetal-hemoglobin trait).
_GWAS_PAYLOAD = {
    "_embedded": {
        "associations": [
            {
                "pvalue": 4e-53,
                "efoTraits": [{"trait": "fetal hemoglobin measurement"}],
                "loci": [{"strongestRiskAlleles": [{"riskAlleleName": "rs1427407-T"}]}],
            }
        ]
    }
}

_EXPECTED_TOOLS = {
    "get_evidence_bundle",
    "score_variant",
    "motif_effect",
    "regulatory_context",
    "gene_target",
    "trait_link",
    "literature",
}


def _stub_scorer() -> ChromBPNetScorer:
    """A small-window, offline stub scorer that fits the synthetic test contig."""
    return ChromBPNetScorer(load_backend(None), window_length=40, model_name="stub(offline)")


class TestRegistration:
    def test_all_tools_registered(self):
        assert {s.name for s in TOOL_SPECS} == _EXPECTED_TOOLS

    def test_primary_interface_listed_first(self):
        # get_evidence_bundle is the documented primary interface.
        assert TOOL_SPECS[0].name == "get_evidence_bundle"

    def test_schemas_well_formed(self):
        for spec in TOOL_SPECS:
            schema = spec.input_schema
            assert schema["type"] == "object"
            assert isinstance(schema["properties"], dict)
            # Every required key must be declared in properties.
            for key in schema["required"]:
                assert key in schema["properties"], (spec.name, key)
            # Every variant-centric tool anchors on a variant except trait_link,
            # which is rsID-keyed (variant is optional context, not required).
            if spec.name != "trait_link":
                assert "variant" in schema["required"]

    def test_trait_link_is_rsid_keyed(self):
        spec = next(s for s in TOOL_SPECS if s.name == "trait_link")
        assert spec.input_schema["required"] == ["rsid"]
        assert "variant" not in spec.input_schema["required"]

    def test_dispatch_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("nope", {})

    def test_dispatch_missing_required_argument_raises(self):
        with pytest.raises(ValueError, match="Missing required argument"):
            dispatch("score_variant", {})

    def test_build_server_registers_tools(self):
        # SDK-wiring smoke test — skip if the optional MCP SDK isn't installed, so the
        # rest of the suite still runs offline without the ".[mcp]" extra.
        pytest.importorskip("mcp")
        server = build_server()
        assert server.name == "reglens"


class TestScoreVariantWrapper:
    def test_roundtrip_offline(self, test_genome: Path, known_locus: dict):
        variant = (
            f"{known_locus['chrom']}:{known_locus['pos']}:"
            f"{known_locus['ref']}>{known_locus['alt']}"
        )
        out = score_variant(
            variant, celltype="test-cell", genome_path=test_genome, scorer=_stub_scorer()
        )
        # Serialized via EvidenceBundle.to_dict — same schema the reasoning layer sees.
        assert set(out) >= {"delta_log_counts", "direction", "ref_log_counts", "model"}
        assert isinstance(out["delta_log_counts"], float)
        assert out["direction"] in {"increase", "decrease", "neutral"}
        assert out["model"] == "stub(offline)"

    def test_missing_genome_returns_actionable_error(self, monkeypatch):
        monkeypatch.delenv(GENOME_ENV_VAR, raising=False)
        monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
        out = score_variant("chr2:60490908:T>G")
        assert "error" in out and "hint" in out
        assert GENOME_ENV_VAR in out["error"]
        assert GENOME_ENV_VAR in out["hint"]
        # It should point the user at the fix, not raise a cryptic genome error.
        assert "hg38" in out["hint"]

    def test_dispatch_score_variant_error_path(self, monkeypatch):
        monkeypatch.delenv(GENOME_ENV_VAR, raising=False)
        out = dispatch("score_variant", {"variant": "chr2:60490908:T>G"})
        assert "error" in out


class TestTraitLinkWrapper:
    def test_rsid_only_roundtrip(self):
        # No variant needed — the GWAS lookup is rsID-keyed.
        out = trait_link_tool("rs1427407", client=_FakeClient(_GWAS_PAYLOAD))
        assert "fetal hemoglobin measurement" in out["top_traits"]
        assert out["associations"][0]["p_value"] == 4e-53

    def test_optional_variant_is_accepted(self):
        out = trait_link_tool(
            "rs1427407", variant="chr2:60490908:T>G", client=_FakeClient(_GWAS_PAYLOAD)
        )
        assert out["top_traits"]
