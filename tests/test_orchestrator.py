"""Offline end-to-end tests for the orchestrator (stub scorer + routing fake client)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.genome import Variant
from reglens.orchestrator import analyze_variant
from reglens.report.render import render_text
from reglens.tools.chrombpnet_score import ChromBPNetScorer, StubBackend

# --- Canned API payloads ----------------------------------------------------
_ENSEMBL_GENES = [
    {"id": "ENSG00000119866", "external_name": "BCL11A", "biotype": "protein_coding",
     "start": 40, "end": 60, "strand": -1},
]
_UCSC_CCRE = {"encodeCcreCombined": [
    {"chromStart": 30, "chromEnd": 45, "name": "EH38E_X", "ucscLabel": "enhD",
     "description": "distal enhancer-like signature"},
]}
_GTEX_VARIANT = {"data": [{"snpId": "rsTEST", "variantId": "chrT_50_C_T_b38"}]}
_GTEX_EQTL = {"data": [{"geneSymbol": "C2orf74", "gencodeId": "ENSGX.8",
                        "tissueSiteDetailId": "Cells_Cultured_fibroblasts",
                        "nes": -0.23, "pValue": 0.0002}]}
_GWAS = {"_embedded": {"associations": [
    {"pvalue": 4e-53, "betaNum": 0.3, "betaUnit": "unit", "betaDirection": "decrease",
     "efoTraits": [{"trait": "fetal hemoglobin measurement"}]},
]}}
_EUROPEPMC = {"hitCount": 5, "resultList": {"result": [
    {"id": "24297846", "source": "MED", "pmid": "24297846", "title": "BCL11A enhancer.",
     "authorString": "Bauer DE", "journalTitle": "Science", "pubYear": "2013"},
]}}


class OrchestratorFake:
    """Dispatches get_json by endpoint (and the Ensembl feature param)."""

    def __init__(self, fail: set[str] | None = None):
        self.fail = fail or set()
        self.queries: list[dict] = []

    def get_json(self, url, params=None):
        params = params or {}
        if "overlap/region" in url:
            is_gene = params.get("feature") == "gene"
            if is_gene and "gene" in self.fail:
                raise RuntimeError("boom-gene")
            return _ENSEMBL_GENES if is_gene else []
        if "getData/track" in url:
            return _UCSC_CCRE
        if "dataset/variant" in url:
            return _GTEX_VARIANT
        if "singleTissueEqtl" in url:
            return _GTEX_EQTL
        if "/associations" in url:
            return _GWAS
        if "europepmc" in url:
            self.queries.append(params)
            if "lit" in self.fail:
                raise RuntimeError("boom-lit")
            return _EUROPEPMC
        raise AssertionError(f"unexpected URL: {url}")


@pytest.fixture
def scorer() -> ChromBPNetScorer:
    return ChromBPNetScorer(StubBackend(seed=1), window_length=40, model_name="stub")


class TestAnalyzeVariant:
    def test_full_bundle(self, scorer, test_genome: Path, known_locus: dict):
        variant = Variant(known_locus["chrom"], known_locus["pos"],
                          known_locus["ref"], known_locus["alt"])
        bundle = analyze_variant(
            variant, rsid="rsTEST", celltype="K562",
            genome_path=test_genome, scorer=scorer, client=OrchestratorFake(), window_length=40,
        )
        assert bundle.chrombpnet is not None
        assert bundle.regulatory is not None and bundle.regulatory.nearest is not None
        assert bundle.gene is not None and bundle.gene.nearest_gene.symbol == "BCL11A"
        assert bundle.gene.eqtls[0].gene_symbol == "C2orf74"
        assert bundle.trait is not None and "fetal hemoglobin" in bundle.trait.summary()
        assert bundle.literature is not None and bundle.literature.citations[0].pmid == "24297846"
        assert bundle.errors == {}

    def test_literature_query_chains_gene(self, scorer, test_genome: Path, known_locus: dict):
        variant = Variant(known_locus["chrom"], known_locus["pos"], "C", "T")
        client = OrchestratorFake()
        analyze_variant(variant, rsid="rsTEST", genome_path=test_genome, scorer=scorer,
                        client=client, window_length=40)
        # The Europe PMC query should include the chained gene symbol and rsID.
        assert client.queries
        q = client.queries[0]["query"]
        assert "rsTEST" in q and "BCL11A" in q

    def test_one_tool_failure_is_isolated(self, scorer, test_genome: Path, known_locus: dict):
        variant = Variant(known_locus["chrom"], known_locus["pos"], "C", "T")
        bundle = analyze_variant(variant, rsid="rsTEST", genome_path=test_genome, scorer=scorer,
                                 client=OrchestratorFake(fail={"gene"}), window_length=40)
        # gene_target failed but the rest still populated.
        assert bundle.gene is None
        assert "gene" in bundle.errors
        assert bundle.trait is not None
        assert bundle.regulatory is not None

    def test_no_genome_skips_sequence_tools(self, known_locus: dict):
        variant = Variant(known_locus["chrom"], known_locus["pos"], "C", "T")
        bundle = analyze_variant(variant, rsid="rsTEST", client=OrchestratorFake())
        assert bundle.chrombpnet is None
        assert bundle.motif is None  # no genome → no windows
        assert bundle.gene is not None  # API tools still run

    def test_render_text_smoke(self, scorer, test_genome: Path, known_locus: dict):
        variant = Variant(known_locus["chrom"], known_locus["pos"], "C", "T")
        bundle = analyze_variant(variant, rsid="rsTEST", genome_path=test_genome, scorer=scorer,
                                 client=OrchestratorFake(), window_length=40)
        text = render_text(bundle)
        assert "RegLens evidence" in text
        assert "ChromBPNet accessibility" in text

    def test_to_dict_is_json_able(self, scorer, test_genome: Path, known_locus: dict):
        import json
        variant = Variant(known_locus["chrom"], known_locus["pos"], "C", "T")
        bundle = analyze_variant(variant, rsid="rsTEST", genome_path=test_genome, scorer=scorer,
                                 client=OrchestratorFake(), window_length=40)
        s = json.dumps(bundle.to_dict())  # must not raise
        assert "chrombpnet" in s and "trait" in s
