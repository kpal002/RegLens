"""Offline tests for the gene-target tool (routing fake HTTP client)."""

from __future__ import annotations

from reglens.genome import Variant
from reglens.tools.gene_target import (
    GeneOverlap,
    _distance,
    _pick_nearest,
    gene_target,
    nearest_genes,
)

# Canned Ensembl overlap response near chr2:60490908 (BCL11A intron).
_ENSEMBL_GENES = [
    {"id": "ENSG00000233953", "external_name": None, "biotype": "lncRNA",
     "start": 60495686, "end": 60499964, "strand": 1},
    {"id": "ENSG00000119866", "external_name": "BCL11A", "biotype": "protein_coding",
     "start": 60450520, "end": 60554467, "strand": -1},
    {"id": "ENSG00000228590", "external_name": "MIR4432HG", "biotype": "lncRNA",
     "start": 60311286, "end": 60439841, "strand": -1},
]
# Canned GTEx variant lookup + eQTL responses for rs1427407.
_GTEX_VARIANT = {"data": [{"snpId": "rs1427407", "variantId": "chr2_60490908_T_G_b38"}]}
_GTEX_EQTL = {"data": [
    {"geneSymbol": "C2orf74", "gencodeId": "ENSG00000237651.8",
     "tissueSiteDetailId": "Cells_Cultured_fibroblasts", "nes": -0.2327, "pValue": 0.000227},
]}


class RoutingFakeClient:
    """Routes get_json by URL substring to canned payloads."""

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.calls: list[str] = []

    def get_json(self, url, params=None):
        self.calls.append(url)
        for needle, payload in self.routes.items():
            if needle in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")


def _client() -> RoutingFakeClient:
    return RoutingFakeClient({
        "overlap/region": _ENSEMBL_GENES,
        "dataset/variant": _GTEX_VARIANT,
        "singleTissueEqtl": _GTEX_EQTL,
    })


class TestDistance:
    def test_inside_is_zero(self):
        assert _distance(50, 10, 100) == 0

    def test_outside_left_and_right(self):
        assert _distance(5, 10, 100) == 5
        assert _distance(130, 10, 100) == 30


class TestNearestGenes:
    def test_sorted_by_distance_and_overlap(self):
        genes = nearest_genes(Variant("chr2", 60490908, "T", "G"), client=_client())
        # BCL11A contains the variant → distance 0, ranked first.
        assert genes[0].symbol == "BCL11A"
        assert genes[0].overlaps

    def test_strips_chr_prefix_in_url(self):
        client = _client()
        nearest_genes(Variant("chr2", 60490908, "T", "G"), client=client)
        assert "/human/2:" in client.calls[0]  # 'chr2' -> '2'


class TestPickNearest:
    def test_prefers_protein_coding(self):
        genes = [
            GeneOverlap("L1", "", "lncRNA", 1, 10, 1, distance=5),
            GeneOverlap("C1", "GENE", "protein_coding", 100, 200, 1, distance=20),
        ]
        assert _pick_nearest(genes).symbol == "GENE"  # coding wins despite farther

    def test_none_when_empty(self):
        assert _pick_nearest([]) is None


class TestGeneTarget:
    def test_end_to_end_with_rsid(self):
        res = gene_target(Variant("chr2", 60490908, "T", "G"), rsid="rs1427407", client=_client())
        assert res.nearest_gene.symbol == "BCL11A"
        assert res.nearest_gene.overlaps
        assert [g.symbol for g in res.overlapping_genes] == ["BCL11A"]
        # GTEx's only eQTL is C2orf74 in fibroblasts (not BCL11A).
        assert len(res.eqtls) == 1
        assert res.eqtls[0].gene_symbol == "C2orf74"
        assert "BCL11A" in res.summary()
        assert "C2orf74" in res.summary()

    def test_skips_gtex_without_rsid(self):
        client = _client()
        res = gene_target(Variant("chr2", 60490908, "T", "G"), client=client)
        assert res.eqtls == []
        assert not any("gtex" in c for c in client.calls)  # no GTEx calls made

    def test_summary_no_eqtl(self):
        client = RoutingFakeClient({"overlap/region": _ENSEMBL_GENES,
                                    "dataset/variant": {"data": []}})
        res = gene_target(Variant("chr2", 60490908, "T", "G"), rsid="rsX", client=client)
        assert "no significant GTEx eQTL" in res.summary()
