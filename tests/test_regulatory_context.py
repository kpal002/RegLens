"""Offline tests for the regulatory-context tool (routing fake HTTP client)."""

from __future__ import annotations

from reglens.genome import Variant
from reglens.tools.regulatory_context import (
    _distance0,
    encode_ccres,
    regulatory_context,
)

# Canned UCSC encodeCcreCombined response near chr2:60490908 (variant NOT inside).
_UCSC_CCRE = {
    "encodeCcreCombined": [
        {"chromStart": 60489435, "chromEnd": 60489636, "name": "EH38E2001344",
         "ucscLabel": "enhD", "description": "distal enhancer-like signature"},
        {"chromStart": 60492646, "chromEnd": 60492868, "name": "EH38E2001348",
         "ucscLabel": "enhD", "description": "distal enhancer-like signature"},
    ]
}
# A case where the variant IS inside a cCRE.
_UCSC_INSIDE = {
    "encodeCcreCombined": [
        {"chromStart": 60490800, "chromEnd": 60491000, "name": "EH38E_INSIDE",
         "ucscLabel": "enhD", "description": "distal enhancer-like signature"},
    ]
}


class RoutingFakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        for needle, payload in self.routes.items():
            if needle in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")


def _client(ccre_payload, ensembl=None):
    return RoutingFakeClient({
        "getData/track": ccre_payload,
        "overlap/region": ensembl if ensembl is not None else [],
    })


class TestDistance0:
    def test_inside(self):
        assert _distance0(50, 10, 100) == 0

    def test_outside(self):
        assert _distance0(5, 10, 100) == 5
        assert _distance0(130, 10, 100) == 31  # to end-1


class TestEncodeCcres:
    def test_parses_and_computes_distance(self):
        elems = encode_ccres(Variant("chr2", 60490908, "T", "G"), client=_client(_UCSC_CCRE))
        assert len(elems) == 2
        assert all(e.source == "ENCODE-SCREEN" for e in elems)
        assert all(not e.overlaps for e in elems)  # variant in the gap between them

    def test_uses_zero_based_coords_in_request(self):
        client = _client(_UCSC_CCRE)
        encode_ccres(Variant("chr2", 60490908, "T", "G"), client=client, window=3000)
        params = client.calls[0][1]
        assert params["start"] == 60490907 - 3000  # pos-1-window
        assert params["track"] == "encodeCcreCombined"


class TestRegulatoryContext:
    def test_not_in_ccre_reports_nearest(self):
        res = regulatory_context(Variant("chr2", 60490908, "T", "G"),
                                 client=_client(_UCSC_CCRE), include_ensembl=False)
        assert res.in_ccre is False
        assert res.nearest is not None
        assert res.nearest.element_type == "enhD"
        assert "not in a cCRE" in res.summary()
        assert "distal enhancer-like" in res.summary()

    def test_inside_ccre(self):
        res = regulatory_context(Variant("chr2", 60490908, "T", "G"),
                                 client=_client(_UCSC_INSIDE), include_ensembl=False)
        assert res.in_ccre is True
        assert res.nearest.overlaps
        assert "inside a distal enhancer-like" in res.summary()

    def test_merges_and_sorts_sources(self):
        ensembl = [{"id": "ENSR_X", "feature_type": "enhancer", "description": "enh",
                    "start": 60490905, "end": 60490915}]  # overlaps -> distance 0
        res = regulatory_context(Variant("chr2", 60490908, "T", "G"),
                                 client=_client(_UCSC_CCRE, ensembl=ensembl))
        # Ensembl element overlaps (distance 0) so it sorts ahead of the far cCREs.
        assert res.nearest.source == "Ensembl"
        assert res.elements == sorted(res.elements, key=lambda e: e.distance)

    def test_empty(self):
        res = regulatory_context(Variant("chr2", 60490908, "T", "G"),
                                 client=_client({"encodeCcreCombined": []}), include_ensembl=False)
        assert res.nearest is None
        assert "no regulatory element" in res.summary()
