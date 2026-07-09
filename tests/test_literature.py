"""Offline tests for the Europe PMC literature tool (fake HTTP client)."""

from __future__ import annotations

import pytest

from reglens.genome import Variant
from reglens.tools.literature import (
    Citation,
    build_query,
    search_literature,
)

# A trimmed but realistic Europe PMC search payload.
_FAKE_PAYLOAD = {
    "hitCount": 2,
    "resultList": {
        "result": [
            {
                "id": "24297846",
                "source": "MED",
                "pmid": "24297846",
                "doi": "10.1126/science.1242088",
                "title": "An erythroid enhancer of BCL11A determines fetal hemoglobin level.",
                "authorString": "Bauer DE, et al.",
                "journalTitle": "Science",
                "pubYear": "2013",
                "citedByCount": 900,
            },
            {
                "id": "PPR123",
                "source": "PPR",
                "title": "A preprint with no PMID",
                "authorString": "Doe J.",
                "journalTitle": "bioRxiv",
                "pubYear": "2024",
            },
        ]
    },
}


class FakeClient:
    """Records the last request and returns a canned payload."""

    def __init__(self, payload):
        self.payload = payload
        self.last_url = None
        self.last_params = None

    def get_json(self, url, params=None):
        self.last_url = url
        self.last_params = params
        return self.payload


class TestSearchLiterature:
    def test_parses_payload(self):
        client = FakeClient(_FAKE_PAYLOAD)
        result = search_literature("rs1427407 AND BCL11A", client=client, page_size=5)
        assert result.hit_count == 2
        assert len(result.citations) == 2
        top = result.citations[0]
        assert top.pmid == "24297846"
        assert top.journal == "Science"
        assert top.title.endswith("level")  # trailing period stripped

    def test_sends_query_and_pagesize(self):
        client = FakeClient(_FAKE_PAYLOAD)
        search_literature("BCL11A", client=client, page_size=3)
        assert client.last_params["query"] == "BCL11A"
        assert client.last_params["pageSize"] == 3
        assert client.last_params["format"] == "json"

    def test_pmids_skips_missing(self):
        client = FakeClient(_FAKE_PAYLOAD)
        result = search_literature("q", client=client)
        assert result.pmids() == ["24297846"]  # preprint w/o PMID excluded

    def test_handles_empty_payload(self):
        result = search_literature("q", client=FakeClient({}))
        assert result.hit_count == 0
        assert result.citations == []


class TestCitation:
    def test_url_prefers_pubmed(self):
        c = Citation("1", "MED", "24297846", None, "t", "a", "j", "2013")
        assert "pubmed.ncbi.nlm.nih.gov/24297846" in c.url

    def test_url_falls_back_to_europepmc(self):
        c = Citation("PPR123", "PPR", None, None, "t", "a", "j", "2024")
        assert "europepmc.org/article/PPR/PPR123" in c.url

    def test_format_includes_identifier(self):
        c = Citation("1", "MED", "24297846", None, "Title", "Bauer DE", "Science", "2013")
        assert "PMID:24297846" in c.format()


class TestBuildQuery:
    def test_rsid_and_gene(self):
        q = build_query(rsid="rs1427407", gene="BCL11A", tf="GATA1", trait="fetal hemoglobin")
        assert "rs1427407" in q
        assert "BCL11A" in q
        assert '"fetal hemoglobin"' in q  # multi-word phrase quoted
        assert " AND " in q  # AND-joined by default for specificity

    def test_or_operator(self):
        q = build_query(gene="BCL11A", tf="GATA1", operator="OR")
        assert q == "BCL11A OR GATA1"

    def test_rejects_bad_operator(self):
        with pytest.raises(ValueError):
            build_query(gene="BCL11A", operator="NOT")

    def test_falls_back_to_variant_locus(self):
        q = build_query(variant=Variant("chr2", 60490908, "T", "G"), gene="BCL11A")
        assert '"chr2:60490908"' in q

    def test_requires_a_term(self):
        with pytest.raises(ValueError):
            build_query()
