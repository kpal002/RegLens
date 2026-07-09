"""Offline tests for the GWAS Catalog trait-link tool (fake HTTP client)."""

from __future__ import annotations

from reglens.tools.trait_link import TraitAssociation, trait_link

# Canned GWAS Catalog associationBySnp payload for rs1427407 (trimmed).
_FAKE = {
    "_embedded": {
        "associations": [
            {
                "pvalue": 2e-07,
                "betaNum": 1.07,
                "betaUnit": "unit",
                "betaDirection": "increase",
                "efoTraits": [{"trait": "fetal hemoglobin measurement"}],
                "loci": [{"strongestRiskAlleles": [{"riskAlleleName": "rs1427407-T"}]}],
            },
            {
                "pvalue": 4e-53,
                "betaNum": 0.3,
                "betaUnit": "unit",
                "betaDirection": "decrease",
                "efoTraits": [{"trait": "fetal hemoglobin measurement"}],
            },
            {
                "pvalue": 4e-09,
                "betaNum": 0.0634,
                "betaUnit": "SD",
                "betaDirection": "decrease",
                "efoTraits": [{"trait": "erythrocyte attribute"}],
            },
        ]
    }
}


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.last_url = None
        self.last_params = None

    def get_json(self, url, params=None):
        self.last_url, self.last_params = url, params
        return self.payload


class TestTraitLink:
    def test_sorted_by_pvalue(self):
        res = trait_link("rs1427407", client=FakeClient(_FAKE))
        pvals = [a.p_value for a in res.associations]
        assert pvals == sorted(pvals)
        assert res.associations[0].p_value == 4e-53  # most significant first

    def test_unique_traits_order(self):
        res = trait_link("rs1427407", client=FakeClient(_FAKE))
        assert res.unique_traits() == ["fetal hemoglobin measurement", "erythrocyte attribute"]

    def test_summary(self):
        res = trait_link("rs1427407", client=FakeClient(_FAKE))
        s = res.summary()
        assert "fetal hemoglobin measurement" in s
        assert "more trait" in s  # 2 traits -> "+1 more trait"

    def test_risk_allele_parsed(self):
        res = trait_link("rs1427407", client=FakeClient(_FAKE))
        # The association with a risk allele carries it through.
        assert any(a.risk_allele == "rs1427407-T" for a in res.associations)

    def test_max_associations(self):
        res = trait_link("rs1427407", client=FakeClient(_FAKE), max_associations=1)
        assert len(res.associations) == 1
        assert res.associations[0].p_value == 4e-53

    def test_sends_projection_param(self):
        client = FakeClient(_FAKE)
        trait_link("rs1427407", client=client)
        assert client.last_params["projection"] == "associationBySnp"
        assert client.last_url.endswith("rs1427407/associations")

    def test_empty_payload(self):
        res = trait_link("rsX", client=FakeClient({}))
        assert res.associations == []
        assert "no GWAS Catalog associations" in res.summary()


class TestEffectStr:
    def test_beta(self):
        a = TraitAssociation(["t"], 1e-5, beta=1.07, beta_unit="unit", beta_direction="increase")
        assert "β=1.07" in a.effect_str() and "increase" in a.effect_str()

    def test_odds_ratio(self):
        a = TraitAssociation(["t"], 1e-5, or_per_copy=1.5)
        assert a.effect_str() == "OR=1.5"

    def test_empty(self):
        assert TraitAssociation(["t"], 1e-5).effect_str() == ""
