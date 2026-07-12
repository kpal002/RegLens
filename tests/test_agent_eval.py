"""Offline tests for the agent-validation harness (reglens.validation.agent_eval).

Covers rsID→coordinate resolution (fake Ensembl payload), TF/gene/trait recovery scoring,
the ablation loop + red-team-effect summary, and the calibration table. No network, no
genome, no API.
"""

from __future__ import annotations

from reglens.agents.interpreter import MechanisticInterpretation
from reglens.genome import Variant
from reglens.validation import agent_eval as ae


def _interp(mechanism="", confidence="low", tf=None, gene=None, trait=None):
    return MechanisticInterpretation(
        mechanism=mechanism, direction="x", tf=tf, gene=gene, trait=trait,
        confidence=confidence,
    )


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params=None):
        return self.payload


class TestResolveVariant:
    def test_parses_grch38_mapping(self):
        payload = {
            "mappings": [
                {"assembly_name": "GRCh37", "seq_region_name": "2", "start": 1,
                 "allele_string": "T/G"},
                {"assembly_name": "GRCh38", "seq_region_name": "2", "start": 60490908,
                 "allele_string": "T/G"},
            ]
        }
        v = ae.resolve_variant("rs1427407", client=_FakeClient(payload))
        assert (v.chrom, v.pos, v.ref, v.alt) == ("chr2", 60490908, "T", "G")

    def test_raises_without_grch38(self):
        import pytest
        payload = {"mappings": [{"assembly_name": "GRCh37", "seq_region_name": "2",
                                 "start": 1, "allele_string": "T/G"}]}
        with pytest.raises(ValueError, match="No GRCh38"):
            ae.resolve_variant("rsX", client=_FakeClient(payload))


class TestRecoveryScoring:
    def _km(self, **kw):
        base = dict(rsid="rs1", gene="BCL11A", trait="fetal hemoglobin", pmid="1",
                    tf="GATA1", trait_terms=("HbF",))
        base.update(kw)
        return ae.KnownMechanism(**base)

    def test_full_recovery(self):
        km = self._km()
        i = _interp(mechanism="Alt disrupts a GATA1 motif in the BCL11A enhancer, "
                              "raising HbF.", tf="GATA1", gene="BCL11A")
        r = ae.score_recovery(i, km)
        assert (r.tf_recovered, r.gene_recovered, r.trait_recovered) == (True, True, True)

    def test_tf_from_prose_when_field_empty(self):
        km = self._km()
        i = _interp(mechanism="The variant weakens a GATA1 site.", gene="BCL11A",
                    trait="fetal hemoglobin")
        assert ae.score_recovery(i, km).tf_recovered is True

    def test_alias_matches(self):
        km = self._km(tf="POU2F1", tf_aliases=("Oct-1",), gene="LCT")
        i = _interp(mechanism="Enhances an Oct-1 site.", gene="LCT", trait="fetal hemoglobin")
        assert ae.score_recovery(i, km).tf_recovered is True

    def test_miss(self):
        km = self._km()
        i = _interp(mechanism="No coherent mechanism.", tf=None)
        r = ae.score_recovery(i, km)
        assert r.tf_recovered is False and r.gene_recovered is False

    def test_tf_none_not_scored(self):
        km = self._km(tf=None)
        r = ae.score_recovery(_interp(mechanism="x", gene="BCL11A", trait="HbF"), km)
        assert r.tf_recovered is None

    def test_recovery_rates(self):
        km = self._km()
        results = [
            ae.score_recovery(_interp(tf="GATA1", gene="BCL11A", trait="HbF",
                                      mechanism="disrupts GATA1"), km),
            ae.score_recovery(_interp(mechanism="no mechanism"), km),
        ]
        rates = ae.recovery_rates(results)
        assert rates["tf"] == "1/2" and rates["gene"] == "1/2"


class _FakeInterp:
    def __init__(self, interp):
        self._interp = interp

    def interpret(self, bundle):
        return self._interp


class _StubBundle:
    chrombpnet = None

    def to_dict(self):
        return {}


class TestAblation:
    def test_redteam_effect_summary(self, monkeypatch):
        monkeypatch.setattr(ae, "analyze_variant", lambda *a, **k: _StubBundle())
        v = Variant("chr2", 1, "C", "T")
        items = [
            (v, None, "null-1", "null"),
            (v, None, "strong-1", "strong"),
        ]
        # null: red-team lowers medium->low; strong: unchanged high.
        rows = ae.run_ablation(
            items,
            single=_FakeInterp(_interp(confidence="medium")),
            multi_no_redteam=_FakeInterp(_interp(confidence="medium")),
            multi_full=_FakeInterp(_interp(confidence="low")),  # applied to all here
        )
        assert len(rows) == 2
        rendered = ae.render_ablation(rows)
        # noRT(medium)→full(low): the red-team lowers each stratum by one.
        assert "red-team    (noRT→full)  : strong 1↓/0↑ of 1   null 1↓/0↑ of 1" in rendered
        # single(medium)→noRT(medium): no change from the specialist structure here.
        assert "multi-agent (single→noRT): strong 0↓/0↑ of 1   null 0↓/0↑ of 1" in rendered


class TestCalibration:
    def test_table_counts(self):
        strata = {
            "strong": [_interp(confidence="high"), _interp(confidence="medium")],
            "null": [_interp(confidence="low"), _interp(confidence="low")],
        }
        table = ae.calibration_table(strata)
        assert table.counts["strong"] == {"high": 1, "medium": 1, "low": 0}
        assert table.counts["null"] == {"high": 0, "medium": 0, "low": 2}
        rendered = ae.render_calibration(table)
        assert "strong" in rendered and "null" in rendered

    def test_curated_set_is_well_formed(self):
        # Every curated record has an rsID, gene, trait, and a PMID.
        assert len(ae.KNOWN_MECHANISMS) >= 10
        for km in ae.KNOWN_MECHANISMS:
            assert km.rsid.startswith("rs")
            assert km.gene and km.trait and km.pmid.isdigit()
