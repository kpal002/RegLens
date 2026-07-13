"""Offline tests for the prospective discovery screen (reglens.validation.discovery)."""

from __future__ import annotations

import types

from reglens.genome import Variant
from reglens.validation import discovery as dsc


def _bundle(delta, motif_delta_score, tf, traits, lit, gene="MYB", rsid="rs1"):
    """Build a minimal fake evidence bundle for the screen."""
    motif = None
    if tf is not None:
        top = types.SimpleNamespace(tf_name=tf, delta_score=motif_delta_score)
        motif = types.SimpleNamespace(top=top)
    return types.SimpleNamespace(
        rsid=rsid,
        variant=Variant("chr6", 135000000, "A", "G"),
        chrombpnet=types.SimpleNamespace(delta_log_counts=delta),
        motif=motif,
        trait=types.SimpleNamespace(unique_traits=lambda: traits),
        literature=types.SimpleNamespace(hit_count=lit),
        gene=types.SimpleNamespace(nearest_gene=types.SimpleNamespace(symbol=gene)),
        errors={},
    )


class TestScreenBundle:
    def test_in_quadrant(self):
        b = _bundle(-0.8, motif_delta_score=-6.0, tf="GATA1",
                    traits=["red blood cell count"], lit=0)
        s = dsc.screen_bundle(b)
        assert s.in_quadrant is True
        assert s.motif_concordant is True   # (-6.0)*(-0.8) > 0
        assert s.blood_gwas is True and s.delta == 0.8

    def test_discordant_motif_excluded(self):
        # motif created (+) but accessibility decreased (−) → not concordant.
        b = _bundle(-0.8, motif_delta_score=+6.0, tf="GATA1",
                    traits=["red blood cell count"], lit=0)
        assert dsc.screen_bundle(b).motif_concordant is False
        assert dsc.screen_bundle(b).in_quadrant is False

    def test_small_delta_excluded(self):
        b = _bundle(0.05, -6.0, "GATA1", ["hemoglobin"], 0)
        assert dsc.screen_bundle(b).in_quadrant is False

    def test_non_blood_trait_excluded(self):
        b = _bundle(-0.8, -6.0, "GATA1", ["LDL cholesterol"], 0)
        s = dsc.screen_bundle(b)
        assert s.blood_gwas is False and s.in_quadrant is False

    def test_dense_literature_excluded(self):
        b = _bundle(-0.8, -6.0, "GATA1", ["platelet count"], lit=25)
        assert dsc.screen_bundle(b).in_quadrant is False

    def test_no_motif_excluded(self):
        b = _bundle(-0.8, 0.0, tf=None, traits=["neutrophil count"], lit=0)
        s = dsc.screen_bundle(b)
        assert s.motif_tf is None and s.in_quadrant is False


class TestRunAndRank:
    def test_ranks_quadrant_first_then_delta(self, monkeypatch):
        cands = [
            dsc.DiscoveryCandidate("rsA"), dsc.DiscoveryCandidate("rsB"),
            dsc.DiscoveryCandidate("rsC"),
        ]
        # Δ and motif-effect share sign → concordant (disruption + accessibility loss).
        bundles = {
            "rsA": _bundle(-0.9, -6.0, "GATA1", ["monocyte count"], lit=30, rsid="rsA"),  # dense
            "rsB": _bundle(-0.5, -6.0, "TAL1", ["platelet count"], lit=0, rsid="rsB"),  # quadrant
            "rsC": _bundle(-0.8, -6.0, "GATA1", ["red blood cell"], lit=1, rsid="rsC"),  # +bigger Δ
        }
        monkeypatch.setattr(dsc, "resolve_variant",
                            lambda rsid, *a, **k: Variant("chr6", 1, "A", "G"))
        monkeypatch.setattr(dsc, "analyze_variant",
                            lambda v, rsid=None, **k: bundles[rsid])
        ranked = dsc.run_discovery_screen(cands, scorer=object(), genome_path="x")
        order = [s.rsid for s, _ in ranked]
        assert order[0] == "rsC" and order[1] == "rsB"   # quadrant, by |Δ|
        assert order[2] == "rsA"                          # non-quadrant last
        rendered = dsc.render_screen(ranked)
        assert "2 candidate(s) in the discovery quadrant" in rendered
        assert "rsC" in rendered

    def test_bad_candidate_is_skipped_not_crashed(self, monkeypatch):
        import urllib.error
        cands = [dsc.DiscoveryCandidate("rsGOOD1"), dsc.DiscoveryCandidate("rsBAD"),
                 dsc.DiscoveryCandidate("rsGOOD2")]

        def fake_resolve(rsid, *a, **k):
            if rsid == "rsBAD":  # e.g. a merged/withdrawn rsID Ensembl 400s on
                raise urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
            return Variant("chr6", 1, "A", "G")

        b = _bundle(-0.8, -6.0, "GATA1", ["platelet count"], lit=0)
        monkeypatch.setattr(dsc, "resolve_variant", fake_resolve)
        monkeypatch.setattr(dsc, "analyze_variant", lambda v, rsid=None, **k: b)
        ranked = dsc.run_discovery_screen(cands, scorer=object(), genome_path="x")
        assert len(ranked) == 2  # the 400 candidate is skipped, the screen finishes

    def test_starter_pool_well_formed(self):
        assert len(dsc.BLOOD_TRAIT_CANDIDATES) >= 6
        for c in dsc.BLOOD_TRAIT_CANDIDATES:
            assert c.rsid.startswith("rs")


class _GwasClient:
    def __init__(self, snps):
        self.snps = snps

    def get_json(self, url, params=None):
        return {"_embedded": {"singleNucleotidePolymorphisms": self.snps}}


class TestFetchGwas:
    def _snps(self):
        return [
            {"rsId": "rs1", "functionalClass": "intron_variant"},
            {"rsId": "rs2", "functionalClass": "intergenic_variant"},
            {"rsId": "rs3", "functionalClass": "missense_variant"},   # coding → dropped
            {"rsId": "rs2", "functionalClass": "intron_variant"},     # dup → dropped
            {"rsId": "rs4", "functionalClass": None},                 # unannotated → kept
        ]

    def test_filters_coding_and_dedupes(self):
        cands = dsc.fetch_gwas_variants("platelet count", client=_GwasClient(self._snps()))
        rsids = {c.rsid for c in cands}
        assert rsids == {"rs1", "rs2", "rs4"}          # rs3 coding, rs2 deduped
        assert all(c.rsid.startswith("rs") for c in cands)

    def test_cap_and_deterministic(self):
        snps = [{"rsId": f"rs{i}", "functionalClass": "intron_variant"} for i in range(50)]
        a = dsc.fetch_gwas_variants("platelet count", client=_GwasClient(snps),
                                    max_variants=10, seed=1)
        b = dsc.fetch_gwas_variants("platelet count", client=_GwasClient(snps),
                                    max_variants=10, seed=1)
        assert len(a) == 10
        assert [c.rsid for c in a] == [c.rsid for c in b]   # seeded → reproducible

    def test_keep_coding_when_disabled(self):
        cands = dsc.fetch_gwas_variants("platelet count", client=_GwasClient(self._snps()),
                                        noncoding_only=False)
        assert "rs3" in {c.rsid for c in cands}
