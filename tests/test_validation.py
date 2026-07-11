"""Offline tests for the validation harness (metrics, dataset, evaluate)."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from reglens.genome import Variant
from reglens.tools.chrombpnet_score import ChromBPNetScorer, StubBackend
from reglens.validation.dataset import LabeledVariant, load_labeled_variants
from reglens.validation.harness import (
    annotation_baseline,
    default_score,
    evaluate,
    evaluate_batched,
)
from reglens.validation.metrics import roc_auc, roc_curve


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc([3, 4, 5], [0, 1, 1]) == pytest.approx(1.0)

    def test_inverted(self):
        assert roc_auc([5, 4, 3], [0, 1, 1]) == pytest.approx(0.0)

    def test_chance_with_ties(self):
        # All-equal scores → every pair tied → AUROC 0.5.
        assert roc_auc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)

    def test_known_value(self):
        # pos={3}, neg={1,2}; pos ranks above both → AUROC 1.0. Add a tie:
        # scores [1,2,2], labels [0,0,1] → pos(2) tied with one neg(2): AUROC 0.75.
        assert roc_auc([1, 2, 2], [0, 0, 1]) == pytest.approx(0.75)

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            roc_auc([1, 2, 3], [1, 1, 1])


class TestRocCurve:
    def test_endpoints_and_monotonic(self):
        fpr, tpr, thr = roc_curve([3, 4, 5, 1], [0, 1, 1, 0])
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


class TestLoadLabeledVariants:
    def test_parses_tsv_with_annotations(self, tmp_path):
        p = tmp_path / "vars.tsv"
        p.write_text(
            "chrom\tpos\tref\talt\tlabel\trsid\tsource\tcadd\n"
            "chr2\t60490908\tT\tG\t1\trs1427407\tMPRA\t22.1\n"
            "chr1\t100\tA\tC\t0\t\tbenign\t3.2\n"
        )
        lvs = load_labeled_variants(p)
        assert len(lvs) == 2
        assert lvs[0].variant.chrom == "chr2" and lvs[0].label == 1
        assert lvs[0].rsid == "rs1427407" and lvs[0].annotations["cadd"] == 22.1
        assert lvs[1].rsid is None

    def test_missing_columns_raises(self, tmp_path):
        p = tmp_path / "bad.tsv"
        p.write_text("chrom\tpos\tref\talt\n")  # no label
        with pytest.raises(ValueError, match="missing required"):
            load_labeled_variants(p)

    def test_bad_label_raises(self, tmp_path):
        p = tmp_path / "bad.tsv"
        p.write_text("chrom\tpos\tref\talt\tlabel\nchr1\t10\tA\tC\t2\n")
        with pytest.raises(ValueError, match="label must be"):
            load_labeled_variants(p)


@pytest.fixture
def val_genome(tmp_path):
    """A controlled 100 bp contig so we know every reference base."""
    seq = "ACGTACGTGGCCAATTACGT" * 5
    fa = tmp_path / "g.fa"
    fa.write_text(">chrV synthetic\n" + "\n".join(textwrap.wrap(seq, 60)) + "\n")
    return fa, seq


def _labeled(seq, pos, label, cadd):
    ref = seq[pos - 1]
    alt = "A" if ref != "A" else "C"
    return LabeledVariant(
        variant=Variant("chrV", pos, ref, alt),
        label=label, source="test", annotations={"cadd": cadd},
    )


class TestEvaluate:
    def _scorer(self):
        return ChromBPNetScorer(StubBackend(seed=3), window_length=20, model_name="stub")

    def test_end_to_end_report(self, val_genome):
        fa, seq = val_genome
        variants = [
            _labeled(seq, 20, 1, cadd=25.0), _labeled(seq, 30, 1, cadd=20.0),
            _labeled(seq, 40, 0, cadd=2.0), _labeled(seq, 50, 0, cadd=1.0),
        ]
        report = evaluate(variants, self._scorer(), genome_path=fa)
        assert report.n_pos == 2 and report.n_neg == 2 and report.errors == 0
        assert 0.0 <= report.model_auroc <= 1.0
        # Baseline CADD perfectly separates by construction → AUROC 1.0.
        assert report.baseline_auroc == pytest.approx(1.0)
        assert "AUROC model=" in report.summary()

    def test_per_variant_error_isolated(self, val_genome):
        fa, seq = val_genome
        good = _labeled(seq, 30, 1, cadd=9.0)
        bad = LabeledVariant(Variant("chrV", 30, "N", "A"), label=0)  # ref mismatch
        report = evaluate([good, bad], self._scorer(), genome_path=fa)
        assert report.errors == 1
        assert any(s.error is not None for s in report.scored)

    def test_no_baseline_when_absent(self, val_genome):
        fa, seq = val_genome
        variants = [  # no annotations → baseline unavailable
            LabeledVariant(_labeled(seq, 20, 1, 0).variant, 1),
            LabeledVariant(_labeled(seq, 40, 0, 0).variant, 0),
        ]
        report = evaluate(variants, self._scorer(), genome_path=fa,
                          baseline=annotation_baseline("cadd"))
        assert report.baseline_auroc is None

    def test_progress_flag_runs(self, val_genome):
        fa, seq = val_genome
        variants = [_labeled(seq, p, lab, 0) for p, lab in [(20, 1), (40, 0)]]
        report = evaluate(variants, self._scorer(), genome_path=fa, progress=True)
        assert report.n_pos == 1 and report.n_neg == 1  # progress path doesn't break it

    def test_roc_points(self, val_genome):
        fa, seq = val_genome
        variants = [_labeled(seq, p, lab, 0) for p, lab in
                    [(20, 1), (30, 1), (40, 0), (50, 0)]]
        report = evaluate(variants, self._scorer(), genome_path=fa)
        pts = report.roc_points()
        assert pts is not None and len(pts[0]) == len(pts[1])

    def test_batched_matches_per_variant(self, val_genome):
        fa, seq = val_genome
        variants = [_labeled(seq, p, lab, 0) for p, lab in
                    [(20, 1), (30, 1), (40, 0), (50, 0), (60, 1), (70, 0)]]
        scorer = self._scorer()
        slow = evaluate(variants, scorer, genome_path=fa)
        fast = evaluate_batched(variants, scorer, genome_path=fa, chunk_size=2)
        # Batched must reproduce the per-variant Δ log-counts (and thus AUROC) exactly.
        for s, f in zip(slow.scored, fast.scored, strict=True):
            assert s.delta_log_counts == pytest.approx(f.delta_log_counts)
        assert fast.model_auroc == pytest.approx(slow.model_auroc)

    def test_batched_isolates_errors(self, val_genome):
        fa, seq = val_genome
        good = _labeled(seq, 30, 1, 0)
        bad = LabeledVariant(Variant("chrV", 30, "N", "A"), label=0)  # ref mismatch
        rep = evaluate_batched([good, bad], self._scorer(), genome_path=fa)
        assert rep.errors == 1 and any(s.error for s in rep.scored)


class TestCurateMpra:
    def _rows(self):
        # Same element 'E1' (matched); mix of releases, SNV/indel, and p-values.
        return [
            {"Chrom": "2", "Pos": "100", "Ref": "C", "Alt": "T", "Coefficient": "-0.9",
             "pValue": "0.001", "Element": "E1", "Release": "GRCh38"},  # positive
            {"Chrom": "2", "Pos": "101", "Ref": "A", "Alt": "G", "Coefficient": "0.02",
             "pValue": "0.8", "Element": "E1", "Release": "GRCh38"},   # negative
            {"Chrom": "2", "Pos": "102", "Ref": "A", "Alt": "G", "Coefficient": "0.3",
             "pValue": "0.05", "Element": "E1", "Release": "GRCh38"},  # ambiguous → drop
            {"Chrom": "2", "Pos": "103", "Ref": "A", "Alt": "-", "Coefficient": "0.9",
             "pValue": "0.001", "Element": "E1", "Release": "GRCh38"}, # deletion → drop
            {"Chrom": "2", "Pos": "100", "Ref": "C", "Alt": "T", "Coefficient": "-0.9",
             "pValue": "0.001", "Element": "E1", "Release": "GRCh37"}, # wrong release → drop
        ]

    def test_labels_and_drops(self):
        from reglens.validation.build_mpra_benchmark import curate
        out, stats = curate(self._rows())
        assert stats.n_pos == 1 and stats.n_neg == 1
        assert stats.dropped_ambiguous == 1
        assert stats.dropped_nonsnv == 1  # the deletion
        assert stats.dropped_wrong_release == 1
        assert stats.per_element["E1"] == (1, 1)  # (pos, neg) — matched in the same element

    def test_adds_chr_prefix(self):
        from reglens.validation.build_mpra_benchmark import curate
        out, _ = curate(self._rows())
        assert all(o["chrom"].startswith("chr") for o in out)
        assert all(o["source"] == "E1" for o in out)  # element = match group

    def test_min_abs_effect_filters_positive(self):
        from reglens.validation.build_mpra_benchmark import curate
        _, stats = curate(self._rows(), min_abs_effect=1.0)  # the positive has |coef|=0.9
        assert stats.n_pos == 0  # dropped below effect threshold → becomes ambiguous


class TestDefaultScore:
    def test_is_abs_delta(self):
        from reglens.genome import SequenceWindow
        from reglens.tools.chrombpnet_score import VariantScore
        vs = VariantScore(Variant("c", 1, "A", "G"), None, 1.0, -0.5, -1.5, "decrease",
                          None, SequenceWindow("c", 0, 4, 2, "ACGT", "ACGT"), "m")
        assert default_score(vs) == pytest.approx(1.5)


class TestPerSourceAndRender:
    def _report(self):
        from reglens.validation.harness import ScoredVariant, ValidationReport

        def sv(src, label, score):
            return ScoredVariant(LabeledVariant(Variant("c", 1, "A", "G"), label, source=src),
                                 model_score=score)
        scored = [
            sv("BCL11A", 1, 0.9), sv("BCL11A", 1, 0.8), sv("BCL11A", 0, 0.2), sv("BCL11A", 0, 0.1),
            sv("SORT1", 1, 0.3), sv("SORT1", 0, 0.7),  # inverted within SORT1
        ]
        return ValidationReport(n_pos=3, n_neg=3, model_auroc=0.7, baseline_auroc=None,
                                model_name="stub", scored=scored)

    def test_per_source_auroc(self):
        per = dict((r[0], r[1:]) for r in self._report().per_source_auroc())
        assert per["BCL11A"] == (pytest.approx(1.0), 2, 2)   # perfectly separated
        assert per["SORT1"] == (pytest.approx(0.0), 1, 1)    # inverted
        # sorted by size → BCL11A (4) before SORT1 (2)
        assert self._report().per_source_auroc()[0][0] == "BCL11A"

    def test_render_validation(self):
        from reglens.report.render import render_validation
        text = render_validation(self._report())
        assert "OVERALL" in text and "per element" in text
        assert "BCL11A" in text and "baseline(CADD)=n/a" in text


class TestCaddMerge:
    def test_annotate_benchmark(self, tmp_path):
        from reglens.validation.cadd import annotate_benchmark, load_cadd_scores
        bench = tmp_path / "b.tsv"
        bench.write_text(
            "chrom\tpos\tref\talt\tlabel\trsid\tsource\n"
            "chr2\t100\tC\tT\t1\t\tBCL11A\n"
            "chr2\t200\tA\tG\t0\t\tBCL11A\n"
        )
        cadd = tmp_path / "cadd.tsv"
        cadd.write_text(
            "## CADD GRCh38\n#Chrom\tPos\tRef\tAlt\tRawScore\tPHRED\n"
            "2\t100\tC\tT\t1.2\t22.4\n"
        )
        scores = load_cadd_scores(cadd)
        assert scores[("2", "100", "C", "T")] == 22.4
        out = tmp_path / "out.tsv"
        n, total = annotate_benchmark(bench, cadd, out)
        assert (n, total) == (1, 2)
        lvs = load_labeled_variants(out)
        assert lvs[0].annotations["cadd"] == 22.4
        assert "cadd" not in lvs[1].annotations  # unmatched → empty, not loaded
