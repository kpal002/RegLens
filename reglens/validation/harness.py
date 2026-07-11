"""Validation harness: does the model rank regulatory variants above benign ones?

Scores each :class:`~reglens.validation.dataset.LabeledVariant` through a
:class:`~reglens.tools.chrombpnet_score.ChromBPNetScorer` (score = ``|Δ log-counts|``,
the predicted effect magnitude) and reports **AUROC** discriminating positives (label 1)
from negatives (label 0), alongside a **naive baseline** (e.g. CADD/phyloP carried on
each variant, or any injected score function). Per-variant failures are isolated so one
bad locus doesn't sink the run.

Honest framing: this validates the **pretrained** model's variant scores; the trained
model is an extensibility demo, not the thing under test.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reglens.tools.chrombpnet_score import ChromBPNetScorer, VariantScore
from reglens.validation.dataset import LabeledVariant
from reglens.validation.metrics import roc_auc, roc_curve

# score_fn: VariantScore → scalar model score (higher = more likely regulatory).
ScoreFn = Callable[[VariantScore], float]
# baseline_fn: LabeledVariant → scalar baseline score, or None if unavailable.
BaselineFn = Callable[[LabeledVariant], float | None]


def default_score(score: VariantScore) -> float:
    """Default model score: the magnitude of the predicted accessibility change."""
    return abs(score.delta_log_counts)


def annotation_baseline(*keys: str) -> BaselineFn:
    """Build a baseline that reads the first available annotation (e.g. CADD/phyloP).

    Args:
        *keys: Annotation names to try in order (e.g. ``"cadd", "phylop"``).

    Returns:
        A baseline function returning the first present annotation, else ``None``.
    """

    def baseline(lv: LabeledVariant) -> float | None:
        for key in keys:
            if key in lv.annotations:
                return lv.annotations[key]
        return None

    return baseline


@dataclass
class ScoredVariant:
    """A labeled variant with its model (and baseline) scores, or an error."""

    labeled: LabeledVariant
    model_score: float | None = None
    baseline_score: float | None = None
    delta_log_counts: float | None = None
    error: str | None = None


@dataclass
class ValidationReport:
    """Outcome of a validation run.

    Attributes:
        n_pos: Number of positive (regulatory/causal) variants scored.
        n_neg: Number of negative (benign) variants scored.
        model_auroc: AUROC of the model score, or ``None`` if not computable.
        baseline_auroc: AUROC of the baseline score, or ``None`` if unavailable.
        model_name: The scoring model's identifier.
        scored: Per-variant results (including failures).
        errors: Count of variants that failed to score.
    """

    n_pos: int
    n_neg: int
    model_auroc: float | None
    baseline_auroc: float | None
    model_name: str
    scored: list[ScoredVariant] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> str:
        """A one-line human-readable summary."""
        m = f"{self.model_auroc:.3f}" if self.model_auroc is not None else "n/a"
        b = f"{self.baseline_auroc:.3f}" if self.baseline_auroc is not None else "n/a"
        return (
            f"AUROC model={m} vs baseline={b} "
            f"({self.n_pos} pos / {self.n_neg} neg, {self.errors} errors) [{self.model_name}]"
        )

    def roc_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Model ROC-curve ``(fpr, tpr)`` for plotting, or ``None`` if not computable."""
        scores, labels = _score_label_arrays(self.scored, use_baseline=False)
        if scores is None:
            return None
        fpr, tpr, _ = roc_curve(scores, labels)
        return fpr, tpr

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary (per-variant scores omitted for brevity)."""
        return {
            "model_name": self.model_name,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "errors": self.errors,
            "model_auroc": self.model_auroc,
            "baseline_auroc": self.baseline_auroc,
        }


def _score_label_arrays(
    scored: list[ScoredVariant], use_baseline: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Collect (scores, labels) for items that have the requested score."""
    xs, ys = [], []
    for s in scored:
        value = s.baseline_score if use_baseline else s.model_score
        if value is not None:
            xs.append(value)
            ys.append(s.labeled.label)
    if not xs or len(set(ys)) < 2:  # need both classes present
        return None, None
    return np.asarray(xs, dtype=np.float64), np.asarray(ys)


def _safe_auroc(scored: list[ScoredVariant], use_baseline: bool) -> float | None:
    """AUROC over the scored items, or None if not computable."""
    scores, labels = _score_label_arrays(scored, use_baseline)
    if scores is None:
        return None
    return roc_auc(scores, labels)


def evaluate(
    variants: list[LabeledVariant],
    scorer: ChromBPNetScorer,
    genome_path: str | os.PathLike[str] | None = None,
    score_fn: ScoreFn = default_score,
    baseline: BaselineFn | None = None,
) -> ValidationReport:
    """Score a labeled variant set and report model vs baseline AUROC.

    Args:
        variants: The labeled variants (positives label 1, negatives label 0).
        scorer: A configured :class:`ChromBPNetScorer` (pretrained model or stub).
        genome_path: hg38 FASTA path used to build each variant's windows.
        score_fn: Maps a :class:`VariantScore` to a scalar model score.
        baseline: Optional baseline score function (e.g.
            :func:`annotation_baseline("cadd")`); defaults to trying ``cadd``/``phylop``.

    Returns:
        A :class:`ValidationReport`.
    """
    if baseline is None:
        baseline = annotation_baseline("cadd", "phylop")

    scored: list[ScoredVariant] = []
    errors = 0
    for lv in variants:
        item = ScoredVariant(labeled=lv, baseline_score=baseline(lv))
        try:
            vs = scorer.score_variant(lv.variant, genome_path=genome_path, celltype=lv.source)
            item.model_score = score_fn(vs)
            item.delta_log_counts = vs.delta_log_counts
        except Exception as exc:  # noqa: BLE001 - isolate per-variant failures
            item.error = f"{type(exc).__name__}: {exc}"
            errors += 1
        scored.append(item)

    return ValidationReport(
        n_pos=sum(1 for s in scored if s.model_score is not None and s.labeled.label == 1),
        n_neg=sum(1 for s in scored if s.model_score is not None and s.labeled.label == 0),
        model_auroc=_safe_auroc(scored, use_baseline=False),
        baseline_auroc=_safe_auroc(scored, use_baseline=True),
        model_name=scorer.model_name,
        scored=scored,
        errors=errors,
    )
