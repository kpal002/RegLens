"""Validation harness: AUROC of variant Δ-scores (regulatory vs benign) vs a baseline.

Validates the **pretrained** ChromBPNet model's ability to rank known regulatory/causal
variants (MPRA / CAGI / fine-mapped) above benign ones, versus a naive baseline
(CADD / phyloP). See :func:`reglens.validation.harness.evaluate`.
"""

from reglens.validation.dataset import LabeledVariant, load_labeled_variants
from reglens.validation.harness import ValidationReport, evaluate
from reglens.validation.metrics import roc_auc, roc_curve

__all__ = [
    "LabeledVariant",
    "ValidationReport",
    "evaluate",
    "load_labeled_variants",
    "roc_auc",
    "roc_curve",
]
