"""Dependency-free ranking metrics for the validation harness.

AUROC via the Mann–Whitney U (rank-sum) identity and a ROC curve for plotting — both
pure numpy, so validation needs no scikit-learn. AUROC here answers "do larger scores
rank the positive (regulatory/causal) variants above the benign ones?"
"""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks with ties assigned their average rank (like scipy)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks_sorted = np.arange(1, len(values) + 1, dtype=np.float64)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:  # tie group [i, j] → average of their 1-based ranks
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity (ties handled).

    Args:
        scores: Per-item scores (higher = more "positive").
        labels: Binary labels (1 = positive, 0 = negative), same length as scores.

    Returns:
        AUROC in ``[0, 1]``: 1.0 = perfect ranking, 0.5 = chance, 0.0 = inverted.

    Raises:
        ValueError: If there are no positives or no negatives (AUROC undefined).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"AUROC needs both classes; got n_pos={n_pos}, n_neg={n_neg}")
    ranks = _average_ranks(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    # U = sum_ranks_pos - n_pos*(n_pos+1)/2; AUROC = U / (n_pos*n_neg).
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC-curve points (fpr, tpr, thresholds) for plotting.

    Args:
        scores: Per-item scores (higher = more "positive").
        labels: Binary labels (1 = positive, 0 = negative).

    Returns:
        ``(fpr, tpr, thresholds)`` arrays, starting at the ``(0, 0)`` origin.

    Raises:
        ValueError: If there are no positives or no negatives.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"ROC needs both classes; got n_pos={n_pos}, n_neg={n_neg}")
    order = np.argsort(-scores, kind="mergesort")  # descending score
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thresholds = np.concatenate([[np.inf], scores[order]])
    return fpr, tpr, thresholds
