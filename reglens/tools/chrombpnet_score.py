"""Score a noncoding variant's chromatin-accessibility effect with ChromBPNet.

This is the core deterministic signal of RegLens. Given a variant and a genome,
it builds reference/alternate sequence windows (:mod:`reglens.genome`), one-hot
encodes them, runs a **pretrained** ChromBPNet model on both, and reports the
difference in predicted log-counts (a proxy for chromatin accessibility):

    Δ log-counts = log_counts(alt) − log_counts(ref)

A negative Δ means the alt allele is predicted to *reduce* accessibility (e.g. by
disrupting a transcription-factor motif); a positive Δ means it increases it.

The model backend is swappable behind the :class:`ModelBackend` protocol so the
same scoring logic works with (a) a real pretrained ChromBPNet Keras model loaded
from a local path or downloaded checkpoint, or (b) a lightweight
:class:`StubBackend` that lets the whole pipeline — and its tests — run offline on
CPU with no TensorFlow install. **We only ever run inference here; no training.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from reglens.genome import (
    DEFAULT_WINDOW_LENGTH,
    SequenceWindow,
    Variant,
    build_sequence_windows,
)

# Fixed nucleotide ordering for one-hot encoding. Any base outside this set
# (e.g. 'N') maps to an all-zero column, matching ChromBPNet's convention.
_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# Below this absolute Δ (in log-count units) we call the effect "neutral" rather
# than over-claiming a direction from prediction noise.
NEUTRAL_THRESHOLD = 1e-3


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence as an ``(L, 4)`` float array.

    Columns are ordered A, C, G, T. Non-ACGT characters (e.g. ``N``) become
    all-zero rows.

    Args:
        sequence: A DNA string (case-insensitive).

    Returns:
        A ``(len(sequence), 4)`` float32 array.
    """
    seq = sequence.upper()
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = _BASE_TO_INDEX.get(base)
        if idx is not None:
            encoded[i, idx] = 1.0
    return encoded


@runtime_checkable
class ModelBackend(Protocol):
    """A ChromBPNet-like model that predicts a scalar log-count per sequence.

    Implementations must accept a batch of one-hot windows and return one
    predicted log-count value per window. This narrow interface is all the
    variant scorer needs, which keeps real Keras models and offline stubs
    interchangeable.
    """

    def predict_log_counts(self, one_hots: np.ndarray) -> np.ndarray:
        """Predict a log-count scalar for each one-hot window.

        Args:
            one_hots: A ``(batch, L, 4)`` float array of one-hot sequences.

        Returns:
            A ``(batch,)`` float array of predicted log-counts.
        """
        ...


@dataclass
class VariantScore:
    """Result of scoring a variant's chromatin-accessibility effect.

    Attributes:
        variant: The scored variant.
        celltype: Free-text cell-type / model context label (for reporting only).
        ref_log_counts: Predicted log-counts on the reference window.
        alt_log_counts: Predicted log-counts on the alternate window.
        delta_log_counts: ``alt_log_counts - ref_log_counts``.
        direction: ``"increase"``, ``"decrease"`` or ``"neutral"`` accessibility.
        window: The sequence windows the score was computed from.
        model_name: Identifier of the backend/model used.
    """

    variant: Variant
    celltype: str | None
    ref_log_counts: float
    alt_log_counts: float
    delta_log_counts: float
    direction: str
    window: SequenceWindow
    model_name: str

    @property
    def effect_size(self) -> float:
        """Absolute magnitude of the accessibility change (``|Δ|``)."""
        return abs(self.delta_log_counts)

    def summary(self) -> str:
        """A one-line human-readable summary of the score."""
        return (
            f"{self.variant} [{self.celltype or 'context: n/a'}] "
            f"Δlog-counts={self.delta_log_counts:+.4f} "
            f"({self.direction} accessibility; model={self.model_name})"
        )


def _direction(delta: float, threshold: float = NEUTRAL_THRESHOLD) -> str:
    """Classify a Δ log-counts value into an accessibility direction."""
    if delta > threshold:
        return "increase"
    if delta < -threshold:
        return "decrease"
    return "neutral"


class ChromBPNetScorer:
    """Scores variants by comparing model predictions on ref vs. alt windows.

    The scorer is backend-agnostic: hand it any :class:`ModelBackend`. Use
    :func:`load_backend` to obtain a real pretrained ChromBPNet backend, or pass a
    :class:`StubBackend` for offline use.
    """

    def __init__(
        self,
        backend: ModelBackend,
        window_length: int = DEFAULT_WINDOW_LENGTH,
        model_name: str = "chrombpnet",
    ) -> None:
        """Initialize the scorer.

        Args:
            backend: The model backend used to predict log-counts.
            window_length: Sequence window width in bp (must match the model's
                expected input length).
            model_name: Label recorded on results for provenance.
        """
        self.backend = backend
        self.window_length = window_length
        self.model_name = model_name

    def score_window(
        self, window: SequenceWindow, variant: Variant, celltype: str | None = None
    ) -> VariantScore:
        """Score a pre-built reference/alternate window pair.

        Args:
            window: The ref/alt :class:`SequenceWindow` to score.
            variant: The variant the window represents (for reporting).
            celltype: Optional cell-type/context label.

        Returns:
            The :class:`VariantScore`.
        """
        # Encode both windows and predict in a single batch so any per-call model
        # overhead is amortized and ref/alt share identical preprocessing.
        batch = np.stack(
            [one_hot_encode(window.ref_seq), one_hot_encode(window.alt_seq)], axis=0
        )
        log_counts = np.asarray(self.backend.predict_log_counts(batch), dtype=np.float64)
        ref_lc, alt_lc = float(log_counts[0]), float(log_counts[1])
        delta = alt_lc - ref_lc
        return VariantScore(
            variant=variant,
            celltype=celltype,
            ref_log_counts=ref_lc,
            alt_log_counts=alt_lc,
            delta_log_counts=delta,
            direction=_direction(delta),
            window=window,
            model_name=self.model_name,
        )

    def score_variant(
        self,
        variant: Variant,
        genome_path: str | os.PathLike[str] | None = None,
        celltype: str | None = None,
    ) -> VariantScore:
        """Build windows for a variant from a genome and score them.

        Args:
            variant: The variant to score.
            genome_path: Path to the genome FASTA (see
                :func:`reglens.genome.resolve_genome_path`).
            celltype: Optional cell-type/context label.

        Returns:
            The :class:`VariantScore`.
        """
        window = build_sequence_windows(
            variant, genome_path=genome_path, window_length=self.window_length
        )
        return self.score_window(window, variant, celltype=celltype)


class StubBackend:
    """A deterministic, dependency-free stand-in for a ChromBPNet model.

    It exists so the full scoring path runs offline (tests, CI, CLI demo) without
    TensorFlow or a downloaded checkpoint. The "prediction" is a fixed linear
    readout over per-position base identity: not biologically meaningful, but
    *deterministic* and *sensitive to a single-base change*, so a ref/alt pair
    yields a nonzero, reproducible Δ — exactly what the scoring logic needs to be
    exercised. Never use it for real interpretation.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialize the stub with a fixed pseudo-random readout.

        Args:
            seed: Seed for the fixed per-base position weights.
        """
        self.seed = seed
        # A fixed (max_len, 4) weight matrix; deterministic given the seed.
        rng = np.random.default_rng(seed)
        self._weights = rng.normal(0.0, 1.0, size=(DEFAULT_WINDOW_LENGTH, 4)).astype(
            np.float32
        )

    def predict_log_counts(self, one_hots: np.ndarray) -> np.ndarray:
        """Return a deterministic scalar 'log-count' per one-hot window.

        Args:
            one_hots: A ``(batch, L, 4)`` one-hot array.

        Returns:
            A ``(batch,)`` array of pseudo log-counts.
        """
        batch, length, _ = one_hots.shape
        weights = self._weights[:length]  # trim to the actual window length
        # Elementwise product then sum over positions and bases → one scalar each.
        # A single-base substitution changes exactly one row, guaranteeing a
        # nonzero Δ between a ref and its alt window.
        scores = np.einsum("blk,lk->b", one_hots, weights)
        # Squash into a plausible log-count range for readable demo output.
        return 5.0 + np.tanh(scores / np.sqrt(length))


class KerasChromBPNetBackend:
    """Backend wrapping a real pretrained ChromBPNet (TF/Keras) model.

    ChromBPNet models have two output heads: a base-resolution *profile* and a
    scalar *log-counts* head. For variant effect scoring we read the counts head.
    TensorFlow is imported lazily so importing RegLens never requires it.

    Note:
        Only used for real inference against a downloaded/pretrained checkpoint;
        it is not exercised by the offline test suite (that uses
        :class:`StubBackend`).
    """

    def __init__(self, model_path: str | os.PathLike[str], counts_head_index: int = 1) -> None:
        """Load a pretrained ChromBPNet Keras model from disk.

        Args:
            model_path: Path to a saved Keras model (``.h5`` or SavedModel dir).
            counts_head_index: Index of the log-counts output head. ChromBPNet's
                standard export orders outputs ``[profile, counts]`` → ``1``.

        Raises:
            ImportError: If TensorFlow is not installed (install the
                ``chrombpnet`` extra).
        """
        try:
            import tensorflow as tf  # noqa: F401  (imported for side effect/availability)
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "TensorFlow is required for the Keras ChromBPNet backend. "
                "Install it with: pip install 'reglens[chrombpnet]'"
            ) from exc

        self.model_path = str(model_path)
        self.counts_head_index = counts_head_index
        # `compile=False`: we only run inference, never optimize.
        self.model = keras.models.load_model(self.model_path, compile=False)

    def predict_log_counts(self, one_hots: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Predict log-counts for a batch of one-hot windows.

        Args:
            one_hots: A ``(batch, L, 4)`` one-hot array.

        Returns:
            A ``(batch,)`` array of predicted log-counts.
        """
        outputs = self.model.predict(one_hots, verbose=0)
        # A two-head model returns a list/tuple; a counts-only model an array.
        counts = outputs[self.counts_head_index] if isinstance(outputs, (list, tuple)) else outputs
        counts = np.asarray(counts, dtype=np.float64)
        # Collapse any trailing singleton dims to a flat (batch,) vector.
        return counts.reshape(counts.shape[0], -1).sum(axis=1)


def load_backend(
    model_path: str | os.PathLike[str] | None = None, *, stub_seed: int = 0
) -> ModelBackend:
    """Return a scoring backend: a real Keras model if a path is given, else a stub.

    Args:
        model_path: Path to a pretrained ChromBPNet Keras model. If ``None``,
            an offline :class:`StubBackend` is returned so the pipeline still runs.
        stub_seed: Seed used when falling back to the stub backend.

    Returns:
        A :class:`ModelBackend` implementation.
    """
    if model_path is None:
        return StubBackend(seed=stub_seed)
    return KerasChromBPNetBackend(model_path)
