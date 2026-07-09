"""Score a noncoding variant's chromatin-accessibility effect with ChromBPNet.

This is the core deterministic signal of RegLens. Given a variant and a genome,
it builds reference/alternate sequence windows (:mod:`reglens.genome`), one-hot
encodes them, runs a **pretrained, bias-corrected** ChromBPNet model on both, and
reports two complementary effects:

    Δ log-counts = log_counts(alt) − log_counts(ref)   (total-accessibility change)
    profile JSD  = Jensen–Shannon distance between the base-resolution
                   softmax profiles of ref vs alt        (footprint-shape change)

A negative Δ means the alt allele is predicted to *reduce* accessibility (e.g. by
disrupting a transcription-factor motif); a positive Δ means it increases it. The
profile JSD captures *where within the window* reads redistribute — a useful
signal for Thursday's motif story even when the total-count change is small.

Model contract (kundajelab ChromBPNet / variant-scorer):
* Load ``chrombpnet_nobias.h5`` — the Tn5 **bias-corrected** model. Loading the
  raw ``chrombpnet.h5`` or the ``bias_model_scaled.h5`` gives a garbage Δ.
* Input: one-hot ``(N, 2114, 4)``.
* Outputs (two heads): profile **logits** ``(N, ~1000)`` and a scalar **logcount**
  ``(N, 1)``. Standard export order is ``[profile, counts]``.

The model backend is swappable behind the :class:`ModelBackend` protocol so the
same scoring logic works with (a) a real pretrained ChromBPNet Keras model, or
(b) a lightweight :class:`StubBackend` that lets the whole pipeline — and its
tests — run offline on CPU with no TensorFlow install. **Inference only; no
training.**
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

# ChromBPNet's default profile-head output length (~1 kb of base-resolution
# predictions from a 2114 bp receptive field). Used to size the stub's profile.
DEFAULT_PROFILE_LENGTH = 1000

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


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def jensen_shannon_distance(profile_logits_a: np.ndarray, profile_logits_b: np.ndarray) -> float:
    """Jensen–Shannon distance between two base-resolution profile logit vectors.

    Mirrors the ``jsd`` metric in kundajelab/variant-scorer: the profile logits are
    softmaxed into probability distributions over positions, then compared. The
    result is the square root of the JS divergence (base 2), so it lies in
    ``[0, 1]`` — 0 for identical footprint shapes, larger as the read distribution
    shifts (e.g. a TF footprint appears/disappears).

    Args:
        profile_logits_a: 1-D profile logits for allele A.
        profile_logits_b: 1-D profile logits for allele B (same length as A).

    Returns:
        The Jensen–Shannon distance as a float in ``[0, 1]``.
    """
    p = _softmax(np.asarray(profile_logits_a, dtype=np.float64).ravel())
    q = _softmax(np.asarray(profile_logits_b, dtype=np.float64).ravel())
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        # Sum only where a > 0; log base 2 so the divergence is bounded by 1.
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js_divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Clamp tiny negatives from floating point before the sqrt.
    return float(np.sqrt(max(js_divergence, 0.0)))


@dataclass
class BackendPrediction:
    """Raw model outputs for a batch of one-hot windows.

    Attributes:
        log_counts: ``(batch,)`` predicted log-counts (natural log of total
            coverage) — one scalar per window.
        profile_logits: ``(batch, P)`` base-resolution profile logits, or ``None``
            if the backend does not expose a profile head.
    """

    log_counts: np.ndarray
    profile_logits: np.ndarray | None = None


@runtime_checkable
class ModelBackend(Protocol):
    """A ChromBPNet-like model predicting a log-count (and optional profile).

    Implementations accept a batch of one-hot windows and return a
    :class:`BackendPrediction`. This narrow interface is all the variant scorer
    needs, which keeps real Keras models and offline stubs interchangeable.
    """

    def predict(self, one_hots: np.ndarray) -> BackendPrediction:
        """Run the model on a batch of one-hot windows.

        Args:
            one_hots: A ``(batch, L, 4)`` float array of one-hot sequences.

        Returns:
            A :class:`BackendPrediction` with per-window log-counts and, when
            available, base-resolution profile logits.
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
        delta_log_counts: ``alt_log_counts - ref_log_counts`` (primary signal).
        direction: ``"increase"``, ``"decrease"`` or ``"neutral"`` accessibility.
        profile_jsd: Jensen–Shannon distance between ref/alt base-resolution
            profiles (footprint-shape change), or ``None`` if the backend has no
            profile head.
        window: The sequence windows the score was computed from.
        model_name: Identifier of the backend/model used.
    """

    variant: Variant
    celltype: str | None
    ref_log_counts: float
    alt_log_counts: float
    delta_log_counts: float
    direction: str
    profile_jsd: float | None
    window: SequenceWindow
    model_name: str

    @property
    def effect_size(self) -> float:
        """Absolute magnitude of the accessibility change (``|Δ|``)."""
        return abs(self.delta_log_counts)

    def summary(self) -> str:
        """A one-line human-readable summary of the score."""
        jsd = f", profileJSD={self.profile_jsd:.4f}" if self.profile_jsd is not None else ""
        return (
            f"{self.variant} [{self.celltype or 'context: n/a'}] "
            f"Δlog-counts={self.delta_log_counts:+.4f} "
            f"({self.direction} accessibility{jsd}; model={self.model_name})"
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
            backend: The model backend used to predict log-counts (+ profile).
            window_length: Sequence window width in bp (must match the model's
                expected input length; ChromBPNet expects 2114).
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
        prediction = self.backend.predict(batch)
        log_counts = np.asarray(prediction.log_counts, dtype=np.float64)
        ref_lc, alt_lc = float(log_counts[0]), float(log_counts[1])
        delta = alt_lc - ref_lc

        # Profile-shape change, if the backend exposed a profile head.
        profile_jsd: float | None = None
        if prediction.profile_logits is not None:
            profiles = np.asarray(prediction.profile_logits, dtype=np.float64)
            profile_jsd = jensen_shannon_distance(profiles[0], profiles[1])

        return VariantScore(
            variant=variant,
            celltype=celltype,
            ref_log_counts=ref_lc,
            alt_log_counts=alt_lc,
            delta_log_counts=delta,
            direction=_direction(delta),
            profile_jsd=profile_jsd,
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
    TensorFlow or a downloaded checkpoint. Both heads are faked from a fixed linear
    readout over per-position base identity: not biologically meaningful, but
    *deterministic* and *sensitive to a single-base change*, so a ref/alt pair
    yields a nonzero, reproducible Δ log-counts **and** profile JSD — exactly what
    the scoring logic needs to be exercised. Never use it for real interpretation.
    """

    def __init__(self, seed: int = 0, profile_length: int = DEFAULT_PROFILE_LENGTH) -> None:
        """Initialize the stub with fixed pseudo-random readouts.

        Args:
            seed: Seed for the fixed per-base position weights.
            profile_length: Length of the fake profile head output.
        """
        self.seed = seed
        self.profile_length = profile_length
        rng = np.random.default_rng(seed)
        # Fixed (max_len, 4) weights for the counts head, deterministic per seed.
        self._count_weights = rng.normal(0.0, 1.0, size=(DEFAULT_WINDOW_LENGTH, 4)).astype(
            np.float32
        )
        # Fixed (max_len, 4, profile_length) weights projecting the sequence onto a
        # base-resolution profile, so a single-base change perturbs the profile.
        self._profile_weights = rng.normal(
            0.0, 1.0, size=(DEFAULT_WINDOW_LENGTH, 4, profile_length)
        ).astype(np.float32)

    def predict(self, one_hots: np.ndarray) -> BackendPrediction:
        """Return deterministic fake log-counts and profile logits.

        Args:
            one_hots: A ``(batch, L, 4)`` one-hot array.

        Returns:
            A :class:`BackendPrediction` with pseudo log-counts and profile logits.
        """
        _, length, _ = one_hots.shape
        count_w = self._count_weights[:length]
        profile_w = self._profile_weights[:length]

        # Counts head: contract sequence with per-base weights → one scalar each.
        # A single-base substitution changes exactly one row → nonzero Δ.
        raw_counts = np.einsum("blk,lk->b", one_hots, count_w)
        log_counts = 5.0 + np.tanh(raw_counts / np.sqrt(length))

        # Profile head: project each window onto `profile_length` logits.
        profile_logits = np.einsum("blk,lkp->bp", one_hots, profile_w) / np.sqrt(length)

        return BackendPrediction(log_counts=log_counts, profile_logits=profile_logits)


class KerasChromBPNetBackend:
    """Backend wrapping a real pretrained, bias-corrected ChromBPNet (TF/Keras) model.

    Load ``chrombpnet_nobias.h5`` (the Tn5 bias-corrected model). ChromBPNet models
    have two output heads exported as ``[profile, counts]``: a base-resolution
    profile (logits) and a scalar log-counts. We read the counts head as the
    primary Δ signal and the profile head for the footprint-shape (JSD) signal.
    TensorFlow is imported lazily so importing RegLens never requires it.

    Note:
        Exercised against a real downloaded checkpoint (see
        ``reglens/model/colab_verify_chrombpnet.ipynb``); the offline test suite
        uses :class:`StubBackend`.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        profile_head_index: int = 0,
        counts_head_index: int = 1,
    ) -> None:
        """Load a pretrained ChromBPNet Keras model from disk.

        Args:
            model_path: Path to ``chrombpnet_nobias.h5`` (or a SavedModel dir).
            profile_head_index: Index of the profile output head. ChromBPNet's
                standard export orders outputs ``[profile, counts]`` → ``0``.
            counts_head_index: Index of the log-counts output head → ``1``.

        Raises:
            ImportError: If TensorFlow is not installed (install the
                ``chrombpnet`` extra).
        """
        try:
            import tensorflow as tf  # noqa: F401  (imported for availability check)
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "TensorFlow is required for the Keras ChromBPNet backend. "
                "Install it with: pip install 'reglens[chrombpnet]'"
            ) from exc

        self.model_path = str(model_path)
        self.profile_head_index = profile_head_index
        self.counts_head_index = counts_head_index
        # `compile=False`: we only run inference, never optimize.
        self.model = keras.models.load_model(self.model_path, compile=False)

    def predict(self, one_hots: np.ndarray) -> BackendPrediction:  # pragma: no cover
        """Predict log-counts and profile logits for a batch of one-hot windows.

        Args:
            one_hots: A ``(batch, L, 4)`` one-hot array (``L`` must be 2114).

        Returns:
            A :class:`BackendPrediction` with counts and profile logits.
        """
        outputs = self.model.predict(one_hots, verbose=0)
        if isinstance(outputs, (list, tuple)):
            profile = np.asarray(outputs[self.profile_head_index], dtype=np.float64)
            counts = np.asarray(outputs[self.counts_head_index], dtype=np.float64)
        else:
            # A counts-only model: no profile head available.
            profile = None
            counts = np.asarray(outputs, dtype=np.float64)
        # Counts head is (batch, 1) log-total-counts → flatten to (batch,).
        log_counts = counts.reshape(counts.shape[0], -1).sum(axis=1)
        return BackendPrediction(log_counts=log_counts, profile_logits=profile)


def load_backend(
    model_path: str | os.PathLike[str] | None = None, *, stub_seed: int = 0
) -> ModelBackend:
    """Return a scoring backend: a real Keras model if a path is given, else a stub.

    Args:
        model_path: Path to a pretrained ``chrombpnet_nobias.h5`` model. If
            ``None``, an offline :class:`StubBackend` is returned so the pipeline
            still runs.
        stub_seed: Seed used when falling back to the stub backend.

    Returns:
        A :class:`ModelBackend` implementation.
    """
    if model_path is None:
        return StubBackend(seed=stub_seed)
    return KerasChromBPNetBackend(model_path)
