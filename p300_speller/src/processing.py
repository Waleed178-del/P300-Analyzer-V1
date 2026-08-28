"""processing.py — filtering and epoching for the P300 Speller.

This module turns the raw, continuous, timestamped EEG stream produced by
:mod:`acquisition` into clean, baseline-corrected epochs aligned to stimulus
flashes. It contains two responsibilities:

1. **Spectral conditioning** — a zero-phase Butterworth band-pass plus an IIR
   notch (mains hum), implemented with :mod:`scipy.signal`. Zero-phase
   (``filtfilt``) filtering is used so the P300 latency is not distorted.
2. **Epoching** — slicing the continuous record into fixed windows relative to
   each flash marker, with pre-stimulus baseline correction.
3. **Artifact rejection** — discarding epochs contaminated by ocular, muscular,
   or movement artifacts using a peak-to-peak amplitude criterion.

The data model throughout is:

* ``timestamps``: ``float64`` array, shape ``(N,)``, host ``perf_counter`` time.
* ``data``: ``float64`` array, shape ``(N, n_channels)``, microvolts.

All functions are pure (no global state) and fully typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from acquisition import StimulusMarker


# --------------------------------------------------------------------------- #
# Filter design + application
# --------------------------------------------------------------------------- #
def design_bandpass(
    low_hz: float, high_hz: float, fs: float, order: int = 4
) -> np.ndarray:
    """Design a Butterworth band-pass as second-order sections (SOS).

    Args:
        low_hz: High-pass cutoff (lower edge of the pass-band).
        high_hz: Low-pass cutoff (upper edge of the pass-band).
        fs: Sampling rate in Hz.
        order: Filter order per edge.

    Returns:
        SOS coefficient array suitable for :func:`scipy.signal.sosfiltfilt`.

    Raises:
        ValueError: If the requested band is invalid for the given ``fs``.
    """
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid band-pass: low={low_hz}Hz high={high_hz}Hz for fs={fs}Hz"
        )
    return butter(order, [low, high], btype="bandpass", output="sos")


def apply_bandpass(
    data: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass along the time axis.

    Args:
        data: Array of shape ``(N, n_channels)`` (or ``(N,)``).
        low_hz: Lower pass-band edge.
        high_hz: Upper pass-band edge.
        fs: Sampling rate in Hz.
        order: Filter order per edge.

    Returns:
        Filtered array of the same shape as ``data``.
    """
    if data.shape[0] == 0:
        return data.copy()
    sos = design_bandpass(low_hz, high_hz, fs, order)
    # axis=0 -> filter along time for each channel column independently.
    return sosfiltfilt(sos, data, axis=0)


def apply_notch(
    data: np.ndarray, freq_hz: float, fs: float, quality_factor: float = 30.0
) -> np.ndarray:
    """Apply a zero-phase IIR notch filter to remove mains interference.

    Args:
        data: Array of shape ``(N, n_channels)`` (or ``(N,)``).
        freq_hz: Notch centre frequency (50 or 60 Hz).
        fs: Sampling rate in Hz.
        quality_factor: ``Q = f0 / bandwidth``. Higher -> narrower notch.

    Returns:
        Filtered array of the same shape as ``data``.
    """
    if data.shape[0] == 0:
        return data.copy()
    # iirnotch returns transfer-function (b, a) coefficients.
    b, a = iirnotch(w0=freq_hz, Q=quality_factor, fs=fs)
    return filtfilt(b, a, data, axis=0)


def precondition(
    data: np.ndarray, fs: float, proc_cfg: dict
) -> np.ndarray:
    """Run the full spectral conditioning chain.

    ORDER OF OPERATIONS — as implemented: **band-pass, then notch.**

    This is the authoritative statement of the order for this codebase; any
    methodology text that asserts a different order contradicts the running
    system and one of the two must be corrected. The order is recorded here
    rather than only in prose so there is a single source of truth.

    Both stages are zero-phase (``sosfiltfilt`` / ``filtfilt``), so the order
    has no effect on phase or latency, and because both are LTI the two
    orderings differ only in numerical conditioning, not in the ideal response.

    On the apparent redundancy: the band-pass upper corner is 30 Hz at order 4,
    so a 50 Hz notch downstream of it is largely redundant *in the analysis
    path* — the mains component has already been attenuated. It is retained
    because (a) it costs nothing, (b) it guards the case where the band is
    widened in ``config.yaml`` without revisiting this function, and (c) the
    display path in :mod:`monitor` uses a causal filter with far less stop-band
    attenuation at 50 Hz, where the notch is not redundant at all.

    Args:
        data: Continuous array, shape ``(N, n_channels)``.
        fs: Sampling rate in Hz.
        proc_cfg: The ``processing`` section of ``config.yaml``.

    Returns:
        Conditioned array of the same shape.
    """
    bp = proc_cfg["bandpass"]
    nf = proc_cfg["notch"]
    out = apply_bandpass(data, bp["low_hz"], bp["high_hz"], fs, bp.get("order", 4))
    out = apply_notch(out, nf["freq_hz"], fs, nf.get("quality_factor", 30.0))
    return out


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #
@dataclass
class Epoch:
    """A single stimulus-locked epoch.

    Attributes:
        data: Array of shape ``(n_times, n_channels)``, baseline-corrected.
        times: Array of shape ``(n_times,)`` in seconds relative to flash onset.
        marker: The :class:`StimulusMarker` that anchored this epoch.
    """

    data: np.ndarray
    times: np.ndarray
    marker: StimulusMarker


def _nearest_index(timestamps: np.ndarray, t: float) -> int:
    """Return the index of the sample whose timestamp is closest to ``t``."""
    pos = int(np.searchsorted(timestamps, t))
    if pos <= 0:
        return 0
    if pos >= len(timestamps):
        return len(timestamps) - 1
    # Choose the closer of the two neighbours straddling ``t``.
    before = timestamps[pos - 1]
    after = timestamps[pos]
    return pos if (after - t) < (t - before) else pos - 1


def epoch_data(
    timestamps: np.ndarray,
    data: np.ndarray,
    markers: Sequence[StimulusMarker],
    fs: float,
    tmin_s: float,
    tmax_s: float,
    baseline: Tuple[float, float] = (-0.1, 0.0),
) -> List[Epoch]:
    """Slice continuous EEG into baseline-corrected epochs around each marker.

    Each epoch spans ``[tmin_s, tmax_s)`` relative to the marker's flash onset.
    A fixed number of samples ``n_times = round((tmax - tmin) * fs)`` is taken
    so all epochs are stackable. Markers whose full window is not contained in
    the buffer are skipped (e.g. flashes at the very tail before the post-roll).

    Baseline correction subtracts, per channel, the mean amplitude over the
    ``baseline`` window so every epoch starts from a common zero.

    Args:
        timestamps: Sample times, shape ``(N,)`` (assumed sorted ascending).
        data: Samples, shape ``(N, n_channels)``.
        markers: Stimulus markers to anchor epochs on.
        fs: Sampling rate in Hz.
        tmin_s: Window start relative to onset (negative = pre-stimulus).
        tmax_s: Window end relative to onset.
        baseline: ``(start_s, end_s)`` window for baseline correction.

    Returns:
        A list of :class:`Epoch`, one per usable marker, in marker order.
    """
    if len(timestamps) == 0 or data.shape[0] == 0:
        return []

    n_times = int(round((tmax_s - tmin_s) * fs))
    if n_times <= 0:
        raise ValueError("Epoch window has non-positive length")

    # Relative time axis shared by every epoch.
    times = tmin_s + np.arange(n_times) / fs

    # Baseline sample mask on the relative axis.
    b0, b1 = baseline
    base_mask = (times >= b0) & (times < b1)
    if not np.any(base_mask):
        # Degenerate baseline -> use the first sample only.
        base_mask = np.zeros(n_times, dtype=bool)
        base_mask[0] = True

    epochs: List[Epoch] = []
    n_total = len(timestamps)
    for marker in markers:
        onset_idx = _nearest_index(timestamps, marker.perf_time)
        # The sample offset of tmin relative to the onset sample.
        start = onset_idx + int(round(tmin_s * fs))
        stop = start + n_times
        if start < 0 or stop > n_total:
            # Window not fully captured -> skip rather than zero-pad.
            continue
        segment = data[start:stop, :].astype(np.float64, copy=True)
        # Baseline correction: subtract per-channel pre-stimulus mean.
        baseline_mean = segment[base_mask, :].mean(axis=0, keepdims=True)
        segment -= baseline_mean
        epochs.append(Epoch(data=segment, times=times, marker=marker))

    return epochs


# --------------------------------------------------------------------------- #
# Artifact rejection
# --------------------------------------------------------------------------- #
# The rejection criterion is *peak-to-peak amplitude within the epoch*, not
# absolute deviation from baseline. The distinction is not cosmetic: a slow
# electrode-drift ramp spanning 90 uV across the epoch never exceeds a +/-75 uV
# absolute-deviation bound once the epoch is baseline-corrected around its own
# mean, yet it is exactly the contamination that must be removed. Peak-to-peak
# catches it; absolute deviation does not.
DEFAULT_REJECTION_THRESHOLD_UV = 100.0
DEFAULT_MAX_SESSION_REJECTION_RATE = 0.30


@dataclass
class RejectionReport:
    """Outcome of applying the peak-to-peak artifact criterion to a set of epochs.

    Attributes:
        n_total: Number of epochs examined.
        n_kept: Number of epochs that passed the criterion.
        n_rejected: Number of epochs discarded.
        rejection_rate: ``n_rejected / n_total`` (0.0 for an empty input).
        keep_mask: Boolean array, shape ``(n_total,)``; ``True`` = kept.
        peak_to_peak_uv: Array of shape ``(n_total, n_channels)`` holding the
            per-epoch, per-channel peak-to-peak amplitude in microvolts.
        threshold_uv: The peak-to-peak threshold that was applied.
        max_session_rate: Rejection rate above which the session is invalid.
        session_valid: ``False`` when ``rejection_rate`` exceeds
            ``max_session_rate``, meaning the recording must be repeated
            rather than analysed.
    """

    n_total: int
    n_kept: int
    n_rejected: int
    rejection_rate: float
    keep_mask: np.ndarray
    peak_to_peak_uv: np.ndarray
    threshold_uv: float
    max_session_rate: float
    session_valid: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        state = "OK" if self.session_valid else "SESSION INVALID"
        return (
            f"rejected {self.n_rejected}/{self.n_total} epochs "
            f"({self.rejection_rate:.1%}) at {self.threshold_uv:.0f} uV p-p "
            f"[{state}]"
        )


def epoch_peak_to_peak(epoch: Epoch) -> np.ndarray:
    """Return the per-channel peak-to-peak amplitude of one epoch.

    Args:
        epoch: The epoch to measure.

    Returns:
        Array of shape ``(n_channels,)`` in microvolts: ``max - min`` computed
        along the time axis independently for each channel.
    """
    return np.ptp(np.asarray(epoch.data, dtype=np.float64), axis=0)


def reject_artifacts(
    epochs: Sequence[Epoch],
    threshold_uv: float = DEFAULT_REJECTION_THRESHOLD_UV,
    max_session_rate: float = DEFAULT_MAX_SESSION_REJECTION_RATE,
) -> Tuple[List[Epoch], RejectionReport]:
    """Drop epochs whose peak-to-peak amplitude exceeds ``threshold_uv``.

    An epoch is rejected when the peak-to-peak amplitude of **any** channel is
    strictly greater than ``threshold_uv``. An epoch sitting exactly on the
    threshold is kept, so the criterion is "reject if it exceeds", not "reject
    if it reaches".

    Args:
        epochs: Epochs to screen (already baseline-corrected).
        threshold_uv: Peak-to-peak rejection threshold in microvolts.
        max_session_rate: Rejection rate above which the whole session is
            declared invalid and must be repeated.

    Returns:
        ``(kept_epochs, report)``. ``kept_epochs`` preserves the input order.

    Raises:
        ValueError: If ``threshold_uv`` is not strictly positive.
    """
    if threshold_uv <= 0.0:
        raise ValueError(f"threshold_uv must be > 0, got {threshold_uv!r}")

    n_total = len(epochs)
    if n_total == 0:
        return [], RejectionReport(
            n_total=0,
            n_kept=0,
            n_rejected=0,
            rejection_rate=0.0,
            keep_mask=np.zeros(0, dtype=bool),
            peak_to_peak_uv=np.zeros((0, 0), dtype=np.float64),
            threshold_uv=float(threshold_uv),
            max_session_rate=float(max_session_rate),
            session_valid=True,
        )

    ptp = np.vstack([epoch_peak_to_peak(ep) for ep in epochs])
    # Reject if ANY channel exceeds the threshold.
    keep_mask = ~np.any(ptp > threshold_uv, axis=1)

    kept = [ep for ep, keep in zip(epochs, keep_mask) if keep]
    n_kept = int(keep_mask.sum())
    n_rejected = n_total - n_kept
    rate = n_rejected / float(n_total)

    report = RejectionReport(
        n_total=n_total,
        n_kept=n_kept,
        n_rejected=n_rejected,
        rejection_rate=float(rate),
        keep_mask=keep_mask,
        peak_to_peak_uv=ptp,
        threshold_uv=float(threshold_uv),
        max_session_rate=float(max_session_rate),
        session_valid=bool(rate <= max_session_rate),
    )
    return kept, report


def reject_artifacts_from_config(
    epochs: Sequence[Epoch], proc_cfg: Optional[dict]
) -> Tuple[List[Epoch], RejectionReport]:
    """Apply :func:`reject_artifacts` using the ``processing`` config section.

    Reads ``processing.artifact_rejection.threshold_uv`` and
    ``processing.artifact_rejection.max_session_rate``. Both fall back to the
    module defaults (100 uV, 0.30) when the section or a key is absent, so an
    older ``config.yaml`` still yields the locked criterion rather than
    silently disabling rejection.

    Args:
        epochs: Epochs to screen.
        proc_cfg: The ``processing`` section of ``config.yaml`` (may be ``None``).

    Returns:
        ``(kept_epochs, report)``.
    """
    ar = (proc_cfg or {}).get("artifact_rejection") or {}
    return reject_artifacts(
        epochs,
        threshold_uv=float(
            ar.get("threshold_uv", DEFAULT_REJECTION_THRESHOLD_UV)
        ),
        max_session_rate=float(
            ar.get("max_session_rate", DEFAULT_MAX_SESSION_REJECTION_RATE)
        ),
    )
