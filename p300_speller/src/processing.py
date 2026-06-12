"""processing.py — filtering and epoching for the P300 Speller.

This module turns the raw, continuous, timestamped EEG stream produced by
:mod:`acquisition` into clean, baseline-corrected epochs aligned to stimulus
flashes. It contains two responsibilities:

1. **Spectral conditioning** — a zero-phase Butterworth band-pass plus an IIR
   notch (mains hum), implemented with :mod:`scipy.signal`. Zero-phase
   (``filtfilt``) filtering is used so the P300 latency is not distorted.
2. **Epoching** — slicing the continuous record into fixed windows relative to
   each flash marker, with pre-stimulus baseline correction.

The data model throughout is:

* ``timestamps``: ``float64`` array, shape ``(N,)``, host ``perf_counter`` time.
* ``data``: ``float64`` array, shape ``(N, n_channels)``, microvolts.

All functions are pure (no global state) and fully typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
    """Run the full spectral conditioning chain (band-pass then notch).

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
