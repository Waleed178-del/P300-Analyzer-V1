"""Unit tests for :mod:`processing` (filtering and epoching).

These establish a behavioural baseline for the spectral-conditioning and
epoching stages: that the band-pass removes DC and out-of-band energy while
preserving the P300 band, that the notch suppresses mains hum, and that epoching
windows the stream correctly with valid baseline correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from acquisition import StimulusMarker
from processing import (
    apply_bandpass,
    apply_notch,
    design_bandpass,
    epoch_data,
)

FS = 250.0


def _sine(freq: float, n: int, fs: float = FS, amp: float = 1.0) -> np.ndarray:
    """Return ``n`` samples of a unit-ish sine at ``freq`` Hz."""
    t = np.arange(n) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


# --------------------------------------------------------------------------- #
# Band-pass
# --------------------------------------------------------------------------- #
def test_bandpass_removes_dc_and_keeps_midband() -> None:
    n = 1500
    signal = (5.0 + _sine(10.0, n))[:, None]  # 5 uV DC offset + 10 Hz tone
    out = apply_bandpass(signal, 0.5, 30.0, FS, order=4)
    assert abs(out.mean()) < 0.05            # DC removed
    assert out[:, 0].std() > 0.6             # 10 Hz tone largely preserved


def test_bandpass_attenuates_out_of_band() -> None:
    n = 2000
    in_band = apply_bandpass(_sine(10.0, n)[:, None], 0.5, 30.0, FS, 4)[:, 0]
    out_band = apply_bandpass(_sine(60.0, n)[:, None], 0.5, 30.0, FS, 4)[:, 0]
    assert out_band.std() < 0.3 * in_band.std()


def test_design_bandpass_rejects_inverted_band() -> None:
    with pytest.raises(ValueError):
        design_bandpass(30.0, 0.5, FS)        # low >= high


def test_bandpass_preserves_shape_multichannel() -> None:
    x = np.random.default_rng(0).normal(size=(500, 3))
    assert apply_bandpass(x, 0.5, 30.0, FS, 4).shape == (500, 3)


# --------------------------------------------------------------------------- #
# Notch
# --------------------------------------------------------------------------- #
def test_notch_suppresses_mains_preserves_signal() -> None:
    n = 2000
    mains = apply_notch(_sine(50.0, n)[:, None], 50.0, FS, 30.0)[:, 0]
    tone = apply_notch(_sine(10.0, n)[:, None], 50.0, FS, 30.0)[:, 0]
    assert mains.std() < 0.2                  # 50 Hz strongly attenuated
    assert tone.std() > 0.6                   # 10 Hz preserved


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #
def _linear_timestamps(n: int, fs: float = FS) -> np.ndarray:
    return np.arange(n) / fs


def test_epoch_extraction_shape_and_count() -> None:
    n = 1000
    ts = _linear_timestamps(n)
    data = np.zeros((n, 3))
    markers = [
        StimulusMarker(code=1, kind="row", index=0, perf_time=ts[200]),
        StimulusMarker(code=2, kind="row", index=1, perf_time=ts[500]),
    ]
    epochs = epoch_data(ts, data, markers, FS, -0.1, 0.8, (-0.1, 0.0))
    expected_len = int(round((0.8 - (-0.1)) * FS))
    assert len(epochs) == 2
    assert epochs[0].data.shape == (expected_len, 3)
    assert epochs[0].times[0] == pytest.approx(-0.1, abs=1e-9)


def test_epoch_baseline_correction_zeros_prestimulus_mean() -> None:
    n = 1000
    ts = _linear_timestamps(n)
    data = np.full((n, 2), 7.0)               # constant 7 uV offset everywhere
    markers = [StimulusMarker(code=1, kind="row", index=0, perf_time=ts[400])]
    epochs = epoch_data(ts, data, markers, FS, -0.1, 0.8, (-0.1, 0.0))
    base = (epochs[0].times >= -0.1) & (epochs[0].times < 0.0)
    assert abs(epochs[0].data[base].mean()) < 1e-9


def test_epoch_skips_window_not_fully_buffered() -> None:
    n = 1000
    ts = _linear_timestamps(n)
    data = np.zeros((n, 1))
    # Marker two samples before the end: the +800 ms tail is unavailable.
    markers = [StimulusMarker(code=1, kind="row", index=0, perf_time=ts[-2])]
    assert epoch_data(ts, data, markers, FS, -0.1, 0.8, (-0.1, 0.0)) == []


def test_epoch_recovers_injected_bump_at_expected_latency() -> None:
    """An epoch must place a post-stimulus bump at the correct relative time."""
    n = 2000
    ts = _linear_timestamps(n)
    data = np.zeros((n, 1))
    onset = 800
    # Inject a sharp positive bump 300 ms (75 samples) after onset.
    data[onset + 75, 0] = 10.0
    markers = [StimulusMarker(code=1, kind="row", index=0, perf_time=ts[onset])]
    epochs = epoch_data(ts, data, markers, FS, -0.1, 0.8, (-0.1, 0.0))
    peak_time = epochs[0].times[int(np.argmax(epochs[0].data[:, 0]))]
    assert peak_time == pytest.approx(0.3, abs=1.0 / FS)
