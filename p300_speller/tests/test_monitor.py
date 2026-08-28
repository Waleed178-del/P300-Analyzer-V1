"""Unit tests for the operator monitor (:mod:`monitor`).

These pin down the property that justifies having two filters in the codebase:
the display filter is causal and therefore delays the waveform, while the
analysis filter is zero-phase and does not. They also check that the monitor
tolerates the messy realities of a fire-and-forget UDP feed — overlapping
frames, malformed frames, and no sender at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from monitor import (
    CausalDisplayFilter,
    MonitorSubscriber,
    render,
)
from processing import apply_bandpass

FS = 250.0
_PORT = 9781        # not the default, so a real monitor never collides


# --------------------------------------------------------------------------- #
# Causal display filter
# --------------------------------------------------------------------------- #
def test_display_filter_removes_dc_and_keeps_band() -> None:
    t = np.arange(1000) / FS
    sig = np.tile((20.0 * np.sin(2 * np.pi * 10.0 * t) + 60.0)[:, None], (1, 3))
    out = CausalDisplayFilter(0.5, 30.0, FS, 3).process(sig)
    # Skip the settling transient before measuring.
    tail = out[400:, :]
    assert abs(tail.mean()) < 1.0                       # 60 uV DC gone
    assert np.ptp(tail[:, 0]) == pytest.approx(40.0, rel=0.1)


def test_display_filter_state_is_continuous_across_blocks() -> None:
    """Filtering in two blocks must equal filtering in one, or the trace would
    show a discontinuity at every datagram seam."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 10.0, size=(600, 2))

    whole = CausalDisplayFilter(0.5, 30.0, FS, 2).process(sig)

    split = CausalDisplayFilter(0.5, 30.0, FS, 2)
    part = np.vstack([split.process(sig[:300]), split.process(sig[300:])])

    np.testing.assert_allclose(part, whole, atol=1e-9)


def test_causal_display_filter_delays_but_zero_phase_does_not() -> None:
    """The reason the display filter must never be used for analysis.

    A causal IIR filter has non-zero group delay and shifts the peak later in
    time. Forward-backward filtering conjugates the phase response, so the net
    group delay is identically zero and the latency is preserved.
    """
    n, onset = 1000, 300
    pulse = np.zeros((n, 1))
    pulse[onset, 0] = 100.0

    causal = CausalDisplayFilter(0.5, 30.0, FS, 1).process(pulse)
    zero_phase = apply_bandpass(pulse, 0.5, 30.0, FS, order=4)

    causal_peak = int(np.argmax(np.abs(causal[:, 0])))
    zero_phase_peak = int(np.argmax(np.abs(zero_phase[:, 0])))

    assert causal_peak > onset                       # delayed by the filter
    assert abs(zero_phase_peak - onset) <= 1         # latency preserved
    assert causal_peak > zero_phase_peak


def test_display_filter_rejects_invalid_band() -> None:
    with pytest.raises(ValueError):
        CausalDisplayFilter(30.0, 0.5, FS, 1)


def test_display_filter_handles_empty_block() -> None:
    empty = np.zeros((0, 3))
    assert CausalDisplayFilter(0.5, 30.0, FS, 3).process(empty).shape == (0, 3)


# --------------------------------------------------------------------------- #
# Subscriber
# --------------------------------------------------------------------------- #
@pytest.fixture()
def subscriber():
    sub = MonitorSubscriber(port=_PORT, window_s=2.0)
    yield sub
    sub.close()


def _signal_frame(t0: float, n: int, n_channels: int = 3) -> dict:
    t = t0 + np.arange(n) / FS
    data = np.tile((20.0 * np.sin(2 * np.pi * 10.0 * t))[:, None], (1, n_channels))
    return {
        "type": "signal",
        "fs": FS,
        "channels": ["Fz", "Cz", "Pz"][:n_channels],
        "t0": float(t0),
        "data": data.tolist(),
    }


def test_overlapping_frames_are_deduplicated(subscriber) -> None:
    """The pipeline republishes overlapping slices of its ring buffer; feeding
    the repeats into a stateful filter would corrupt it."""
    subscriber._apply(_signal_frame(0.0, 250))
    subscriber._apply(_signal_frame(0.0, 500))       # first 250 are repeats
    assert subscriber.state.n_samples_seen == 500


def test_quality_frame_updates_alarm_state(subscriber) -> None:
    subscriber._apply(
        {
            "type": "quality",
            "n_total": 96,
            "n_kept": 60,
            "n_rejected": 36,
            "rejection_rate": 0.375,
            "threshold_uv": 100.0,
            "session_valid": False,
        }
    )
    st = subscriber.state
    assert st.rejection_rate == pytest.approx(0.375)
    assert st.session_valid is False
    assert "SESSION INVALID" in render(st)


def test_event_frame_is_recorded(subscriber) -> None:
    subscriber._apply({"type": "event", "event": "character_cue", "text": "A"})
    assert "character_cue A" == subscriber.state.last_event


def test_malformed_frames_are_ignored(subscriber) -> None:
    for bad in (
        {"type": "signal"},                       # no data
        {"type": "signal", "data": []},           # empty
        {"type": "signal", "data": [1, 2, 3]},    # 1-D, not (n, ch)
        {"type": "unknown"},
        {},
    ):
        subscriber._apply(bad)
    assert subscriber.state.n_samples_seen == 0


def test_poll_times_out_without_a_sender(subscriber) -> None:
    assert subscriber.poll() is False


def test_render_before_any_data_is_informative(subscriber) -> None:
    out = render(subscriber.state)
    assert "waiting for signal frames" in out


def test_alarm_renders_even_before_the_first_signal_frame(subscriber) -> None:
    """A quality frame can arrive first. The alarm must not be swallowed by the
    'waiting for data' screen."""
    subscriber._apply(
        {"type": "quality", "n_total": 10, "n_rejected": 9,
         "rejection_rate": 0.9, "threshold_uv": 100.0, "session_valid": False}
    )
    assert subscriber.state.channels == []
    assert "SESSION INVALID" in render(subscriber.state)


def test_render_flags_channels_over_threshold(subscriber) -> None:
    frame = _signal_frame(0.0, 600)
    data = np.asarray(frame["data"])
    data[:, 1] *= 20.0                             # drive Cz far over the limit
    frame["data"] = data.tolist()
    subscriber._apply(frame)
    assert "OVER" in render(subscriber.state)
