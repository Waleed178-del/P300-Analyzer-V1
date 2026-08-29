"""Unit tests for the operator monitor (:mod:`monitor`).

These pin down the property that justifies having two filters in the codebase:
the display filter is causal and therefore delays the waveform, while the
analysis filter is zero-phase and does not. They also cover the binary wire
protocol, the independent rejection accounting the panel performs, and the
requirement that a fire-and-forget publisher never disturbs a recording.

No test here opens a window; ``draw()`` needs pygame and a display, so the
rendering layer is exercised only through the state it reads.
"""

from __future__ import annotations

import socket
import struct

import numpy as np
import pytest

from monitor import (
    MAGIC_EPOCH,
    MAGIC_STREAM,
    REJECT_UV,
    SCALE_UV_PER_COUNT,
    CausalDisplayFilter,
    MonitorPublisher,
    MonitorState,
    drain,
    make_socket,
)
from processing import apply_bandpass

FS = 250.0
_PORT = 9787        # not the default, so a real monitor never collides


def _state(n_ch: int = 3, window_s: float = 2.0) -> MonitorState:
    return MonitorState(n_ch, FS, window_s, ["Fz", "Cz", "Pz"][:n_ch])


# --------------------------------------------------------------------------- #
# Causal display filter
# --------------------------------------------------------------------------- #
def test_display_filter_removes_dc_and_keeps_band() -> None:
    t = np.arange(1500) / FS
    sig = np.tile(20.0 * np.sin(2 * np.pi * 10.0 * t) + 300.0, (3, 1))
    out = CausalDisplayFilter(FS, 3)(sig)
    tail = out[:, 600:]                      # skip the settling transient
    assert abs(tail.mean()) < 2.0            # 300 uV DC removed
    assert np.ptp(tail[0]) == pytest.approx(40.0, rel=0.15)


def test_display_filter_attenuates_mains() -> None:
    """The display chain includes a 50 Hz notch; unlike the analysis path it has
    no steep 30 Hz low-pass in front of it, so the notch is not redundant."""
    t = np.arange(2000) / FS
    mains = np.tile(20.0 * np.sin(2 * np.pi * 50.0 * t), (1, 1))
    out = CausalDisplayFilter(FS, 1)(mains)
    assert out[0, 800:].std() < 0.25 * mains[0, 800:].std()


def test_display_filter_state_is_continuous_across_chunks() -> None:
    """Filtering in two chunks must equal filtering in one, or the trace shows a
    discontinuity at every datagram seam."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 10.0, size=(2, 600))

    whole = CausalDisplayFilter(FS, 2)(sig)

    split = CausalDisplayFilter(FS, 2)
    part = np.hstack([split(sig[:, :300]), split(sig[:, 300:])])

    np.testing.assert_allclose(part, whole, atol=1e-9)


def test_causal_display_filter_delays_but_zero_phase_does_not() -> None:
    """The reason the display filter must never be used for analysis.

    A causal IIR filter has non-zero group delay and shifts the peak later in
    time. Forward-backward filtering conjugates the phase response, so the net
    group delay is identically zero and latency is preserved.
    """
    n, onset = 1200, 300
    pulse = np.zeros((1, n))
    pulse[0, onset] = 100.0

    causal = CausalDisplayFilter(FS, 1)(pulse)
    zero_phase = apply_bandpass(pulse.T, 0.5, 30.0, FS, order=4)[:, 0]

    causal_peak = int(np.argmax(np.abs(causal[0])))
    zero_phase_peak = int(np.argmax(np.abs(zero_phase)))

    assert causal_peak > onset                   # delayed by the filter
    assert abs(zero_phase_peak - onset) <= 1     # latency preserved
    assert causal_peak > zero_phase_peak


# --------------------------------------------------------------------------- #
# Rolling state
# --------------------------------------------------------------------------- #
def test_push_scrolls_the_window_and_records_measured_rate() -> None:
    st = _state(window_s=1.0)
    width = st.raw.shape[1]
    st.push(np.ones((3, 100)) * 5.0, fs_measured=249.4)
    assert st.raw.shape[1] == width               # window length is fixed
    assert np.allclose(st.raw[:, -100:], 5.0)
    assert np.allclose(st.raw[:, : width - 100], 0.0)
    assert st.fs_measured == pytest.approx(249.4)


def test_push_ignores_empty_chunks() -> None:
    st = _state()
    st.push(np.zeros((3, 0)), FS)
    assert st.fs_measured == 0.0


def test_push_clamps_a_chunk_longer_than_the_window() -> None:
    st = _state(window_s=1.0)
    width = st.raw.shape[1]
    st.push(np.arange(3 * 5000, dtype=float).reshape(3, 5000), FS)
    assert st.raw.shape[1] == width


# --------------------------------------------------------------------------- #
# Independent rejection accounting on the panel
# --------------------------------------------------------------------------- #
def _epoch(ptp_uv: float, n_ch: int = 3, n_times: int = 225) -> np.ndarray:
    ramp = np.linspace(-ptp_uv / 2.0, ptp_uv / 2.0, n_times)
    return np.tile(ramp, (n_ch, 1))


def test_panel_applies_the_same_100uv_criterion() -> None:
    st = _state()
    st.push_epoch(_epoch(40.0), label=1)
    st.push_epoch(_epoch(400.0), label=1)
    assert st.epochs_seen == 2
    assert st.epochs_rejected == 1
    assert st.rejection_pct == pytest.approx(50.0)


def test_rejected_epochs_never_enter_the_erp_average() -> None:
    st = _state()
    st.push_epoch(_epoch(400.0), label=1)         # rejected
    assert st.erp_mean(1) is None
    st.push_epoch(_epoch(40.0), label=1)          # kept
    assert st.erp_mean(1) is not None
    assert st.erp_count[1] == 1


def test_erp_averages_are_kept_separate_per_label() -> None:
    st = _state()
    for _ in range(3):
        st.push_epoch(_epoch(40.0), label=1)
    for _ in range(5):
        st.push_epoch(_epoch(20.0), label=0)
    assert st.erp_count[1] == 3
    assert st.erp_count[0] == 5
    assert np.ptp(st.erp_mean(1)) > np.ptp(st.erp_mean(0))


def test_rejection_pct_is_zero_before_any_epoch() -> None:
    assert _state().rejection_pct == 0.0


def test_reset_counters_clears_everything() -> None:
    st = _state()
    st.push_epoch(_epoch(40.0), label=1)
    st.push_epoch(_epoch(400.0), label=0)
    st.reset_counters()
    assert st.epochs_seen == 0
    assert st.epochs_rejected == 0
    assert st.erp_mean(1) is None


def test_saturation_flag_tracks_the_adc_ceiling() -> None:
    st = _state(window_s=1.0)
    st.push(np.zeros((3, 50)), FS)
    assert not st.saturated(0)
    full_scale = 32767 * SCALE_UV_PER_COUNT
    st.push(np.full((3, 50), full_scale), FS)
    assert st.saturated(0)


def test_dc_offset_is_reported_in_microvolts() -> None:
    st = _state(window_s=1.0)
    width = st.raw.shape[1]
    st.push(np.full((3, width), 1000.0), FS)
    assert st.dc_offset_uv(0) == pytest.approx(1000.0)


def test_locked_constants_match_the_project() -> None:
    """The panel's constants must not drift from config.yaml / the gain divisor."""
    assert REJECT_UV == 100.0
    # 4.096 V / 32768 counts / 949 gain, in microvolts.
    assert SCALE_UV_PER_COUNT == pytest.approx(
        (4.096 / 32768.0) * 1e6 / 949.0, rel=1e-4
    )


# --------------------------------------------------------------------------- #
# Wire protocol
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sock():
    s = make_socket(_PORT)
    yield s
    s.close()


def test_publisher_to_state_roundtrip(sock) -> None:
    pub = MonitorPublisher(port=_PORT, enabled=True)
    st = _state(window_s=2.0)

    chunk = np.tile(np.linspace(-10.0, 10.0, 120), (3, 1))
    pub.publish_chunk(chunk, fs_measured=249.8)
    pub.publish_epoch(_epoch(40.0), label=1)
    pub.publish_epoch(_epoch(400.0), label=0)

    drain(sock, st)
    pub.close()

    assert st.fs_measured == pytest.approx(249.8)
    np.testing.assert_allclose(st.raw[:, -120:], chunk, rtol=1e-5)
    assert st.epochs_seen == 2
    assert st.epochs_rejected == 1


def test_drain_returns_quietly_when_nothing_is_queued(sock) -> None:
    st = _state()
    drain(sock, st)
    assert st.epochs_seen == 0


def test_malformed_datagrams_are_dropped(sock) -> None:
    st = _state()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for payload in (
        MAGIC_STREAM + b"\x00\x01",                        # truncated header
        MAGIC_EPOCH + struct.pack("<BII", 1, 3, 999) + b"\x00" * 8,  # short body
        b"X" + b"garbage",                                 # unknown tag
        b"",                                               # empty
    ):
        sender.sendto(payload, ("127.0.0.1", _PORT))
    drain(sock, st)
    sender.close()
    assert st.epochs_seen == 0
    assert st.fs_measured == 0.0


def test_oversized_payload_is_dropped_not_raised() -> None:
    """A frame beyond the UDP limit must be discarded silently."""
    pub = MonitorPublisher(port=_PORT, enabled=True)
    pub.publish_chunk(np.zeros((3, 100000)), FS)     # far over MAX_DATAGRAM
    pub.close()


def test_publisher_never_raises_without_a_listener() -> None:
    """Nothing is bound to this port; sends must still be harmless."""
    pub = MonitorPublisher(port=9788, enabled=True)
    pub.publish_chunk(np.zeros((3, 50)), FS)
    pub.publish_epoch(_epoch(40.0), label=1)
    pub.close()
    # Post-close publishing is a no-op rather than an error.
    pub.publish_chunk(np.zeros((3, 50)), FS)


def test_disabled_publisher_sends_nothing(sock) -> None:
    pub = MonitorPublisher(port=_PORT, enabled=False)
    pub.publish_chunk(np.ones((3, 50)) * 99.0, FS)
    pub.publish_epoch(_epoch(40.0), label=1)
    st = _state()
    drain(sock, st)
    assert st.epochs_seen == 0
    assert np.allclose(st.raw, 0.0)
