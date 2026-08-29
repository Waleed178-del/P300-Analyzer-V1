"""Integration tests for the shared data path in :mod:`main_pipeline`.

Defining ``reject_artifacts`` is not the same as performing artifact rejection.
These tests exercise :meth:`P300Pipeline._collect_epochs` — the single path used
by both calibration and free spelling — and assert that a contaminated epoch is
actually removed there, that the session-level rejection tally accumulates, and
that the >30% session alarm fires. Without them the methodology could describe a
procedure the running system never applies.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pytest
import yaml

from acquisition import BaseEEGSource, StimulusMarker
from main_pipeline import P300Pipeline
from monitor import MonitorState, drain, make_socket

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "config.yaml",
)
FS = 250.0


# --------------------------------------------------------------------------- #
# A deterministic, offline stand-in for a real EEG source
# --------------------------------------------------------------------------- #
class _StubSource(BaseEEGSource):
    """Replays a fixed buffer; never starts a thread."""

    def __init__(self, timestamps: np.ndarray, data: np.ndarray) -> None:
        super().__init__(FS, ["Fz", "Cz", "Pz"], ring_buffer_s=30.0)
        self._fixed_ts = timestamps
        self._fixed_data = data

    def get_buffer(self):
        return self._fixed_ts.copy(), self._fixed_data.copy()

    def _acquire_loop(self) -> None:  # pragma: no cover - never started
        raise AssertionError("stub source must not be started")


@pytest.fixture()
def config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["speller"]["headless"] = True
    return cfg


def _buffer(n_seconds: float = 12.0):
    """A clean, low-amplitude continuous recording."""
    n = int(n_seconds * FS)
    ts = np.arange(n) / FS
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 2.0, size=(n, 3))
    return ts, data


def _contaminate(data: np.ndarray, centre_s: float, amplitude_uv: float) -> None:
    """Inject a large deflection around ``centre_s`` (in place)."""
    idx = int(centre_s * FS)
    half = int(0.15 * FS)
    lo, hi = max(0, idx - half), min(data.shape[0], idx + half)
    data[lo:hi, 0] += np.linspace(0.0, amplitude_uv, hi - lo)


def _markers(times_s: List[float]) -> List[StimulusMarker]:
    return [
        StimulusMarker(code=i + 1, kind="row", index=i % 6, perf_time=t)
        for i, t in enumerate(times_s)
    ]


def _pipeline(config: dict, ts, data) -> P300Pipeline:
    return P300Pipeline(config, source=_StubSource(ts, data))


# --------------------------------------------------------------------------- #
# Rejection is live inside the shared path
# --------------------------------------------------------------------------- #
def test_collect_epochs_applies_rejection(config: dict) -> None:
    ts, data = _buffer()
    onsets = [2.0, 4.0, 6.0, 8.0]
    _contaminate(data, 4.2, 900.0)          # contaminates the epoch at t=4.0
    pipe = _pipeline(config, ts, data)

    epochs, report = pipe._collect_epochs(_markers(onsets))

    assert report.n_total == 4
    assert report.n_rejected == 1
    assert len(epochs) == 3
    # The surviving epochs are the clean ones, in order.
    assert [round(ep.marker.perf_time, 3) for ep in epochs] == [2.0, 6.0, 8.0]


def test_collect_epochs_keeps_clean_data_intact(config: dict) -> None:
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    epochs, report = pipe._collect_epochs(_markers([2.0, 4.0, 6.0]))
    assert report.n_rejected == 0
    assert len(epochs) == 3
    assert report.session_valid


def test_rejection_uses_the_configured_threshold(config: dict) -> None:
    """The threshold actually applied comes from config.yaml, not a literal."""
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    _, report = pipe._collect_epochs(_markers([2.0]))
    assert report.threshold_uv == 100.0
    assert report.max_session_rate == 0.30


# --------------------------------------------------------------------------- #
# Session-level tally and alarm
# --------------------------------------------------------------------------- #
def test_session_tally_accumulates_across_characters(config: dict) -> None:
    ts, data = _buffer()
    _contaminate(data, 4.2, 900.0)
    pipe = _pipeline(config, ts, data)

    pipe._collect_epochs(_markers([2.0, 4.0]))
    pipe._collect_epochs(_markers([6.0, 8.0]))

    assert pipe.session_epochs_seen == 4
    assert pipe.session_epochs_rejected == 1
    assert pipe.session_rejection_rate() == pytest.approx(0.25)


def test_session_alarm_fires_above_thirty_percent(config: dict) -> None:
    ts, data = _buffer()
    for centre in (2.2, 4.2, 6.2):
        _contaminate(data, centre, 900.0)
    pipe = _pipeline(config, ts, data)

    events: List[Dict[str, object]] = []
    pipe.event_listener = lambda kind, payload: events.append(
        {"kind": kind, **payload}
    )

    pipe._collect_epochs(_markers([2.0, 4.0, 6.0, 8.0]))   # 3 of 4 rejected
    assert pipe.session_rejection_rate() == pytest.approx(0.75)

    assert pipe._check_session_quality() is False
    kinds = [e["kind"] for e in events]
    assert "session_invalid" in kinds
    assert "epochs_rejected" in kinds


def test_session_valid_when_within_budget(config: dict) -> None:
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    pipe._collect_epochs(_markers([2.0, 4.0, 6.0, 8.0]))
    assert pipe._check_session_quality() is True


def test_rejection_rate_is_zero_before_any_epochs(config: dict) -> None:
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    assert pipe.session_rejection_rate() == 0.0
    assert pipe._check_session_quality() is True


def test_empty_buffer_returns_empty_report(config: dict) -> None:
    pipe = _pipeline(config, np.empty(0), np.empty((0, 3)))
    epochs, report = pipe._collect_epochs(_markers([1.0]))
    assert epochs == []
    assert report.n_total == 0
    assert report.session_valid


# --------------------------------------------------------------------------- #
# Operator monitor feed
# --------------------------------------------------------------------------- #
_MON_PORT = 9789


def test_monitor_disabled_by_default(config: dict) -> None:
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    assert pipe.monitor.enabled is False
    # A full collect with the monitor off must not raise.
    pipe._collect_epochs(_markers([2.0, 4.0]), target_rc=(0, 0))


def test_pipeline_publishes_raw_signal_and_labelled_epochs(config: dict) -> None:
    """The end-to-end feed: the pipeline's publisher must speak the protocol the
    monitor's receiver actually parses."""
    config["monitor"] = {"enabled": True, "host": "127.0.0.1", "port": _MON_PORT}
    ts, data = _buffer()
    _contaminate(data, 4.2, 900.0)

    sock = make_socket(_MON_PORT)
    try:
        pipe = _pipeline(config, ts, data)
        assert pipe.monitor.enabled is True
        pipe._collect_epochs(_markers([2.0, 4.0, 6.0, 8.0]), target_rc=(0, 0))

        state = MonitorState(3, FS, 2.0, ["Fz", "Cz", "Pz"])
        drain(sock, state)
    finally:
        sock.close()

    # The raw chunk arrived, with a measured (not nominal) rate.
    assert state.fs_measured == pytest.approx(FS, rel=1e-6)
    assert not np.allclose(state.raw, 0.0)

    # All four epochs were published, including the contaminated one, so the
    # panel's own rejection counter is meaningful.
    assert state.epochs_seen == 4
    assert state.epochs_rejected == 1
    # Marker 0 is a row-0 flash, so exactly one epoch is a target.
    assert state.erp_count.get(1, 0) == 1


def test_spelling_publishes_no_epochs(config: dict) -> None:
    """During free spelling the label is unknown, so epochs must not be pushed
    into the ERP average as guesses."""
    config["monitor"] = {"enabled": True, "host": "127.0.0.1", "port": _MON_PORT}
    ts, data = _buffer()

    sock = make_socket(_MON_PORT)
    try:
        pipe = _pipeline(config, ts, data)
        pipe._collect_epochs(_markers([2.0, 4.0]))      # no target_rc
        state = MonitorState(3, FS, 2.0, ["Fz", "Cz", "Pz"])
        drain(sock, state)
    finally:
        sock.close()

    assert not np.allclose(state.raw, 0.0)   # raw stream still flows
    assert state.epochs_seen == 0            # but no labelled epochs


def test_measured_rate_reflects_the_actual_timestamps(config: dict) -> None:
    """A front-end running slow must be visible to the operator, so the measured
    rate is derived from timestamps rather than echoed from config."""
    n = 2000
    slow_ts = np.arange(n) / 200.0            # 200 Hz against a 250 Hz nominal
    data = np.random.default_rng(0).normal(0.0, 2.0, size=(n, 3))
    pipe = _pipeline(config, slow_ts, data)
    assert pipe._measured_fs(slow_ts) == pytest.approx(200.0, rel=1e-6)
    assert pipe._measured_fs(np.empty(0)) == pytest.approx(FS)
    assert pipe._measured_fs(np.zeros(5)) == pytest.approx(FS)
