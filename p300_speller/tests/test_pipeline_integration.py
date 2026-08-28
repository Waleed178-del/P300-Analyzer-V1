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
from main_pipeline import MonitorPublisher, P300Pipeline

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
# Monitor publisher must never disturb a recording
# --------------------------------------------------------------------------- #
def test_monitor_disabled_by_default(config: dict) -> None:
    ts, data = _buffer()
    pipe = _pipeline(config, ts, data)
    assert pipe.monitor.enabled is False
    # Publishing while disabled is a silent no-op, not an error.
    pipe.monitor.publish_event("noop")
    pipe.monitor.publish_signal(ts, data, FS, ["Fz", "Cz", "Pz"])


def test_monitor_publish_never_raises_without_a_listener() -> None:
    """Nothing is bound to the port; UDP sends must still be harmless."""
    pub = MonitorPublisher(port=9751, enabled=True)
    ts, data = _buffer(2.0)
    pub.publish_signal(ts, data, FS, ["Fz", "Cz", "Pz"])
    pub.publish_event("character_cue", "A")
    pub.close()
    # Closing twice is safe, and post-close publishing is a no-op.
    pub.close()
    pub.publish_event("after_close")
