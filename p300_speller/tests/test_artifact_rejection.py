"""Unit tests for the peak-to-peak artifact-rejection stage in :mod:`processing`.

The criterion under test is deliberately *peak-to-peak within the epoch*, not
absolute deviation from baseline. Several tests below exist specifically to pin
that distinction down, because the two criteria disagree on exactly the
contamination that matters most: slow electrode drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from acquisition import StimulusMarker
from processing import (
    DEFAULT_MAX_SESSION_REJECTION_RATE,
    DEFAULT_REJECTION_THRESHOLD_UV,
    Epoch,
    RejectionReport,
    epoch_peak_to_peak,
    reject_artifacts,
    reject_artifacts_from_config,
)

FS = 250.0
N_TIMES = 225           # 0.9 s window at 250 Hz
N_CHANNELS = 3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _marker(index: int = 0) -> StimulusMarker:
    return StimulusMarker(code=index + 1, kind="row", index=index, perf_time=0.0)


def _epoch(data: np.ndarray, index: int = 0) -> Epoch:
    """Wrap a ``(n_times, n_channels)`` array in an :class:`Epoch`."""
    times = -0.1 + np.arange(data.shape[0]) / FS
    return Epoch(data=np.asarray(data, dtype=np.float64), times=times,
                 marker=_marker(index))


def _flat_epoch(amplitude_uv: float = 5.0, n_channels: int = N_CHANNELS) -> Epoch:
    """A benign epoch: a small sinusoid with a known peak-to-peak amplitude."""
    t = np.arange(N_TIMES) / FS
    wave = (amplitude_uv / 2.0) * np.sin(2.0 * np.pi * 5.0 * t)
    return _epoch(np.tile(wave[:, None], (1, n_channels)))


def _drift_epoch(span_uv: float, n_channels: int = N_CHANNELS) -> Epoch:
    """A slow monotonic drift ramp spanning ``span_uv`` peak-to-peak.

    Centred on zero, so its maximum *absolute deviation* is only half the
    peak-to-peak span. This is the case that separates the two criteria.
    """
    ramp = np.linspace(-span_uv / 2.0, span_uv / 2.0, N_TIMES)
    return _epoch(np.tile(ramp[:, None], (1, n_channels)))


# --------------------------------------------------------------------------- #
# 1-3. Basic behaviour
# --------------------------------------------------------------------------- #
def test_clean_epochs_are_all_kept() -> None:
    epochs = [_flat_epoch(10.0) for _ in range(5)]
    kept, report = reject_artifacts(epochs, threshold_uv=100.0)
    assert len(kept) == 5
    assert report.n_rejected == 0
    assert report.rejection_rate == 0.0
    assert report.session_valid


def test_epoch_exceeding_threshold_is_rejected() -> None:
    epochs = [_flat_epoch(10.0), _drift_epoch(400.0), _flat_epoch(10.0)]
    kept, report = reject_artifacts(epochs, threshold_uv=100.0)
    assert len(kept) == 2
    assert report.n_rejected == 1
    assert list(report.keep_mask) == [True, False, True]


def test_rejection_triggers_when_any_single_channel_exceeds() -> None:
    """One bad channel condemns the whole epoch; the others being clean is
    irrelevant, because the feature vector concatenates all channels."""
    data = np.tile(np.linspace(-5.0, 5.0, N_TIMES)[:, None], (1, N_CHANNELS))
    data[:, 1] = np.linspace(-200.0, 200.0, N_TIMES)   # channel 1 only
    kept, report = reject_artifacts([_epoch(data)], threshold_uv=100.0)
    assert kept == []
    assert report.n_rejected == 1
    # The clean channels are still measured and reported.
    assert report.peak_to_peak_uv[0, 0] == pytest.approx(10.0)
    assert report.peak_to_peak_uv[0, 1] == pytest.approx(400.0)


# --------------------------------------------------------------------------- #
# 4-6. Peak-to-peak versus absolute deviation
# --------------------------------------------------------------------------- #
def test_peak_to_peak_catches_drift_that_absolute_deviation_misses() -> None:
    """A 90 uV drift ramp: |deviation| never exceeds 45 uV, so a +/-75 uV
    absolute bound keeps it. Peak-to-peak sees the full 90 uV span."""
    epoch = _drift_epoch(90.0)
    ptp = epoch_peak_to_peak(epoch)
    max_abs_deviation = np.max(np.abs(epoch.data))

    assert max_abs_deviation == pytest.approx(45.0)      # passes +/-75 uV
    assert np.all(ptp == pytest.approx(90.0))            # 90 uV peak-to-peak
    # Under the locked 100 uV p-p criterion this epoch is (correctly) kept...
    _, report_100 = reject_artifacts([epoch], threshold_uv=100.0)
    assert report_100.n_rejected == 0
    # ...but an 80 uV p-p criterion rejects it, while a +/-75 uV absolute
    # criterion never would. The two families are not interchangeable.
    _, report_80 = reject_artifacts([epoch], threshold_uv=80.0)
    assert report_80.n_rejected == 1


def test_drift_beyond_threshold_is_rejected_despite_small_deviation() -> None:
    """A 150 uV drift has only 75 uV maximum absolute deviation — right on a
    +/-75 uV bound — yet is unambiguously an artifact by peak-to-peak."""
    epoch = _drift_epoch(150.0)
    assert np.max(np.abs(epoch.data)) == pytest.approx(75.0)
    kept, report = reject_artifacts([epoch], threshold_uv=100.0)
    assert kept == []
    assert report.n_rejected == 1


def test_peak_to_peak_is_measured_within_epoch_not_against_zero() -> None:
    """A large constant offset has zero peak-to-peak and must not be rejected;
    it is removed by baseline correction, not by artifact rejection."""
    data = np.full((N_TIMES, N_CHANNELS), 5000.0)
    kept, report = reject_artifacts([_epoch(data)], threshold_uv=100.0)
    assert len(kept) == 1
    assert np.all(report.peak_to_peak_uv == pytest.approx(0.0))


# --------------------------------------------------------------------------- #
# 7-8. Exact boundary behaviour
# --------------------------------------------------------------------------- #
def test_epoch_exactly_at_threshold_is_kept() -> None:
    """The criterion is 'reject if it exceeds', not 'reject if it reaches'."""
    epoch = _drift_epoch(100.0)
    assert np.all(epoch_peak_to_peak(epoch) == pytest.approx(100.0))
    kept, report = reject_artifacts([epoch], threshold_uv=100.0)
    assert len(kept) == 1
    assert report.n_rejected == 0


def test_epoch_just_above_threshold_is_rejected() -> None:
    epoch = _drift_epoch(100.0 + 1e-6)
    kept, report = reject_artifacts([epoch], threshold_uv=100.0)
    assert kept == []
    assert report.n_rejected == 1


# --------------------------------------------------------------------------- #
# 9-11. Session-level rejection-rate alarm
# --------------------------------------------------------------------------- #
def test_rejection_rate_is_reported_correctly() -> None:
    epochs = [_flat_epoch(10.0)] * 6 + [_drift_epoch(400.0)] * 4
    _, report = reject_artifacts(epochs, threshold_uv=100.0)
    assert report.n_total == 10
    assert report.n_kept == 6
    assert report.rejection_rate == pytest.approx(0.4)


def test_session_invalid_when_rejection_rate_exceeds_limit() -> None:
    """More than 30% rejected means the recording is unusable and must be
    repeated, not analysed."""
    epochs = [_flat_epoch(10.0)] * 6 + [_drift_epoch(400.0)] * 4   # 40%
    _, report = reject_artifacts(epochs, threshold_uv=100.0, max_session_rate=0.30)
    assert report.rejection_rate == pytest.approx(0.4)
    assert report.session_valid is False


def test_session_valid_exactly_at_rate_limit() -> None:
    """Exactly 30% is within budget; the alarm is for rates *above* the limit."""
    epochs = [_flat_epoch(10.0)] * 7 + [_drift_epoch(400.0)] * 3   # 30%
    _, report = reject_artifacts(epochs, threshold_uv=100.0, max_session_rate=0.30)
    assert report.rejection_rate == pytest.approx(0.3)
    assert report.session_valid is True


# --------------------------------------------------------------------------- #
# 12-14. Config wiring and contract
# --------------------------------------------------------------------------- #
def test_config_wiring_reads_threshold_and_rate() -> None:
    proc_cfg = {
        "artifact_rejection": {"threshold_uv": 50.0, "max_session_rate": 0.10}
    }
    epochs = [_flat_epoch(10.0)] * 8 + [_drift_epoch(60.0)] * 2   # 20% rejected
    kept, report = reject_artifacts_from_config(epochs, proc_cfg)
    assert report.threshold_uv == 50.0
    assert report.max_session_rate == 0.10
    assert len(kept) == 8
    assert report.session_valid is False       # 20% > 10% limit


def test_config_falls_back_to_locked_defaults_when_section_absent() -> None:
    """An older config without the section must still get the locked criterion,
    never a silently disabled one."""
    for cfg in ({}, None, {"bandpass": {}}):
        _, report = reject_artifacts_from_config([_flat_epoch(10.0)], cfg)
        assert report.threshold_uv == DEFAULT_REJECTION_THRESHOLD_UV == 100.0
        assert (
            report.max_session_rate
            == DEFAULT_MAX_SESSION_REJECTION_RATE
            == 0.30
        )


def test_empty_input_yields_empty_valid_report() -> None:
    kept, report = reject_artifacts([])
    assert kept == []
    assert isinstance(report, RejectionReport)
    assert report.n_total == 0
    assert report.rejection_rate == 0.0
    assert report.session_valid is True
    assert report.keep_mask.shape == (0,)


def test_non_positive_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        reject_artifacts([_flat_epoch(10.0)], threshold_uv=0.0)


def test_report_shapes_match_input() -> None:
    epochs = [_flat_epoch(10.0) for _ in range(4)]
    _, report = reject_artifacts(epochs)
    assert report.keep_mask.shape == (4,)
    assert report.peak_to_peak_uv.shape == (4, N_CHANNELS)
