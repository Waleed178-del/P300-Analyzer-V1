"""Unit tests for :mod:`features` (spatial filtering, downsampling, flattening).

These pin down the deterministic feature-vector contract the classifier relies
on: decimation factor and length, block-average semantics, common-average
reference behaviour for both multi- and single-channel montages, and the final
design-matrix shape produced by :func:`features.epochs_to_matrix`.
"""

from __future__ import annotations

import numpy as np
import pytest

from acquisition import StimulusMarker
from features import (
    apply_spatial_filter,
    common_average_reference,
    decimate_epoch,
    epoch_to_features,
    epochs_to_matrix,
)
from processing import Epoch

FS = 250.0


def _epoch(n_times: int, n_channels: int, fill: float = 1.0) -> Epoch:
    """Construct a constant-valued epoch for shape/length invariance tests."""
    marker = StimulusMarker(code=1, kind="row", index=0, perf_time=0.0)
    times = np.arange(n_times) / FS
    data = np.full((n_times, n_channels), fill, dtype=np.float64)
    return Epoch(data=data, times=times, marker=marker)


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #
def test_decimate_reduces_length_by_integer_factor() -> None:
    epoch = np.arange(100 * 3, dtype=float).reshape(100, 3)
    out = decimate_epoch(epoch, FS, target_hz=25.0)   # factor = 10
    assert out.shape == (10, 3)


def test_decimate_block_average_preserves_overall_mean() -> None:
    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(120, 2))
    out = decimate_epoch(epoch, FS, target_hz=25.0)   # factor = 10, exact fit
    assert np.isclose(out.mean(), epoch.mean(), atol=1e-9)


def test_decimate_is_noop_when_target_ge_source() -> None:
    epoch = np.ones((10, 1))
    out = decimate_epoch(epoch, FS, target_hz=FS)
    assert out.shape == (10, 1)


# --------------------------------------------------------------------------- #
# Spatial filtering
# --------------------------------------------------------------------------- #
def test_car_zero_sums_across_channels() -> None:
    rng = np.random.default_rng(1)
    epoch = rng.normal(size=(50, 3))
    out = common_average_reference(epoch)
    assert np.allclose(out.sum(axis=1), 0.0, atol=1e-9)


def test_car_single_channel_is_noop() -> None:
    epoch = np.full((10, 1), 3.0)
    assert np.allclose(common_average_reference(epoch), epoch)


def test_spatial_filter_none_passthrough() -> None:
    epoch = np.arange(12, dtype=float).reshape(6, 2)
    assert np.allclose(apply_spatial_filter(epoch, "none"), epoch)


def test_spatial_filter_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        apply_spatial_filter(np.ones((4, 2)), "bogus")


# --------------------------------------------------------------------------- #
# Flattening / design matrix
# --------------------------------------------------------------------------- #
def test_epoch_to_features_length_matches_decimated_channels() -> None:
    epoch = _epoch(n_times=225, n_channels=3)        # 0.9 s at 250 Hz
    feat = epoch_to_features(epoch, FS, downsample_hz=20.0, spatial_filter="none")
    factor = round(FS / 20.0)                         # 12 -> 225 // 12 = 18
    assert feat.shape == (18 * 3,)


def test_epochs_to_matrix_shape_multichannel() -> None:
    epochs = [_epoch(225, 3) for _ in range(5)]
    cfg = {"downsample_hz": 20, "spatial_filter": "none"}
    matrix = epochs_to_matrix(epochs, FS, cfg)
    assert matrix.ndim == 2 and matrix.shape[0] == 5


def test_epochs_to_matrix_supports_single_channel() -> None:
    epochs = [_epoch(225, 1) for _ in range(4)]
    cfg = {"downsample_hz": 20, "spatial_filter": "car"}
    matrix = epochs_to_matrix(epochs, FS, cfg)
    assert matrix.shape[0] == 4


def test_epochs_to_matrix_empty_input() -> None:
    cfg = {"downsample_hz": 20, "spatial_filter": "none"}
    assert epochs_to_matrix([], FS, cfg).shape == (0, 0)
