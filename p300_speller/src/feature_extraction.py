"""feature_extraction.py — turn epochs into classifier feature vectors.

The classifier in :mod:`classifier` expects fixed-length, scaled numeric
vectors. This module bridges :class:`signal_processing.Epoch` objects to that
representation through three deterministic steps:

1. **Spatial filtering** (optional) — combine information across channels. The
   common-average reference (CAR) subtracts the instantaneous mean across
   channels, attenuating spatially diffuse noise that is shared by all
   electrodes. For single-channel montages this is a no-op.
2. **Temporal downsampling** — the P300 is a slow (~0.3 s) deflection, so the
   full 250 Hz resolution is redundant and inflates dimensionality. Each epoch
   is decimated to ``downsample_hz`` by integer-factor block averaging, which
   also acts as a mild anti-alias low-pass.
3. **Flattening** — the ``(n_times_ds, n_channels)`` epoch is raveled into a
   single 1-D feature vector ``[t0c0, t0c1, ..., t1c0, ...]``.

The public entry point :func:`epochs_to_matrix` converts a list of epochs into
a single ``(n_epochs, n_features)`` design matrix ready for scikit-learn.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from signal_processing import Epoch


# --------------------------------------------------------------------------- #
# Spatial filtering
# --------------------------------------------------------------------------- #
def common_average_reference(epoch: np.ndarray) -> np.ndarray:
    """Apply the common-average reference across channels.

    Args:
        epoch: Array of shape ``(n_times, n_channels)``.

    Returns:
        Re-referenced array of the same shape. Each sample has the across-
        channel mean removed.
    """
    if epoch.shape[1] < 2:
        # CAR is undefined / pointless for a single channel.
        return epoch.copy()
    mean_across_channels = epoch.mean(axis=1, keepdims=True)
    return epoch - mean_across_channels


def apply_spatial_filter(epoch: np.ndarray, mode: str) -> np.ndarray:
    """Dispatch to the configured spatial filter.

    Args:
        epoch: Array of shape ``(n_times, n_channels)``.
        mode: ``"none"`` or ``"car"``.

    Returns:
        Spatially filtered epoch.

    Raises:
        ValueError: For an unknown ``mode``.
    """
    if mode == "none":
        return epoch
    if mode == "car":
        return common_average_reference(epoch)
    raise ValueError(f"Unknown spatial_filter mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Temporal downsampling
# --------------------------------------------------------------------------- #
def decimate_epoch(epoch: np.ndarray, fs: float, target_hz: float) -> np.ndarray:
    """Downsample an epoch by integer-factor block averaging.

    Block averaging (a boxcar decimation) is used instead of naive striding so
    that energy between retained samples is not discarded and high-frequency
    content is attenuated rather than aliased.

    Args:
        epoch: Array of shape ``(n_times, n_channels)``.
        fs: Original sampling rate in Hz.
        target_hz: Desired post-decimation rate in Hz. If it is >= ``fs`` the
            epoch is returned unchanged.

    Returns:
        Array of shape ``(n_times // factor, n_channels)``.
    """
    factor = max(1, int(round(fs / float(target_hz))))
    if factor == 1:
        return epoch.copy()
    n_times, n_channels = epoch.shape
    usable = (n_times // factor) * factor
    if usable == 0:
        # Epoch shorter than one block -> collapse to its mean.
        return epoch.mean(axis=0, keepdims=True)
    trimmed = epoch[:usable, :]
    blocks = trimmed.reshape(usable // factor, factor, n_channels)
    return blocks.mean(axis=1)


# --------------------------------------------------------------------------- #
# Epoch -> feature vector
# --------------------------------------------------------------------------- #
def epoch_to_features(
    epoch: Epoch, fs: float, downsample_hz: float, spatial_filter: str
) -> np.ndarray:
    """Convert one :class:`Epoch` to a flat feature vector.

    Args:
        epoch: The source epoch.
        fs: Original sampling rate in Hz.
        downsample_hz: Target decimation rate.
        spatial_filter: ``"none"`` or ``"car"``.

    Returns:
        1-D feature vector of length ``n_times_ds * n_channels``.
    """
    spatial = apply_spatial_filter(epoch.data, spatial_filter)
    decimated = decimate_epoch(spatial, fs, downsample_hz)
    # Row-major flatten: time-major blocks of channels.
    return decimated.reshape(-1).astype(np.float64, copy=False)


def epochs_to_matrix(
    epochs: Sequence[Epoch], fs: float, feat_cfg: dict
) -> np.ndarray:
    """Convert a list of epochs into a design matrix.

    Args:
        epochs: Epochs to vectorise (all must share the same shape).
        fs: Original sampling rate in Hz.
        feat_cfg: The ``features`` section of ``config.yaml``.

    Returns:
        Array of shape ``(n_epochs, n_features)``. Empty input yields a
        ``(0, 0)`` array.
    """
    if len(epochs) == 0:
        return np.empty((0, 0), dtype=np.float64)

    downsample_hz = feat_cfg.get("downsample_hz", 20)
    spatial_filter = feat_cfg.get("spatial_filter", "car")

    vectors: List[np.ndarray] = [
        epoch_to_features(ep, fs, downsample_hz, spatial_filter) for ep in epochs
    ]
    feature_len = vectors[0].shape[0]
    if any(v.shape[0] != feature_len for v in vectors):
        raise ValueError(
            "Inconsistent feature lengths; epochs must share an identical "
            "time/channel shape before vectorisation."
        )
    return np.vstack(vectors)
