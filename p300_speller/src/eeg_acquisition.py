"""eeg_acquisition.py — EEG data sources for the P300 Speller.

This module abstracts *where* EEG samples come from behind a single interface,
:class:`BaseEEGSource`, so the rest of the pipeline never needs to know whether
it is talking to real silicon or a synthetic generator.

Two concrete sources are provided:

* :class:`SerialAcquisition` — reads ASCII data packets from the Arduino /
  ADS1115 front-end (see ``arduino/p300_eeg_acquisition.ino``) on a background
  thread and decodes them into a timestamped ring buffer.
* :class:`Simulator` — generates synthetic multi-channel EEG with realistic
  background activity (alpha rhythm + broadband noise) and injects an
  artificial P300 deflection ~300 ms after any flash that targets the user's
  attended symbol. Indispensable for offline development and CI.

Both sources share a common contract:

* Sampling happens on a background thread once :meth:`start` is called.
* Samples land in a fixed-size ring buffer keyed by a monotonic
  ``time.perf_counter`` timestamp captured on the host. The *same* clock is
  used by the speller to timestamp flashes, so epoching needs no clock
  conversion between the stimulus and neural streams.
* The host pushes stimulus markers with :meth:`push_marker`. A source may use
  the marker for bookkeeping (Serial forwards a sync byte to the Arduino) or to
  drive synthetic physiology (the Simulator schedules a P300).

All public methods are thread-safe.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pyserial is only required for the hardware path.
    import serial  # type: ignore
except Exception:  # pragma: no cover - exercised only without pyserial present
    serial = None  # type: ignore


# --------------------------------------------------------------------------- #
# Stimulus markers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StimulusMarker:
    """A single row/column flash event, timestamped on the host clock.

    Attributes:
        code: Integer sync code sent to the hardware. Rows map to ``1..n_rows``
            and columns to ``n_rows + 1 .. n_rows + n_cols`` (0 is reserved for
            "no marker").
        kind: Either ``"row"`` or ``"col"``.
        index: Zero-based row or column index within the speller matrix.
        perf_time: ``time.perf_counter`` value captured at flash onset.
    """

    code: int
    kind: str
    index: int
    perf_time: float


# --------------------------------------------------------------------------- #
# Base source
# --------------------------------------------------------------------------- #
class BaseEEGSource:
    """Abstract, thread-safe EEG source backed by a ring buffer.

    Subclasses implement :meth:`_acquire_loop`, appending decoded samples via
    :meth:`_push_sample`. Consumers read the accumulated stream with
    :meth:`get_buffer` and the marker log with :meth:`get_markers`.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        ring_buffer_s: float,
    ) -> None:
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.channel_names = list(channel_names)
        self.n_channels = len(self.channel_names)
        self._capacity = max(1, int(ring_buffer_s * self.sampling_rate_hz))

        # Pre-allocated ring buffers. ``_count`` tracks total samples ever
        # written; the physical index is ``_count % _capacity``.
        self._ts = np.zeros(self._capacity, dtype=np.float64)
        self._data = np.zeros((self._capacity, self.n_channels), dtype=np.float64)
        self._count = 0

        self._markers: List[StimulusMarker] = []
        self._lock = threading.RLock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        """Begin background acquisition. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._acquire_loop, name=type(self).__name__, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background acquisition and join the worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "BaseEEGSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- producer side (called from the acquisition thread) ----------------- #
    def _push_sample(self, perf_time: float, channels: np.ndarray) -> None:
        """Append one multi-channel sample to the ring buffer."""
        with self._lock:
            idx = self._count % self._capacity
            self._ts[idx] = perf_time
            self._data[idx, :] = channels
            self._count += 1

    # -- consumer side ------------------------------------------------------ #
    def get_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the buffered stream in chronological order.

        Returns:
            ``(timestamps, data)`` where ``timestamps`` has shape ``(N,)`` and
            ``data`` has shape ``(N, n_channels)``. ``N`` is at most the ring
            capacity; older samples are dropped once it is full.
        """
        with self._lock:
            n = min(self._count, self._capacity)
            if n == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty((0, self.n_channels), dtype=np.float64),
                )
            if self._count <= self._capacity:
                return self._ts[:n].copy(), self._data[:n, :].copy()
            # Buffer has wrapped: unroll starting at the oldest sample.
            start = self._count % self._capacity
            order = np.arange(start, start + self._capacity) % self._capacity
            return self._ts[order].copy(), self._data[order, :].copy()

    def push_marker(self, marker: StimulusMarker) -> None:
        """Record a stimulus marker. Subclasses may extend this."""
        with self._lock:
            self._markers.append(marker)

    def get_markers(self) -> List[StimulusMarker]:
        """Return a copy of the marker log in insertion order."""
        with self._lock:
            return list(self._markers)

    def clear_markers(self) -> None:
        """Drop all recorded markers (call between characters)."""
        with self._lock:
            self._markers.clear()

    # -- to be implemented by subclasses ------------------------------------ #
    def _acquire_loop(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Hardware source
# --------------------------------------------------------------------------- #
class SerialAcquisition(BaseEEGSource):
    """Read timestamped EEG packets from the Arduino / ADS1115 over Serial.

    The expected line protocol (emitted by ``p300_eeg_acquisition.ino``) is::

        D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]\\n

    Raw ADC counts are converted to microvolts using the configured PGA
    full-scale range. The host applies its own ``perf_counter`` timestamp on
    packet arrival; the Arduino ``micros`` field is retained only for jitter
    diagnostics.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        ring_buffer_s: float,
        adc_fullscale_v: float = 4.096,
        adc_resolution_bits: int = 16,
        read_timeout_s: float = 1.0,
    ) -> None:
        super().__init__(sampling_rate_hz, channel_names, ring_buffer_s)
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed; install it or set "
                "acquisition.use_simulator: true in config.yaml"
            )
        self._port = port
        self._baud = baud_rate
        self._read_timeout_s = read_timeout_s
        # Counts -> microvolts. For a signed 16-bit ADC the positive full scale
        # is 2**(bits-1) counts mapping to adc_fullscale_v volts.
        full_scale_counts = float(2 ** (adc_resolution_bits - 1))
        self._counts_to_uv = (adc_fullscale_v / full_scale_counts) * 1e6
        self._serial: Optional["serial.Serial"] = None

    def _open(self) -> "serial.Serial":
        return serial.Serial(
            port=self._port, baudrate=self._baud, timeout=self._read_timeout_s
        )

    def push_marker(self, marker: StimulusMarker) -> None:
        """Record the marker locally and forward a sync code to the Arduino."""
        super().push_marker(marker)
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.write(f"M{marker.code}\n".encode("ascii"))
            except Exception:
                # A dropped sync byte is non-fatal: the host timestamp still
                # aligns the epoch. Do not crash the acquisition thread.
                pass

    def _acquire_loop(self) -> None:
        self._serial = self._open()
        try:
            # Discard any partial line already in the OS buffer.
            self._serial.reset_input_buffer()
            while not self._stop_event.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                perf_time = time.perf_counter()
                sample = self._decode(raw)
                if sample is not None:
                    self._push_sample(perf_time, sample)
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            finally:
                self._serial = None

    def _decode(self, raw: bytes) -> Optional[np.ndarray]:
        """Decode one ASCII data line into a microvolt channel vector."""
        try:
            line = raw.decode("ascii", errors="ignore").strip()
        except Exception:
            return None
        if not line or line[0] != "D":
            return None  # header ('H'), error ('E') or noise
        parts = line.split(",")
        # D, seq, micros, marker, ch0[, ch1, ch2]
        expected = 4 + self.n_channels
        if len(parts) != expected:
            return None
        try:
            counts = np.array(
                [int(parts[4 + c]) for c in range(self.n_channels)],
                dtype=np.float64,
            )
        except ValueError:
            return None
        return counts * self._counts_to_uv


# --------------------------------------------------------------------------- #
# Synthetic source
# --------------------------------------------------------------------------- #
@dataclass
class _ScheduledP300:
    """A pending synthetic P300 deflection to be mixed into the stream."""

    onset_time: float          # perf_counter time of the flash that elicited it
    channels: np.ndarray       # per-channel amplitude scaling (Pz strongest)


class Simulator(BaseEEGSource):
    """Synthetic EEG generator with attention-dependent P300 injection.

    The generator runs a real-time loop producing samples at the configured
    rate. Background activity is broadband noise plus a posterior alpha rhythm.
    When :meth:`push_marker` reports a flash whose row/column contains the
    *secret attended target*, a Gaussian P300 bump is scheduled at
    ``p300_latency_ms`` after the flash, weighted toward central/posterior
    channels. This reproduces the essential statistic the classifier must learn:
    target flashes carry a delayed positive deflection, non-target flashes do
    not.

    The attended target is supplied by the pipeline via :meth:`set_target`
    (during calibration it is the known letter; during free spelling it is the
    letter a simulated user "intends", enabling end-to-end offline tests).
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        ring_buffer_s: float,
        noise_uv: float = 12.0,
        p300_amplitude_uv: float = 6.0,
        p300_latency_ms: float = 300.0,
        p300_width_ms: float = 120.0,
        alpha_amplitude_uv: float = 8.0,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(sampling_rate_hz, channel_names, ring_buffer_s)
        self._noise_uv = noise_uv
        self._p300_amp = p300_amplitude_uv
        self._p300_latency_s = p300_latency_ms / 1000.0
        self._p300_width_s = p300_width_ms / 1000.0
        self._alpha_amp = alpha_amplitude_uv
        self._rng = np.random.default_rng(seed)

        # Per-channel spatial weighting of the P300 (posterior emphasis). If the
        # montage is unknown, fall back to a uniform profile.
        self._channel_gain = self._build_channel_gain(self.channel_names)

        # Secret attended target the simulated user is focusing on.
        self._target_row: Optional[int] = None
        self._target_col: Optional[int] = None

        self._scheduled: List[_ScheduledP300] = []
        self._alpha_phase = 0.0

    @staticmethod
    def _build_channel_gain(names: Sequence[str]) -> np.ndarray:
        """Relative P300 amplitude per channel (Pz > Cz > Fz)."""
        profile = {"fz": 0.5, "cz": 0.8, "pz": 1.0}
        gains = [profile.get(n.strip().lower(), 0.8) for n in names]
        return np.asarray(gains, dtype=np.float64)

    def set_target(self, row: Optional[int], col: Optional[int]) -> None:
        """Set the row/column the simulated user is attending to.

        Pass ``(None, None)`` to disable P300 injection (e.g. to model a user
        who is not attending).
        """
        with self._lock:
            self._target_row = row
            self._target_col = col

    def push_marker(self, marker: StimulusMarker) -> None:
        """Record the marker and, if it hits the target, schedule a P300."""
        super().push_marker(marker)
        with self._lock:
            is_target = (
                marker.kind == "row" and marker.index == self._target_row
            ) or (
                marker.kind == "col" and marker.index == self._target_col
            )
            if is_target:
                self._scheduled.append(
                    _ScheduledP300(
                        onset_time=marker.perf_time,
                        channels=self._channel_gain.copy(),
                    )
                )

    def _acquire_loop(self) -> None:
        period = 1.0 / self.sampling_rate_hz
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                # Sleep the bulk of the remaining time, then spin briefly for
                # sub-millisecond pacing accuracy.
                remaining = next_t - now
                if remaining > 0.002:
                    time.sleep(remaining - 0.001)
                continue
            next_t += period
            self._push_sample(now, self._generate_sample(now))

    def _generate_sample(self, t: float) -> np.ndarray:
        """Synthesize one multi-channel sample at host time ``t``."""
        # Background: white-ish noise + shared posterior alpha rhythm.
        self._alpha_phase += 2.0 * np.pi * 10.0 / self.sampling_rate_hz
        alpha = self._alpha_amp * np.sin(self._alpha_phase)
        noise = self._rng.normal(0.0, self._noise_uv, size=self.n_channels)
        sample = noise + alpha * self._channel_gain

        # Sum any active scheduled P300 deflections (Gaussian envelope).
        with self._lock:
            if self._scheduled:
                still_active: List[_ScheduledP300] = []
                two_sigma_tail = self._p300_latency_s + 4.0 * self._p300_width_s
                for ev in self._scheduled:
                    dt = t - ev.onset_time
                    if dt < 0:
                        still_active.append(ev)
                        continue
                    if dt <= two_sigma_tail:
                        env = np.exp(
                            -0.5
                            * ((dt - self._p300_latency_s) / self._p300_width_s) ** 2
                        )
                        sample += self._p300_amp * env * ev.channels
                        still_active.append(ev)
                    # else: fully decayed -> drop it
                self._scheduled = still_active

        return sample


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_source(config: dict) -> BaseEEGSource:
    """Construct the EEG source selected by ``config['acquisition']``.

    Args:
        config: Parsed ``config.yaml`` as a nested dict.

    Returns:
        A started-on-demand :class:`BaseEEGSource` (the caller must
        :meth:`~BaseEEGSource.start` it).
    """
    acq = config["acquisition"]
    channels = acq["channels"]
    fs = acq["sampling_rate_hz"]
    ring = acq["ring_buffer_s"]

    if acq.get("use_simulator", True):
        sim = acq.get("simulator", {})
        return Simulator(
            sampling_rate_hz=fs,
            channel_names=channels,
            ring_buffer_s=ring,
            noise_uv=sim.get("noise_uv", 12.0),
            p300_amplitude_uv=sim.get("p300_amplitude_uv", 6.0),
            p300_latency_ms=sim.get("p300_latency_ms", 300.0),
            p300_width_ms=sim.get("p300_width_ms", 120.0),
            alpha_amplitude_uv=sim.get("alpha_amplitude_uv", 8.0),
            seed=sim.get("seed", 42),
        )

    ser = acq["serial"]
    return SerialAcquisition(
        port=ser["port"],
        baud_rate=ser["baud_rate"],
        sampling_rate_hz=fs,
        channel_names=channels,
        ring_buffer_s=ring,
        adc_fullscale_v=acq.get("adc_fullscale_v", 4.096),
        adc_resolution_bits=acq.get("adc_resolution_bits", 16),
        read_timeout_s=ser.get("read_timeout_s", 1.0),
    )
