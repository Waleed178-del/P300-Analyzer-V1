"""monitor.py — standalone operator display for live signal-quality checking.

WHY THIS IS A SEPARATE PROCESS
------------------------------
Only one process may hold the serial port, and redrawing a scrolling trace must
never compete with the pygame stimulus loop or the serial reader thread. The
recording process therefore *publishes* its raw buffer as UDP datagrams on
loopback (see :class:`main_pipeline.MonitorPublisher`) and this module listens.
Publishing is fire-and-forget: if the monitor is not running, is slow, or dies,
the recording is unaffected.

DISPLAY FILTER vs ANALYSIS FILTER — THE IMPORTANT PART
------------------------------------------------------
This module filters with :func:`scipy.signal.sosfilt`, a **causal, stateful**
filter whose internal state ``zi`` is carried from one datagram to the next so
the displayed trace is continuous. :mod:`processing`, which produces every
number that is ever reported, filters with :func:`scipy.signal.sosfiltfilt`, a
**zero-phase** filter that runs the signal forwards and then backwards.

The two are not interchangeable:

* A causal IIR filter has a non-zero, frequency-dependent group delay
  ``tau(w) = -d(phase(w))/dw``. It shifts and disperses the waveform in time,
  which would bias any P300 latency measurement taken from it.
* Forward-backward filtering conjugates the phase response, so the net phase is
  identically zero at every frequency and therefore the group delay is
  identically zero. Latency is preserved, which is why it is used for analysis.

A causal filter is nonetheless the correct choice *here*, because a display must
show the operator what is happening now and cannot wait for the future samples
that a backward pass requires.

**No reported result is attributable to the display filter.** This module is an
operator aid only. It never writes to the model, the epoch store, or the result
files; it consumes a copy of the raw stream and renders it.

Usage::

    # Terminal 1 — enable `monitor.enabled: true` in config.yaml, then record:
    python p300_speller/run.py train

    # Terminal 2 — watch the live signal-quality readout:
    python p300_speller/src/monitor.py --config p300_speller/configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9750
# Max UDP payload; frames larger than this are dropped by the transport anyway.
_RECV_BUFSIZE = 65535


# --------------------------------------------------------------------------- #
# Causal display filter
# --------------------------------------------------------------------------- #
class CausalDisplayFilter:
    """Stateful causal band-pass for continuous display.

    Unlike the zero-phase :func:`processing.apply_bandpass` used for analysis,
    this filter runs forwards only and preserves its delay state across calls,
    so successive datagrams join without a discontinuity at the seam. It
    introduces a real group delay; that is acceptable for a display and
    unacceptable for latency analysis.

    Args:
        low_hz: Lower pass-band edge.
        high_hz: Upper pass-band edge.
        fs: Sampling rate in Hz.
        n_channels: Number of channels (one filter state per channel).
        order: Butterworth order per edge.
    """

    def __init__(
        self,
        low_hz: float,
        high_hz: float,
        fs: float,
        n_channels: int,
        order: int = 4,
    ) -> None:
        nyq = 0.5 * fs
        low, high = low_hz / nyq, high_hz / nyq
        if not (0.0 < low < high < 1.0):
            raise ValueError(
                f"Invalid display band: {low_hz}-{high_hz} Hz at fs={fs} Hz"
            )
        self.sos = butter(order, [low, high], btype="bandpass", output="sos")
        self.n_channels = n_channels
        # One independent delay-line state per channel, primed to the steady
        # state so the first block does not open with a large transient. With
        # axis=0 and input (n_samples, n_channels), sosfilt requires zi of
        # shape (n_sections, 2, n_channels).
        zi_unit = sosfilt_zi(self.sos)                      # (n_sections, 2)
        self._zi = np.repeat(zi_unit[:, :, None], n_channels, axis=2)
        self._primed = False

    def reset(self) -> None:
        """Drop the filter state (use when the stream is discontinuous)."""
        self._primed = False

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filter one contiguous block, carrying state across calls.

        Args:
            block: Array of shape ``(n_samples, n_channels)``.

        Returns:
            Filtered array of the same shape.
        """
        if block.size == 0:
            return block
        if not self._primed:
            # Scale the steady-state condition by the first sample so the
            # filter starts settled at the current DC level.
            zi_unit = sosfilt_zi(self.sos)
            self._zi = np.stack(
                [zi_unit * block[0, c] for c in range(block.shape[1])], axis=2
            )
            self._primed = True
        out, self._zi = sosfilt(self.sos, block, axis=0, zi=self._zi)
        return out


# --------------------------------------------------------------------------- #
# Rolling state
# --------------------------------------------------------------------------- #
@dataclass
class MonitorState:
    """Everything the renderer needs to draw one frame."""

    channels: List[str] = field(default_factory=list)
    fs: float = 250.0
    last_sample_time: float = -np.inf
    n_samples_seen: int = 0
    rms_uv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ptp_uv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Latest artifact-rejection snapshot published by the pipeline.
    rejection_rate: float = 0.0
    n_rejected: int = 0
    n_total: int = 0
    threshold_uv: float = 100.0
    session_valid: bool = True
    last_event: str = ""
    last_update: float = 0.0


# --------------------------------------------------------------------------- #
# Subscriber
# --------------------------------------------------------------------------- #
class MonitorSubscriber:
    """Receives pipeline datagrams and maintains a :class:`MonitorState`.

    Args:
        host: Bind address.
        port: Bind port.
        band: ``(low_hz, high_hz)`` for the causal display filter.
        window_s: Length of the rolling window used for the RMS / peak-to-peak
            readout.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        band: tuple = (0.5, 30.0),
        window_s: float = 2.0,
    ) -> None:
        self.band = band
        self.window_s = window_s
        self.state = MonitorState()
        self._filter: Optional[CausalDisplayFilter] = None
        self._window: Optional[np.ndarray] = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.25)

    def close(self) -> None:
        """Release the socket."""
        try:
            self._sock.close()
        except Exception:
            pass

    def poll(self) -> bool:
        """Read and apply one pending datagram.

        Returns:
            ``True`` if a datagram was processed, ``False`` on timeout.
        """
        try:
            raw, _addr = self._sock.recvfrom(_RECV_BUFSIZE)
        except socket.timeout:
            return False
        except OSError:
            return False
        try:
            msg = json.loads(raw.decode("utf-8"))
        except Exception:
            # Malformed frame: a display must never crash on bad input.
            return False
        self._apply(msg)
        return True

    def _apply(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "signal":
            self._apply_signal(msg)
        elif kind == "quality":
            st = self.state
            st.n_total = int(msg.get("n_total", 0))
            st.n_rejected = int(msg.get("n_rejected", 0))
            st.rejection_rate = float(msg.get("rejection_rate", 0.0))
            st.threshold_uv = float(msg.get("threshold_uv", 100.0))
            st.session_valid = bool(msg.get("session_valid", True))
            st.last_update = time.time()
        elif kind == "event":
            text = str(msg.get("text", ""))
            self.state.last_event = (
                f"{msg.get('event', '')} {text}".strip()
            )
            self.state.last_update = time.time()

    def _apply_signal(self, msg: dict) -> None:
        """Filter and accumulate the genuinely new samples in a signal frame.

        The pipeline publishes overlapping slices of its ring buffer. Feeding
        overlapping samples into a stateful filter would corrupt its state, so
        only samples newer than the last one seen are processed.
        """
        data = np.asarray(msg.get("data", []), dtype=np.float64)
        if data.ndim != 2 or data.size == 0:
            return
        fs = float(msg.get("fs", 250.0))
        t0 = float(msg.get("t0", 0.0))
        channels = list(msg.get("channels", []))

        st = self.state
        if channels and channels != st.channels:
            # Montage changed (or first frame): rebuild the filter and window.
            st.channels = channels
            st.fs = fs
            self._filter = CausalDisplayFilter(
                self.band[0], self.band[1], fs, data.shape[1]
            )
            self._window = None
            st.last_sample_time = -np.inf

        if self._filter is None:
            self._filter = CausalDisplayFilter(
                self.band[0], self.band[1], fs, data.shape[1]
            )

        times = t0 + np.arange(data.shape[0]) / fs
        fresh = times > st.last_sample_time
        if not np.any(fresh):
            return
        block = data[fresh, :]
        st.last_sample_time = float(times[fresh][-1])
        st.n_samples_seen += int(block.shape[0])

        filtered = self._filter.process(block)
        self._window = (
            filtered
            if self._window is None
            else np.vstack([self._window, filtered])
        )
        keep = int(self.window_s * fs)
        if self._window.shape[0] > keep:
            self._window = self._window[-keep:, :]

        st.rms_uv = np.sqrt(np.mean(self._window ** 2, axis=0))
        st.ptp_uv = np.ptp(self._window, axis=0)
        st.last_update = time.time()


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #
def _bar(value: float, limit: float, width: int = 28) -> str:
    """Render ``value`` as a fixed-width ASCII bar scaled to ``limit``."""
    if limit <= 0:
        return " " * width
    filled = int(np.clip(value / limit, 0.0, 1.0) * width)
    return "#" * filled + "-" * (width - filled)


def render(state: MonitorState) -> str:
    """Format the current state as a terminal dashboard."""
    lines: List[str] = []
    lines.append("=" * 62)
    lines.append("  P300 OPERATOR MONITOR   (causal display filter — not analysis)")
    lines.append("=" * 62)

    thr = state.threshold_uv
    if not state.channels:
        # No signal frame yet. The per-channel table cannot be drawn, but the
        # quality block below still must be: a session-invalidity alarm can
        # arrive before the first signal frame and must never be swallowed.
        lines.append("  waiting for signal frames ...")
        lines.append("  (enable `monitor.enabled: true` in config.yaml, then")
        lines.append("   start a session with `python p300_speller/run.py train`)")
    else:
        lines.append(
            f"  {'ch':<5}{'RMS uV':>9}{'p-p uV':>9}   p-p vs {thr:.0f} uV limit"
        )
        for i, name in enumerate(state.channels):
            rms = state.rms_uv[i] if i < state.rms_uv.size else 0.0
            ptp = state.ptp_uv[i] if i < state.ptp_uv.size else 0.0
            flag = " OVER" if ptp > thr else ""
            lines.append(
                f"  {name:<5}{rms:>9.1f}{ptp:>9.1f}   {_bar(ptp, thr)}{flag}"
            )

    lines.append("-" * 62)
    if state.n_total:
        lines.append(
            f"  rejected {state.n_rejected}/{state.n_total} epochs "
            f"({state.rejection_rate:.1%}) at {thr:.0f} uV peak-to-peak"
        )
    else:
        lines.append("  no epochs screened yet")

    if not state.session_valid:
        lines.append("  *** SESSION INVALID — rejection rate over budget ***")
        lines.append("  *** repeat the recording; do not report results  ***")

    lines.append(f"  samples seen: {state.n_samples_seen}")
    if state.last_event:
        lines.append(f"  last event  : {state.last_event}")
    lines.append("=" * 62)
    return "\n".join(lines)


def run_monitor(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    band: tuple = (0.5, 30.0),
    refresh_s: float = 0.5,
) -> int:
    """Listen and render until interrupted.

    Returns:
        Process exit status (0 on a clean Ctrl-C).
    """
    sub = MonitorSubscriber(host=host, port=port, band=band)
    print(f"[monitor] listening on {host}:{port} — Ctrl-C to stop")
    last_draw = 0.0
    try:
        while True:
            sub.poll()
            now = time.time()
            if now - last_draw >= refresh_s:
                # Clear screen without requiring curses.
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(render(sub.state) + "\n")
                sys.stdout.flush()
                last_draw = now
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
        return 0
    finally:
        sub.close()


def _load_monitor_config(path: Optional[str]) -> dict:
    """Read the ``monitor`` and band settings from ``config.yaml`` if given."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    mon = cfg.get("monitor") or {}
    bp = (cfg.get("processing") or {}).get("bandpass") or {}
    return {
        "host": mon.get("host", DEFAULT_HOST),
        "port": mon.get("port", DEFAULT_PORT),
        "low_hz": bp.get("low_hz", 0.5),
        "high_hz": bp.get("high_hz", 30.0),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description=(
            "Live operator display for the P300 speller. Runs in its own "
            "process and never touches the analysis path."
        ),
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help="bind port")
    parser.add_argument("--refresh", type=float, default=0.5,
                        help="redraw interval in seconds")
    args = parser.parse_args(argv)

    from_cfg = _load_monitor_config(args.config)
    host = args.host or from_cfg.get("host", DEFAULT_HOST)
    port = args.port or from_cfg.get("port", DEFAULT_PORT)
    band = (from_cfg.get("low_hz", 0.5), from_cfg.get("high_hz", 30.0))
    return run_monitor(host=host, port=port, band=band, refresh_s=args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
