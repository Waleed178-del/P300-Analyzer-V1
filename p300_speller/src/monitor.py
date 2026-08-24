"""
Operator acquisition monitor for the P300 speller.

Runs as a SEPARATE OS PROCESS from run.py. The pipeline publishes small UDP
datagrams to loopback; this process receives them and draws the operator panel.

Rationale for the process split
-------------------------------
1. Only one process may hold the serial port. This module NEVER opens it.
2. Drawing must not compete with the pygame stimulus loop or the serial reader
   thread that carry the timing-critical path. Publishing costs one non-blocking
   sendto per chunk.

Display filtering
-----------------
This module uses its own CAUSAL, STATEFUL filter (sosfilt with zi), which is a
DIFFERENT filter from the zero-phase sosfiltfilt used in src/processing.py for
analysis. Zero-phase filtering requires future samples and cannot be applied to
a live buffer. Nothing drawn here ever feeds the classifier or the results.

Usage
-----
    python src/monitor.py                       # listen for the pipeline
    python src/monitor.py --demo                # SIMULATED DATA, no hardware
    python src/monitor.py --port 9911 --window 5

Keys: [ and ] change the vertical scale, r resets counters, q quits.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

# --------------------------------------------------------------------------
# Wire protocol. Little endian, no padding.
#   b'S' + <I n_ch><I n_samp><d fs_measured> + float32[n_ch * n_samp]  raw uV
#   b'P' + <B label><I n_ch><I n_samp>       + float32[n_ch * n_samp]  epoch uV
# label: 1 = target flash, 0 = non-target flash
# --------------------------------------------------------------------------

MAGIC_STREAM = b"S"
MAGIC_EPOCH = b"P"
DEFAULT_PORT = 9911
MAX_DATAGRAM = 65507

# Locked project constants. Keep in sync with configs/config.yaml.
REJECT_UV = 100.0            # peak-to-peak per epoch, np.ptp, per channel
SCALE_UV_PER_COUNT = 0.131718  # includes the 949x analogue gain divisor
ADC_FULLSCALE_COUNTS = 32767
SATURATION_UV = 0.95 * ADC_FULLSCALE_COUNTS * SCALE_UV_PER_COUNT

OKABE_ITO = [(230, 159, 0), (0, 114, 178), (0, 158, 115),
             (204, 121, 167), (86, 180, 233), (213, 94, 0)]


# ==========================================================================
# Publisher. Import this from main_pipeline.py. It is deliberately tiny.
# ==========================================================================

class MonitorPublisher:
    """Fire-and-forget UDP sender. Never raises, never blocks, never retries."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 enabled: bool = True) -> None:
        self.addr = (host, port)
        self.enabled = enabled
        self._sock: socket.socket | None = None
        if enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setblocking(False)
            except OSError:
                self.enabled = False

    def _send(self, payload: bytes) -> None:
        if not self.enabled or self._sock is None:
            return
        if len(payload) > MAX_DATAGRAM:
            return
        try:
            self._sock.sendto(payload, self.addr)
        except (BlockingIOError, OSError):
            pass  # dropping a display frame is always cheaper than blocking

    def publish_chunk(self, data_uv: np.ndarray, fs_measured: float) -> None:
        """data_uv: (n_channels, n_new_samples), RAW microvolts, unfiltered."""
        arr = np.ascontiguousarray(np.atleast_2d(data_uv), dtype=np.float32)
        header = MAGIC_STREAM + struct.pack("<IId", arr.shape[0], arr.shape[1],
                                            float(fs_measured))
        self._send(header + arr.tobytes())

    def publish_epoch(self, epoch_uv: np.ndarray, label: int) -> None:
        """epoch_uv: (n_channels, n_times), baseline-corrected microvolts."""
        arr = np.ascontiguousarray(np.atleast_2d(epoch_uv), dtype=np.float32)
        header = MAGIC_EPOCH + struct.pack("<BII", int(label) & 1,
                                           arr.shape[0], arr.shape[1])
        self._send(header + arr.tobytes())

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# ==========================================================================
# Causal display filter. NOT the analysis filter.
# ==========================================================================

class CausalDisplayFilter:
    """Order-2 Butterworth band-pass plus notch, stateful across chunks."""

    def __init__(self, fs: float, n_channels: int, low: float = 1.0,
                 high: float = 30.0, notch_hz: float = 50.0, q: float = 30.0):
        nyq = fs / 2.0
        high = min(high, nyq * 0.95)
        sos_bp = butter(2, [low / nyq, high / nyq], btype="band", output="sos")
        b, a = iirnotch(notch_hz / nyq, q)
        self.sos = np.vstack([sos_bp, tf2sos(b, a)])
        zi_unit = sosfilt_zi(self.sos)
        self.zi = np.repeat(zi_unit[:, None, :], n_channels, axis=1)
        self.primed = np.zeros(n_channels, dtype=bool)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        out = np.empty_like(chunk, dtype=np.float64)
        for ch in range(chunk.shape[0]):
            if not self.primed[ch] and chunk.shape[1] > 0:
                self.zi[:, ch, :] *= chunk[ch, 0]
                self.primed[ch] = True
            out[ch], self.zi[:, ch, :] = sosfilt(
                self.sos, chunk[ch], zi=self.zi[:, ch, :])
        return out


# ==========================================================================
# State
# ==========================================================================

@dataclass
class MonitorState:
    n_channels: int
    fs_nominal: float
    window_s: float
    labels: list[str]

    fs_measured: float = 0.0
    last_packet_t: float = 0.0

    epochs_seen: int = 0
    epochs_rejected: int = 0

    erp_sum: dict = field(default_factory=dict)
    erp_count: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(round(self.fs_nominal * self.window_s))
        self.raw = np.zeros((self.n_channels, n), dtype=np.float64)
        self.disp = np.zeros((self.n_channels, n), dtype=np.float64)
        self.filt = CausalDisplayFilter(self.fs_nominal, self.n_channels)

    def push(self, chunk_uv: np.ndarray, fs_measured: float) -> None:
        if chunk_uv.size == 0:
            return
        n = chunk_uv.shape[1]
        if n >= self.raw.shape[1]:
            chunk_uv = chunk_uv[:, -self.raw.shape[1]:]
            n = chunk_uv.shape[1]
        self.raw = np.roll(self.raw, -n, axis=1)
        self.raw[:, -n:] = chunk_uv
        self.disp = np.roll(self.disp, -n, axis=1)
        self.disp[:, -n:] = self.filt(chunk_uv)
        self.fs_measured = fs_measured
        self.last_packet_t = time.perf_counter()

    def push_epoch(self, epoch_uv: np.ndarray, label: int) -> None:
        self.epochs_seen += 1
        if float(np.max(np.ptp(epoch_uv, axis=1))) > REJECT_UV:
            self.epochs_rejected += 1
            return  # rejected epochs never enter the ERP average
        cz = min(1, epoch_uv.shape[0] - 1)
        trace = epoch_uv[cz].astype(np.float64)
        if label not in self.erp_sum or self.erp_sum[label].shape != trace.shape:
            self.erp_sum[label] = np.zeros_like(trace)
            self.erp_count[label] = 0
        self.erp_sum[label] += trace
        self.erp_count[label] += 1

    def erp_mean(self, label: int) -> np.ndarray | None:
        if self.erp_count.get(label, 0) == 0:
            return None
        return self.erp_sum[label] / self.erp_count[label]

    @property
    def rejection_pct(self) -> float:
        if self.epochs_seen == 0:
            return 0.0
        return 100.0 * self.epochs_rejected / self.epochs_seen

    def dc_offset_uv(self, ch: int) -> float:
        return float(np.mean(self.raw[ch]))

    def saturated(self, ch: int) -> bool:
        return bool(np.max(np.abs(self.raw[ch])) >= SATURATION_UV)

    def reset_counters(self) -> None:
        self.epochs_seen = 0
        self.epochs_rejected = 0
        self.erp_sum.clear()
        self.erp_count.clear()


# ==========================================================================
# Receiver
# ==========================================================================

def make_socket(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    s.bind(("127.0.0.1", port))
    s.setblocking(False)
    return s


def drain(sock: socket.socket, state: MonitorState, budget: int = 64) -> None:
    for _ in range(budget):
        try:
            payload, _ = sock.recvfrom(MAX_DATAGRAM)
        except (BlockingIOError, OSError):
            return
        if not payload:
            continue
        tag, body = payload[:1], payload[1:]
        try:
            if tag == MAGIC_STREAM:
                n_ch, n_s, fs = struct.unpack("<IId", body[:16])
                arr = np.frombuffer(body[16:], dtype=np.float32, count=n_ch * n_s)
                state.push(arr.reshape(n_ch, n_s).astype(np.float64), fs)
            elif tag == MAGIC_EPOCH:
                label, n_ch, n_s = struct.unpack("<BII", body[:9])
                arr = np.frombuffer(body[9:], dtype=np.float32, count=n_ch * n_s)
                state.push_epoch(arr.reshape(n_ch, n_s).astype(np.float64), label)
        except (struct.error, ValueError):
            continue  # malformed datagram, drop it


# ==========================================================================
# Drawing
# ==========================================================================

BG = (18, 18, 20)
FG = (232, 232, 228)
MUTED = (140, 140, 136)
GRID = (48, 48, 52)
BAND = (58, 58, 66)
GOOD = (0, 158, 115)
WARN = (230, 159, 0)
BAD = (213, 94, 0)

W, H = 1180, 780
TRACE_X0, TRACE_X1 = 150, 830
PANEL_X = 855


def draw(screen, font, big, state: MonitorState, uv_per_div: float,
         demo: bool) -> None:
    import pygame

    screen.fill(BG)
    n_ch = state.n_channels
    top, bottom = 70, 470
    lane_h = (bottom - top) / n_ch
    px_per_uv = (lane_h * 0.42) / uv_per_div

    stale = (time.perf_counter() - state.last_packet_t) > 1.0
    fs_txt = f"{state.fs_measured:7.2f} Hz measured   {state.fs_nominal:.0f} Hz nominal"
    fs_bad = state.fs_measured > 0 and abs(
        state.fs_measured - state.fs_nominal) > 0.02 * state.fs_nominal
    screen.blit(big.render("Acquisition monitor", True, FG), (20, 18))
    screen.blit(font.render(fs_txt, True, BAD if fs_bad else FG), (300, 24))
    screen.blit(font.render(f"scale {uv_per_div:.0f} uV/div  [ ]", True, MUTED),
                (660, 24))
    if demo:
        screen.blit(big.render("SIMULATED DATA", True, BAD), (940, 18))
    elif stale:
        screen.blit(big.render("NO DATA", True, BAD), (990, 18))

    n_pts = state.disp.shape[1]
    xs = np.linspace(TRACE_X0, TRACE_X1, n_pts)
    step = max(1, n_pts // (TRACE_X1 - TRACE_X0))

    for ch in range(n_ch):
        mid = top + lane_h * (ch + 0.5)
        half = (REJECT_UV / 2.0) * px_per_uv
        pygame.draw.rect(screen, BAND,
                         (TRACE_X0, mid - half, TRACE_X1 - TRACE_X0, 2 * half), 1)
        pygame.draw.line(screen, GRID, (TRACE_X0, mid), (TRACE_X1, mid), 1)

        colour = OKABE_ITO[ch % len(OKABE_ITO)]
        ys = mid - np.clip(state.disp[ch] * px_per_uv, -lane_h / 2, lane_h / 2)
        pts = list(zip(xs[::step], ys[::step]))
        if len(pts) > 1:
            pygame.draw.lines(screen, colour, False, pts, 1)

        name = state.labels[ch] if ch < len(state.labels) else f"ch{ch}"
        screen.blit(big.render(name, True, colour), (20, mid - 26))
        dc_mv = state.dc_offset_uv(ch) / 1000.0
        sat = state.saturated(ch)
        screen.blit(font.render(f"DC {dc_mv:+7.2f} mV", True,
                                WARN if abs(dc_mv) > 20 else MUTED),
                    (20, mid - 4))
        screen.blit(font.render("SATURATED" if sat else "in range", True,
                                BAD if sat else GOOD), (20, mid + 16))

    screen.blit(font.render(f"{REJECT_UV:.0f} uV p-p rejection band", True, MUTED),
                (TRACE_X0, bottom + 8))

    # counters
    pct = state.rejection_pct
    col = BAD if pct > 30 else (WARN if pct > 15 else GOOD)
    screen.blit(big.render("Epoch quality", True, FG), (20, 520))
    screen.blit(font.render(f"seen      {state.epochs_seen}", True, FG), (20, 552))
    screen.blit(font.render(f"rejected  {state.epochs_rejected}", True, FG), (20, 574))
    screen.blit(big.render(f"{pct:5.1f} %", True, col), (20, 600))
    if pct > 30 and state.epochs_seen >= 20:
        screen.blit(font.render("ABORT: session too noisy", True, BAD), (20, 636))

    # ERP panel
    ex0, ey0, ew, eh = PANEL_X, 520, 300, 200
    pygame.draw.rect(screen, GRID, (ex0, ey0, ew, eh), 1)
    screen.blit(big.render("Running ERP at Cz", True, FG), (ex0, ey0 - 30))
    mid_y = ey0 + eh / 2
    pygame.draw.line(screen, GRID, (ex0, mid_y), (ex0 + ew, mid_y), 1)
    for label, colour, tag in ((1, OKABE_ITO[0], "target"), (0, OKABE_ITO[1], "non-target")):
        mean = state.erp_mean(label)
        n = state.erp_count.get(label, 0)
        screen.blit(font.render(f"{tag}  n={n}", True, colour),
                    (ex0 + 6, ey0 + 6 + (0 if label == 1 else 20)))
        if mean is None or mean.size < 2:
            continue
        scale = (eh * 0.4) / max(8.0, float(np.max(np.abs(mean))))
        exs = np.linspace(ex0 + 4, ex0 + ew - 4, mean.size)
        eys = mid_y - mean * scale
        pygame.draw.lines(screen, colour, False, list(zip(exs, eys)), 2)

    screen.blit(font.render(
        "display filter is causal and separate from the analysis filter",
        True, MUTED), (PANEL_X, 740))


# ==========================================================================
# Demo source. SIMULATED DATA. Never for results.
# ==========================================================================

class DemoSource:
    def __init__(self, n_ch: int, fs: float, seed: int = 42):
        self.n_ch, self.fs = n_ch, fs
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.last = time.perf_counter()

    def next_chunk(self):
        now = time.perf_counter()
        n = int((now - self.last) * self.fs)
        if n < 1:
            return None
        self.last = now
        t = self.t + np.arange(n) / self.fs
        self.t = t[-1] + 1.0 / self.fs
        out = np.empty((self.n_ch, n))
        for ch in range(self.n_ch):
            out[ch] = (12 * self.rng.standard_normal(n)
                       + 8 * np.sin(2 * np.pi * 10 * t)
                       + 6 * np.sin(2 * np.pi * 50 * t)
                       + 300 * (ch + 1))
        return out


# ==========================================================================
# Main
# ==========================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="P300 speller operator monitor")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--fs", type=float, default=250.0)
    p.add_argument("--window", type=float, default=5.0)
    p.add_argument("--channels", default="Fz,Cz,Pz")
    p.add_argument("--uv-per-div", type=float, default=50.0)
    p.add_argument("--demo", action="store_true",
                   help="SIMULATED DATA. Verifies the panel without hardware.")
    args = p.parse_args(argv)

    import pygame

    labels = [c.strip() for c in args.channels.split(",") if c.strip()]
    state = MonitorState(len(labels), args.fs, args.window, labels)
    demo = DemoSource(len(labels), args.fs) if args.demo else None
    sock = None if args.demo else make_socket(args.port)

    pygame.init()
    pygame.display.set_caption("P300 acquisition monitor")
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
    big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 19, bold=True)
    clock = pygame.time.Clock()

    uv_per_div = args.uv_per_div
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_LEFTBRACKET:
                    uv_per_div = max(5.0, uv_per_div / 1.5)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    uv_per_div = min(2000.0, uv_per_div * 1.5)
                elif ev.key == pygame.K_r:
                    state.reset_counters()

        if demo is not None:
            chunk = demo.next_chunk()
            if chunk is not None:
                state.push(chunk, args.fs)
        else:
            drain(sock, state)

        draw(screen, font, big, state, uv_per_div, args.demo)
        pygame.display.flip()
        clock.tick(15)  # hard cap. Dropping frames is correct behaviour.

    pygame.quit()
    if sock is not None:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
