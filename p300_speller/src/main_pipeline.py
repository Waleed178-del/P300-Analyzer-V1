"""main_pipeline.py — end-to-end orchestration of the P300 Speller.

This module wires the components into the two operating modes a clinical speller
needs:

* **Calibration / training** (:meth:`P300Pipeline.train`) — the user is cued to
  attend a sequence of known letters. Because the targets are known, every
  resulting epoch is automatically labelled target / non-target, a supervised
  data set is assembled, and a :class:`~classifier.P300Classifier` is fit and
  persisted.

* **Free spelling** (:meth:`P300Pipeline.spell`) — for each character the matrix
  is flashed, the trained model scores every flash epoch, scores are averaged
  per row and per column across repetitions, and the intersection of the
  best-scoring row and column yields the selected symbol.

The same data path (acquire -> precondition -> epoch -> reject -> vectorise) is
shared by both modes via :meth:`P300Pipeline._collect_epochs`, guaranteeing
train/test consistency. Artifact rejection sits *inside* that shared path, so
the criterion described in the methodology is the criterion the system actually
applies, in both calibration and free spelling.

Timing model: the speller and the EEG source share ``time.perf_counter``; the
pipeline simply pulls the continuous buffer after each character and epochs it
against the flash markers that were recorded during presentation.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from classifier import (
    DEFAULT_LEAKAGE_GATE_AUC,
    NON_TARGET,
    TARGET,
    P300Classifier,
    TrainingReport,
    decode_character_scores,
)
from acquisition import BaseEEGSource, Simulator, StimulusMarker, build_source
from features import epochs_to_matrix
from processing import (
    Epoch,
    RejectionReport,
    epoch_data,
    precondition,
    reject_artifacts_from_config,
)
from stimulus import SpellerMatrix


def load_config(path: str) -> dict:
    """Load and return the YAML configuration document as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class MonitorPublisher:
    """Best-effort UDP publisher feeding the standalone operator monitor.

    The operator display (:mod:`monitor`) must run in its own process: only one
    process may hold the serial port, and redrawing a scrolling trace must never
    compete with the pygame stimulus loop or the serial reader thread for the
    GIL. The two are therefore decoupled by a datagram socket on loopback — the
    pipeline pushes, the monitor listens, and neither blocks the other.

    Every operation is deliberately failure-tolerant: a monitor that is not
    running, a full socket buffer, or an oversized frame must never disturb an
    in-progress recording. Publishing is fire-and-forget by construction (UDP),
    so no acknowledgement is awaited.

    Args:
        host: Destination address (loopback by default).
        port: Destination UDP port.
        enabled: When ``False`` every method is a no-op.
        max_samples: Newest-N samples included per data frame, bounding the
            datagram well below the 65507-byte UDP payload limit.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9750,
        enabled: bool = False,
        max_samples: int = 500,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self.max_samples = int(max_samples)
        self._sock: Optional[socket.socket] = None
        if self.enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setblocking(False)
            except Exception:
                self._sock = None
                self.enabled = False

    def _send(self, payload: dict) -> None:
        """Serialise and emit one datagram, swallowing every transport error."""
        if not self.enabled or self._sock is None:
            return
        try:
            self._sock.sendto(
                json.dumps(payload).encode("utf-8"), (self.host, self.port)
            )
        except Exception:
            # BlockingIOError (full buffer), ENETUNREACH, oversized frame — all
            # are non-events for the recording. Never propagate.
            pass

    def publish_signal(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
        fs: float,
        channel_names: List[str],
    ) -> None:
        """Publish the newest slice of the continuous buffer for display."""
        if not self.enabled or data.shape[0] == 0:
            return
        n = min(self.max_samples, data.shape[0])
        self._send(
            {
                "type": "signal",
                "fs": float(fs),
                "channels": list(channel_names),
                "t0": float(timestamps[-n]),
                # Rounded to 3 dp: display precision, not analysis precision.
                "data": np.round(data[-n:, :], 3).tolist(),
            }
        )

    def publish_quality(self, report: RejectionReport) -> None:
        """Publish the artifact-rejection state of the last epoch batch."""
        if not self.enabled:
            return
        self._send(
            {
                "type": "quality",
                "n_total": report.n_total,
                "n_kept": report.n_kept,
                "n_rejected": report.n_rejected,
                "rejection_rate": report.rejection_rate,
                "threshold_uv": report.threshold_uv,
                "session_valid": report.session_valid,
                "max_ptp_uv": (
                    float(np.max(report.peak_to_peak_uv))
                    if report.peak_to_peak_uv.size
                    else 0.0
                ),
            }
        )

    def publish_event(self, event_type: str, text: str = "") -> None:
        """Publish a short lifecycle annotation (cue, selection, phase change)."""
        self._send({"type": "event", "event": event_type, "text": text})

    def close(self) -> None:
        """Release the socket. Safe to call repeatedly."""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        self.enabled = False


@dataclass
class CharacterResult:
    """Outcome of decoding a single character during free spelling.

    Attributes:
        symbol: The decoded symbol.
        row: Decoded target row index.
        col: Decoded target column index.
        row_scores: Mean decision score per row.
        col_scores: Mean decision score per column.
    """

    symbol: str
    row: int
    col: int
    row_scores: np.ndarray
    col_scores: np.ndarray


class P300Pipeline:
    """High-level controller binding acquisition, stimulus, and classification.

    Args:
        config: Parsed configuration document.
        source: Optional pre-built EEG source (mainly for tests). If ``None`` a
            source is constructed from ``config`` via
            :func:`acquisition.build_source`.
    """

    def __init__(self, config: dict, source: Optional[BaseEEGSource] = None) -> None:
        self.config = config
        self.fs: float = config["acquisition"]["sampling_rate_hz"]
        self.proc_cfg = config["processing"]
        self.feat_cfg = config["features"]
        self.clf_cfg = config["classifier"]
        self.epoch_cfg = self.proc_cfg["epoch"]

        self.source: BaseEEGSource = source if source is not None else build_source(config)
        self.speller = SpellerMatrix(config)

        # Operator monitor feed. Opt-in via the `monitor` config section; when
        # absent or disabled every publish call is a no-op.
        mon_cfg = config.get("monitor") or {}
        self.monitor = MonitorPublisher(
            host=mon_cfg.get("host", "127.0.0.1"),
            port=mon_cfg.get("port", 9750),
            enabled=bool(mon_cfg.get("enabled", False)),
            max_samples=int(mon_cfg.get("max_samples", 500)),
        )

        # Cumulative artifact-rejection tally for the whole session, so the
        # >30% session-invalidity alarm is evaluated over the session rather
        # than per character.
        self.session_epochs_seen = 0
        self.session_epochs_rejected = 0

        # Optional observer invoked with ``(event_type, data_dict)`` at every
        # significant lifecycle point (see :meth:`_emit`). The session manager
        # registers one to drive user prompts and persistent logging. When no
        # listener is attached the pipeline prints a concise default line so the
        # bare CLI / self-test remains informative.
        self.event_listener: Optional[Callable[[str, Dict[str, object]], None]] = None

    # ------------------------------------------------------------------ #
    # Event emission
    # ------------------------------------------------------------------ #
    def _emit(self, event_type: str, **data: object) -> None:
        """Dispatch a lifecycle event to the listener (or a default printer).

        Args:
            event_type: One of ``calibration_start``, ``character_cue``,
                ``character_trained``, ``calibration_complete``, ``spell_start``,
                ``character_decoded``, ``spell_complete``.
            **data: Event-specific payload forwarded verbatim to the listener.
        """
        # Mirror the annotation to the operator monitor (no-op when disabled).
        self.monitor.publish_event(
            event_type, str(data.get("symbol", data.get("text", "")))
        )
        if self.event_listener is not None:
            self.event_listener(event_type, data)
        else:
            self._default_report(event_type, data)

    @staticmethod
    def _default_report(event_type: str, data: Dict[str, object]) -> None:
        """Concise stdout fallback when no listener is registered."""
        if event_type == "character_decoded":
            result = data["result"]  # type: ignore[index]
            text = data["text"]  # type: ignore[index]
            print(
                f"[spell] selected '{result.symbol}' "  # type: ignore[attr-defined]
                f"(row={result.row}, col={result.col}) -> \"{text}\""  # type: ignore[attr-defined]
            )
        elif event_type == "calibration_complete":
            print(f"[train] calibration complete: {data['report']}")
        elif event_type == "epochs_rejected":
            print(f"[quality] {data['report']}")
        elif event_type == "session_invalid":
            print(
                f"[quality] *** SESSION INVALID *** rejection rate "
                f"{float(data['rejection_rate']):.1%} exceeds the "
                f"{float(data['max_session_rate']):.0%} limit "
                f"({data['n_rejected']}/{data['n_total']} epochs). "
                "Repeat the recording; do not report results from this session."
            )

    # ------------------------------------------------------------------ #
    # Shared data path
    # ------------------------------------------------------------------ #
    def _post_roll_s(self) -> float:
        """Seconds to wait after the last flash so its epoch is fully buffered."""
        return float(self.epoch_cfg["tmax_s"]) + 0.2

    def _collect_epochs(
        self, markers: List[StimulusMarker]
    ) -> Tuple[List[Epoch], RejectionReport]:
        """Acquire, precondition, epoch, and artifact-screen around ``markers``.

        Artifact rejection is applied here, in the one path shared by
        calibration and free spelling, so a contaminated epoch can never reach
        the classifier through either route.

        Args:
            markers: The flash markers recorded for the current character.

        Returns:
            ``(epochs, rejection_report)`` where ``epochs`` are the
            baseline-corrected epochs that survived the peak-to-peak criterion.
        """
        timestamps, data = self.source.get_buffer()
        if timestamps.size == 0:
            return [], reject_artifacts_from_config([], self.proc_cfg)[1]

        # The monitor receives the RAW buffer, not the conditioned one: the
        # display applies its own causal filter, and keeping the two paths
        # independent is what allows the claim that no reported result depends
        # on the display filter.
        self.monitor.publish_signal(
            timestamps, data, self.fs, self.source.channel_names
        )
        conditioned = precondition(data, self.fs, self.proc_cfg)

        epochs = epoch_data(
            timestamps=timestamps,
            data=conditioned,
            markers=markers,
            fs=self.fs,
            tmin_s=self.epoch_cfg["tmin_s"],
            tmax_s=self.epoch_cfg["tmax_s"],
            baseline=(
                self.epoch_cfg["baseline_tmin_s"],
                self.epoch_cfg["baseline_tmax_s"],
            ),
        )

        kept, report = reject_artifacts_from_config(epochs, self.proc_cfg)
        self.session_epochs_seen += report.n_total
        self.session_epochs_rejected += report.n_rejected
        self.monitor.publish_quality(report)
        if report.n_rejected:
            self._emit("epochs_rejected", report=report)
        return kept, report

    # ------------------------------------------------------------------ #
    # Session-level data-quality gate
    # ------------------------------------------------------------------ #
    def session_rejection_rate(self) -> float:
        """Fraction of all epochs rejected so far this session (0.0 if none)."""
        if self.session_epochs_seen == 0:
            return 0.0
        return self.session_epochs_rejected / float(self.session_epochs_seen)

    def _max_session_rejection_rate(self) -> float:
        ar = self.proc_cfg.get("artifact_rejection") or {}
        return float(ar.get("max_session_rate", 0.30))

    def _check_session_quality(self) -> bool:
        """Emit the session-invalidity alarm when too much data was rejected.

        Returns:
            ``True`` if the session is within the configured rejection budget.
        """
        rate = self.session_rejection_rate()
        limit = self._max_session_rejection_rate()
        valid = rate <= limit
        if not valid:
            self._emit(
                "session_invalid",
                rejection_rate=rate,
                max_session_rate=limit,
                n_total=self.session_epochs_seen,
                n_rejected=self.session_epochs_rejected,
            )
        return valid

    def _set_simulator_target(self, symbol: Optional[str]) -> None:
        """If running on the Simulator, steer its synthetic attention.

        During calibration this is the known cue. During simulated free spelling
        it lets an offline run produce a decodable signal. With real hardware the
        source is not a Simulator and this is a no-op.
        """
        if not isinstance(self.source, Simulator):
            return
        if symbol is None:
            self.source.set_target(None, None)
            return
        row, col = self.speller.symbol_to_rc[symbol]
        self.source.set_target(row, col)

    # ------------------------------------------------------------------ #
    # Training / calibration
    # ------------------------------------------------------------------ #
    def train(
        self,
        words: Optional[List[str]] = None,
        n_sequences: Optional[int] = None,
    ) -> TrainingReport:
        """Run calibration over known words and fit the classifier.

        Args:
            words: Calibration words; defaults to ``session.calibration_words``.
            n_sequences: Override the per-character sequence count.

        Returns:
            The :class:`TrainingReport` from fitting the model.

        Raises:
            RuntimeError: If no epochs were collected (e.g. acquisition failure).
        """
        words = words if words is not None else self.config["session"]["calibration_words"]
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        group_parts: List[np.ndarray] = []
        # One group per cued character block. Every epoch collected while the
        # user attends a single letter shares drift, impedance, and attentional
        # state, so the block — not the epoch — is the independent unit that
        # cross-validation must split on.
        block_index = 0

        self.source.start()
        self.speller.open()
        try:
            # Allow the acquisition buffer to prime before the first flash.
            time.sleep(1.0)
            self._emit("calibration_start", words=list(words))
            for word in words:
                for symbol in word.upper():
                    if symbol not in self.speller.symbol_to_rc:
                        # Skip characters not present in the matrix (e.g. space).
                        continue
                    self._emit("character_cue", symbol=symbol, word=word)
                    Xc, yc = self._train_one_character(symbol, n_sequences)
                    if Xc.size:
                        X_parts.append(Xc)
                        y_parts.append(yc)
                        group_parts.append(
                            np.full(Xc.shape[0], block_index, dtype=int)
                        )
                        block_index += 1
                        self._emit(
                            "character_trained",
                            symbol=symbol,
                            epochs=int(Xc.shape[0]),
                            targets=int((yc == TARGET).sum()),
                        )
                    self._inter_character_pause()
        finally:
            self.speller.close()
            self.source.stop()

        if not X_parts:
            raise RuntimeError("Calibration produced no epochs; check acquisition.")

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        groups = np.concatenate(group_parts)

        # Data-quality gate before any model is fit or saved.
        self._check_session_quality()

        classifier = P300Classifier(
            model_type=self.clf_cfg.get("model_type", "lda"),
            lda_shrinkage=self.clf_cfg.get("lda_shrinkage", "auto"),
            svm_c=self.clf_cfg.get("svm_c", 0.1),
            class_weight=self.clf_cfg.get("class_weight", "balanced"),
        )
        report = classifier.fit(
            X,
            y,
            groups=groups,
            cv_folds=int(self.clf_cfg.get("cv_folds", 5)),
            leakage_gate_auc=float(
                self.clf_cfg.get("leakage_gate_auc", DEFAULT_LEAKAGE_GATE_AUC)
            ),
        )
        classifier.save(self.clf_cfg["model_path"])
        self._emit(
            "calibration_complete",
            report=report,
            model_path=self.clf_cfg["model_path"],
            rejection_rate=self.session_rejection_rate(),
        )
        return report

    def _train_one_character(
        self, symbol: str, n_sequences: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flash one cued character and return its labelled feature matrix."""
        target_row, target_col = self.speller.symbol_to_rc[symbol]

        self._set_simulator_target(symbol)
        self.source.clear_markers()
        self.speller.show_cue(symbol)

        markers = self.speller.run_character_flashes(self.source, n_sequences)
        time.sleep(self._post_roll_s())

        epochs, _rejection = self._collect_epochs(markers)
        if not epochs:
            return np.empty((0, 0)), np.empty((0,))

        X = epochs_to_matrix(epochs, self.fs, self.feat_cfg)
        y = np.array(
            [
                TARGET
                if (
                    (ep.marker.kind == "row" and ep.marker.index == target_row)
                    or (ep.marker.kind == "col" and ep.marker.index == target_col)
                )
                else NON_TARGET
                for ep in epochs
            ],
            dtype=int,
        )
        return X, y

    # ------------------------------------------------------------------ #
    # Free spelling
    # ------------------------------------------------------------------ #
    def spell(
        self,
        n_characters: Optional[int] = None,
        n_sequences: Optional[int] = None,
        simulated_intent: Optional[str] = None,
    ) -> str:
        """Free-spell characters until stopped, returning the decoded text.

        Args:
            n_characters: Stop after this many characters. ``None`` -> run until
                interrupted (Ctrl-C / window close).
            n_sequences: Override the per-character sequence count.
            simulated_intent: When running on the Simulator, the text a synthetic
                user "intends", consumed one character at a time so an offline run
                decodes a meaningful string. Ignored on real hardware.

        Returns:
            The decoded text.
        """
        classifier = P300Classifier.load(self.clf_cfg["model_path"])
        decoded: List[str] = []

        self.source.start()
        self.speller.open()
        try:
            time.sleep(1.0)
            self._emit("spell_start")
            idx = 0
            while True:
                if n_characters is not None and idx >= n_characters:
                    break
                intent_symbol = self._intent_for_index(simulated_intent, idx)
                self._set_simulator_target(intent_symbol)

                result = self._spell_one_character(classifier, n_sequences)
                if result is None:
                    break
                decoded.append(result.symbol)
                self._announce_selection(result, decoded)
                self._inter_character_pause()
                idx += 1
                if simulated_intent is not None and idx >= len(simulated_intent):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.speller.close()
            self.source.stop()

        text = "".join(decoded)
        self._write_output(text)
        self._emit("spell_complete", text=text)
        return text

    @staticmethod
    def _intent_for_index(simulated_intent: Optional[str], idx: int) -> Optional[str]:
        """Return the simulated target symbol for character ``idx`` (or None)."""
        if simulated_intent is None or idx >= len(simulated_intent):
            return None
        return simulated_intent[idx].upper()

    def _spell_one_character(
        self, classifier: P300Classifier, n_sequences: Optional[int]
    ) -> Optional[CharacterResult]:
        """Flash, score, and decode one character."""
        self.source.clear_markers()
        markers = self.speller.run_character_flashes(self.source, n_sequences)
        time.sleep(self._post_roll_s())

        epochs, _rejection = self._collect_epochs(markers)
        if not epochs:
            return None

        X = epochs_to_matrix(epochs, self.fs, self.feat_cfg)
        scores = classifier.decision_scores(X)

        # Accumulate mean decision score per row and per column.
        row_sum = np.zeros(self.speller.n_rows, dtype=np.float64)
        row_cnt = np.zeros(self.speller.n_rows, dtype=np.float64)
        col_sum = np.zeros(self.speller.n_cols, dtype=np.float64)
        col_cnt = np.zeros(self.speller.n_cols, dtype=np.float64)
        for ep, score in zip(epochs, scores):
            if ep.marker.kind == "row":
                row_sum[ep.marker.index] += score
                row_cnt[ep.marker.index] += 1
            else:
                col_sum[ep.marker.index] += score
                col_cnt[ep.marker.index] += 1

        row_scores = row_sum / np.maximum(row_cnt, 1.0)
        col_scores = col_sum / np.maximum(col_cnt, 1.0)

        row, col = decode_character_scores(row_scores, col_scores)
        symbol = self.speller.symbols[row][col]
        return CharacterResult(
            symbol=symbol,
            row=row,
            col=col,
            row_scores=row_scores,
            col_scores=col_scores,
        )

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def _inter_character_pause(self) -> None:
        pause = self.config["speller"].get("inter_character_pause_s", 2.0)
        if pause > 0:
            time.sleep(pause)

    def _announce_selection(self, result: CharacterResult, decoded: List[str]) -> None:
        """Flash the decoded symbol as visual feedback and emit a decode event."""
        try:
            self.speller.show_cue(result.symbol, duration_s=1.0)
        except Exception:
            # Visual feedback is best-effort; never let a render glitch abort
            # an in-progress spelling session.
            pass
        self._emit("character_decoded", result=result, text="".join(decoded))

    def _write_output(self, text: str) -> None:
        import os

        path = self.config["session"].get("output_text_path")
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
