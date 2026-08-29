"""session.py — top-level session orchestration for the P300 Speller.

:class:`P300Pipeline` (in :mod:`main_pipeline`) owns the *signal* flow:
acquisition -> processing -> classification -> stimulus. This module sits one
level above it and owns the *session* flow: deciding whether to run a
calibration or a free-spelling phase, cycling the operator/user through the
characters with clear on-screen prompts, and persisting both a structured
event log and the final decoded text to disk.

The two layers communicate through the pipeline's lightweight event hook
(:meth:`main_pipeline.P300Pipeline._emit`). :class:`SessionManager` registers
itself as the listener, so every lifecycle moment — a calibration character cue,
a decoded selection, completion of a phase — is turned into a human-readable
prompt and a timestamped log record without duplicating any of the numerical
pipeline logic.

Prompts are deliberately non-blocking (printed, not ``input()``-driven) so a
session can run unattended in headless / CI environments while still producing
the same operator-facing narration a clinician would see on a real run.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Optional

from classifier import TrainingReport
from main_pipeline import CharacterResult, P300Pipeline


class SessionLogger:
    """Timestamped logger that mirrors records to stdout and a log file.

    Args:
        log_path: Destination log file. Parent directories are created as
            needed. If ``None``, logging is console-only.
        echo: When ``True`` (default) records are also printed to stdout.
    """

    def __init__(self, log_path: Optional[str], echo: bool = True) -> None:
        self.log_path = log_path
        self.echo = echo
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            # Touch the file and write a session banner.
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== P300 session started {self._stamp()} =====\n")

    @staticmethod
    def _stamp() -> str:
        """Return an ISO-8601 second-resolution UTC timestamp."""
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message: str) -> None:
        """Write one timestamped line to the console and/or the log file."""
        line = f"[{self._stamp()}] {message}"
        if self.echo:
            print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class SessionManager:
    """Conducts calibration and free-spelling sessions end to end.

    The manager constructs (or adopts) a :class:`P300Pipeline`, attaches itself
    as the pipeline's event listener, and exposes two high-level operations:
    :meth:`calibrate` and :meth:`free_spell`. It owns user prompting, character
    cycling oversight, and persistent logging; the pipeline owns the signal math.

    Args:
        config: Parsed configuration document (see ``configs/config.yaml``).
        pipeline: Optional pre-built pipeline (useful for tests). When omitted a
            new :class:`P300Pipeline` is created from ``config``.
        log_path: Optional explicit session-log path. When omitted a timestamped
            log is created next to ``session.output_text_path``.
    """

    def __init__(
        self,
        config: dict,
        pipeline: Optional[P300Pipeline] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline if pipeline is not None else P300Pipeline(config)
        # Route all pipeline lifecycle events through this manager.
        self.pipeline.event_listener = self._on_event

        self.logger = SessionLogger(log_path or self._default_log_path(config))

        # Running record of what has been spelled this session.
        self.spelled_history: List[str] = []

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_log_path(config: dict) -> str:
        """Derive a timestamped log path from the configured output directory."""
        out_text = config.get("session", {}).get(
            "output_text_path", "output/spelled_text.txt"
        )
        out_dir = os.path.dirname(os.path.abspath(out_text)) or "output"
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(out_dir, f"session_{stamp}.log")

    # ------------------------------------------------------------------ #
    # Public phases
    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        words: Optional[List[str]] = None,
        n_sequences: Optional[int] = None,
    ) -> TrainingReport:
        """Run a calibration phase and persist the trained model.

        Args:
            words: Calibration words; defaults to ``session.calibration_words``.
            n_sequences: Override the per-character sequence count.

        Returns:
            The :class:`TrainingReport` produced by fitting the classifier.
        """
        words = words if words is not None else self.config["session"]["calibration_words"]
        self.logger.log(
            "CALIBRATION phase requested for words: " + " ".join(words)
        )
        self.logger.log(
            "Instruct the user: silently COUNT each flash of the highlighted "
            "(green) letter; ignore all other flashes."
        )
        report = self.pipeline.train(words=words, n_sequences=n_sequences)
        self.logger.log(f"CALIBRATION complete -> {report}")
        return report

    def free_spell(
        self,
        n_characters: Optional[int] = None,
        n_sequences: Optional[int] = None,
        simulated_intent: Optional[str] = None,
    ) -> str:
        """Run a free-spelling phase and persist the decoded text.

        Args:
            n_characters: Stop after this many characters (``None`` = until the
                window is closed / interrupted).
            n_sequences: Override the per-character sequence count.
            simulated_intent: Simulator-only target text for unattended runs.

        Returns:
            The decoded text for this phase.
        """
        self.logger.log("SPELLING phase requested.")
        self.logger.log(
            "Instruct the user: focus on and COUNT the flashes of the single "
            "letter they wish to type; a selection is made after each block."
        )
        text = self.pipeline.spell(
            n_characters=n_characters,
            n_sequences=n_sequences,
            simulated_intent=simulated_intent,
        )
        self.spelled_history.append(text)
        self.logger.log(f"SPELLING complete -> {text!r}")
        return text

    # ------------------------------------------------------------------ #
    # Pipeline event handler
    # ------------------------------------------------------------------ #
    def _on_event(self, event_type: str, data: Dict[str, object]) -> None:
        """Translate a pipeline lifecycle event into a prompt + log record.

        This is registered as :attr:`P300Pipeline.event_listener`; the pipeline
        calls it synchronously from the acquisition/stimulus thread context.
        Each branch is intentionally small and side-effect-free beyond logging.
        """
        if event_type == "calibration_start":
            words = data.get("words", [])
            self.logger.log(f"--- calibration started ({len(words)} word(s)) ---")

        elif event_type == "character_cue":
            symbol = data["symbol"]
            word = data.get("word", "")
            self.logger.log(f"CUE: focus on '{symbol}' (word '{word}')")

        elif event_type == "character_trained":
            self.logger.log(
                f"  collected '{data['symbol']}': {data['epochs']} epochs "
                f"({data['targets']} target)"
            )

        elif event_type == "epochs_rejected":
            self.logger.log(f"  artifact rejection: {data['report']}")

        elif event_type == "session_invalid":
            self.logger.log(
                "*** SESSION INVALID *** artifact rejection rate "
                f"{float(data['rejection_rate']):.1%} exceeds the "
                f"{float(data['max_session_rate']):.0%} limit "
                f"({data['n_rejected']}/{data['n_total']} epochs). "
                "Repeat the recording; results from this session must not be "
                "reported."
            )

        elif event_type == "calibration_complete":
            self.logger.log(
                f"model saved to {data.get('model_path')} | {data['report']}"
            )

        elif event_type == "spell_start":
            self.logger.log("--- spelling started ---")

        elif event_type == "character_decoded":
            result = data["result"]
            text = data["text"]
            assert isinstance(result, CharacterResult)
            self.logger.log(
                f"SELECTED '{result.symbol}' (row={result.row}, "
                f"col={result.col}) | text so far: \"{text}\""
            )

        elif event_type == "spell_complete":
            self.logger.log(f"--- spelling finished: \"{data['text']}\" ---")

        else:  # forward-compatible: never drop an unknown event silently
            self.logger.log(f"event:{event_type} {data}")
