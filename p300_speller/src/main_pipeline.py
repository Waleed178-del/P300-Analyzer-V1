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

The same data path (acquire -> precondition -> epoch -> vectorise) is shared by
both modes via :meth:`P300Pipeline._collect_epochs`, guaranteeing train/test
consistency.

Timing model: the speller and the EEG source share ``time.perf_counter``; the
pipeline simply pulls the continuous buffer after each character and epochs it
against the flash markers that were recorded during presentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

from classifier import (
    NON_TARGET,
    TARGET,
    P300Classifier,
    TrainingReport,
    decode_character_scores,
)
from eeg_acquisition import BaseEEGSource, Simulator, StimulusMarker, build_source
from feature_extraction import epochs_to_matrix
from signal_processing import Epoch, epoch_data, precondition
from speller_matrix import SpellerMatrix


def load_config(path: str) -> dict:
    """Load and return the YAML configuration document as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
            :func:`eeg_acquisition.build_source`.
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

    # ------------------------------------------------------------------ #
    # Shared data path
    # ------------------------------------------------------------------ #
    def _post_roll_s(self) -> float:
        """Seconds to wait after the last flash so its epoch is fully buffered."""
        return float(self.epoch_cfg["tmax_s"]) + 0.2

    def _collect_epochs(self, markers: List[StimulusMarker]) -> List[Epoch]:
        """Acquire, precondition, and epoch the buffer around ``markers``.

        Args:
            markers: The flash markers recorded for the current character.

        Returns:
            Baseline-corrected epochs aligned to each usable marker.
        """
        timestamps, data = self.source.get_buffer()
        if timestamps.size == 0:
            return []
        conditioned = precondition(data, self.fs, self.proc_cfg)
        return epoch_data(
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

        self.source.start()
        self.speller.open()
        try:
            # Allow the acquisition buffer to prime before the first flash.
            time.sleep(1.0)
            for word in words:
                for symbol in word.upper():
                    if symbol not in self.speller.symbol_to_rc:
                        # Skip characters not present in the matrix (e.g. space).
                        continue
                    Xc, yc = self._train_one_character(symbol, n_sequences)
                    if Xc.size:
                        X_parts.append(Xc)
                        y_parts.append(yc)
                    self._inter_character_pause()
        finally:
            self.speller.close()
            self.source.stop()

        if not X_parts:
            raise RuntimeError("Calibration produced no epochs; check acquisition.")

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        classifier = P300Classifier(
            model_type=self.clf_cfg.get("model_type", "lda"),
            lda_shrinkage=self.clf_cfg.get("lda_shrinkage", "auto"),
            svm_c=self.clf_cfg.get("svm_c", 0.1),
            class_weight=self.clf_cfg.get("class_weight", "balanced"),
        )
        report = classifier.fit(X, y)
        classifier.save(self.clf_cfg["model_path"])
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

        epochs = self._collect_epochs(markers)
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

        epochs = self._collect_epochs(markers)
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
        """Flash the decoded symbol as a cue and print progress."""
        try:
            self.speller.show_cue(result.symbol, duration_s=1.0)
        except Exception:
            pass
        print(
            f"[spell] selected '{result.symbol}' "
            f"(row={result.row}, col={result.col}) -> \"{''.join(decoded)}\""
        )

    def _write_output(self, text: str) -> None:
        import os

        path = self.config["session"].get("output_text_path")
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
