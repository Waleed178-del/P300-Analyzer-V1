"""speller_matrix.py — the visual 6x6 Row/Column stimulus presenter.

Implements the Farwell & Donchin (1988) Row-Column paradigm with :mod:`pygame`.
A grid of symbols is drawn dim; to elicit P300s an entire random row or entire
random column is briefly intensified ("flashed"). The user attends (counts) the
flashes overlapping their intended symbol, so target flashes — and only target
flashes — evoke a P300.

Responsibilities:

* Render a 6x6 matrix of A-Z, 1-9 and ``_``.
* Drive the flash schedule: every "sequence" flashes all 6 rows and 6 columns
  exactly once, in a randomised order that never repeats the same stimulus
  back-to-back.
* Timestamp each flash with :func:`time.perf_counter` at the instant the
  intensified frame is presented, and push a :class:`~eeg_acquisition.Stimulus\
Marker` to the EEG source so the neural epoch can be aligned to it.
* Support a *headless* mode (SDL dummy video driver) so the full timing
  pipeline runs on a server / CI without a physical display.

The marker code convention (shared with the Arduino firmware) is:

* rows ``0..n_rows-1``    -> codes ``1 .. n_rows``
* columns ``0..n_cols-1`` -> codes ``n_rows+1 .. n_rows+n_cols``
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional, Tuple

from eeg_acquisition import BaseEEGSource, StimulusMarker


def _precise_sleep(duration_s: float) -> None:
    """Sleep for ``duration_s`` with sub-millisecond accuracy.

    ``time.sleep`` alone is too coarse for stimulus timing. We sleep most of the
    interval, then busy-wait the remainder against ``perf_counter``.
    """
    if duration_s <= 0:
        return
    end = time.perf_counter() + duration_s
    coarse = duration_s - 0.0015
    if coarse > 0:
        time.sleep(coarse)
    while time.perf_counter() < end:
        pass


class SpellerMatrix:
    """Renders the speller grid and runs the row/column flash schedule.

    Args:
        config: Parsed ``config.yaml`` (the whole document).
    """

    def __init__(self, config: dict) -> None:
        sp = config["speller"]
        self.config = config
        self.n_rows: int = sp["rows"]
        self.n_cols: int = sp["cols"]
        self.symbols: List[List[str]] = sp["symbols"]
        self._validate_symbols()

        self.flash_duration_s: float = sp["flash_duration_ms"] / 1000.0
        self.isi_s: float = sp["inter_stimulus_interval_ms"] / 1000.0
        self.n_sequences_default: int = sp["n_sequences"]

        self.width: int = sp["window_width_px"]
        self.height: int = sp["window_height_px"]
        self.bg_color = tuple(sp["background_color"])
        self.symbol_color = tuple(sp["symbol_color"])
        self.highlight_color = tuple(sp["highlight_color"])
        self.font_size: int = sp["font_size_px"]
        self.headless: bool = sp.get("headless", False)

        self._rng = random.Random(config.get("session", {}).get("random_seed", 7))

        # Symbol -> (row, col) reverse lookup.
        self.symbol_to_rc: Dict[str, Tuple[int, int]] = {}
        for r, row in enumerate(self.symbols):
            for c, sym in enumerate(row):
                self.symbol_to_rc[sym] = (r, c)

        # pygame handles, initialised lazily in :meth:`open`.
        self._pygame = None
        self._screen = None
        self._font = None
        self._cell_w = self.width / self.n_cols
        self._cell_h = self.height / self.n_rows
        self._opened = False

    # -- validation --------------------------------------------------------- #
    def _validate_symbols(self) -> None:
        if len(self.symbols) != self.n_rows:
            raise ValueError("symbols row count does not match speller.rows")
        for row in self.symbols:
            if len(row) != self.n_cols:
                raise ValueError("symbols column count does not match speller.cols")

    # -- marker code helpers ------------------------------------------------ #
    def _row_code(self, index: int) -> int:
        return index + 1

    def _col_code(self, index: int) -> int:
        return self.n_rows + index + 1

    # -- lifecycle ---------------------------------------------------------- #
    def open(self) -> None:
        """Initialise pygame and create the rendering surface. Idempotent."""
        if self._opened:
            return
        if self.headless:
            # Must be set before pygame.display is imported/initialised.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported lazily so the dummy driver env var takes effect

        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("P300 Speller")
        self._screen = pygame.display.set_mode((self.width, self.height))
        self._font = pygame.font.SysFont("consolas,monospace", self.font_size, bold=True)
        self._opened = True
        self.draw_grid()

    def close(self) -> None:
        """Tear down pygame."""
        if self._opened and self._pygame is not None:
            self._pygame.quit()
        self._opened = False
        self._screen = None
        self._font = None

    def __enter__(self) -> "SpellerMatrix":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- rendering ---------------------------------------------------------- #
    def _pump_events(self) -> None:
        """Service the OS event queue so the window stays responsive."""
        if self._pygame is None:
            return
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                raise KeyboardInterrupt("Speller window closed by user")

    def draw_grid(
        self,
        highlight: Optional[Tuple[str, int]] = None,
        cue: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Render the full grid.

        Args:
            highlight: ``("row", idx)`` or ``("col", idx)`` to intensify, or
                ``None`` for the resting (all-dim) frame.
            cue: ``(row, col)`` of a symbol to mark in green (training cue), or
                ``None``.
        """
        if not self._opened:
            return
        pygame = self._pygame
        self._screen.fill(self.bg_color)

        hl_kind, hl_index = (highlight if highlight is not None else (None, -1))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                intensified = (hl_kind == "row" and r == hl_index) or (
                    hl_kind == "col" and c == hl_index
                )
                if cue is not None and cue == (r, c):
                    color = (0, 220, 0)  # green calibration cue
                elif intensified:
                    color = self.highlight_color
                else:
                    color = self.symbol_color

                glyph = self._font.render(self.symbols[r][c], True, color)
                cx = int(c * self._cell_w + self._cell_w / 2)
                cy = int(r * self._cell_h + self._cell_h / 2)
                rect = glyph.get_rect(center=(cx, cy))
                self._screen.blit(glyph, rect)

        pygame.display.flip()

    # -- cueing (training) -------------------------------------------------- #
    def show_cue(self, symbol: str, duration_s: float = 1.5) -> None:
        """Highlight ``symbol`` in green to tell the user where to focus.

        Args:
            symbol: The target symbol for this calibration character.
            duration_s: How long to display the cue.
        """
        if symbol not in self.symbol_to_rc:
            raise ValueError(f"Symbol {symbol!r} is not in the speller matrix")
        rc = self.symbol_to_rc[symbol]
        self.draw_grid(cue=rc)
        self._pump_events()
        _precise_sleep(duration_s)
        self.draw_grid()
        _precise_sleep(0.5)  # brief blank-ish settle before flashing begins

    # -- flash schedule ----------------------------------------------------- #
    def _build_sequence_order(self) -> List[Tuple[str, int]]:
        """Return one randomised pass over all rows and columns.

        Produces ``n_rows + n_cols`` stimuli, each row and column exactly once,
        shuffled so no stimulus repeats consecutively across the boundary of two
        sequences is avoided by the caller.
        """
        stimuli: List[Tuple[str, int]] = [("row", r) for r in range(self.n_rows)]
        stimuli += [("col", c) for c in range(self.n_cols)]
        self._rng.shuffle(stimuli)
        return stimuli

    def flash_once(
        self, kind: str, index: int, eeg_source: BaseEEGSource
    ) -> StimulusMarker:
        """Intensify one row/column, timestamp it, and notify the EEG source.

        The marker timestamp is captured immediately after ``display.flip``
        returns, i.e. as close as possible to the photons reaching the user.

        Args:
            kind: ``"row"`` or ``"col"``.
            index: Row/column index.
            eeg_source: The active EEG source to receive the marker.

        Returns:
            The :class:`StimulusMarker` that was pushed.
        """
        code = self._row_code(index) if kind == "row" else self._col_code(index)

        self._pump_events()
        self.draw_grid(highlight=(kind, index))
        onset = time.perf_counter()
        marker = StimulusMarker(code=code, kind=kind, index=index, perf_time=onset)
        eeg_source.push_marker(marker)

        _precise_sleep(self.flash_duration_s)

        self.draw_grid()  # return to resting frame
        _precise_sleep(self.isi_s)
        return marker

    def run_character_flashes(
        self,
        eeg_source: BaseEEGSource,
        n_sequences: Optional[int] = None,
    ) -> List[StimulusMarker]:
        """Run the full flash schedule for a single character selection.

        Args:
            eeg_source: The active EEG source.
            n_sequences: Number of full row+column passes; defaults to the
                configured value.

        Returns:
            All :class:`StimulusMarker` objects emitted, in presentation order.
        """
        n = n_sequences if n_sequences is not None else self.n_sequences_default
        emitted: List[StimulusMarker] = []
        last_stimulus: Optional[Tuple[str, int]] = None

        for _ in range(n):
            order = self._build_sequence_order()
            # Avoid an immediate repeat across the sequence boundary.
            if last_stimulus is not None and order and order[0] == last_stimulus:
                order.append(order.pop(0))
            for kind, index in order:
                emitted.append(self.flash_once(kind, index, eeg_source))
                last_stimulus = (kind, index)

        return emitted
