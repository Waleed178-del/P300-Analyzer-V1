"""Pytest configuration: make the ``src/`` package importable in tests.

The project keeps its modules under ``src/`` and imports them by bare name
(``import processing``) at runtime via the ``run.py`` path shim. Tests need the
same path setup, so this conftest — auto-discovered by pytest — prepends
``src/`` to ``sys.path`` before any test module is collected.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
