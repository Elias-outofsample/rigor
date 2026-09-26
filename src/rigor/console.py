"""Console helpers shared by the CLI and the ``python -m rigor.project.*`` entry points."""
from __future__ import annotations

import contextlib
import sys


def force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 (``errors='replace'``).

    A strategy name like ``'… Niño-3.4 → RenaissanceRe'`` — or any report glyph —
    must never crash output on the default Windows console codepage (cp1252). The
    ``rigor`` CLI calls this; the tool modules call it from their own ``main()`` too,
    so running ``python -m rigor.project.audit`` directly is equally safe. Best-effort
    and idempotent.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # best-effort; older/odd streams
            # reconfigure() exists on the real TextIOWrapper at runtime but is
            # absent from the TextIO stub; guarded by suppress() above.
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
