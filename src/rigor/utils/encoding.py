"""JSON encoding and display-formatting helpers.

Prevents crashes when serialising numpy arrays / scalars to JSON and provides
consistent display formatting for percentages and numbers across reports.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


def _sanitize(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert numpy / pandas types and replace non-finite floats with None.

    This is a pre-pass so that CPython's C-level float encoder never sees nan/inf
    before we can replace them with JSON null.
    """
    if isinstance(obj, bool):
        # bool is a subclass of int — check before int
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    """JSONEncoder subclass that handles numpy and pandas types cleanly.

    Usage::

        json.dumps(obj, cls=NumpyEncoder)

    Conversions applied:

    * ``numpy.integer``  → ``int``
    * ``numpy.floating`` → ``float`` (non-finite → ``null``)
    * ``numpy.bool_``   → ``bool``
    * ``numpy.ndarray`` → ``list`` (elements go through the same rules)
    * ``pandas.Timestamp`` → ISO-8601 string via ``isoformat()``
    * Python ``float`` nan/inf → ``null``

    Implementation note: CPython's C-level float encoder intercepts nan/inf
    before any Python hook fires, so we pre-sanitize the entire object tree
    in ``encode`` rather than relying on ``default`` or ``iterencode``.
    """

    def encode(self, obj: Any) -> str:  # noqa: ANN401
        """Pre-sanitize *obj* then encode with the standard encoder."""
        return super().encode(_sanitize(obj))

    def iterencode(self, obj: Any, _one_shot: bool = False):  # noqa: ANN001, ANN201
        """Pre-sanitize *obj* then iterencode with the standard encoder."""
        return super().iterencode(_sanitize(obj), _one_shot)

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        # Fallback for any type not caught by _sanitize (e.g. custom objects).
        return super().default(obj)


def coerce_float(v: Any, default: float | None = None) -> float | None:  # noqa: ANN401
    """Parse *v* to float, returning *default* on any failure or non-finite result.

    Accepts Python numbers, numpy scalars, and string-encoded floats.

    Examples
    --------
    >>> coerce_float("1.5")
    1.5
    >>> coerce_float("x", default=None) is None
    True
    >>> coerce_float(float("inf"), default=None) is None
    True
    """
    try:
        result = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return result


def format_pct(x: Any, dp: int = 2) -> str:  # noqa: ANN401
    """Format *x* as a percentage string, e.g. ``"12.34%"``.

    Returns ``"—"`` for None, NaN, or any non-finite value.

    Examples
    --------
    >>> format_pct(0.1234)
    '12.34%'
    >>> format_pct(None)
    '—'
    """
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:.{dp}f}%"


def format_number(x: Any, dp: int = 2) -> str:  # noqa: ANN401
    """Format *x* as a plain decimal string, e.g. ``"1.23"``.

    Returns ``"—"`` for None, NaN, or any non-finite value.

    Examples
    --------
    >>> format_number(1.2345)
    '1.23'
    >>> format_number(float("nan"))
    '—'
    """
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.{dp}f}"
