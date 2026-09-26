"""rigor.utils — shared helpers that prevent recurring bug classes.

Sub-modules
-----------
math_utils  Numerical helpers: safe_divide, clip_safe, max_consecutive.
encoding    JSON / display: NumpyEncoder, coerce_float, format_pct, format_number.
align       Index-alignment: align_series, align_frame, align_mask.
"""

from __future__ import annotations

from rigor.utils.align import align_frame, align_mask, align_series
from rigor.utils.encoding import NumpyEncoder, coerce_float, format_number, format_pct
from rigor.utils.math_utils import clip_safe, max_consecutive, safe_divide

__all__ = [
    "safe_divide",
    "clip_safe",
    "max_consecutive",
    "NumpyEncoder",
    "coerce_float",
    "format_pct",
    "format_number",
    "align_series",
    "align_frame",
    "align_mask",
]
