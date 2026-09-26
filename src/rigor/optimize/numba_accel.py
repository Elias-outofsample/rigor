"""Optional CPU acceleration for the optimiser's composite-score hot path.

Identical results; enable with the ``accel`` extra (``pip install -e ".[accel]"``).

The 5-pillar composite score (``optimize.scoring.score_configs``) reduces an
``(n_configs, n_pillars)`` matrix to one weighted-geometric-mean score *per
config*:

    score[c] = exp( sum_p( w[p] * log(pillar[c, p]) ) / sum_p(w[p]) )

For a large grid search this row reduction is the inner numerical loop that
scales with the number of configurations evaluated. This module provides an
optional Numba-JIT kernel for it **plus a pure-NumPy fallback that runs the
identical algorithm**, so the framework's cross-machine reproducibility is
preserved exactly (mirroring ``rigor.validation.accel``):

    The kernel is a *speed* knob, never a *results* knob. The NumPy fallback and
    the Numba kernel compute the same weighted geomean, in the same accumulation
    order, and agree to the last bit (asserted in ``tests/test_numba_accel.py``).

Dispatch (``weighted_geomean``) is **opt-in**:

  * Default = the NumPy fallback. Continuous Integration ships **no Numba**, so
    CI always exercises the fallback and the default path never changes.
  * The Numba kernel is used only when Numba is importable **and** acceleration
    is enabled — via the ``enable`` argument, or globally through the
    ``RIGOR_OPTIMIZE_ACCEL`` environment variable (``1``/``true``/``yes``/``on``).

Any failure to import or compile Numba silently leaves the fallback in place; the
optimiser never crashes because acceleration was requested.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["weighted_geomean", "numba_available", "accel_enabled", "active_impl"]

try:  # Numba has no type stubs — ``ignore_missing_imports`` (pyproject) covers it.
    import numba  # type: ignore

    _HAS_NUMBA = True
except Exception:  # noqa: BLE001 - absence is the normal CI case, never an error
    _HAS_NUMBA = False

# Env flag that turns the kernel on globally (opt-in). The default-off behaviour
# is what keeps CI (no Numba) on the byte-identical NumPy fallback.
_ENV_FLAG = "RIGOR_OPTIMIZE_ACCEL"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def numba_available() -> bool:
    """Whether the optional Numba accelerator is importable on this machine."""
    return _HAS_NUMBA


def accel_enabled() -> bool:
    """Whether acceleration is requested via the ``RIGOR_OPTIMIZE_ACCEL`` env flag."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


def active_impl(enable: bool | None = None) -> str:
    """Resolve which implementation ``weighted_geomean`` will use: ``"numba"`` or
    ``"numpy"``. ``enable`` overrides the env flag when given; ``None`` defers to
    ``RIGOR_OPTIMIZE_ACCEL``. Numba is used only when it is *both* importable and
    requested — otherwise the NumPy fallback (the default)."""
    want = accel_enabled() if enable is None else enable
    return "numba" if (want and _HAS_NUMBA) else "numpy"


def _geomean_numpy(pillars: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pure-NumPy weighted geometric mean of each row of ``pillars`` (the
    reference algorithm). ``pillars`` is ``(n_configs, n_pillars)`` and strictly
    positive; ``weights`` is ``(n_pillars,)``.

    The column reduction is an explicit *sequential* left-to-right accumulation
    (vectorised across configs) rather than ``np.sum(axis=1)``. ``np.sum`` uses
    pairwise summation, whose accumulation order differs from the Numba kernel's
    plain ``for j`` loop and diverges at the last bit. Accumulating column by
    column here matches the kernel exactly, so the two paths are byte-identical.
    """
    k = pillars.shape[1]
    wsum = 0.0
    for j in range(k):
        wsum += float(weights[j])
    acc = np.zeros(pillars.shape[0], dtype="float64")
    for j in range(k):
        acc += np.log(pillars[:, j]) * weights[j]
    return np.exp(acc / wsum)


if _HAS_NUMBA:

    @numba.njit(cache=True, fastmath=False)  # serial + no fastmath -> deterministic
    def _geomean_numba(pillars, weights):  # pragma: no cover - only with numba present
        # Same maths and the same left-to-right (``for j``) accumulation order as
        # ``_geomean_numpy``'s column-by-column reduction, so the two paths are
        # bit-for-bit identical (asserted in tests/test_numba_accel.py).
        n, k = pillars.shape
        wsum = 0.0
        for j in range(k):
            wsum += weights[j]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(k):
                acc += np.log(pillars[i, j]) * weights[j]
            out[i] = np.exp(acc / wsum)
        return out


def weighted_geomean(
    pillars: np.ndarray, weights: np.ndarray, *, enable: bool | None = None
) -> np.ndarray:
    """Weighted geometric mean of each row of the ``(n_configs, n_pillars)``
    pillar matrix — the optimiser's composite-score hot path.

    Returns an ``(n_configs,)`` float64 array. Optionally accelerated with Numba;
    the NumPy fallback is the default and produces byte-identical results.

    Parameters
    ----------
    pillars:
        ``(n_configs, n_pillars)`` strictly-positive score matrix.
    weights:
        ``(n_pillars,)`` calibration weights.
    enable:
        Force the kernel on/off for this call. ``None`` (default) defers to the
        ``RIGOR_OPTIMIZE_ACCEL`` env flag. The Numba kernel is used only when it is
        also importable; otherwise the NumPy fallback runs.
    """
    pillars = np.ascontiguousarray(pillars, dtype="float64")
    weights = np.ascontiguousarray(weights, dtype="float64")
    if active_impl(enable) == "numba":
        try:
            return _geomean_numba(pillars, weights)
        except Exception:  # noqa: BLE001 - any JIT failure -> identical NumPy fallback
            pass
    return _geomean_numpy(pillars, weights)
