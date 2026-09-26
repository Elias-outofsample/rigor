"""Compute backend — make the bootstrap-heavy stages faster, without changing results.

The data-snooping tests (White / Hansen / StepM) and the falsification noise test
spend almost all their time resampling: thousands of stationary-bootstrap passes,
each a gather-and-mean over the returns. This module accelerates that, with a hard
rule that protects the framework's cross-machine reproducibility:

    The random draws are generated ONCE with deterministic NumPy. The backends
    (Numba CPU, CUDA GPU) only accelerate the *arithmetic reduction* over those
    fixed draws — so the numbers are identical whether or not Numba/CUDA is
    installed. The backend is a speed knob, never a results knob.

This is the safe version of what the prior research library did with ``numba
prange`` (which was non-deterministic across machines because the parallel
reduction order varied). Here the reduction is order-independent (a mean), and
the indices are fixed up front, so every backend agrees to the last bit.

  * ``numpy``  — always available (vectorised, low-dependency).
  * ``numba``  — JIT-compiled serial kernel; avoids the big temporary and the
    Python per-bootstrap loop. Used automatically when ``numba`` is importable.
  * ``cuda``   — CuPy batched reduction on a GPU; used when ``cupy`` imports and
    a device is present. Falls back to CPU otherwise (most laptops have no GPU —
    that is expected, not an error).

Also provides **Sobol** quasi-random sampling (``sobol_pick``): when a grid is too
big to evaluate in full, a low-discrepancy sequence covers the space more evenly
than uniform random, so the sample is more representative for the same budget.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "active_backend", "backends_available", "bootstrap_index_batch",
    "bootstrap_column_means", "batched_sharpes", "sobol_pick",
]

try:
    import numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # noqa: BLE001
    _HAS_NUMBA = False

try:  # cupy + an actual device
    import cupy as _cp  # type: ignore
    _HAS_CUDA = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    _cp = None
    _HAS_CUDA = False


def backends_available() -> dict:
    """Which backends are usable on this machine."""
    return {"numpy": True, "numba": _HAS_NUMBA, "cuda": _HAS_CUDA}


def active_backend(prefer: str = "auto") -> str:
    """Resolve the backend to use. ``auto`` picks cuda > numba > numpy."""
    if prefer == "numpy":
        return "numpy"
    if prefer == "cuda":
        return "cuda" if _HAS_CUDA else ("numba" if _HAS_NUMBA else "numpy")
    if prefer == "numba":
        return "numba" if _HAS_NUMBA else "numpy"
    return "cuda" if _HAS_CUDA else ("numba" if _HAS_NUMBA else "numpy")


def bootstrap_index_batch(n: int, n_boot: int, block_size: float, seed: int) -> np.ndarray:
    """Deterministic, fully-vectorised stationary bootstrap (Politis-Romano 1994).

    Returns an ``(n_boot, n)`` int array of resampling indices. Geometric block
    lengths via a per-step restart probability ``1/block_size``; between restarts
    the index increments by 1 (mod n). Generated with NumPy's default_rng so it
    is identical on every machine and every backend.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / block_size
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.broadcast_to(np.arange(n), (n_boot, n))
    last_restart = np.maximum.accumulate(np.where(restart, pos, -1), axis=1)
    base = np.take_along_axis(starts, last_restart, axis=1)
    return (base + (pos - last_restart)) % n


if _HAS_NUMBA:
    @numba.njit(cache=True, fastmath=False)  # serial -> deterministic
    def _means_numba(excess, idx):  # pragma: no cover - exercised only with numba
        n_boot, n = idx.shape
        k = excess.shape[1]
        out = np.zeros((n_boot, k))
        for b in range(n_boot):
            for i in range(n):
                r = idx[b, i]
                for j in range(k):
                    out[b, j] += excess[r, j]
            for j in range(k):
                out[b, j] /= n
        return out


def _means_cupy(excess, idx):  # pragma: no cover - exercised only with a GPU
    ex = _cp.asarray(excess)
    out = _cp.empty((idx.shape[0], excess.shape[1]))
    chunk = 64
    for s in range(0, idx.shape[0], chunk):
        ix = _cp.asarray(idx[s:s + chunk])
        out[s:s + chunk] = ex[ix].mean(axis=1)
    return _cp.asnumpy(out)


def bootstrap_column_means(
    excess: np.ndarray, n_boot: int, *, block_size: float = 10.0, seed: int = 42,
    backend: str = "auto",
) -> np.ndarray:
    """``(n_boot, k)`` array of stationary-bootstrap column means of ``excess``.

    The per-bootstrap means are the shared primitive of every snooping test. The
    indices are deterministic; only the reduction is accelerated, so the result
    is backend-independent.
    """
    excess = np.ascontiguousarray(excess, dtype="float64")
    n = excess.shape[0]
    idx = bootstrap_index_batch(n, n_boot, block_size, seed)
    be = active_backend(backend)
    if be == "cuda":
        return _means_cupy(excess, idx)
    if be == "numba":
        return _means_numba(excess, idx)
    return np.stack([excess[idx[b]].mean(axis=0) for b in range(n_boot)])


def batched_sharpes(series_2d: np.ndarray, ppy: int) -> np.ndarray:
    """Annualised Sharpe of each row of a ``(m, n)`` matrix (vectorised)."""
    mu = series_2d.mean(axis=1)
    sd = series_2d.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mu / sd * np.sqrt(ppy), 0.0)


def sobol_pick(n_total: int, n_pick: int, *, seed: int = 0) -> np.ndarray:
    """Low-discrepancy indices in ``[0, n_total)`` (Sobol) — even coverage of a
    big grid for a fixed budget. Falls back to evenly-spaced if SciPy's QMC or
    the request is degenerate."""
    if n_pick >= n_total:
        return np.arange(n_total)
    try:
        import warnings

        from scipy.stats import qmc
        with warnings.catch_warnings():  # "n not a power of 2" balance note — fine here
            warnings.simplefilter("ignore")
            pts = qmc.Sobol(d=1, scramble=True, seed=seed).random(n_pick).ravel()
        idx = np.unique((pts * n_total).astype(int))
        # top up any collisions deterministically
        if len(idx) < n_pick:
            extra = np.setdiff1d(np.linspace(0, n_total - 1, n_pick).astype(int), idx)
            idx = np.union1d(idx, extra[: n_pick - len(idx)])
        return np.sort(idx[:n_pick])
    except Exception:  # noqa: BLE001
        return np.linspace(0, n_total - 1, n_pick).astype(int)
