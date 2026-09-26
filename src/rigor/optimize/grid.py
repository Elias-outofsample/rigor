"""Grid construction: enumerate, down-sample, and coarsen a ``param_grid``.

The full cartesian product of a ``param_grid()`` can be huge. Two tools keep the
search tractable on modest hardware:

  * ``sample`` — a deterministic cap on the number of configs actually run
    (always keeping the default config), used by the one-pass optimiser.
  * ``decimate`` / ``refine_around`` — the two-pass coarse->fine search: thin
    each axis to a coarse grid first (``decimate``), find the neighbourhood of
    the winner, then re-expand only that neighbourhood (``refine_around``). This
    visits a fraction of the full grid while keeping the span intact.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..validation.accel import sobol_pick

__all__ = ["expand", "sample", "decimate", "refine_around"]

_MIN_COARSE_LEVELS = 3  # keep >=3 levels per axis so curvature stays estimable


def expand(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a ``{param: [values]}`` grid -> list of config dicts."""
    if not grid:
        return []
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(grid[k] for k in keys))]


def sample(combos: list[dict], max_combos: int, *, default: dict, seed: int = 0) -> list[dict]:
    """Down-sample a huge config list with a Sobol low-discrepancy pick (even
    coverage of the grid), always keeping ``default``. Deterministic."""
    if len(combos) <= max_combos:
        return combos
    rest = [c for c in combos if c != default]
    idx = sobol_pick(len(rest), max_combos - 1, seed=seed)
    return [default, *[rest[i] for i in idx]]


def _even_subset(values: list, target_len: int) -> list:
    """Evenly-spaced subset of ``values`` (endpoints kept), preserving span."""
    n = len(values)
    if target_len >= n:
        return list(values)
    if target_len <= 1:
        return [values[0]]
    idx = sorted(set(np.linspace(0, n - 1, target_len).round().astype(int).tolist()))
    return [values[i] for i in idx]


def _grid_size(ranges: dict[str, list]) -> int:
    size = 1
    for v in ranges.values():
        size *= max(len(v), 1)
    return size


def decimate(grid: dict[str, list], max_combos: int) -> dict[str, list]:
    """Coarsen ``grid`` so its product <= ``max_combos`` (Pass 1 of two-pass).

    Thins the *resolution* of each axis (never its span — endpoints are always
    kept), reducing the largest axis first but not below ``_MIN_COARSE_LEVELS``
    until that floor itself is too large. Pass 1 only decides which axes matter;
    ``refine_around`` re-expands the winner's neighbourhood at full resolution.
    """
    dec = {k: list(v) for k, v in grid.items()}
    if _grid_size(dec) <= max_combos:
        return dec
    floors = {k: min(len(v), _MIN_COARSE_LEVELS) for k, v in grid.items()}
    while _grid_size(dec) > max_combos:
        over = [k for k in dec if len(dec[k]) > floors[k]]
        if not over:
            break
        k = max(over, key=lambda kk: len(dec[kk]))
        dec[k] = _even_subset(grid[k], max(floors[k], (len(dec[k]) + 1) // 2))
    while _grid_size(dec) > max_combos:  # extreme grids: go below the floor
        reducible = [k for k in dec if len(dec[k]) > 2]
        if not reducible:
            break
        k = max(reducible, key=lambda kk: len(dec[kk]))
        dec[k] = _even_subset(grid[k], max(2, (len(dec[k]) + 1) // 2))
    return dec


def refine_around(grid: dict[str, list], winner: dict, *, span: int = 1) -> dict[str, list]:
    """Re-expand a fine grid around ``winner`` (Pass 2 of two-pass).

    For each axis, keep the values within ``span`` index positions of the
    winning value in the *full* ordered range — so the second pass searches the
    full-resolution neighbourhood of the coarse winner, not the coarse grid.
    """
    fine: dict[str, list] = {}
    for k, values in grid.items():
        vals = list(values)
        if k not in winner or winner[k] not in vals:
            fine[k] = vals
            continue
        i = vals.index(winner[k])
        lo, hi = max(0, i - span), min(len(vals), i + span + 1)
        fine[k] = vals[lo:hi]
    return fine
