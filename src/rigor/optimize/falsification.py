"""Falsification — attack the edge, try to make it disappear.

Complements the distribution-level overfit tests (PBO, DSR, snooping) by probing
whether the edge survives transformations that destroy exploitable structure
while preserving benign statistical features. Ported faithfully from the prior
research library (Theiler et al. 1992 surrogate data).

  * ``noise_injection_test`` — add Gaussian noise of std ``k·σ`` to the returns
    and watch the Sharpe decay. A robust edge degrades gracefully; an overfit
    one collapses. ``fragility`` = relative Sharpe loss per unit noise. Always
    runs (returns-level).
  * ``surrogate_price_test`` — phase-randomise the *price* spectrum (keeps the
    autocorrelation/trend spectrum, destroys deterministic structure), re-run
    the strategy on each surrogate path, and compare Sharpe. p = fraction of
    surrogates beating the real Sharpe. Needs a ``resimulate_fn(prices)->returns``
    hook; skipped when the optimiser has no such hook (same as the source — it
    is degenerate on returns, so it never runs returns-level).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import metrics as _m
from ..validation.accel import batched_sharpes

__all__ = [
    "phase_randomize", "noise_injection_test", "surrogate_price_test",
    "run_falsification",
]


def _clean(arr) -> np.ndarray:
    a = np.asarray(arr.values if isinstance(arr, pd.Series) else arr, dtype=np.float64)
    return a[np.isfinite(a)]


def _sharpe(returns, ppy: int) -> float:
    r = _clean(returns)
    return _m.compute_sharpe(r, ppy) if len(r) >= 2 else 0.0


def phase_randomize(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One phase-randomised surrogate preserving the power spectrum (Theiler 1992).

    rfft -> keep magnitudes, replace interior phases with uniform random angles
    (DC and, for even length, Nyquist stay real) -> irfft. Same autocorrelation
    as the input but no higher-order structure; mean and variance preserved.
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    spectrum = np.fft.rfft(x)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n)


def noise_injection_test(
    returns, ppy: int, *, noise_mults: tuple[float, ...] = (0.25, 0.5, 1.0),
    n_trials: int = 200, seed: int = 42,
) -> dict:
    """Add Gaussian noise of std ``k·σ(returns)`` and measure Sharpe decay.

    ``fragility`` = relative Sharpe loss per unit noise at the largest level;
    higher means more brittle to small perturbations / estimation error.
    """
    r = _clean(returns)
    n = len(r)
    if n < 32:
        return {"fragility": float("nan"), "base_sharpe": 0.0, "decay_curve": {}, "n_obs": n}
    base = _m.compute_sharpe(r, ppy)
    sigma = float(np.std(r, ddof=1))
    rng = np.random.default_rng(seed)
    decay: dict[float, float] = {}
    for k in noise_mults:  # vectorised: all trials at once -> batched Sharpe
        noisy = r[None, :] + rng.normal(0.0, k * sigma, size=(n_trials, n))
        decay[float(k)] = float(np.mean(batched_sharpes(noisy, ppy)))
    k_max = max(noise_mults)
    fragility = float(max((base - decay[float(k_max)]) / base / k_max, 0.0)) if base > 1e-9 \
        else float("nan")
    return {"base_sharpe": float(base), "decay_curve": decay, "fragility": fragility,
            "n_trials": int(n_trials), "n_obs": int(n)}


def surrogate_price_test(
    prices, resimulate_fn: Callable[[np.ndarray], np.ndarray], ppy: int, *,
    n_surrogates: int = 200, seed: int = 42,
) -> dict:
    """Phase-randomised surrogate-PRICE test: re-run the strategy on price paths
    with the real spectrum but randomised phases. Low p => the edge does NOT
    survive on structureless same-spectrum data => genuine."""
    p = _clean(prices)
    if len(p) < 32 or np.any(p <= 0):
        return {"p_value": 1.0, "is_significant": False, "n_obs": int(len(p)),
                "error": "need >= 32 positive prices"}
    log_ret = np.diff(np.log(p))
    p0 = float(p[0])
    try:
        observed = _sharpe(resimulate_fn(p), ppy)
    except Exception as exc:  # noqa: BLE001
        return {"p_value": 1.0, "is_significant": False, "error": f"resimulate failed: {exc}"}
    rng = np.random.default_rng(seed)
    surr = np.full(n_surrogates, np.nan)
    count = valid = 0
    for i in range(n_surrogates):
        sr = phase_randomize(log_ret, rng)
        path = p0 * np.exp(np.cumsum(np.concatenate([[0.0], sr])))
        try:
            sh = _sharpe(resimulate_fn(path), ppy)
        except Exception:  # noqa: BLE001
            continue
        surr[i] = sh
        valid += 1
        count += sh >= observed
    if valid < 5:
        return {"p_value": 1.0, "is_significant": False, "observed_sharpe": float(observed),
                "n_surrogates": valid, "error": "too few valid surrogates"}
    finite = surr[np.isfinite(surr)]
    p_value = (count + 1) / (valid + 1)
    return {"p_value": float(p_value), "is_significant": bool(p_value < 0.05),
            "observed_sharpe": float(observed), "surrogate_sharpe_mean": float(np.mean(finite)),
            "surrogate_sharpe_p95": float(np.percentile(finite, 95)),
            "n_surrogates": int(valid), "n_obs": int(len(p))}


def run_falsification(
    returns, ppy: int, *, prices=None, resimulate_fn: Callable | None = None,
    n_surrogates: int = 200, n_noise_trials: int = 200, seed: int = 42,
    fragility_max: float = 0.8,
) -> dict:
    """Falsification battery. ``noise_injection`` always runs; the surrogate-price
    test only runs when both ``prices`` and ``resimulate_fn`` are supplied."""
    noise = noise_injection_test(returns, ppy, n_trials=n_noise_trials, seed=seed)
    out: dict = {"noise_injection": noise}
    frag = noise.get("fragility")
    if prices is not None and resimulate_fn is not None:
        fft = surrogate_price_test(prices, resimulate_fn, ppy, n_surrogates=n_surrogates, seed=seed)
        out["surrogate_price"] = fft
        out["suspect"] = bool(
            (frag is not None and np.isfinite(frag) and frag >= fragility_max)
            or fft.get("p_value", 1.0) > 0.05)
    else:
        out["surrogate_price"] = {"skipped": "no prices/resimulate_fn supplied"}
        out["suspect"] = bool(frag is not None and np.isfinite(frag) and frag >= fragility_max) \
            if (frag is not None and np.isfinite(frag)) else None
    return out
