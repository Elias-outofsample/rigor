"""Kelly-criterion position sizing (#18).

Optimal-growth bet fractions for log-utility maximisation, in five flavours that
trade growth against estimation risk:

  * ``kelly_full`` — the classic discrete bet ``f* = W - (1-W)/R`` (Kelly 1956).
  * ``kelly_half`` / ``kelly_fractional`` — the textbook risk-reduced multiples;
    half-Kelly keeps ~75% of the growth for ~50% of the variance.
  * ``kelly_continuous`` — the continuous-return form ``f* = mu / sigma^2`` (the
    Merton fraction), estimated from a return series.
  * ``kelly_multi_asset`` — the vector solution ``f* = Sigma^{-1} mu`` (Thorp
    2006), the per-asset growth-optimal allocation.
  * ``kelly_shrinkage`` — ``kelly_multi_asset`` shrunk toward the naive estimate
    by the Kan & Zhou (2007) optimal factor to bleed off in-sample estimation
    error; the plug-in ``Sigma^{-1} mu`` over-bets badly when ``T`` is small.

Pure NumPy + stdlib — deterministic on every machine (no Numba, no threads, no
RNG, no optional C-extension dependencies). Every function guards its degenerate
inputs (non-positive variance, singular covariance, too-few observations) and
returns a finite value (or a zero vector) rather than ``NaN`` or an exception, so
a sizing call never blows up a backtest. Long-only clipping (``f* >= 0``) is
applied to the discrete ``kelly_full`` only, matching the source convention; the
continuous and vector forms are returned signed (a negative fraction is a short).

References
----------
Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System
Technical Journal*.
Thorp, E. O. (2006). The Kelly Criterion in Blackjack, Sports Betting, and the
Stock Market. *Handbook of Asset and Liability Management*.
Kan, R. & Zhou, G. (2007). Optimal Portfolio Choice with Parameter Uncertainty.
*Journal of Financial and Quantitative Analysis*, 42(3), 621-656.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "kelly_full",
    "kelly_half",
    "kelly_fractional",
    "kelly_continuous",
    "kelly_multi_asset",
    "kelly_shrinkage",
]


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    return r[np.isfinite(r)]


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """``a / b`` guarded against a zero/non-finite denominator or result."""
    if b == 0 or not np.isfinite(b):
        return default
    result = a / b
    return result if np.isfinite(result) else default


def kelly_full(win_rate: float, payoff_ratio: float) -> float:
    """Classic discrete Kelly fraction ``f* = W - (1-W)/R`` (Kelly 1956).

    ``win_rate`` ``W`` is the win probability; ``payoff_ratio`` ``R`` is the
    win/loss size ratio (won amount per unit risked). Long-only: a negative
    optimum (no edge) is clipped to ``0.0``. A non-positive payoff ratio returns
    ``0.0``.
    """
    if payoff_ratio <= 0 or not np.isfinite(payoff_ratio):
        return 0.0
    w = float(np.clip(win_rate, 0.0, 1.0))
    f = w - (1.0 - w) / payoff_ratio
    return float(max(f, 0.0))


def kelly_half(win_rate: float, payoff_ratio: float) -> float:
    """Half-Kelly: ``0.5 * kelly_full`` (textbook variance reduction)."""
    return 0.5 * kelly_full(win_rate, payoff_ratio)


def kelly_fractional(win_rate: float, payoff_ratio: float, fraction: float = 0.5) -> float:
    """Fractional Kelly: ``fraction * kelly_full``.

    ``fraction`` scales the full Kelly bet (``0.5`` = half-Kelly, ``0.25`` =
    quarter-Kelly). It is clamped to ``[0, 1]`` so the result never exceeds the
    full-Kelly bet or flips sign.
    """
    frac = float(np.clip(fraction, 0.0, 1.0))
    return frac * kelly_full(win_rate, payoff_ratio)


def kelly_continuous(returns) -> float:
    """Continuous Kelly fraction ``f* = mu / sigma^2`` (the Merton fraction).

    Optimal leverage for log-utility on a return series: the sample mean over the
    sample variance (``ddof=1``). Returned signed (negative = a short tilt).
    Fewer than two finite observations, or zero variance, returns ``0.0``.
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    # A constant series has zero variance, but float ``var`` can return a tiny
    # roundoff residual (~1e-36); dividing by it would blow up. Treat a
    # negligible spread (relative to the value scale) as genuinely zero vol.
    scale = float(np.max(np.abs(r)))
    if var <= 0 or (scale > 0 and var <= (scale * 1e-12) ** 2):
        return 0.0
    return _safe_div(mu, var)


def kelly_multi_asset(expected_returns, covariance_matrix) -> np.ndarray:
    """Multi-asset Kelly fractions ``f* = Sigma^{-1} mu`` (Thorp 2006).

    The growth-optimal per-asset allocation given a mean vector ``mu`` and
    covariance ``Sigma``. Returned signed (a negative weight is a short). A
    singular covariance returns a zero vector rather than raising.

    Raises ``ValueError`` only for a genuine shape mismatch (``mu`` not 1-D,
    ``Sigma`` not a matching square matrix).
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    sigma = np.asarray(covariance_matrix, dtype=np.float64)
    if mu.ndim != 1 or sigma.ndim != 2:
        raise ValueError("expected_returns must be 1-D and covariance_matrix 2-D")
    if mu.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("expected_returns and covariance_matrix dimensions disagree")
    if mu.size == 0:
        return np.zeros(0, dtype=np.float64)
    try:
        f = np.linalg.solve(sigma, mu)
    except np.linalg.LinAlgError:
        return np.zeros_like(mu)
    if not np.all(np.isfinite(f)):
        return np.zeros_like(mu)
    return f


def kelly_shrinkage(returns_matrix, n_obs: int | None = None) -> np.ndarray:
    """Shrinkage multi-asset Kelly (Kan & Zhou 2007).

    The plug-in ``Sigma^{-1} mu`` over-bets when the sample is short because it
    treats noisy estimates as truth. Kan & Zhou (2007, Theorem) shrink the
    full-Kelly vector toward the naive (zero-edge) estimate by

        ``c* = T * psi^2 / (T * psi^2 + N + 2)``

    where ``T`` is the sample length, ``N`` the number of assets and
    ``psi^2 = mu^T Sigma^{-1} mu`` the in-sample squared Sharpe. ``c*`` lies in
    ``[0, 1)`` and rises toward ``1`` as the sample (and thus the confidence in
    the edge) grows, so the result is always a contraction of ``kelly_multi_asset``
    toward zero.

    ``returns_matrix`` is ``(T, N)`` (a 1-D array is treated as one asset).
    ``n_obs`` overrides ``T`` for the shrinkage factor when the effective sample
    differs from the matrix length. Too few observations (``<= N + 2``), a
    singular covariance, or a non-positive in-sample Sharpe all return a zero
    vector.
    """
    r = np.asarray(returns_matrix, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.ndim != 2:
        raise ValueError("returns_matrix must be 1-D or 2-D")
    t_obs, n_assets = r.shape
    if n_obs is None:
        n_obs = t_obs

    # Need T > N + 2 for the sample covariance to invert and the factor to be
    # well-defined (Kan & Zhou's regularity condition).
    if n_assets == 0 or n_obs <= n_assets + 2 or not np.all(np.isfinite(r)):
        return np.zeros(n_assets, dtype=np.float64)

    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False, ddof=1)
    sigma = np.atleast_2d(sigma)

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.zeros(n_assets, dtype=np.float64)

    f_full = sigma_inv @ mu
    if not np.all(np.isfinite(f_full)):
        return np.zeros(n_assets, dtype=np.float64)

    sharpe_sq = float(mu @ sigma_inv @ mu)
    if not np.isfinite(sharpe_sq) or sharpe_sq <= 0:
        return np.zeros(n_assets, dtype=np.float64)

    denom = n_obs * sharpe_sq + (n_assets + 2.0)
    shrinkage = _safe_div(n_obs * sharpe_sq, denom)
    return f_full * shrinkage
