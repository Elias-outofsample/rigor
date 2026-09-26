"""Allocation weight methods — long-only, sum-to-one weights from a returns matrix.

The six standard modes the app offers, implemented in plain NumPy (no cvxpy):

  * ``equal_weight``  — 1/N.
  * ``inverse_vol``   — proportional to 1/σ_i (risk-balanced, ignores correlation).
  * ``risk_parity``   — equal risk *contribution* (iterative, uses the covariance).
  * ``min_variance``  — minimise wᵀΣw (projected-gradient, long-only simplex).
  * ``max_sharpe``    — tangency portfolio Σ⁻¹μ, long-only normalised.
  * ``hrp``           — Hierarchical Risk Parity (López de Prado 2016): quasi-diagonal
    correlation clustering + recursive bisection. Robust to ill-conditioned Σ.
  * ``risk_budget``   — arbitrary risk-budget targets; generalises risk_parity via
    scipy SLSQP. Falls back to risk_parity on solver failure.

Each returns a weight vector aligned to the matrix columns; ``max_weight`` caps any
single weight and redistributes the excess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = (
    "equal_weight",
    "inverse_vol",
    "risk_parity",
    "min_variance",
    "max_sharpe",
    "hrp",
    "risk_budget",
)


def _cap_and_normalise(w: np.ndarray, max_weight: float | None) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.ones_like(w) / len(w)
    if not max_weight or max_weight >= 1.0:
        return w
    for _ in range(100):  # iterative water-filling onto the cap
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def inverse_vol(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv = 1.0 / vol
    return inv / inv.sum()


def risk_parity(cov: np.ndarray, *, iters: int = 500, tol: float = 1e-8) -> np.ndarray:
    w = inverse_vol(cov)  # good starting point
    for _ in range(iters):
        rc = w * (cov @ w)              # risk contribution
        target = rc.mean()
        grad = rc - target
        w = np.clip(w - 0.01 * grad / (np.diag(cov) + 1e-12), 1e-6, None)
        w /= w.sum()
        if np.abs(grad).max() < tol:
            break
    return w


def risk_budget(
    cov: np.ndarray,
    target_budget: np.ndarray,
    *,
    max_weight: float | None = None,
) -> np.ndarray:
    """Return long-only weights whose risk-contribution shares match ``target_budget``.

    Risk contribution for asset *i* is defined as::

        RCᵢ = wᵢ · (Σw)ᵢ / sqrt(wᵀΣw)

    The solver minimises ``Σᵢ (RCᵢ/σ_p − bᵢ)²`` subject to ``sum(w) = 1``, ``w ≥ 0``,
    where ``b = target_budget / target_budget.sum()`` (budget normalised internally).

    Parameters
    ----------
    cov:
        (n × n) covariance matrix.
    target_budget:
        Non-negative risk-budget shares (need not sum to 1; normalised internally).
        When uniform (``1/n`` each), the result closely matches :func:`risk_parity`.
    max_weight:
        Optional per-asset upper bound.  Passed to :func:`_cap_and_normalise`.

    Returns
    -------
    np.ndarray
        Long-only weight vector summing to 1.

    Notes
    -----
    * Warm-started from :func:`inverse_vol` (good first guess for any budget).
    * On solver failure or non-convergence falls back to :func:`risk_parity` silently.
    * :func:`_cap_and_normalise` is applied after the solve.
    """
    from scipy.optimize import minimize

    n = cov.shape[0]
    budget = np.asarray(target_budget, dtype=float)
    if budget.sum() <= 0 or not np.all(budget >= 0):
        budget = np.ones(n) / n
    else:
        budget = budget / budget.sum()

    w0 = inverse_vol(cov)
    ub = float(max_weight) if max_weight and max_weight < 1.0 else 1.0

    def _objective(w: np.ndarray) -> float:
        portfolio_var = float(w @ cov @ w)
        if portfolio_var <= 0:
            return 1e9
        sigma_p = portfolio_var**0.5
        rc = w * (cov @ w) / sigma_p  # shape (n,)
        diff = rc / sigma_p - budget   # normalised RC share − budget share
        return float(diff @ diff)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, ub)] * n

    try:
        result = minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success or result.fun < 1e-6:
            w_opt = np.clip(result.x, 0.0, None)
            s = w_opt.sum()
            if s > 0:
                return _cap_and_normalise(w_opt, max_weight)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: equal risk parity
    return _cap_and_normalise(risk_parity(cov), max_weight)


def min_variance(cov: np.ndarray, *, iters: int = 1000) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 1.0 / (np.trace(cov) + 1e-12)
    for _ in range(iters):
        grad = cov @ w
        w = w - step * grad
        w = _project_simplex(w)
    return w


def max_sharpe(cov: np.ndarray, mean: np.ndarray) -> np.ndarray:
    try:
        raw = np.linalg.solve(cov + 1e-8 * np.eye(len(mean)), np.clip(mean, 0, None))
    except np.linalg.LinAlgError:
        raw = np.clip(mean, 0, None)
    raw = np.clip(raw, 0.0, None)
    return raw / raw.sum() if raw.sum() > 0 else np.ones(len(mean)) / len(mean)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(len(v)) + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    theta = cssv[rho[-1]] / (rho[-1] + 1.0)
    return np.clip(v - theta, 0.0, None)


def _hrp(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado 2016)."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    order = _quasi_diag(dist)
    w = np.ones(n)
    clusters = [list(order)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    w = iv / iv.sum()
    return float(w @ sub @ w)


def _quasi_diag(dist: np.ndarray) -> list[int]:
    """Single-linkage-style ordering that clusters correlated columns together."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    if dist.shape[0] <= 2:
        return list(range(dist.shape[0]))
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    return list(leaves_list(link))


def allocate_weights(
    returns_matrix: pd.DataFrame,
    method: str,
    *,
    max_weight: float | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Compute long-only weights for ``method`` over a (bars x strategies) matrix.

    Extra keyword arguments are forwarded to the underlying solver where applicable:

    * ``target_budget`` (``np.ndarray | None``) — used by ``"risk_budget"`` to supply
      per-asset risk-budget shares.  Defaults to uniform (equal risk parity).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    rm = returns_matrix.dropna(how="any")
    n = rm.shape[1]
    if n == 0:
        return np.array([])
    if method == "equal_weight" or len(rm) < 2:
        return _cap_and_normalise(equal_weight(n), max_weight)
    cov = rm.cov().to_numpy()
    if method == "inverse_vol":
        w = inverse_vol(cov)
    elif method == "risk_parity":
        w = risk_parity(cov)
    elif method == "min_variance":
        w = min_variance(cov)
    elif method == "max_sharpe":
        w = max_sharpe(cov, rm.mean().to_numpy())
    elif method == "risk_budget":
        raw_budget = kwargs.get("target_budget")
        tb = (
            np.asarray(raw_budget, dtype=float)
            if raw_budget is not None
            else np.ones(n) / n
        )
        return risk_budget(cov, tb, max_weight=max_weight)
    else:  # hrp
        try:
            w = _hrp(cov, rm.corr().to_numpy())
        except Exception:  # noqa: BLE001
            # Degenerate correlation/covariance (e.g. a zero-variance book makes the
            # HRP linkage distance non-finite) — fall back to inverse-variance weights.
            w = inverse_vol(cov)
    return _cap_and_normalise(w, max_weight)
