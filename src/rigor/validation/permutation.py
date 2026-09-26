"""Data-snooping tests for a searched *family* of configurations.

When you search a grid and keep the best configuration, its performance is the
**maximum** of many noisy trials — it looks good partly by luck. The deflated
Sharpe (``overfit.deflated_sharpe_ratio``) corrects a single winner for the
number of trials *assuming the trials are independent*; these tests are the
complementary, distribution-free view that uses the trials' actual correlation
structure (a stationary bootstrap), following Sullivan-Timmermann-White (1999).

They take the **whole family** of searched configs and a **benchmark** (the
untuned default, by default) and ask: does the *best of the family* beat the
benchmark once you account for having searched the family? — the honest
optimiser question.

  * **White Reality Check** (White 2000) — single bootstrap p-value for
    "no config in the family beats the benchmark".
  * **Hansen SPA** (Hansen 2005) — studentized, less sensitive to poor
    configs padding the family; reports lower/consistent/upper p-values.
  * **Romano-Wolf StepM** (Romano & Wolf 2005) — stepdown that reports *how
    many* configs beat the benchmark with strong family-wise error control.
  * **multiple_testing_adjustment** — Bonferroni / Holm / BH / BY p-adjustment.

All are plain NumPy and reuse a stationary bootstrap (Politis & Romano 1994) so
serial correlation in daily returns is preserved. Each takes
``family`` (list of config return series) and ``benchmark`` (a return series, or
``None`` for a zero/cash benchmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .accel import bootstrap_column_means

__all__ = [
    "white_reality_check",
    "hansen_spa_test",
    "stepm_test",
    "multiple_testing_adjustment",
    "sign_permutation_test",
]


def _as_array(returns) -> np.ndarray:
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    return r.astype(np.float64)


def _excess_matrix(family: list, benchmark=None) -> np.ndarray:
    """(n bars x k configs) matrix of (config_k - benchmark).

    ``n`` is the shortest length across the family; ``benchmark=None`` means a
    zero (cash) benchmark.
    """
    cols = [_as_array(f) for f in family]
    n = min(len(c) for c in cols)
    bench = np.zeros(n) if benchmark is None else _as_array(benchmark)[:n]
    return np.column_stack([c[:n] - bench for c in cols])


def white_reality_check(
    family: list, benchmark=None, *,
    n_bootstrap: int = 1000, block_size: float = 10.0, seed: int = 42, backend: str = "auto",
) -> dict:
    """White (2000) Reality Check for Data Snooping, *Econometrica* 68(5).

    Test statistic ``V_n = max_k sqrt(n) * dbar_k`` where ``dbar_k`` is the mean
    of (config_k - benchmark) over the searched ``family``. The bootstrap is
    **recentered** by the observed means so it mimics the null; ``p = P(V*_n >=
    V_n)``. A small p means the best config genuinely beats the benchmark even
    after accounting for having searched the whole family.
    """
    if len(family) < 2:
        return {"p_value": float("nan"), "note": "need >= 2 configs in the family"}
    excess = _excess_matrix(family, benchmark)
    n = excess.shape[0]
    observed_means = excess.mean(axis=0)
    sqrt_n = np.sqrt(n)
    t_observed = float(np.max(sqrt_n * observed_means))

    boot_means = bootstrap_column_means(excess, n_bootstrap, block_size=block_size,
                                        seed=seed, backend=backend)
    boot_max = (sqrt_n * (boot_means - observed_means)).max(axis=1)
    count = int(np.sum(boot_max >= t_observed))
    p_value = (count + 1) / (n_bootstrap + 1)
    return {
        "p_value": float(p_value),
        "t_statistic": t_observed,
        "n_configs": len(family),
        "is_significant": p_value < 0.05,
    }


def hansen_spa_test(
    family: list, benchmark=None, *,
    n_bootstrap: int = 1000, block_size: float = 10.0, seed: int = 42, backend: str = "auto",
) -> dict:
    """Hansen (2005) Test for Superior Predictive Ability, *JBES* 23(4).

    Studentized statistic ``T = max_k sqrt(n) * dbar_k / omega_k`` over the
    searched ``family`` (config_k - benchmark), with the three Hansen
    null-recenterings (lower / consistent / upper). The *consistent* p-value is
    the headline; it is less sensitive than the Reality Check to poor configs
    padding the family.
    """
    if len(family) < 2:
        return {"p_consistent": float("nan"), "note": "need >= 2 configs in the family"}
    excess = _excess_matrix(family, benchmark)
    n = excess.shape[0]
    means = excess.mean(axis=0)
    se = excess.std(axis=0, ddof=1) / np.sqrt(n)
    se_safe = np.where(se > 0, se, 1.0)
    t_observed = float(np.max(np.where(se > 0, means / se_safe, 0.0)))

    sqrt_n = np.sqrt(n)
    omega = se * sqrt_n
    omega_safe = np.where(omega > 0, omega, 1.0)
    a_n = np.sqrt(2.0 * np.log(np.log(max(n, 3))))
    mu_lower = means.copy()
    mu_upper = np.zeros_like(means)
    mu_consistent = np.where(sqrt_n * means / omega_safe >= -a_n, means, 0.0)

    boot_means = bootstrap_column_means(excess, n_bootstrap, block_size=block_size,
                                        seed=seed, backend=backend)
    mask = se > 0

    def _zmax(mu):  # ``se`` is the SE of the mean; observed stat is means/se (no sqrt(n)).
        z = np.where(mask, (boot_means - mu) / se_safe, 0.0)
        return z.max(axis=1)

    p_lower = float(np.mean(_zmax(mu_lower) >= t_observed))
    p_consistent = float(np.mean(_zmax(mu_consistent) >= t_observed))
    p_upper = float(np.mean(_zmax(mu_upper) >= t_observed))
    return {
        "p_lower": p_lower,
        "p_consistent": p_consistent,
        "p_upper": p_upper,
        "t_statistic": t_observed,
        "n_configs": len(family),
        "is_significant": p_consistent < 0.05,
    }


def stepm_test(
    family: list, benchmark=None, *,
    n_bootstrap: int = 1000, block_size: float = 10.0, alpha: float = 0.05, seed: int = 42,
    backend: str = "auto",
) -> dict:
    """Romano & Wolf (2005) StepM stepdown, *Econometrica* 73(4).

    Stepwise multiple testing with strong family-wise error control: returns how
    many of the searched ``family`` provably beat the benchmark (``n_rejected``).
    Bootstrap statistics are recentered by the observed means to mimic the null.
    """
    if len(family) < 2:
        return {"n_rejected": 0, "n_total": len(family), "note": "need >= 2 configs"}
    excess = _excess_matrix(family, benchmark)
    n, k = excess.shape
    means = excess.mean(axis=0)
    se = excess.std(axis=0, ddof=1) / np.sqrt(n)
    se_safe = np.where(se > 0, se, 1.0)
    t_stats = np.where(se > 0, means / se_safe, 0.0)

    boot_means = bootstrap_column_means(excess, n_bootstrap, block_size=block_size,
                                        seed=seed, backend=backend)
    boot_t = np.where(se > 0, (boot_means - means) / se_safe, 0.0)

    rejected = [False] * k
    remaining = list(range(k))
    while remaining:
        boot_max = boot_t[:, remaining].max(axis=1)
        crit = float(np.percentile(boot_max, (1.0 - alpha) * 100))
        new = [j for j in remaining if t_stats[j] > crit]
        if not new:
            break
        for j in new:
            rejected[j] = True
            remaining.remove(j)
    return {
        "n_rejected": int(sum(rejected)),
        "n_total": k,
        "is_significant": any(rejected),
    }


def sign_permutation_test(returns, *, n_perm: int = 5000, seed: int = 42) -> dict:
    """Sign-shuffle permutation test: preserves magnitudes, randomises direction.
    p = P(permuted mean >= observed mean)."""
    r = _as_array(returns)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return {"p_value": 1.0, "is_significant": False, "note": "too few observations"}
    observed = float(np.mean(r))
    rng = np.random.default_rng(seed)
    count = sum(np.mean(r * rng.choice([-1, 1], size=len(r))) >= observed for _ in range(n_perm))
    p_value = (count + 1) / (n_perm + 1)
    return {"p_value": float(p_value), "is_significant": p_value < 0.05}


def multiple_testing_adjustment(p_values, method: str = "bh") -> np.ndarray:
    """Adjust a vector of p-values for multiple testing.

    ``bonferroni`` / ``holm`` (FWER) or ``bh`` / ``by`` (FDR). Returns the
    adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([])
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method == "holm":
        order = np.argsort(p)
        adjusted = np.empty(n)
        cummax = 0.0
        for i, idx in enumerate(order):
            cummax = max(min(p[idx] * (n - i), 1.0), cummax)
            adjusted[idx] = cummax
        return adjusted
    if method == "bh":
        order = np.argsort(p)[::-1]
        sorted_p = np.sort(p)
        adjusted = np.empty(n)
        prev = 1.0
        for idx in order:
            rank = int(np.searchsorted(sorted_p, p[idx])) + 1
            prev = min(min(p[idx] * n / rank, prev), 1.0)
            adjusted[idx] = prev
        return adjusted
    if method == "by":
        harmonic = sum(1.0 / k for k in range(1, n + 1))
        return np.minimum(p * n * harmonic, 1.0)
    raise ValueError(f"unknown method {method!r} (use bonferroni/holm/bh/by)")
