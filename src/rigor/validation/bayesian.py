"""Bayesian Sharpe-ratio posterior via a Normal-Inverse-Gamma conjugate prior.

No PyMC dependency — scipy only. For Gaussian returns the NIG prior is conjugate:
the posterior of (mu, sigma²)|data is NIG, so the posterior of SR = mu/sigma is a
scaled non-central t. We draw from it and summarise.

  * ``bayesian_sharpe_posterior`` — posterior mean/CI of the annualised Sharpe.
  * ``prob_sharpe_above_threshold`` — P(SR > threshold) from the draws.
  * ``decision_rule`` — DEPLOY / WAIT / REJECT from the posterior probability.

Reference: Harvey & Liu (2019) "Detecting Repeatable Performance".
"""
from __future__ import annotations

import numpy as np

__all__ = ["bayesian_sharpe_posterior", "prob_sharpe_above_threshold", "decision_rule"]

# Uninformative NIG hyperparameters. beta_0 kept tiny so the posterior is
# data-driven on financial-scale returns (a larger beta_0 shrinks SR sharply).
_PRIOR_MU_0 = 0.0
_PRIOR_KAPPA_0 = 0.01
_PRIOR_ALPHA_0 = 2.0
_PRIOR_BETA_0 = 1e-6


def bayesian_sharpe_posterior(
    returns, n_samples: int = 10_000, annualisation: float = 252.0,
    prior_mu: float = _PRIOR_MU_0, prior_kappa: float = _PRIOR_KAPPA_0,
    prior_alpha: float = _PRIOR_ALPHA_0, prior_beta: float = _PRIOR_BETA_0,
    seed: int = 42,
) -> dict:
    """Posterior of the annualised Sharpe via NIG conjugate update + sampling."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return {"mean": float("nan"), "std": float("nan"), "ci_5": float("nan"),
                "ci_95": float("nan"), "samples": np.array([], dtype=np.float64), "n_obs": n}

    r_bar, s2 = float(np.mean(r)), float(np.var(r, ddof=1))
    kappa_n = prior_kappa + n
    mu_n = (prior_kappa * prior_mu + n * r_bar) / kappa_n
    alpha_n = prior_alpha + n / 2.0
    beta_n = (prior_beta + 0.5 * (n - 1) * s2
              + (prior_kappa * n * (r_bar - prior_mu) ** 2) / (2.0 * kappa_n))

    sigma2 = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n, size=n_samples)
    mu = rng.normal(loc=mu_n, scale=np.sqrt(sigma2 / kappa_n), size=n_samples)
    sharpe = (mu / np.sqrt(sigma2)) * np.sqrt(annualisation)
    sharpe = sharpe[np.isfinite(sharpe)]
    return {"mean": float(np.mean(sharpe)), "std": float(np.std(sharpe, ddof=1)),
            "ci_5": float(np.percentile(sharpe, 5)), "ci_95": float(np.percentile(sharpe, 95)),
            "median": float(np.median(sharpe)), "samples": sharpe, "n_obs": n,
            "posterior_mu_mean": float(mu_n), "posterior_alpha": float(alpha_n),
            "posterior_beta": float(beta_n)}


def prob_sharpe_above_threshold(posterior: dict, threshold: float = 1.0) -> float:
    """P(SR > threshold) from posterior samples."""
    samples = posterior.get("samples", np.array([]))
    if len(samples) == 0:
        return float("nan")
    return float(np.mean(samples > threshold))


def decision_rule(posterior: dict, deploy_threshold_prob: float = 0.80,
                  sr_threshold: float = 1.0, reject_prob: float = 0.20) -> dict:
    """Three-way deployment decision from posterior P(SR > sr_threshold)."""
    p = prob_sharpe_above_threshold(posterior, threshold=sr_threshold)
    if np.isnan(p):
        decision = "INSUFFICIENT_DATA"
    elif p >= deploy_threshold_prob:
        decision = "DEPLOY"
    elif p < reject_prob:
        decision = "REJECT"
    else:
        decision = "WAIT"
    return {"decision": decision, "p_above_threshold": float(p) if not np.isnan(p) else None,
            "sr_threshold": float(sr_threshold),
            "deploy_threshold_prob": float(deploy_threshold_prob),
            "posterior_mean": posterior.get("mean"), "ci_5": posterior.get("ci_5"),
            "ci_95": posterior.get("ci_95"), "n_obs": posterior.get("n_obs")}
