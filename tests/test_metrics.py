import numpy as np

from rigor import metrics as m


def test_sharpe_sign_and_zero_vol():
    up = np.full(252, 0.001)
    assert m.compute_sharpe(up) > 0
    assert m.compute_sharpe(np.zeros(252)) == 0.0  # zero vol -> 0, not nan


def test_cagr_and_drawdown_known():
    # +100% then -50% nets to flat over 2 bars.
    r = np.array([1.0, -0.5])
    assert abs(m.compute_cagr(r, periods_per_year=1)) < 1e-9
    assert abs(m.compute_max_drawdown(r) - (-0.5)) < 1e-9


def test_core_metrics_deterministic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 1000)
    assert m.compute_core_metrics(r) == m.compute_core_metrics(r)


def test_empty_returns_safe():
    out = m.compute_core_metrics(np.array([]))
    assert out["n_obs"] == 0 and out["sharpe"] == 0.0
