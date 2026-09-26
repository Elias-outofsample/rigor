import pandas as pd

from rigor.engine import result_from_returns, simulate_weights


def _panel():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)  # +10%/bar
    return idx, prices


def test_no_lookahead_and_weighting():
    idx, prices = _panel()
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    res = simulate_weights(weights, prices, commission_bps=0)
    # First row dropped (no prior weight); two +10% bars remain.
    assert list(res.returns.round(6)) == [0.10, 0.10]
    assert abs(res.metrics["total_return"] - 0.21) < 1e-9


def test_turnover_cost_charged_only_on_change():
    idx, prices = _panel()
    w_flat = pd.DataFrame({"A": [1.0, 1, 1]}, index=idx)
    w_chg = pd.DataFrame({"A": [0.0, 1, 1]}, index=idx)
    flat = simulate_weights(w_flat, prices, commission_bps=10)
    changing = simulate_weights(w_chg, prices, commission_bps=10)
    assert flat.costs.sum() == 0.0
    assert changing.costs.sum() > 0.0


def test_result_from_returns_metrics():
    r = pd.Series([0.01, -0.02, 0.03])
    res = result_from_returns(r)
    assert res.metrics["n_obs"] == 3
    assert res.equity.iloc[-1] > 0
