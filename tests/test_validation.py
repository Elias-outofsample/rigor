import numpy as np
import pandas as pd

from rigor.engine import result_from_returns
from rigor.validation import validate
from rigor.validation.cpcv import cpcv_pbo
from rigor.validation.overfit import deflated_sharpe_ratio, probabilistic_sharpe_ratio


def _series(mu, sd, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2010-01-01", periods=n))


def test_dsr_deflates_below_psr_for_many_trials():
    psr = probabilistic_sharpe_ratio(0.1, 0.0, 2500)
    dsr = deflated_sharpe_ratio(0.1, 200, 2500)
    assert dsr < psr  # deflation for multiple testing lowers the probability


def test_strong_strategy_not_rejected():
    rep = validate(result_from_returns(_series(0.0008, 0.008, n=3000)), n_trials=1)
    assert rep["verdict"]["verdict"] in ("PROMOTE", "CONDITIONAL")
    assert rep["overfit"]["dsr"] > 0.5


def test_negative_sharpe_rejected_on_critical_gate():
    rep = validate(result_from_returns(_series(-0.0006, 0.01)), n_trials=1)
    assert rep["verdict"]["verdict"] == "REJECT"
    assert any(g["name"] == "Sharpe > 0" and not g["passed"] for g in rep["verdict"]["gates"])


def test_validation_is_deterministic():
    res = result_from_returns(_series(0.0006, 0.009))
    a, b = validate(res, n_trials=5)["verdict"], validate(res, n_trials=5)["verdict"]
    # Compare the decision (gate values may contain NaN for neutral gates, and
    # NaN != NaN would break a naive dict equality).
    assert (a["verdict"], a["grade"], a["score"]) == (b["verdict"], b["grade"], b["score"])
    assert [g["passed"] for g in a["gates"]] == [g["passed"] for g in b["gates"]]


def test_cpcv_pbo_in_range_and_none_for_single_config():
    rng = np.random.default_rng(1)
    mat = rng.normal(0.0005, 0.01, (12, 1000))
    out = cpcv_pbo(mat, n_groups=6, n_test_groups=2)
    assert out is not None and 0.0 <= out["pbo"] <= 1.0
    assert cpcv_pbo(rng.normal(0, 0.01, (1, 1000)), n_groups=6) is None  # need >=2 configs


def test_annual_periods_safeguard():
    """annual_periods=1 is a no-op; annual_periods=ppy down-scales an annualised
    Sharpe to match a manual SR/sqrt(ppy) call so PSR/DSR don't saturate to 1.0."""
    assert probabilistic_sharpe_ratio(0.1, 0.0, 2500, annual_periods=1) == \
        probabilistic_sharpe_ratio(0.1, 0.0, 2500)
    assert deflated_sharpe_ratio(0.1, 200, 2500, annual_periods=1) == \
        deflated_sharpe_ratio(0.1, 200, 2500)
    sr_pp = 1.5 / np.sqrt(252.0)
    assert abs(probabilistic_sharpe_ratio(1.5, 0.0, 2500, annual_periods=252)
               - probabilistic_sharpe_ratio(sr_pp, 0.0, 2500)) < 1e-12
    assert abs(deflated_sharpe_ratio(1.5, 200, 2500, annual_periods=252)
               - deflated_sharpe_ratio(sr_pp, 200, 2500)) < 1e-12
    # the raw annualised call saturates far higher than the correctly-scaled one
    assert (probabilistic_sharpe_ratio(1.5, 0.0, 2500)
            > probabilistic_sharpe_ratio(1.5, 0.0, 2500, annual_periods=252))
