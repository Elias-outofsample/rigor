"""Tests for rigor.analysis.distribution — stationarity/autocorrelation battery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.analysis import distribution as dist


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Jarque-Bera
# ---------------------------------------------------------------------------


def test_jarque_bera_normal_vs_fat_tailed(rng):
    normal = rng.normal(0.0, 1.0, 3000)
    fat = rng.standard_t(2, 3000)
    assert dist.jarque_bera(normal)["is_normal"] is True
    assert dist.jarque_bera(fat)["is_normal"] is False


def test_jarque_bera_degenerate():
    assert dist.jarque_bera([1.0, 2.0])["status"].startswith("insufficient")
    const = dist.jarque_bera(np.ones(50))
    assert const["status"] == "constant series (no variation)"
    assert np.isnan(const["statistic"])


# ---------------------------------------------------------------------------
# Runs test
# ---------------------------------------------------------------------------


def test_runs_test_random_vs_streaky(rng):
    random = rng.normal(0.0, 1.0, 1000)
    assert dist.runs_test(random)["is_random"] is True
    # Highly persistent (each block of 50 same sign) → far too few runs → not random.
    streaky = np.concatenate([np.full(50, 1.0 if k % 2 == 0 else -1.0)
                              for k in range(20)])
    res = dist.runs_test(streaky)
    assert res["is_random"] is False
    assert res["z_stat"] < 0  # fewer runs than expected


def test_runs_test_degenerate():
    short = dist.runs_test([0.1, 0.2, 0.3])
    assert short["is_random"] is True and short["p_value"] == 1.0
    # All identical → degenerate median split, neutral result, no raise.
    const = dist.runs_test(np.ones(40))
    assert const["is_random"] is True


# ---------------------------------------------------------------------------
# Variance ratio
# ---------------------------------------------------------------------------


def test_variance_ratio_iid_is_random_walk(rng):
    iid = rng.normal(0.0, 0.01, 4000)
    res = dist.variance_ratio(iid, q=2)
    assert res["interpretation"] == "random_walk"
    assert abs(res["vr"] - 1.0) < 0.1


def test_variance_ratio_mean_reverting_and_trending(rng):
    # Negative AR(1) → mean-reverting (VR < 1, significant).
    mr = np.zeros(4000)
    for i in range(1, 4000):
        mr[i] = -0.6 * mr[i - 1] + rng.normal(0.0, 0.01)
    res_mr = dist.variance_ratio(mr, q=2)
    assert res_mr["interpretation"] == "mean_reverting"
    assert res_mr["vr"] < 0.8 and res_mr["z_stat"] < -1.96

    # Positive AR(1) → trending (VR > 1).
    tr = np.zeros(4000)
    for i in range(1, 4000):
        tr[i] = 0.5 * tr[i - 1] + rng.normal(0.0, 0.01)
    res_tr = dist.variance_ratio(tr, q=2)
    assert res_tr["interpretation"] == "trending"
    assert res_tr["vr"] > 1.2 and res_tr["z_stat"] > 1.96


def test_variance_ratio_degenerate():
    assert dist.variance_ratio([0.1, 0.2], q=2)["status"].startswith("insufficient")
    assert dist.variance_ratio(np.ones(100), q=2)["status"] == "zero variance"


# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------


def test_durbin_watson_iid_near_two(rng):
    iid = rng.normal(0.0, 1.0, 2000)
    dw = dist.durbin_watson(iid)
    assert abs(dw["statistic"] - 2.0) < 0.2
    assert dw["has_autocorrelation"] is False


def test_durbin_watson_positively_autocorrelated(rng):
    x = np.zeros(2000)
    for i in range(1, 2000):
        x[i] = 0.8 * x[i - 1] + rng.normal(0.0, 1.0)
    dw = dist.durbin_watson(x)
    assert dw["statistic"] < 1.5
    assert dw["has_autocorrelation"] is True


def test_durbin_watson_degenerate():
    assert dist.durbin_watson([1.0])["statistic"] == 2.0
    assert dist.durbin_watson(np.zeros(10))["statistic"] == 2.0


# ---------------------------------------------------------------------------
# ARCH-LM / White
# ---------------------------------------------------------------------------


def test_arch_lm_detects_clustered_vol(rng):
    iid = rng.normal(0.0, 1.0, 3000)
    assert dist.arch_lm_test(iid)["has_arch_effects"] is False
    # ARCH(1)-style volatility clustering.
    e = np.zeros(3000)
    h = 1e-4
    for i in range(3000):
        if i > 0:
            h = 5e-6 + 0.9 * (e[i - 1] ** 2)
        e[i] = np.sqrt(max(h, 1e-12)) * rng.normal()
    assert dist.arch_lm_test(e)["has_arch_effects"] is True


def test_arch_lm_degenerate():
    res = dist.arch_lm_test([0.1, 0.2, 0.3])
    assert res["has_arch_effects"] is False
    assert res["status"].startswith("insufficient")


def test_white_hetero_degenerate_and_trend():
    res = dist.white_hetero_test(np.arange(5, dtype=float))
    assert res["has_hetero"] is False
    assert res["status"].startswith("insufficient")
    # 50 obs, runs without raising.
    out = dist.white_hetero_test(np.random.default_rng(0).normal(0, 1, 200))
    assert out["status"] == "ok"


# ---------------------------------------------------------------------------
# ADF / KPSS (statsmodels-backed; degrade gracefully when absent)
# ---------------------------------------------------------------------------


def test_adf_random_walk_vs_stationary(rng):
    sm = pytest.importorskip("statsmodels")  # noqa: F841
    rw = np.cumsum(rng.normal(0.0, 1.0, 800))
    white = rng.normal(0.0, 1.0, 800)
    assert dist.adf_test(rw)["is_stationary"] is False
    assert dist.adf_test(white)["is_stationary"] is True


def test_kpss_stationary(rng):
    pytest.importorskip("statsmodels")
    white = rng.normal(0.0, 1.0, 800)
    assert dist.kpss_test(white)["is_stationary"] is True


def test_adf_kpss_degrade_without_statsmodels(monkeypatch, rng):
    """When statsmodels cannot be imported, ADF/KPSS return NaN + a clear status."""
    def fake_import(name):
        # Simulate statsmodels being absent; nothing else is queried here.
        return None

    monkeypatch.setattr(dist, "optional_import", fake_import)
    series = rng.normal(0.0, 1.0, 500)
    adf = dist.adf_test(series)
    kpss = dist.kpss_test(series)
    assert np.isnan(adf["statistic"]) and np.isnan(adf["p_value"])
    assert adf["is_stationary"] is False
    assert "statsmodels not installed" in adf["status"]
    assert np.isnan(kpss["statistic"])
    assert "statsmodels not installed" in kpss["status"]


def test_adf_kpss_degenerate():
    assert dist.adf_test(np.ones(50))["status"] == "constant series (no variation)"
    assert dist.kpss_test([0.1, 0.2])["status"].startswith("insufficient")


# ---------------------------------------------------------------------------
# Ljung-Box (always works — statsmodels or pure fallback)
# ---------------------------------------------------------------------------


def test_ljung_box_detects_autocorrelation(rng):
    iid = rng.normal(0.0, 1.0, 1000)
    assert dist.ljung_box_test(iid)["has_autocorrelation"] is False
    ar = np.zeros(1000)
    for i in range(1, 1000):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0.0, 1.0)
    assert dist.ljung_box_test(ar)["has_autocorrelation"] is True


def test_ljung_box_pure_fallback_when_absent(monkeypatch, rng):
    monkeypatch.setattr(dist, "optional_import", lambda name: None)
    ar = np.zeros(1000)
    for i in range(1, 1000):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0.0, 1.0)
    res = dist.ljung_box_test(ar)
    assert "pure fallback" in res["status"]
    assert res["has_autocorrelation"] is True


def test_ljung_box_degenerate():
    assert dist.ljung_box_test([0.1, 0.2])["status"].startswith("insufficient")


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------


def test_analyze_distribution_bundle(rng):
    r = pd.Series(rng.normal(0.0, 0.01, 1500))
    out = dist.analyze_distribution(r)
    for key in ("jarque_bera", "adf", "kpss", "ljung_box", "variance_ratio",
                "runs_test", "durbin_watson", "arch_lm", "white_hetero", "stats"):
        assert key in out
    assert out["stats"]["n_obs"] == 1500
    assert "statsmodels_available" in out


def test_analyze_distribution_insufficient():
    assert dist.analyze_distribution([])["error"] == "insufficient data"
    assert dist.analyze_distribution([0.1] * 5)["error"] == "insufficient data"


def test_analyze_distribution_pure_path_without_statsmodels(monkeypatch, rng):
    """The pure path still produces results; only ADF/KPSS report the fallback."""
    monkeypatch.setattr(dist, "optional_import", lambda name: None)
    out = dist.analyze_distribution(rng.normal(0.0, 0.01, 800))
    assert out["jarque_bera"]["status"] == "ok"
    assert out["variance_ratio"]["status"] == "ok"
    assert out["runs_test"]["status"] == "ok"
    assert out["arch_lm"]["status"] == "ok"
    assert "statsmodels not installed" in out["adf"]["status"]
    assert "statsmodels not installed" in out["kpss"]["status"]
    assert out["statsmodels_available"] is False
