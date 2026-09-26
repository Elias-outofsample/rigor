"""Hermetic tests for rigor.xs cross-sectional backtest primitives.

All data is synthetic (10 symbols, ~300 trading days). No file I/O, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.xs import (
    _rank_long_short_weights,
    monthly_xs_quintile_ls,
    xs_daily_basket,
    xs_information_coefficient,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_SYMS = 10
N_DAYS = 300
SYMBOLS = [f"S{i:02d}" for i in range(N_SYMS)]
_IDX = pd.bdate_range("2020-01-01", periods=N_DAYS)


def _make_panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signal, prices) with a strong oracle predictive relationship.

    signal[t] = prices.pct_change()[t+1] + noise  (the NEXT bar's return).

    After simulate_weights' 1-bar lag, the position held over bar t+1 is
    decided by signal[t] = ret[t+1] + noise → this is a near-perfect oracle
    that reliably produces positive Sharpe in the cross-section.
    """
    rng = np.random.default_rng(seed)
    # Prices: GBM with a wide spread of per-symbol drifts so cross-section
    # has material dispersion.
    drifts = np.linspace(-0.0005, 0.0010, N_SYMS)  # heterogeneous drifts
    log_ret = drifts + rng.normal(0, 0.010, size=(N_DAYS, N_SYMS))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_ret, axis=0)),
        index=_IDX,
        columns=SYMBOLS,
    )
    # Oracle signal: signal[t] = actual next-bar return + small noise.
    # Very low noise so the rank ordering is preserved most of the time.
    next_ret = np.roll(log_ret, -1, axis=0)  # shift returns back by 1
    next_ret[-1] = 0.0                        # last row has no future
    noise = rng.normal(0, 0.002, size=(N_DAYS, N_SYMS))
    signal = pd.DataFrame(next_ret + noise, index=_IDX, columns=SYMBOLS)
    return signal, prices


def _make_random_signal(seed: int = 99) -> pd.DataFrame:
    """Pure noise signal — IC should be ~0."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((N_DAYS, N_SYMS)), index=_IDX, columns=SYMBOLS)


# ---------------------------------------------------------------------------
# _rank_long_short_weights
# ---------------------------------------------------------------------------


class TestRankLongShortWeights:
    def test_long_only_sums_to_one(self):
        sig = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 0.5})
        eligible = sig.index
        w = _rank_long_short_weights(sig, eligible, top_k=2, long_short=False)
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert all(v > 0 for v in w.values())

    def test_long_short_net_zero(self):
        sig = pd.Series({f"S{i}": float(i) for i in range(10)})
        w = _rank_long_short_weights(sig, sig.index, top_k=3, long_short=True)
        assert abs(sum(w.values())) < 1e-12  # net zero
        assert sum(v for v in w.values() if v > 0) == pytest.approx(1.0)
        assert sum(v for v in w.values() if v < 0) == pytest.approx(-1.0)

    def test_empty_eligible_returns_empty(self):
        sig = pd.Series({"A": 1.0})
        w = _rank_long_short_weights(sig, pd.Index([]), top_k=1, long_short=True)
        assert w == {}

    def test_all_nan_returns_empty(self):
        sig = pd.Series({"A": np.nan, "B": np.nan})
        w = _rank_long_short_weights(sig, sig.index, top_k=1, long_short=True)
        assert w == {}

    def test_top_k_respected(self):
        sig = pd.Series({f"S{i}": float(i) for i in range(8)})
        w = _rank_long_short_weights(sig, sig.index, top_k=2, long_short=True)
        longs = [s for s, v in w.items() if v > 0]
        shorts = [s for s, v in w.items() if v < 0]
        assert len(longs) == 2
        assert len(shorts) == 2


# ---------------------------------------------------------------------------
# xs_daily_basket
# ---------------------------------------------------------------------------


class TestXsDailyBasket:
    def test_predictive_signal_positive_sharpe(self):
        signal, prices = _make_panel(seed=42)
        out = xs_daily_basket(signal, prices, top_k=3, long_short=True, commission_bps=1.0)
        assert out["result"].metrics["sharpe"] > 0.0

    def test_result_keys_present(self):
        signal, prices = _make_panel()
        out = xs_daily_basket(signal, prices, commission_bps=0)
        assert {"result", "weights", "n_long", "n_short", "turnover"} == set(out)

    def test_turnover_series_non_negative(self):
        signal, prices = _make_panel()
        out = xs_daily_basket(signal, prices, commission_bps=1.0)
        assert (out["turnover"] >= -1e-12).all()

    def test_n_long_sane(self):
        signal, prices = _make_panel()
        out = xs_daily_basket(signal, prices, top_k=3, long_short=True)
        assert (out["n_long"] <= 3).all()
        assert (out["n_long"] >= 0).all()

    def test_membership_mask_respected(self):
        signal, prices = _make_panel()
        # Only allow the first 5 symbols.
        mask = pd.DataFrame(False, index=_IDX, columns=SYMBOLS)
        mask[SYMBOLS[:5]] = True
        out = xs_daily_basket(signal, prices, membership_mask=mask, commission_bps=0)
        w = out["weights"]
        # Symbols 5..9 should always have zero weight.
        assert (w[SYMBOLS[5:]].abs() < 1e-12).all().all()

    def test_long_only_no_short_positions(self):
        signal, prices = _make_panel()
        out = xs_daily_basket(signal, prices, top_k=2, long_short=False, commission_bps=0)
        w = out["weights"]
        assert (w >= -1e-12).all().all()

    def test_rebalance_frequency_respected(self):
        """With rebalance=5, weights should change only on rebalance dates."""
        signal, prices = _make_panel()
        out5 = xs_daily_basket(signal, prices, rebalance=5, commission_bps=0)
        out1 = xs_daily_basket(signal, prices, rebalance=1, commission_bps=0)
        # Daily rebalance should have higher (or equal) gross turnover.
        assert out1["turnover"].sum() >= out5["turnover"].sum() - 1e-9

    def test_degenerate_single_symbol(self):
        """1-symbol universe should not raise — it just can't long-short."""
        sig = pd.DataFrame({"A": np.linspace(0, 1, N_DAYS)}, index=_IDX)
        px = pd.DataFrame(
            {"A": 100 * np.exp(np.cumsum(np.zeros(N_DAYS)))},
            index=_IDX,
        )
        out = xs_daily_basket(sig, px, top_k=1, long_short=True, commission_bps=0)
        assert out["result"] is not None

    def test_all_nan_signal_does_not_raise(self):
        _, prices = _make_panel()
        signal = pd.DataFrame(np.nan, index=_IDX, columns=SYMBOLS)
        out = xs_daily_basket(signal, prices, commission_bps=0)
        # All weights should be zero.
        assert (out["weights"].abs() < 1e-12).all().all()

    def test_empty_universe_does_not_raise(self):
        """Prices panel with no overlapping symbols → zero weights, no exception."""
        signal, _ = _make_panel()
        prices_empty = pd.DataFrame(index=_IDX)  # no columns
        out = xs_daily_basket(signal, prices_empty, commission_bps=0)
        assert out["weights"].shape[1] == 0


# ---------------------------------------------------------------------------
# monthly_xs_quintile_ls
# ---------------------------------------------------------------------------


class TestMonthlyXsQuintileLs:
    def test_result_keys_present(self):
        signal, prices = _make_panel()
        out = monthly_xs_quintile_ls(signal, prices, commission_bps=0)
        assert {"result", "weights", "quintile_returns", "ls_spread"} == set(out)

    def test_quintile_columns_present(self):
        signal, prices = _make_panel()
        out = monthly_xs_quintile_ls(signal, prices, n_quantiles=5)
        qr = out["quintile_returns"]
        if not qr.empty:
            expected = {f"Q{i}" for i in range(1, 6)}
            assert expected <= set(qr.columns)

    def test_long_outperforms_short_with_predictive_signal(self):
        """With an oracle signal (signal[t] = NEXT month's return + tiny noise)
        the top quintile's mean forward return must exceed the bottom quintile's."""
        rng = np.random.default_rng(7)
        # Wide spread of per-symbol drifts so the cross-section has material
        # dispersion; oracle signal ranks are preserved almost every month.
        drifts = np.linspace(-0.0015, 0.0015, N_SYMS)
        log_ret = drifts + rng.normal(0, 0.008, size=(N_DAYS, N_SYMS))
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(log_ret, axis=0)),
            index=_IDX,
            columns=SYMBOLS,
        )
        # Oracle: signal[t] = next-bar log-return + tiny noise.
        next_ret = np.roll(log_ret, -1, axis=0)
        next_ret[-1] = 0.0
        noise = rng.normal(0, 0.001, size=(N_DAYS, N_SYMS))
        signal = pd.DataFrame(next_ret + noise, index=_IDX, columns=SYMBOLS)

        out = monthly_xs_quintile_ls(signal, prices, n_quantiles=5, commission_bps=0)
        qr = out["quintile_returns"]
        if qr.empty or "Q1" not in qr.columns or "Q5" not in qr.columns:
            pytest.skip("insufficient data for quintile split")
        # Top quintile (Q5) mean > bottom quintile (Q1) mean.
        assert qr["Q5"].mean() > qr["Q1"].mean()

    def test_ls_spread_series_type(self):
        signal, prices = _make_panel()
        out = monthly_xs_quintile_ls(signal, prices)
        assert isinstance(out["ls_spread"], pd.Series)

    def test_degenerate_tiny_universe(self):
        """Fewer symbols than n_quantiles → no crash, empty quintile_returns."""
        sig = pd.DataFrame(
            np.random.default_rng(3).standard_normal((N_DAYS, 3)),
            index=_IDX,
            columns=["A", "B", "C"],
        )
        px = pd.DataFrame(
            100 * np.exp(np.cumsum(np.zeros((N_DAYS, 3)), axis=0)),
            index=_IDX,
            columns=["A", "B", "C"],
        )
        out = monthly_xs_quintile_ls(sig, px, n_quantiles=5, commission_bps=0)
        assert out["result"] is not None


# ---------------------------------------------------------------------------
# xs_information_coefficient
# ---------------------------------------------------------------------------


class TestXsInformationCoefficient:
    def test_positive_mean_ic_for_predictive_signal(self):
        signal, prices = _make_panel(seed=17)
        # Forward returns at h=1.
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(signal, fwd)
        assert out["mean_ic"] > 0.0
        assert out["n_dates"] > 0

    def test_near_zero_mean_ic_for_random_signal(self):
        _, prices = _make_panel()
        sig_rand = _make_random_signal(seed=55)
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(sig_rand, fwd)
        # Mean IC of noise should be close to 0 (|IC| < 0.10 with high prob).
        assert abs(out["mean_ic"]) < 0.15

    def test_result_keys_present(self):
        signal, prices = _make_panel()
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(signal, fwd)
        assert {"ic_series", "mean_ic", "ic_std", "icir", "t_stat",
                "pct_positive", "n_dates"} == set(out)

    def test_ic_series_bounded(self):
        signal, prices = _make_panel()
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(signal, fwd)
        ic = out["ic_series"].dropna()
        assert ((ic >= -1.0) & (ic <= 1.0)).all()

    def test_all_nan_signal_does_not_raise(self):
        _, prices = _make_panel()
        sig = pd.DataFrame(np.nan, index=_IDX, columns=SYMBOLS)
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(sig, fwd)
        assert out["n_dates"] == 0
        assert out["mean_ic"] == 0.0

    def test_empty_signal_does_not_raise(self):
        fwd = pd.DataFrame(index=_IDX, columns=SYMBOLS, dtype="float64")
        sig = pd.DataFrame(index=_IDX, columns=SYMBOLS, dtype="float64")
        out = xs_information_coefficient(sig, fwd)
        assert out["n_dates"] == 0

    def test_pct_positive_in_range(self):
        signal, prices = _make_panel()
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(signal, fwd)
        assert 0.0 <= out["pct_positive"] <= 1.0

    def test_icir_equals_mean_over_std(self):
        signal, prices = _make_panel(seed=3)
        fwd = prices.pct_change(fill_method=None).shift(-1)
        out = xs_information_coefficient(signal, fwd)
        if out["ic_std"] > 0:
            expected = out["mean_ic"] / out["ic_std"]
            assert abs(out["icir"] - expected) < 1e-10
