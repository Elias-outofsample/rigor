"""Regime detection & performance (#12).

Leak-aware regime labelling so a regime filter can be used live without lookahead:

  * ``detect_regimes_hmm`` — Hamilton (1989) Gaussian HMM with BIC regime-count
    selection (needs the optional ``hmmlearn``; returns an ``error`` key if absent).
  * ``detect_regimes_hmm_rolling`` — walk-forward HMM: refit on an expanding window,
    label only the forward chunk (strict point-in-time, no leak).
  * ``detect_regimes_sma`` / ``detect_vix_regimes`` / ``detect_market_vol_regime`` /
    ``detect_market_regime`` — transparent rule-based regimes (always available;
    the HMM-free fallback).
  * ``compute_regime_performance`` / ``compute_regime_timeline`` — per-regime stats
    and the transition timeline.
  * ``bai_perron_breaks`` — BIC-selected multiple mean-shift segmentation.

The HMM functions degrade to a clear ``error`` message when ``hmmlearn`` is not
installed; the rule-based regimes need no optional deps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = [
    "detect_regimes_hmm", "detect_regimes_hmm_rolling", "detect_regimes_sma",
    "detect_vix_regimes", "detect_market_vol_regime", "detect_market_regime",
    "compute_regime_performance", "compute_regime_timeline", "bai_perron_breaks",
]


def _ppy(returns: pd.Series) -> int:
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        return _m.periods_per_year_of(returns.index)
    return _m.TRADING_DAYS


def detect_regimes_hmm(returns: pd.Series, n_regimes: int = 2, auto_select: bool = True) -> dict:
    """Hamilton (1989) Gaussian-HMM regimes, BIC-selected; needs optional ``hmmlearn``."""
    r = returns.dropna().to_numpy().astype(np.float64).reshape(-1, 1)
    if len(r) < 100:
        return {"regimes": pd.Series(dtype=int), "error": "insufficient data"}
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {"regimes": pd.Series(dtype=int),
                "error": "hmmlearn not installed (pip install -e '.[analysis]')"}

    best_bic, best_model, best_n, T = np.inf, None, n_regimes, len(r)
    for n in (range(2, 5) if auto_select else [n_regimes]):
        for _ in range(5):
            try:
                model = GaussianHMM(n_components=n, covariance_type="full",
                                    n_iter=200, random_state=42)
                model.fit(r)
                k_params = n * n + n - 1
                bic = -2.0 * model.score(r) + k_params * np.log(T)
                if bic < best_bic:
                    best_bic, best_model, best_n = bic, model, n
            except Exception:
                continue
    if best_model is None:
        return {"regimes": pd.Series(dtype=int), "error": "HMM fit failed"}

    labels = best_model.predict(r)
    means = [np.mean(r[labels == i]) for i in range(best_n)]
    order = np.argsort(means)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[lab] for lab in labels])
    return {
        "regimes": pd.Series(labels, index=returns.dropna().index), "n_regimes": best_n,
        "means": [float(best_model.means_[order[i]][0]) for i in range(best_n)],
        "vols": [float(np.sqrt(best_model.covars_[order[i]][0, 0])) for i in range(best_n)],
        "transition_matrix": best_model.transmat_[np.ix_(order, order)].tolist(),
        "bic": float(best_bic),
    }


def detect_regimes_hmm_rolling(returns: pd.Series, n_regimes: int = 2, min_history: int = 504,
                               refit_freq: int = 21, auto_select: bool = False) -> dict:
    """Walk-forward HMM: refit on an expanding window, label the next chunk (no lookahead).

    Bars before ``min_history`` get label ``-1`` (no regime estimable).
    """
    r = returns.dropna()
    n = len(r)
    if n < min_history + refit_freq:
        return {"regimes": pd.Series(-1, index=r.index, dtype=int),
                "error": f"insufficient data (need >= {min_history + refit_freq})"}
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {"regimes": pd.Series(dtype=int),
                "error": "hmmlearn not installed (pip install -e '.[analysis]')"}

    arr = r.to_numpy().astype(np.float64).reshape(-1, 1)
    labels = np.full(n, -1, dtype=int)
    fit_log: list[dict] = []
    cursor = min_history
    last_means = None
    while cursor < n:
        train = arr[:cursor]
        chunk_end = min(cursor + refit_freq, n)
        try:
            best_n, best_model, best_bic = n_regimes, None, np.inf
            for nc in (range(2, 5) if auto_select else [n_regimes]):
                for _ in range(3):
                    try:
                        model = GaussianHMM(n_components=nc, covariance_type="full",
                                            n_iter=200, random_state=42)
                        model.fit(train)
                        k_params = nc * nc + nc - 1
                        bic = -2.0 * model.score(train) + k_params * np.log(len(train))
                        if bic < best_bic:
                            best_bic, best_model, best_n = bic, model, nc
                    except Exception:
                        continue
            if best_model is None:
                cursor = chunk_end
                continue
            train_labels = best_model.predict(train)
            means = np.array([np.mean(train[train_labels == i]) if (train_labels == i).any()
                              else 0.0 for i in range(best_n)])
            order = np.argsort(means)
            remap = {int(old): int(new) for new, old in enumerate(order)}
            chunk_labels = best_model.predict(arr[cursor:chunk_end])
            labels[cursor:chunk_end] = [remap[int(lab)] for lab in chunk_labels]
            last_means = means[order]
            fit_log.append({"refit_at": int(cursor), "train_size": int(len(train)),
                            "n_regimes": int(best_n), "bic": float(best_bic),
                            "regime_means": means[order].tolist()})
        except Exception as exc:
            fit_log.append({"refit_at": int(cursor), "error": str(exc)})
        cursor = chunk_end

    regime_series = pd.Series(labels, index=r.index, dtype=int)
    return {"regimes": regime_series, "n_regimes": int(n_regimes), "n_refits": len(fit_log),
            "coverage_pct": round(float((regime_series >= 0).mean()), 4),
            "min_history": int(min_history), "refit_freq": int(refit_freq), "fit_log": fit_log,
            "last_regime_means": last_means.tolist() if last_means is not None else None}


def detect_regimes_sma(returns: pd.Series, benchmark_prices: pd.Series | None = None,
                       sma_period: int = 200) -> pd.Series:
    """SMA regime: 1 when price is above its SMA, 0 below."""
    prices = benchmark_prices if benchmark_prices is not None else (1 + returns.fillna(0)).cumprod()
    sma = prices.rolling(sma_period, min_periods=sma_period).mean()
    return (prices > sma).astype(int).reindex(returns.index).ffill().fillna(0).astype(int)


def detect_vix_regimes(returns: pd.Series, vix_data: pd.Series,
                       thresholds: tuple[float, float] = (15.0, 25.0)) -> pd.Series:
    """VIX-level regime: 0 low / 1 medium / 2 high."""
    vix = vix_data.reindex(returns.index).ffill()
    regime = pd.Series(1, index=returns.index, dtype=int)
    regime[vix <= thresholds[0]] = 0
    regime[vix >= thresholds[1]] = 2
    return regime


def detect_market_vol_regime(returns: pd.Series, vol_window: int = 63,
                             thresholds: tuple[float, float] = (0.12, 0.20)) -> pd.Series:
    """Realised-vol regime from rolling annualised vol: 0 low / 1 mid / 2 high."""
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(_ppy(returns))
    regime = pd.Series(1, index=returns.index, dtype=int)
    regime[rolling_vol <= thresholds[0]] = 0
    regime[rolling_vol >= thresholds[1]] = 2
    return regime


def detect_market_regime(returns: pd.Series, benchmark_prices: pd.Series | None = None,
                         vol_window: int = 63, ma_window: int = 200, mom_short: int = 21,
                         mom_long: int = 252, vol_thresholds: tuple[float, float] = (0.12, 0.20),
                         regime_method: str = "fixed") -> pd.Series:
    """6-bucket Trend×Vol regime (Bull/Bear × Calm/Mid/Stress), lagged 1 bar (no self-label)."""
    ppy = _ppy(returns)
    if regime_method == "percentile":
        if benchmark_prices is not None and len(benchmark_prices) >= ma_window:
            proxy = benchmark_prices.pct_change().fillna(0)
        else:
            proxy = returns.fillna(0)
        long_vol = (proxy.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(ppy)).dropna()
        if len(long_vol) >= 30:
            vol_thresholds = (float(long_vol.quantile(0.33)), float(long_vol.quantile(0.67)))

    vol_low, vol_high = vol_thresholds
    if benchmark_prices is not None and len(benchmark_prices) >= ma_window:
        prices = benchmark_prices.reindex(returns.index).ffill()
    else:
        prices = (1 + returns.fillna(0)).cumprod()
    prices = prices.astype(np.float64)

    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    is_bull = prices > ma
    ret_proxy = prices.pct_change().fillna(0)
    rolling_vol = ret_proxy.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(ppy)
    vol_regime = pd.Series("Mid", index=returns.index, dtype=object)
    vol_regime[rolling_vol <= vol_low] = "Calm"
    vol_regime[rolling_vol >= vol_high] = "Stress"

    trend = pd.Series(np.where(is_bull.reindex(returns.index).fillna(False), "Bull", "Bear"),
                      index=returns.index, dtype=object)
    labels = trend + "-" + vol_regime.reindex(returns.index).fillna("Mid")
    labels[ma.reindex(returns.index).isna()] = "Bear-Mid"
    return labels.shift(1).fillna("Bear-Mid")


def compute_regime_performance(returns: pd.Series, regime_series: pd.Series) -> dict:
    """Per-regime Sharpe / CAGR / MaxDD / vol / win-rate / time-share."""
    aligned = pd.DataFrame({"ret": returns, "regime": regime_series}).dropna()
    results = {}
    for reg in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == reg
        r = aligned.loc[mask, "ret"]
        if len(r) < 5:
            continue
        key: int | str
        try:
            key = int(reg)
        except (ValueError, TypeError):
            key = str(reg)
        results[key] = {
            "sharpe": _m.compute_sharpe(r), "cagr": _m.compute_cagr(r),
            "max_dd": _m.compute_max_drawdown(r), "volatility": _m.compute_volatility(r),
            "win_rate": float((r > 0).mean()), "n_obs": len(r),
            "pct_time": float(mask.sum() / len(aligned)),
        }
    return results


def compute_regime_timeline(regime_series: pd.Series) -> list[dict]:
    """Regime transition timeline (contiguous runs with start/end/duration)."""
    if regime_series.empty:
        return []
    timeline = []
    current, start, start_idx = regime_series.iloc[0], regime_series.index[0], 0
    for i in range(1, len(regime_series)):
        if regime_series.iloc[i] != current:
            timeline.append({"regime": int(current), "start": str(start),
                             "end": str(regime_series.index[i - 1]), "duration": i - start_idx})
            current, start, start_idx = regime_series.iloc[i], regime_series.index[i], i
    timeline.append({"regime": int(current), "start": str(start),
                     "end": str(regime_series.index[-1]),
                     "duration": len(regime_series) - start_idx})
    return timeline


def bai_perron_breaks(returns, max_breaks: int = 5, min_segment_size: int | None = None) -> dict:
    """Bai-Perron (2003) multiple mean-shift breaks, greedy + BIC model selection."""
    if not hasattr(returns, "values"):
        returns = pd.Series(np.asarray(returns, dtype=np.float64))
    r = returns.dropna()
    n = len(r)
    if min_segment_size is None:
        min_segment_size = 252
        if hasattr(r, "index") and len(r) > 1:
            try:
                gap = (r.index[1:] - r.index[:-1]).to_series().median()
                gap_days = float(gap.total_seconds() / 86400.0)
                min_segment_size = 12 if gap_days >= 25 else 52 if gap_days >= 5 else 252
            except Exception:
                min_segment_size = 252
    if n < 2 * min_segment_size:
        return {"n_breaks": 0, "break_dates": [], "segments": [], "bic_curve": [],
                "verdict": "INSUFFICIENT_DATA"}

    arr = r.to_numpy().astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    csqsum = np.concatenate([[0.0], np.cumsum(arr**2)])

    def _ssr(i: int, j: int) -> float:
        if j - i <= 0:
            return 0.0
        s = csum[j] - csum[i]
        return float((csqsum[j] - csqsum[i]) - (s**2) / (j - i))

    def _ssr_breaks(brks: list[int]) -> float:
        bs = sorted([0, *brks, n])
        return sum(_ssr(bs[i], bs[i + 1]) for i in range(len(bs) - 1))

    breaks: list[int] = []
    sigma2_0 = _ssr(0, n) / n
    bic_curve = [float(n * np.log(max(sigma2_0, 1e-12)) + 2 * np.log(n))]
    for _ in range(max_breaks):
        best_pos, best_ssr = None, float("inf")
        existing = sorted([0, *breaks, n])
        for pos in range(min_segment_size, n - min_segment_size):
            if any(abs(pos - eb) < min_segment_size for eb in existing):
                continue
            cand = _ssr_breaks([*breaks, pos])
            if cand < best_ssr:
                best_ssr, best_pos = cand, pos
        if best_pos is None:
            break
        breaks.append(best_pos)
        m = len(breaks)
        bic_curve.append(float(n * np.log(max(best_ssr / n, 1e-12)) + (2 * m + 1) * np.log(n)))

    m_star = int(np.argmin(bic_curve))
    selected = sorted(breaks[:m_star])
    segs_b = sorted([0, *selected, n])
    segments, break_dates = [], []
    for i in range(len(segs_b) - 1):
        a, b = segs_b[i], segs_b[i + 1]
        s = arr[a:b]
        segments.append({"start_idx": a, "end_idx": b, "n": len(s), "mean": float(s.mean()),
                         "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                         "start": str(r.index[a]) if hasattr(r, "index") else a,
                         "end": str(r.index[b - 1]) if hasattr(r, "index") else b - 1})
    for bp in selected:
        break_dates.append(str(r.index[bp]) if hasattr(r, "index") else bp)
    verdict = "STABLE" if m_star == 0 else "SHIFTING" if m_star <= 2 else "FRAGMENTED"
    return {"n_breaks": m_star, "break_dates": break_dates, "segments": segments,
            "bic_curve": bic_curve, "verdict": verdict}
