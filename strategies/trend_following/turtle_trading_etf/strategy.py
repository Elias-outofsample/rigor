"""Turtle Trading (S16) — multi-instrument ETF trend following.

Port of an earlier research version (TurtleTrading_S16, "Multi-Instrument" variant): the classic
Dennis/Eckhardt Turtle system across uncorrelated ETF proxies for diversification.

Two breakout systems run together:
  - System 1: enter on a 20-day high, exit on a 10-day low (with the Turtle "skip if
    the last S1 trade won" filter).
  - System 2: enter on a 55-day high, exit on a 20-day low (every signal taken).
N = ATR(20). Position sizing is 1% equity risk per "unit" (shares = 0.01*equity / N).
Pyramiding: add a unit each time price advances ½N past the last add, up to 4 units
per instrument; the stop is 2N below the last add. Correlation-group caps (equity
SPY/QQQ, bonds TLT/IEF, commodities GLD): max 6 units/group, 12 total. Signals at the
close, fills at the next open (+slippage); era-based costs. Long-only.

Event-driven with unit pyramiding and ATR-stop accounting, so it is simulated
directly. Deviations (documented): EODHD split-adjusted prices (original used
yfinance); otherwise channels, sizing, pyramiding, the S1 filter, correlation caps,
and next-open execution match the canonical config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.strategy import StrategyBase, StrategyConfig


def _wilder_atr(high, low, close, n):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.instruments = e.get("instruments", ["SPY", "QQQ", "GLD", "TLT", "IEF"])
        self.groups = e.get("groups", {"equity": ["SPY", "QQQ"], "bonds": ["TLT", "IEF"],
                                       "commodities": ["GLD"]})
        self.s1_entry = int(e.get("s1_entry_channel", 20))
        self.s1_exit = int(e.get("s1_exit_channel", 10))
        self.s2_entry = int(e.get("s2_entry_channel", 55))
        self.s2_exit = int(e.get("s2_exit_channel", 20))
        self.atr_p = int(e.get("atr_period", 20))
        self.risk = float(e.get("risk_per_unit", 0.01))
        self.stop_mult = float(e.get("stop_multiplier", 2.0))
        self.pyramid_interval = float(e.get("pyramid_interval", 0.5))
        self.max_units_inst = int(e.get("max_units_per_instrument", 4))
        self.max_units_group = int(e.get("max_units_per_group", 6))
        self.max_units_total = int(e.get("max_total_units", 12))
        self.s1_filter = bool(e.get("s1_filter_enabled", True))
        self.slippage = float(e.get("slippage_bps", 0.0005))
        self.cost_eras = e.get("cost_eras", [
            [2002, 2007, 15], [2007, 2015, 10], [2015, 2019, 5], [2019, 2030, 3]])
        self.first_trade = e.get("first_trade_date", "2005-01-01")
        self.init_capital = float(config.initial_capital)

    def _cost(self, year):
        for lo, hi, bps in self.cost_eras:
            if lo <= year < hi:
                return bps / 1e4
        return self.cost_eras[-1][2] / 1e4

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        spy = self.data.prices("SPY", start=start, end=end)
        cal = pd.DatetimeIndex(spy["date"])
        inst = self.instruments
        op, hi, lo, cl = {}, {}, {}, {}
        for s in inst:
            df = self.data.prices(s, start=start, end=end, method="split")
            df = df.set_index(pd.DatetimeIndex(df["date"]))
            op[s], hi[s], lo[s], cl[s] = df["adj_open"], df["adj_high"], df["adj_low"], df["adj_close"]
        Op = pd.DataFrame(op).reindex(index=cal, columns=inst)
        H = pd.DataFrame(hi).reindex(index=cal, columns=inst)
        L = pd.DataFrame(lo).reindex(index=cal, columns=inst)
        C = pd.DataFrame(cl).reindex(index=cal, columns=inst)

        s1u = H.shift(1).rolling(self.s1_entry, min_periods=self.s1_entry).max()
        s1l = L.shift(1).rolling(self.s1_exit, min_periods=self.s1_exit).min()
        s2u = H.shift(1).rolling(self.s2_entry, min_periods=self.s2_entry).max()
        s2l = L.shift(1).rolling(self.s2_exit, min_periods=self.s2_exit).min()
        atr = _wilder_atr(H, L, C, self.atr_p)

        return {
            "cal": cal,
            "O": Op.to_numpy(), "C": C.to_numpy(), "Cff": C.ffill().to_numpy(),
            "atr": atr.to_numpy(),
            "s1_entry": (s1u < C).to_numpy(), "s2_entry": (s2u < C).to_numpy(),
            "s1_exit": (s1l > C).to_numpy(), "s2_exit": (s2l > C).to_numpy(),
            "cost_bps": np.array([self._cost(d.year) for d in cal]),
            "first_idx": int(cal.searchsorted(pd.Timestamp(self.first_trade))),
        }

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any], trade_log=None) -> pd.Series:
        _p = trade_log or {}
        pyramid_interval = float(_p.get("pyramid_interval", self.pyramid_interval))
        cal = cache["cal"]
        Op, C, Cff, atr = cache["O"], cache["C"], cache["Cff"], cache["atr"]
        s1e, s2e, s1x, s2x = cache["s1_entry"], cache["s2_entry"], cache["s1_exit"], cache["s2_exit"]
        cost_bps, first_idx, n = cache["cost_bps"], cache["first_idx"], len(cal)
        ninst = len(self.instruments)
        slip, risk, smult, pyr = self.slippage, self.risk, self.stop_mult, pyramid_interval
        igroup = {}
        for g, members in self.groups.items():
            for m in members:
                if m in self.instruments:
                    igroup[self.instruments.index(m)] = g

        cash = prev_equity = self.init_capital
        pos: dict[int, dict] = {}
        s1f: dict[int, dict] = {}
        pend_entries: list[tuple] = []
        pend_exits: list[tuple] = []
        daily = np.zeros(n)

        def tot_units():
            return sum(len(p["units"]) for p in pos.values())

        def grp_units(g):
            return sum(len(p["units"]) for i, p in pos.items() if igroup.get(i) == g)

        def equity(i):
            eq = cash
            for idx, p in pos.items():
                px = Cff[i, idx]
                for u in p["units"]:
                    eq += u["shares"] * (px if np.isfinite(px) else u["entry"])
            return eq

        def fill_px(i, idx):
            px = Op[i, idx]
            if not np.isfinite(px) or px <= 0:
                px = C[i, idx]
            return px if np.isfinite(px) and px > 0 else None

        for i in range(first_idx, n):
            bps = cost_bps[i]
            # PHASE 1 — exits at open
            for idx, reason in pend_exits:
                if idx not in pos:
                    continue
                px = fill_px(i, idx)
                if px is None:
                    px = Cff[i, idx]
                slipped = px * (1 - slip)
                p = pos[idx]
                pnl = 0.0
                for u in p["units"]:
                    sv = u["shares"] * slipped
                    pnl += (sv - sv * bps) - u["shares"] * u["entry"]
                    cash += sv - sv * bps
                if p["system"] == "S1" and self.s1_filter:
                    f = s1f.get(idx, {"prof": None, "skip": False})
                    f["prof"] = pnl > 0
                    f["skip"] = False
                    s1f[idx] = f
                if trade_log is not None:
                    sym = self.instruments[idx]
                    exit_price = slipped
                    exit_idx = i
                    for unit in p["units"]:
                        trade_log.record(
                            symbol=sym, side="long",
                            entry_date=cal[unit["entry_idx"]], exit_date=cal[exit_idx],
                            entry_price=unit["entry"], exit_price=exit_price,
                            bar_prices=pd.Series(
                                C[unit["entry_idx"]:exit_idx + 1, idx],
                                index=cal[unit["entry_idx"]:exit_idx + 1],
                            ),
                            system=str(p.get("system", "")),
                        )
                del pos[idx]
            pend_exits = []

            # PHASE 2 — entries at open (pyramids first, then new)
            pyramids = [e for e in pend_entries if e[3]]
            news = [e for e in pend_entries if not e[3]]
            for idx, a, system, _ in pyramids:
                if idx not in pos or len(pos[idx]["units"]) >= self.max_units_inst:
                    continue
                g = igroup.get(idx)
                if g and grp_units(g) >= self.max_units_group:
                    continue
                if tot_units() >= self.max_units_total:
                    continue
                px = fill_px(i, idx)
                if px is None or not (np.isfinite(a) and a > 0):
                    continue
                fill = px * (1 + slip)
                eq = equity(i)
                shares = int(eq * risk / a)
                if shares < 1:
                    continue
                bv = shares * fill
                if cash - bv - bv * bps < 0:
                    continue
                pos[idx]["units"].append({"entry_idx": i, "entry": fill, "shares": shares, "atr": a})
                pos[idx]["stop"] = fill - smult * a
                cash -= bv + bv * bps
            news.sort(key=lambda e: e[1], reverse=True)
            for idx, a, system, _ in news:
                if idx in pos or tot_units() >= self.max_units_total:
                    continue
                g = igroup.get(idx)
                if g and grp_units(g) >= self.max_units_group:
                    continue
                px = fill_px(i, idx)
                if px is None or not (np.isfinite(a) and a > 0):
                    continue
                fill = px * (1 + slip)
                eq = equity(i)
                shares = int(eq * risk / a)
                if shares < 1:
                    continue
                bv = shares * fill
                if cash - bv - bv * bps < 0:
                    continue
                pos[idx] = {"units": [{"entry_idx": i, "entry": fill, "shares": shares, "atr": a}],
                            "stop": fill - smult * a, "system": system}
                cash -= bv + bv * bps
            pend_entries = []

            # PHASE 3 — exit checks at close (stop / channel)
            for idx, p in pos.items():
                if any(u["entry_idx"] == i for u in p["units"]):
                    continue
                cc = C[i, idx]
                if not np.isfinite(cc):
                    continue
                if cc <= p["stop"]:
                    pend_exits.append((idx, "stop"))
                elif (p["system"] == "S1" and s1x[i, idx]) or (p["system"] == "S2" and s2x[i, idx]):
                    pend_exits.append((idx, "channel"))

            # PHASE 3.5 — pyramid checks at close
            exiting = {idx for idx, _ in pend_exits}
            for idx, p in pos.items():
                if idx in exiting or len(p["units"]) >= self.max_units_inst:
                    continue
                if any(u["entry_idx"] == i for u in p["units"]):
                    continue
                g = igroup.get(idx)
                if (g and grp_units(g) >= self.max_units_group) or tot_units() >= self.max_units_total:
                    continue
                cc, a = C[i, idx], atr[i, idx]
                last = p["units"][-1]
                if np.isfinite(cc) and cc >= last["entry"] + pyr * last["atr"] and np.isfinite(a) and a > 0 and i < n - 1:
                    pend_entries.append((idx, a, p["system"], True))

            # PHASE 4 — new entry signals at close
            if tot_units() < self.max_units_total and i < n - 1:
                held_exit = set(pos) | exiting
                s2sig = {j for j in range(ninst) if j not in held_exit and s2e[i, j]}
                s1sig = {j for j in range(ninst) if j not in held_exit and s1e[i, j]} - s2sig
                s1acc = set()
                for j in s1sig:
                    if self.s1_filter:
                        f = s1f.get(j, {"prof": None, "skip": False})
                        if f["prof"] is True and not f["skip"]:
                            f["skip"] = True
                            s1f[j] = f
                            continue
                        f["skip"] = False
                        s1f[j] = f
                    s1acc.add(j)
                sigs = {}
                for j in s2sig:
                    a = atr[i, j]
                    if np.isfinite(a) and a > 0:
                        sigs[j] = ("S2", a)
                for j in s1acc:
                    a = atr[i, j]
                    if j not in sigs and np.isfinite(a) and a > 0:
                        sigs[j] = ("S1", a)
                for j, (system, a) in sorted(sigs.items(), key=lambda x: x[1][1], reverse=True):
                    g = igroup.get(j)
                    if g and grp_units(g) >= self.max_units_group:
                        continue
                    pend_entries.append((j, a, system, False))

            eq = equity(i)
            daily[i] = eq / prev_equity - 1.0 if prev_equity > 0 else 0.0
            prev_equity = eq

        out = pd.Series(daily[first_idx:], index=cal[first_idx:], name="returns")
        return out.iloc[1:]

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: sweep the knobs run_backtest reads; canonical value first.
        return {
            'pyramid_interval': [0.5, 0.25, 0.75],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
