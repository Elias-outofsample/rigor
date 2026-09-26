"""IBS mean-reversion (Pagonidis) — directional intraday on liquid ETFs.
IBS = (Close - Low) / (High - Low) on yesterday's daily bar. When yesterday closed near its LOW
(IBS < thr, default 0.20), the asset reverts UP today. Go LONG. Exit mode (default 'oc' = intraday
open->close, fits the intraday-position mandate; 'overnight' = exit next 09:31 fillable; 'close' = hold
to today's close from prior close).
Documented cross-market mean-reversion. Decorrelated from everything (even overnight_reversal: ρ≈0).
QQQ next-o→c +1.23, SMH +2.59 (posYr 100%), SOXL +2.51, XLF +1.73; overnight versions ρ≈0 to all.
"""
from __future__ import annotations
from typing import Any
import numpy as np, pandas as pd
from rigor.strategy import StrategyBase, StrategyConfig

class Strategy(StrategyBase):
    def __init__(self, config, data) -> None:
        super().__init__(config); self.data = data
        e = config.extra; self.symbol = str(e.get("symbol","QQQ")); self._e = dict(e)
        self.commission_bps = float(config.commission_bps)
        self.start_date = config.start_date; self.end_date = config.end_date

    def _p(self, p, k, d): return p[k] if k in p else self._e.get(k, d)
    def param_grid(self): return {"ibs_thr": [0.20, 0.15, 0.25], "exit_mode": ["oc", "overnight", "close"]}

    def build_cache(self) -> dict[str, Any]:
        df = self.data.intraday(self.symbol, interval="1m")
        if df.empty: return {"empty": True, "dates": pd.DatetimeIndex([])}
        df = df.sort_values("dt").reset_index(drop=True)
        if self.start_date: df = df[df["dt"] >= pd.Timestamp(self.start_date)]
        if self.end_date: df = df[df["dt"] <= pd.Timestamp(self.end_date) + pd.Timedelta(days=1)]
        df = df.reset_index(drop=True)
        o,h,l,c = (df[k].to_numpy("float64") for k in ("open","high","low","close"))
        sess = df["session"].to_numpy("datetime64[ns]")
        new = np.empty(len(df), bool); new[0]=True; new[1:] = sess[1:] != sess[:-1]
        starts = np.flatnonzero(new); ends = np.append(starts[1:], len(df))
        sdates = pd.DatetimeIndex(sess[starts]); n=len(starts)
        s_open=np.array([o[int(starts[i])] for i in range(n)])
        s_high=np.array([h[int(starts[i]):int(ends[i])].max() for i in range(n)])
        s_low =np.array([l[int(starts[i]):int(ends[i])].min() for i in range(n)])
        s_close=np.array([c[int(ends[i])-1] for i in range(n)])
        return {"empty": False, "o":o,"starts":starts,"ends":ends,
                "s_open":s_open,"s_high":s_high,"s_low":s_low,"s_close":s_close,
                "sdates":sdates,"dates":sdates}

    def run_backtest(self, cache, params):
        if cache.get("empty"): return pd.Series(dtype="float64")
        thr = float(self._p(params,"ibs_thr",0.20)); mode = str(self._p(params,"exit_mode","oc"))
        per_side = self.commission_bps / 1e4
        o=cache["o"]; starts,ends=cache["starts"],cache["ends"]
        so,sh,sl,sc=cache["s_open"],cache["s_high"],cache["s_low"],cache["s_close"]
        sd=cache["sdates"]; n=len(starts)
        ibs=(sc-sl)/np.where(sh-sl==0,np.nan,sh-sl)
        rets=np.zeros(n)
        for i in range(n-1):
            if not np.isfinite(ibs[i]) or ibs[i]>=thr: continue
            i0n,i1n=int(starts[i+1]),int(ends[i+1])
            if i1n-i0n<5: continue
            if mode=="oc":
                rets[i+1]=(sc[i+1]/so[i+1]-1) - 2*per_side
            elif mode=="overnight":
                if i0n+1>=i1n: continue
                rets[i+1]=(o[i0n+1]/sc[i]-1) - 2*per_side
            else:  # close: prior close -> next close
                rets[i+1]=(sc[i+1]/sc[i]-1) - 2*per_side
        ser=pd.Series(rets,index=sd,name="returns"); ser.index=pd.DatetimeIndex(ser.index); return ser

def build(config, data): return Strategy(config, data)
