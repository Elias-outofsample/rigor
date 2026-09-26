"""Regenerate the synthetic SPY snapshot used by ``test_reproducibility.py``.

The reproducibility test needs *fixed* input data in the exact on-disk format of the
offline cache (``<as_of>/eod|div|splits/<SYMBOL>.json``). Real vendor data cannot be
redistributed, so this script writes a deterministic synthetic history instead:
geometric Brownian motion on the NYSE-like business-day calendar from 1993-01-29,
quarterly dividends, no splits. Same seed -> byte-identical files.

    python tests/fixtures/repro/make_synthetic_spy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = "2026-06-01"
OUT = Path(__file__).resolve().parent / AS_OF
SEED = 20260601


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("1993-01-29", "2026-05-29")
    n = len(dates)
    daily_mu, daily_sigma = 0.09 / 252, 0.18 / np.sqrt(252)
    log_ret = rng.normal(daily_mu - 0.5 * daily_sigma**2, daily_sigma, n)
    log_ret[0] = 0.0
    close = 44.0 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(20_000_000, 120_000_000, n)

    # Quarterly dividends (~1.8%/yr) on the third Friday of Mar/Jun/Sep/Dec.
    divs, factor = [], np.ones(n)
    for d in pd.date_range(dates[0], dates[-1], freq="WOM-3FRI"):
        if d.month in (3, 6, 9, 12) and d in dates:
            i = dates.get_loc(d)
            value = round(float(close[i - 1]) * 0.0045, 5)
            divs.append({"currency": "USD", "date": d.strftime("%Y-%m-%d"), "declarationDate": None,
                         "paymentDate": None, "period": "Quarterly", "recordDate": None,
                         "unadjustedValue": value, "value": value})
            factor[:i] *= 1 - value / close[i - 1]
    eod = [
        {"adjusted_close": round(float(c * f), 4), "close": round(float(c), 4),
         "date": d.strftime("%Y-%m-%d"), "high": round(float(h), 4), "low": round(float(lo), 4),
         "open": round(float(o), 4), "volume": int(v)}
        for d, o, h, lo, c, f, v in zip(dates, open_, high, low, close, factor, volume, strict=True)
    ]
    for kind, payload in (("eod", eod), ("div", divs), ("splits", [])):
        path = OUT / kind / "SPY.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(eod)} bars, {len(divs)} dividends to {OUT}")


if __name__ == "__main__":
    main()
