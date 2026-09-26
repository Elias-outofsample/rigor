import numpy as np
import pandas as pd

from rigor.data.adjust import adjust_ohlcv
from rigor.data.cache import RawCache
from rigor.data.loader import (
    clean_nonpositive_prices,
    parse_earnings,
    parse_fred,
    parse_fundamentals,
    parse_funding,
    parse_intraday,
    parse_order_flow,
    parse_perp,
)
from rigor.data.pit_membership import TickerMapper, _candidates


def test_eodhd_client_satisfies_datasource_protocol():
    """The default vendor client structurally satisfies the DataSource protocol, so
    DataLoader can accept an alternative source via its ``source`` argument."""
    from rigor.data.client import EODHDClient
    from rigor.data.datasource import DataSource
    assert issubclass(EODHDClient, DataSource)

    class _Incomplete:
        def eod(self, symbol):  # only one of the ten methods
            return []

    assert not issubclass(_Incomplete, DataSource)


def test_reserved_device_names_are_escaped(tmp_path):
    # "CON.XETRA" (Continental) -> "CON.XETRA.json" IS the Windows console
    # device, not a file: reading it blocks forever. The cache must escape any
    # symbol whose stem is a reserved device name so it becomes an ordinary file.
    cache = RawCache(tmp_path, "2026-06-01")
    for sym in ("CON.XETRA", "PRN.US", "nul", "COM1.X", "AUX"):
        name = cache._path("eod", sym).name
        assert name.split(".", 1)[0].upper() not in {
            "CON", "PRN", "AUX", "NUL", "COM1"
        }, f"{sym} -> {name} still resolves to a reserved device"
    # Ordinary tickers are left untouched (no spurious underscore prefix).
    assert cache._path("eod", "ADS.XETRA").name == "ADS.XETRA.json"
    assert cache._path("eod", "AAPL.US").name == "AAPL.US.json"


def test_parse_funding_pit_safe():
    raw = [
        {"fundingTime": "2021-01-01T08:00:00", "fundingRate": 0.0001},
        {"fundingTime": "2021-01-01T16:00:00", "fundingRate": 0.0002},
        {"fundingTime": "2026-09-01T00:00:00", "fundingRate": 0.0003},  # after as-of -> dropped
    ]
    df = parse_funding(raw, as_of_date="2026-06-01")
    assert len(df) == 2
    assert list(df.columns) == ["funding_time", "funding_rate"]
    assert df["funding_time"].is_monotonic_increasing
    assert parse_funding([], as_of_date="2026-06-01").empty


def _bar(date, close, vol=10.0):
    return {"date": date, "open": 1.0, "high": 2.0, "low": 0.5, "close": close, "volume": vol}


def test_parse_perp_pit_safe():
    raw = [
        _bar("2021-01-02", 1.5),
        _bar("2021-01-01", 1.2),
        _bar("2021-01-02", 1.5),   # duplicate date -> collapsed
        _bar("2026-09-01", 9.0),   # after as-of -> dropped
    ]
    df = parse_perp(raw, as_of_date="2026-06-01")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert len(df) == 2                                   # dup collapsed, future bar dropped
    assert df["date"].is_monotonic_increasing
    assert (df["adj_close"] == df["close"]).all()         # perps have no corporate actions
    assert df["date"].max() <= pd.Timestamp("2026-06-01")
    assert parse_perp([], as_of_date="2026-06-01").empty


def test_dataloader_perp_caches_and_parses(tmp_path):
    """DataLoader.perp routes through the cache + parse_perp over a stub source,
    proving the perp price route works end-to-end without a live fetch."""
    from rigor.data.config import DataConfig
    from rigor.data.loader import DataLoader

    klines = [_bar("2024-01-01", 105.0), _bar("2024-01-02", 112.0), _bar("2030-01-01", 1.0)]
    calls: list[tuple[str, str, str]] = []

    class _StubSource:
        def perp(self, symbol, *, interval="1d", venue="binance", start="2019-09-01", end=None):
            calls.append((symbol, interval, venue))
            return klines

    cfg = DataConfig(api_key="x", as_of="2026-06-01", cache_dir=tmp_path)
    dl = DataLoader(cfg, source=_StubSource())
    df = dl.perp("BTCUSDT")
    assert list(df["close"]) == [105.0, 112.0]            # future bar dropped, ascending
    assert df["adj_close"].equals(df["close"])
    dl.perp("BTCUSDT")                                     # served from cache
    assert calls == [("BTCUSDT", "1d", "binance")]

    # A different venue is a different cache namespace (same symbol must not collide).
    dl.perp("BTCUSDT", venue="bybit")
    assert calls == [("BTCUSDT", "1d", "binance"), ("BTCUSDT", "1d", "bybit")]
    assert (tmp_path / "2026-06-01" / "perp_1d" / "BTCUSDT.json").exists()
    assert (tmp_path / "2026-06-01" / "perp_bybit_1d" / "BTCUSDT.json").exists()


def _of_bar(dt, close, buy, sell):
    return {"datetime": dt, "open": 1.0, "high": 2.0, "low": 0.5, "close": close,
            "volume": buy + sell, "buy_vol": buy, "sell_vol": sell, "trades": 7}


def test_parse_order_flow_delta_session_and_pit():
    # 24h UTC, no cash-session filter; delta = buy - sell; session = UTC day;
    # one bar after the as-of date is dropped.
    raw = [
        _of_bar("2024-06-03 00:01:00", 100.0, 6.0, 4.0),   # delta +2, session 06-03
        _of_bar("2024-06-03 23:50:00", 101.0, 3.0, 5.0),   # delta -2, still 06-03
        _of_bar("2024-06-04 02:00:00", 102.0, 1.0, 1.0),   # 02:00 UTC kept (no cash filter)
        _of_bar("2026-09-01 00:00:00", 9.0, 1.0, 1.0),     # after as-of -> dropped
    ]
    df = parse_order_flow(raw, as_of_date="2026-06-01")
    assert list(df.columns) == [
        "dt", "session", "open", "high", "low", "close",
        "volume", "buy_vol", "sell_vol", "delta", "trades"]
    assert len(df) == 3                                    # future bar dropped
    assert list(df["delta"]) == [2.0, -2.0, 0.0]           # buy - sell
    assert df["session"].nunique() == 2                    # 06-03 and 06-04
    assert df["dt"].is_monotonic_increasing
    assert df["dt"].max() <= pd.Timestamp("2026-06-02")
    assert parse_order_flow([], as_of_date="2026-06-01").empty


def test_dataloader_order_flow_caches_and_parses(tmp_path):
    """DataLoader.order_flow routes through the cache + parse_order_flow over a stub
    source, proving the order-flow route works end-to-end without a live download."""
    from rigor.data.config import DataConfig
    from rigor.data.loader import DataLoader

    bars = [_of_bar("2024-06-03 00:01:00", 100.0, 6.0, 4.0),
            _of_bar("2030-01-01 00:00:00", 1.0, 1.0, 1.0)]
    calls: list[tuple[str, str, str]] = []

    class _StubSource:
        def order_flow(self, symbol, *, interval="1m", venue="binance",
                       start=None, end=None):
            calls.append((symbol, interval, venue))
            return bars

    cfg = DataConfig(api_key="x", as_of="2026-06-01", cache_dir=tmp_path)
    dl = DataLoader(cfg, source=_StubSource())
    df = dl.order_flow("BTCUSDT")
    assert list(df["delta"]) == [2.0]                      # future bar dropped
    dl.order_flow("BTCUSDT")                                # served from cache
    assert calls == [("BTCUSDT", "1m", "binance")]
    assert (tmp_path / "2026-06-01" / "order_flow_1m" / "BTCUSDT.json").exists()


def test_parse_intraday_session_and_tz():
    # UTC datetimes: 13:30Z = 09:30 ET (in-session), 13:25Z = 09:25 ET (pre-open, dropped),
    # 20:55Z = 16:55 ET (after close, dropped), and one after the as-of date (dropped).
    raw = [
        {"datetime": dt, "open": o, "high": h, "low": lo, "close": c, "volume": v}
        for dt, o, h, lo, c, v in [
            ("2024-06-03 13:25:00", 1, 1, 1, 1, 10),
            ("2024-06-03 13:30:00", 2, 3, 1, 2, 20),
            ("2024-06-03 19:55:00", 2, 2, 2, 2, 30),
            ("2024-06-03 20:55:00", 9, 9, 9, 9, 40),
            ("2026-09-01 14:00:00", 5, 5, 5, 5, 50),
        ]
    ]
    df = parse_intraday(raw, as_of_date="2026-06-01")
    assert list(df.columns) == ["dt", "session", "open", "high", "low", "close", "volume"]
    assert len(df) == 2  # 09:30 and 15:55 ET kept; pre-open / after-close / future dropped
    assert df["dt"].iloc[0] == pd.Timestamp("2024-06-03 09:30:00")
    assert (df["session"] == pd.Timestamp("2024-06-03")).all()
    assert parse_intraday([], as_of_date="2026-06-01").empty


def test_parse_fundamentals_pit_safe():
    raw = {
        "Financials": {
            "Balance_Sheet": {"quarterly": {
                "2020-03-31": {"date": "2020-03-31", "filing_date": "2020-05-01",
                               "totalAssets": "100", "totalLiab": "40"},
                "2026-03-31": {"date": "2026-03-31", "filing_date": "2026-09-01",  # after as-of
                               "totalAssets": "200", "totalLiab": "80"},
            }},
            "Income_Statement": {"quarterly": {
                "2020-03-31": {"date": "2020-03-31", "filing_date": "2020-05-01",
                               "grossProfit": "30", "ebit": "20"}}},
            "Cash_Flow": {"quarterly": {}},
        },
        "outstandingShares": {"quarterly": {
            "0": {"dateFormatted": "2020-03-31", "shares": "1000"},
            "1": {"dateFormatted": "2026-09-30", "shares": "2000"}}},  # after as-of
    }
    f = parse_fundamentals(raw, as_of_date="2026-06-01")
    assert len(f["balance"]) == 1  # only the 2020 filing (2026 filed after as-of -> dropped)
    assert f["balance"]["totalAssets"].iloc[0] == 100.0
    assert f["income"]["grossProfit"].iloc[0] == 30.0
    assert len(f["shares"]) == 1 and f["shares"].iloc[0] == 1000.0


def test_parse_fred_pit_safe():
    raw = [
        {"date": "2020-01-01", "value": 1.5},
        {"date": "2020-01-02", "value": None},        # missing obs -> dropped
        {"date": "2020-01-03", "value": 1.7},
        {"date": "2026-09-01", "value": 9.9},         # after as-of -> dropped
    ]
    s = parse_fred(raw, as_of_date="2026-06-01")
    assert len(s) == 2
    assert s.index.is_monotonic_increasing
    assert s.loc["2020-01-03"] == 1.7
    # publication lag shifts the effective date forward
    lagged = parse_fred(raw, as_of_date="2026-06-01", pub_lag_days=1)
    assert lagged.index.min() == pd.Timestamp("2020-01-02")
    assert parse_fred([], as_of_date="2026-06-01").empty


def test_parse_earnings_pit_safe():
    raw = {
        "2020-03-31": {"reportDate": "2020-04-30", "date": "2020-03-31", "epsActual": 0.9,
                       "epsEstimate": 0.8},
        "2020-06-30": {"reportDate": "2020-07-30", "date": "2020-06-30", "epsActual": 1.1,
                       "epsEstimate": 1.0},
        "2026-09-30": {"reportDate": "2026-10-30", "date": "2026-09-30", "epsActual": None,
                       "epsEstimate": 2.0},   # future / estimate-only -> dropped
    }
    df = parse_earnings(raw, as_of_date="2020-12-31")
    assert list(df["period"]) == ["2020-03-31", "2020-06-30"]
    assert list(df["eps_actual"]) == [0.9, 1.1]
    # an announcement after the as-of date is excluded (no look-ahead)
    assert parse_earnings(raw, as_of_date="2020-05-15")["period"].tolist() == ["2020-03-31"]
    assert parse_earnings({}, as_of_date="2020-12-31").empty


def test_clean_nonpositive_prices_neutralizes_bad_ticks():
    # A 0.0 (or negative) price bar yields an inf return that would explode a
    # backtest; cleaning NaNs it so no inf survives, while good returns remain.
    prices = [10.0, 11.0, 0.0, 12.0, -1.0, 13.0]
    df = pd.DataFrame({"adj_close": prices, "close": list(prices)})

    dirty = df["adj_close"].pct_change(fill_method=None)
    assert np.isinf(dirty).any()  # the 0.0 bar produces +inf without cleaning

    out = clean_nonpositive_prices(df.copy())
    assert np.isnan(out["adj_close"].iloc[2]) and np.isnan(out["adj_close"].iloc[4])
    rets = out["adj_close"].pct_change(fill_method=None)
    assert not np.isinf(rets).any()                       # no inf remains
    assert abs(rets.iloc[1] - 0.1) < 1e-9                 # good return survives


def test_split_adjustment_is_continuous_and_anchored():
    # 2:1 split between the two bars: raw close 100 -> 50, adjusted should be flat.
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02"],
        "open": [100.0, 50.0], "high": [100.0, 50.0],
        "low": [100.0, 50.0], "close": [100.0, 50.0], "volume": [10, 20],
    })
    splits = [{"date": "2020-01-02", "split": "2.000000/1.000000"}]
    out = adjust_ohlcv(df, splits, [], method="split")
    assert abs(out["adj_close"].iloc[0] - 50.0) < 1e-9   # pre-split halved
    assert abs(out["adj_close"].iloc[-1] - 50.0) < 1e-9  # latest anchored to raw


def test_ticker_mapper_candidates_and_reuse_suffix():
    cands = _candidates("BSC1")
    assert "BSC" in cands and "BSC_old" in cands
    mapper = TickerMapper({"BSC_old", "AAPL"})
    assert mapper.map("BSC1") == "BSC_old.US"
    assert mapper.map("AAPL") == "AAPL.US"
    assert mapper.map("ZZZZ") is None


def test_alias_takes_priority():
    mapper = TickerMapper({"LEH"}, aliases={"LEHMQ": "LEH"})
    assert mapper.map("LEHMQ") == "LEH.US"


def test_candidates_are_deterministic_ordered_list():
    # Must be an ordered list (a set would iterate in PYTHONHASHSEED-random order,
    # making the universe — and every backtest — non-reproducible across machines).
    c = _candidates("BSC1")
    assert isinstance(c, list)
    assert c.index("BSC") < c.index("BSC_old")      # active form precedes _old fallback
    assert _candidates("BSC1") == c                 # stable across calls
    # When both the active and _old codes exist, the active one wins deterministically.
    assert TickerMapper({"AWK", "AWK_old"}).map("AWK1") == "AWK.US"


# --- FRED: official API (when keyed) vs public CSV fallback ---------------

class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_fred_api_and_csv_normalise_identically(monkeypatch):
    from rigor.data.client import EODHDClient
    from rigor.data.config import DataConfig

    api_json = ('{"observations":[{"date":"2020-01-01","value":"1.5"},'
                '{"date":"2020-01-02","value":"."}]}')
    csv_text = "DATE,DGS3MO\n2020-01-01,1.5\n2020-01-02,.\n"
    expected = [{"date": "2020-01-01", "value": 1.5},
                {"date": "2020-01-02", "value": None}]

    # keyed -> official API (asserts the right endpoint + PIT bound + key)
    capi = EODHDClient(DataConfig(api_key="x", fred_api_key="abc", as_of="2026-06-01"))

    def get_api(url, params=None, timeout=None):
        assert "api.stlouisfed.org" in url
        assert params["api_key"] == "abc" and params["observation_end"] == "2026-06-01"
        return _Resp(200, api_json)

    monkeypatch.setattr(capi._session, "get", get_api)

    # unkeyed -> public graph CSV (coed = as-of)
    ccsv = EODHDClient(DataConfig(api_key="x", as_of="2026-06-01"))

    def get_csv(url, params=None, timeout=None):
        assert "fredgraph.csv" in url and params["coed"] == "2026-06-01"
        return _Resp(200, csv_text)

    monkeypatch.setattr(ccsv._session, "get", get_csv)

    assert capi.fred("DGS3MO") == ccsv.fred("DGS3MO") == expected


def test_fred_falls_back_to_csv_when_api_fails(monkeypatch):
    from rigor.data.client import EODHDClient
    from rigor.data.config import DataConfig

    capi = EODHDClient(DataConfig(api_key="x", fred_api_key="bad", as_of="2026-06-01"))

    def get(url, params=None, timeout=None):
        if "api.stlouisfed.org" in url:
            return _Resp(400, "bad api_key")          # API rejects -> should fall back
        return _Resp(200, "DATE,X\n2020-01-01,2.0\n")  # public CSV works

    monkeypatch.setattr(capi._session, "get", get)
    assert capi.fred("X") == [{"date": "2020-01-01", "value": 2.0}]
