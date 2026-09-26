"""Phase 1: market-data helpers degrade gracefully and resolve sectors."""
from __future__ import annotations

import json

from portfolio_engine.data import strategy_sector


def test_sector_name_heuristic():
    assert strategy_sector("btc_basis", name="BTC Basis") == {"Crypto": 1.0}
    assert strategy_sector("carbon_credits", name="Carbon Credits") == {"Energy": 1.0}
    assert strategy_sector("mystery", name="Mystery Strategy") == {"Unknown": 1.0}


def test_sector_override_file_wins(tmp_path):
    (tmp_path / "sectors.json").write_text(json.dumps({"btc_basis": "Digital Assets"}))
    out = strategy_sector("btc_basis", name="BTC Basis", strategy_root=tmp_path)
    assert out == {"Digital Assets": 1.0}


def test_benchmark_loader_is_offline_safe(monkeypatch):
    # with no loader available, returns None rather than raising
    import portfolio_engine.data as d
    monkeypatch.setattr(d, "_loader", lambda data=None: None)
    d.clear_caches()
    assert d.load_benchmark_returns("SPY") is None
    d.clear_caches()
