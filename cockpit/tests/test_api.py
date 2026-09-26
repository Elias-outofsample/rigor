"""API smoke tests — FastAPI TestClient over synthetic strategies (fully offline).

The strategy book and the benchmark/factor network fetches are monkeypatched, so
these run anywhere without a data key and exercise the real build → serialise path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from portfolio_engine import StrategyCandidate


def _synth(n=4, bars=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=bars)
    return [StrategyCandidate(f"strat_{i}", ".", slug=f"strat_{i}", category="x",
                              returns=pd.Series(0.0004 + 0.0001 * i
                                                + 0.01 * rng.standard_normal(bars), index=idx))
            for i in range(n)]


@pytest.fixture
def client(monkeypatch):
    import cockpit_api.engine as eng
    import portfolio_engine.cockpit as core_cockpit
    cands = _synth()
    monkeypatch.setattr(eng, "candidates", lambda refresh=False: cands)
    monkeypatch.setattr(core_cockpit, "load_benchmark_returns", lambda *a, **k: None)
    monkeypatch.setattr(core_cockpit, "build_factor_proxies", lambda *a, **k: {})
    from cockpit_api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["n_strategies"] == 4


def test_strategies(client):
    r = client.get("/api/strategies")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 4
    assert {"slug", "sharpe", "verdict", "start_year"} <= rows[0].keys()
    assert all(row["start_year"] == 2017 for row in rows)   # synthetic book starts 2017


def test_search_min_start_year_filter(client):
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    # all start 2017; requiring data since ≤ 2010 leaves nothing → 400
    body = {"slugs": slugs, "n_samples": 600, "constraints": {}}
    r = client.post("/api/portfolio/search", json={**body, "min_start_year": 2010})
    assert r.status_code == 400
    # since ≤ 2020 keeps them all
    r2 = client.post("/api/portfolio/search", json={**body, "min_start_year": 2020})
    assert r2.status_code == 200 and r2.json()["n"] >= 1


def test_search_common_window(client):
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    r = client.post("/api/portfolio/search",
                    json={"slugs": slugs, "n_samples": 600, "min_strats": 3,
                          "common_window": True, "constraints": {}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["common_window"] is True
    assert d["candidates"] and all(c["n_obs"] > 0 for c in d["candidates"])


def test_methods(client):
    r = client.get("/api/portfolio/methods")
    assert "robust (grid)" in r.json()["methods"]


def test_build_returns_full_dossier(client):
    body = {"slugs": ["strat_0", "strat_1", "strat_2"], "method": "equal_weight",
            "constraints": {"min_sharpe": 0.1}}
    r = client.post("/api/portfolio/build", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["verdict"]["verdict"] in ("PROMOTE", "CONDITIONAL", "REJECT")
    assert d["metrics"]["sharpe"] is not None
    assert len(d["gates"]) == 1                       # one constraint set → one gate
    assert d["equity"]["portfolio"]                   # equity points present
    assert len(d["weights"]) == 3
    assert len(d["attribution"]) == 3


def test_build_rejects_too_few(client):
    r = client.post("/api/portfolio/build", json={"slugs": ["strat_0"]})
    assert r.status_code == 422                        # pydantic min_length=2


def test_search_lists_candidates(client):
    body = {"slugs": ["strat_0", "strat_1", "strat_2", "strat_3"], "n_samples": 800,
            "constraints": {"min_sharpe": 0.0}}
    r = client.post("/api/portfolio/search", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] >= 1 and len(d["candidates"]) == d["n"]
    c0 = d["candidates"][0]
    assert {"rank", "weights", "sharpe", "composite", "n_strats"} <= c0.keys()
    assert all(k.startswith("strat_") for k in c0["weights"])   # weights keyed by slug


def test_search_min_strats_filter(client):
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    body = {"slugs": slugs, "n_samples": 1500, "min_strats": 3, "constraints": {}}
    r = client.post("/api/portfolio/search", json=body)
    assert r.status_code == 200, r.text
    cands = r.json()["candidates"]
    assert cands                                                 # some portfolios use ≥ 3 legs
    assert all(c["n_strats"] >= 3 for c in cands)                # cardinality floor enforced


def test_detail_only_includes_book_legs(client):
    """The opened dossier is about the portfolio's actual legs — not every selected
    strategy — so weights/attribution/components/correlation aren't padded or empty."""
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    s = client.post("/api/portfolio/search",
                    json={"slugs": slugs, "n_samples": 800, "min_strats": 2, "max_strats": 2,
                          "constraints": {}})
    top = s.json()["candidates"][0]
    n_legs = len(top["weights"])
    assert n_legs == 2
    d = client.post("/api/portfolio/detail",
                    json={"slugs": slugs, "weights": top["weights"], "constraints": {}}).json()
    assert len(d["weights"]) == n_legs
    assert len(d["attribution"]) == n_legs
    assert len(d["components"]) == n_legs
    assert len(d["correlation"]["names"]) == n_legs
    assert d["equity"]["strategies"]                # per-leg equity present for the chart


def test_search_max_strats_bounds_legs(client):
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    r = client.post("/api/portfolio/search",
                    json={"slugs": slugs, "n_samples": 1200, "min_strats": 2,
                          "max_strats": 3, "constraints": {}})
    assert r.status_code == 200, r.text
    cands = r.json()["candidates"]
    assert cands
    assert all(2 <= c["n_strats"] <= 3 for c in cands)           # focused k-of-N books


def test_detail_opens_a_chosen_candidate(client):
    slugs = ["strat_0", "strat_1", "strat_2", "strat_3"]
    res = client.post("/api/portfolio/search",
                      json={"slugs": slugs, "n_samples": 800, "constraints": {}})
    cands = res.json()["candidates"]
    assert cands
    r = client.post("/api/portfolio/detail",
                    json={"slugs": slugs, "weights": cands[0]["weights"],
                          "constraints": {"min_sharpe": 0.1}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["verdict"]["verdict"] in ("PROMOTE", "CONDITIONAL", "REJECT")
    assert len(d["gates"]) == 1
    assert d["equity"]["portfolio"]
    assert len(d["weights"]) >= 1
