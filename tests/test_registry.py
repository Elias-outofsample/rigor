"""Tests for rigor.project.registry — hermetic, uses tmp_path only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rigor.project.registry import (
    ModelRecord,
    fingerprint_strategy,
    latest,
    lineage,
    load,
    make_model_id,
    query,
    record,
    record_strategy_run,
    to_sqlite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    slug: str = "test_strat",
    category: str = "momentum",
    version: str = "v1",
    status: str = "idle",
    created_at: str | None = None,
    parent_model_id: str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
) -> ModelRecord:
    ts = created_at or datetime.now(tz=UTC).isoformat()
    p = params or {}
    mid = make_model_id(slug, version, p, parent_model_id)
    return ModelRecord(
        model_id=mid,
        slug=slug,
        category=category,
        version=version,
        params=p,
        metrics=metrics or {},
        dataset_hash=None,
        code_fingerprint=None,
        config_fingerprint=None,
        parent_model_id=parent_model_id,
        status=status,
        note="",
        created_at=ts,
    )


def _jsonl_path(tmp_path: Path) -> Path:
    return tmp_path / "model_registry.jsonl"


# ---------------------------------------------------------------------------
# 1. round-trip
# ---------------------------------------------------------------------------

def test_record_load_roundtrip(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)
    rec = _make_record(slug="roundtrip_strat", metrics={"sharpe": 1.23})
    returned = record(rec, path=p)

    assert returned.model_id == rec.model_id
    assert returned.slug == "roundtrip_strat"

    loaded = load(p)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.model_id == rec.model_id
    assert r.slug == rec.slug
    assert r.category == rec.category
    assert r.version == rec.version
    assert r.params == rec.params
    assert r.metrics == rec.metrics
    assert r.status == rec.status
    assert r.created_at == rec.created_at


# ---------------------------------------------------------------------------
# 2. append-only — two records, first still present
# ---------------------------------------------------------------------------

def test_append_only(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)
    r1 = _make_record(slug="strat_a", created_at="2026-01-01T00:00:00+00:00")
    r2 = _make_record(slug="strat_b", created_at="2026-01-02T00:00:00+00:00")
    record(r1, path=p)
    record(r2, path=p)

    loaded = load(p)
    assert len(loaded) == 2
    ids = {r.model_id for r in loaded}
    assert r1.model_id in ids
    assert r2.model_id in ids


# ---------------------------------------------------------------------------
# 3. fingerprint deterministic and changes on content
# ---------------------------------------------------------------------------

def test_fingerprint_deterministic_and_changes(tmp_path: Path) -> None:
    # Create a minimal strategy dir
    s_dir = tmp_path / "my_strat"
    s_dir.mkdir()
    config = s_dir / "config.json"
    strategy = s_dir / "strategy.py"
    config.write_text('{"slug": "my_strat"}', encoding="utf-8")
    strategy.write_text("# strategy v1\n", encoding="utf-8")

    fp1 = fingerprint_strategy(s_dir)
    fp2 = fingerprint_strategy(s_dir)

    assert fp1["code_fingerprint"] is not None
    assert fp1["config_fingerprint"] is not None
    # Same content → same fingerprint (deterministic)
    assert fp1["code_fingerprint"] == fp2["code_fingerprint"]
    assert fp1["config_fingerprint"] == fp2["config_fingerprint"]

    # Modify strategy.py → code fingerprint must change
    strategy.write_text("# strategy v2\n", encoding="utf-8")
    fp3 = fingerprint_strategy(s_dir)
    assert fp3["code_fingerprint"] != fp1["code_fingerprint"]
    # config.json unchanged → config fingerprint same
    assert fp3["config_fingerprint"] == fp1["config_fingerprint"]

    # Missing files → None
    s_dir2 = tmp_path / "empty_strat"
    s_dir2.mkdir()
    fp4 = fingerprint_strategy(s_dir2)
    assert fp4["code_fingerprint"] is None
    assert fp4["config_fingerprint"] is None


# ---------------------------------------------------------------------------
# 4. lineage chain: root → child → grandchild
# ---------------------------------------------------------------------------

def test_lineage_chain(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)

    root = _make_record(slug="chain_strat", created_at="2026-01-01T00:00:00+00:00")
    child = _make_record(
        slug="chain_strat",
        created_at="2026-02-01T00:00:00+00:00",
        parent_model_id=root.model_id,
    )
    grandchild = _make_record(
        slug="chain_strat",
        created_at="2026-03-01T00:00:00+00:00",
        parent_model_id=child.model_id,
    )

    record(root, path=p)
    record(child, path=p)
    record(grandchild, path=p)

    chain = lineage(grandchild.model_id, path=p)
    assert len(chain) == 3
    assert chain[0].model_id == root.model_id
    assert chain[1].model_id == child.model_id
    assert chain[2].model_id == grandchild.model_id


# ---------------------------------------------------------------------------
# 5. latest returns newest record for slug
# ---------------------------------------------------------------------------

def test_latest_returns_newest(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)

    older = _make_record(slug="dated_strat", created_at="2025-01-01T00:00:00+00:00")
    newer = _make_record(slug="dated_strat", created_at="2026-06-01T00:00:00+00:00")

    record(older, path=p)
    record(newer, path=p)

    result = latest("dated_strat", path=p)
    assert result is not None
    assert result.created_at == newer.created_at
    assert result.model_id == newer.model_id


# ---------------------------------------------------------------------------
# 6. to_sqlite count matches load()
# ---------------------------------------------------------------------------

def test_to_sqlite_count(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)
    for i in range(5):
        r = _make_record(
            slug=f"strat_{i}",
            created_at=f"2026-0{i + 1}-01T00:00:00+00:00",
        )
        record(r, path=p)

    loaded_count = len(load(p))
    conn = to_sqlite(p)
    db_count = conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
    assert db_count == loaded_count == 5
    conn.close()


# ---------------------------------------------------------------------------
# 7. record_strategy_run — minimal synthetic strategy dir
# ---------------------------------------------------------------------------

def test_record_strategy_run(tmp_path: Path) -> None:
    reg_path = _jsonl_path(tmp_path)
    s_dir = tmp_path / "cat_test_strat"
    s_dir.mkdir()

    # Minimal strategy.py
    (s_dir / "strategy.py").write_text("# placeholder\n", encoding="utf-8")

    # config.json using the real schema
    config = {
        "name": "Test Strategy",
        "slug": "test_strat_run",
        "category": "seasonal",
        "start_date": "2010-01-01",
        "end_date": None,
        "initial_capital": 100000.0,
        "commission_bps": 5.0,
        "long_short": "long",
        "rebalance_freq": "monthly",
        "role": "alpha",
        "status": "paper",
        "version": "v2",
        "extra": {"lookback": 20, "vol_target": 0.1},
    }
    (s_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    # artifacts/<slug>_summary.json with nested metrics (as produced by the real engine)
    art_dir = s_dir / "artifacts"
    art_dir.mkdir()
    summary = {
        "as_of": "2026-06-07",
        "start": "2010-01-05",
        "end": "2026-06-05",
        "metrics": {
            "sharpe": 1.45,
            "cagr": 0.12,
            "max_drawdown": -0.08,
        },
        "verdict": {"verdict": "PROMOTE", "score": 80.0},
    }
    (art_dir / "test_strat_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    rec = record_strategy_run(s_dir, note="integration test", path=reg_path)

    assert rec.slug == "test_strat_run"
    assert rec.category == "seasonal"
    assert rec.version == "v2"
    assert rec.status == "paper"
    assert rec.params == {"lookback": 20, "vol_target": 0.1}
    assert rec.metrics.get("sharpe") == pytest.approx(1.45)
    assert rec.metrics.get("cagr") == pytest.approx(0.12)
    assert rec.code_fingerprint is not None
    assert rec.config_fingerprint is not None
    assert rec.note == "integration test"

    # Must also be persisted
    loaded = load(reg_path)
    assert len(loaded) == 1
    assert loaded[0].model_id == rec.model_id


# ---------------------------------------------------------------------------
# 8. query filtering
# ---------------------------------------------------------------------------

def test_query_filters(tmp_path: Path) -> None:
    p = _jsonl_path(tmp_path)
    r1 = _make_record(slug="alpha", category="momentum", status="live",
                      created_at="2026-01-01T00:00:00+00:00")
    r2 = _make_record(slug="beta", category="carry", status="idle",
                      created_at="2026-01-02T00:00:00+00:00")
    r3 = _make_record(slug="gamma", category="momentum", status="idle",
                      created_at="2026-01-03T00:00:00+00:00")
    for r in (r1, r2, r3):
        record(r, path=p)

    assert len(query(p, slug="alpha")) == 1
    assert len(query(p, category="momentum")) == 2
    assert len(query(p, status="idle")) == 2
    assert len(query(p, category="momentum", status="idle")) == 1
    assert query(p, category="momentum", status="idle")[0].slug == "gamma"


# ---------------------------------------------------------------------------
# 9. NaN in summary.json is handled
# ---------------------------------------------------------------------------

def test_nan_in_summary_json(tmp_path: Path) -> None:
    """summary.json with NaN values should not crash record_strategy_run."""
    reg_path = _jsonl_path(tmp_path)
    s_dir = tmp_path / "nan_strat"
    s_dir.mkdir()

    config = {
        "name": "NaN Strategy",
        "slug": "nan_strat",
        "category": "other",
        "status": "idle",
        "version": "v1",
        "extra": {},
    }
    (s_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    art_dir = s_dir / "artifacts"
    art_dir.mkdir()
    # Write summary with a literal NaN (not valid strict JSON, but produced by
    # the real engine for PBO when CSCV returns no value).
    nan_text = '{"metrics": {"sharpe": 1.0, "pbo": NaN}}'
    (art_dir / "nan_strat_summary.json").write_text(nan_text, encoding="utf-8")

    rec = record_strategy_run(s_dir, path=reg_path)
    assert rec.metrics.get("sharpe") == pytest.approx(1.0)
    # NaN → None (null in JSON)
    assert rec.metrics.get("pbo") is None
