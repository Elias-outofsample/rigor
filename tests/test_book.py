"""Tests for rigor.project.book — hermetic, no real strategies/ tree required."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rigor.project.book import (
    build_all,
    build_master,
    build_strategy_snapshot,
    decimate_equity,
    load_master,
)

# ---------------------------------------------------------------------------
# Helpers for building fake strategy trees
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = {
    "name": "Test Strategy A",
    "category": "Momentum",
    "status": "live",
    "version": "v1",
}

_SUMMARY_TEMPLATE: dict = {
    "as_of": "2024-12-31",
    "start": "2020-01-01",
    "end": "2024-12-31",
    "name": "Test Strategy A",
    "metrics": {
        "sharpe": 1.5,
        "cagr": 0.12,
        "max_drawdown": -0.05,
        "volatility": 0.08,
        "sortino": 2.1,
        "calmar": 2.4,
        "n_obs": 1260,
    },
    "verdict": {"score": 75.0, "verdict": "PROMOTE"},
}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_named_returns_csv(path: Path, n: int = 30) -> None:
    """Write a returns CSV with an explicit ``date`` column header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0003, 0.01, n)
    df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "returns": rets})
    df.to_csv(path, index=False)


def _make_unnamed_returns_csv(path: Path, n: int = 30) -> None:
    """Write a returns CSV where the index column has no name (unnamed positional format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    rets = rng.normal(0.0003, 0.01, n)
    df = pd.DataFrame({"returns": rets})
    # pandas writes this with an empty index label: ",returns\n0,..."
    df.to_csv(path)


def _make_strategy(
    tmp_root: Path,
    slug: str,
    category: str = "Momentum",
    *,
    named_returns: bool = True,
    n_returns: int = 30,
    include_returns: bool = True,
    config_override: dict | None = None,
    summary_override: dict | None = None,
) -> Path:
    """Create a fake strategy directory under ``tmp_root/Strategy/<category>/<slug>/``."""
    strat_dir = tmp_root / "strategies" / category / slug
    arts = strat_dir / "artifacts"

    cfg = dict(_CONFIG_TEMPLATE)
    cfg["name"] = slug.replace("_", " ").title()
    cfg["category"] = category
    if config_override:
        cfg.update(config_override)

    summ = dict(_SUMMARY_TEMPLATE)
    summ["name"] = cfg["name"]
    if summary_override:
        # Deep merge for nested "metrics"/"verdict" keys
        for k, v in summary_override.items():
            if isinstance(v, dict) and isinstance(summ.get(k), dict):
                summ[k] = {**summ[k], **v}  # type: ignore[assignment]
            else:
                summ[k] = v

    _write_json(strat_dir / "config.json", cfg)
    _write_json(arts / f"{slug}_summary.json", summ)

    if include_returns:
        ret_path = arts / f"{slug}_returns.csv"
        if named_returns:
            _make_named_returns_csv(ret_path, n=n_returns)
        else:
            _make_unnamed_returns_csv(ret_path, n=n_returns)

    return strat_dir


def _make_fake_tree(tmp_root: Path) -> None:
    """Build the canonical fake tree described in the task spec."""
    (tmp_root / "catalog").mkdir(parents=True, exist_ok=True)

    _make_strategy(tmp_root, "test_strat_a", named_returns=True)
    _make_strategy(tmp_root, "test_strat_b", named_returns=False)
    # Missing returns — only config + summary, no CSV
    _make_strategy(tmp_root, "test_strat_missing", include_returns=False)


# ---------------------------------------------------------------------------
# decimate_equity tests
# ---------------------------------------------------------------------------


def test_decimate_equity_named(tmp_path):
    """1000 returns with DatetimeIndex, n=50 → len <= 50, first + last preserved."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-02", periods=1000, freq="B")
    rets = pd.Series(rng.normal(0.0003, 0.01, 1000), index=idx)

    dates, values = decimate_equity(rets, n=50)

    assert len(dates) <= 50
    assert len(dates) == len(values)
    # First date should be one day before the first return date (origin label)
    assert dates[-1] == idx[-1].strftime("%Y-%m-%d")
    # Values start at 1.0 (origin)
    assert abs(values[0] - 1.0) < 1e-9


def test_decimate_equity_unnamed():
    """Array returns (no DatetimeIndex) → str integer position labels returned."""
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0003, 0.01, 200)

    dates, values = decimate_equity(rets, n=50)

    assert len(dates) <= 50
    # All labels should be parseable as integers
    for label in dates:
        int(label)
    assert len(dates) == len(values)
    # First label is "0" (the origin)
    assert dates[0] == "0"
    # Last label is str(len(equity)-1) = str(200) since equity has 201 points
    assert dates[-1] == "200"


def test_decimate_equity_preserves_small_series():
    """When total points <= n, all points are returned unchanged."""
    rets = np.array([0.01, -0.005, 0.003, 0.002, -0.001])
    dates, values = decimate_equity(rets, n=50)
    # 5 returns + 1 origin = 6 points, all returned
    assert len(dates) == 6
    assert len(values) == 6


def test_decimate_equity_first_value_is_origin():
    """Equity curve always starts at 1.0 regardless of input."""
    rets = np.array([0.1, -0.05, 0.03])
    _, values = decimate_equity(rets, n=100)
    assert abs(values[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# build_master tests
# ---------------------------------------------------------------------------


def test_build_master(tmp_path):
    """DataFrame has correct columns; skips strategy with missing returns."""
    _make_fake_tree(tmp_path)

    df = build_master(tmp_path)

    expected_cols = {
        "slug", "name", "category", "status", "version",
        "sharpe", "cagr", "max_drawdown", "volatility", "sortino",
        "calmar", "n_obs", "verdict", "score", "as_of",
    }
    assert expected_cols <= set(df.columns)

    # test_strat_a and test_strat_b should appear; test_strat_missing should be skipped
    slugs = set(df["slug"].tolist())
    assert "test_strat_a" in slugs
    assert "test_strat_b" in slugs
    assert "test_strat_missing" not in slugs
    assert len(df) == 2


def test_build_master_correct_values(tmp_path):
    """Metrics are read correctly from summary.json."""
    _make_strategy(tmp_path, "strat_x", summary_override={"metrics": {"sharpe": 2.5}})

    df = build_master(tmp_path)

    row = df[df["slug"] == "strat_x"].iloc[0]
    assert abs(row["sharpe"] - 2.5) < 1e-6
    assert row["verdict"] == "PROMOTE"
    assert abs(row["score"] - 75.0) < 1e-6


# ---------------------------------------------------------------------------
# load_master roundtrip tests
# ---------------------------------------------------------------------------


def test_load_master_roundtrip(tmp_path):
    """Write with build_master then read back with load_master — same shape."""
    _make_strategy(tmp_path, "strat_a")
    _make_strategy(tmp_path, "strat_b")

    (tmp_path / "catalog").mkdir(parents=True, exist_ok=True)
    df_written = build_master(tmp_path)
    df_read = load_master(tmp_path)

    assert df_written.shape == df_read.shape
    assert list(df_written.columns) == list(df_read.columns)


def test_load_master_absent(tmp_path):
    """Returns an empty DataFrame when the parquet file does not exist."""
    df = load_master(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# build_strategy_snapshot tests
# ---------------------------------------------------------------------------


def test_build_strategy_snapshot(tmp_path):
    """Snapshot file is written as gzip JSON with equity + metrics keys."""
    strat_dir = _make_strategy(tmp_path, "snap_test")

    snap = build_strategy_snapshot(strat_dir)

    # Return value has required keys
    for key in ("slug", "category", "name", "as_of", "metrics", "verdict", "equity"):
        assert key in snap

    assert snap["slug"] == "snap_test"
    assert "dates" in snap["equity"]
    assert "values" in snap["equity"]
    assert len(snap["equity"]["dates"]) == len(snap["equity"]["values"])

    # Artifact file must exist and be loadable
    out_path = strat_dir / "extra" / "snap_test_snapshot.json.gz"
    assert out_path.exists()

    with gzip.open(out_path, "rb") as fh:
        loaded = json.loads(fh.read().decode("utf-8"))

    assert loaded["slug"] == "snap_test"
    assert loaded["equity"]["dates"]
    assert loaded["equity"]["values"]


def test_build_strategy_snapshot_missing_returns(tmp_path):
    """Raises FileNotFoundError when returns CSV is absent."""
    strat_dir = _make_strategy(tmp_path, "no_returns", include_returns=False)
    with pytest.raises(FileNotFoundError):
        build_strategy_snapshot(strat_dir)


# ---------------------------------------------------------------------------
# build_all tests
# ---------------------------------------------------------------------------


def test_build_all_skips_missing(tmp_path):
    """build_all with a missing-returns strategy doesn't raise; skips that slug."""
    _make_fake_tree(tmp_path)

    result = build_all(tmp_path)

    assert "n_strategies" in result
    assert "master_path" in result
    assert "snapshots" in result

    # Only strategies WITH returns CSV should appear in snapshots
    assert "test_strat_a" in result["snapshots"]
    assert "test_strat_b" in result["snapshots"]
    assert "test_strat_missing" not in result["snapshots"]

    # n_strategies matches master row count (2 valid, 1 skipped)
    assert result["n_strategies"] == 2


def test_build_all_returns_correct_types(tmp_path):
    """Result dict types are as documented."""
    _make_strategy(tmp_path, "only_one")
    (tmp_path / "catalog").mkdir(parents=True, exist_ok=True)

    result = build_all(tmp_path)

    assert isinstance(result["n_strategies"], int)
    assert isinstance(result["master_path"], str)
    assert isinstance(result["snapshots"], list)


# ---------------------------------------------------------------------------
# Unnamed returns format tests
# ---------------------------------------------------------------------------


def test_unnamed_returns_format(tmp_path):
    """The ',returns' (unnamed index) CSV format produces a correct snapshot."""
    strat_dir = _make_strategy(tmp_path, "unnamed_fmt", named_returns=False, n_returns=50)

    snap = build_strategy_snapshot(strat_dir)

    assert snap["slug"] == "unnamed_fmt"
    dates = snap["equity"]["dates"]
    values = snap["equity"]["values"]

    # Dates should be positional string integers when no DatetimeIndex
    for d in dates:
        # Must be castable to int (positional labels)
        int(d)

    assert len(dates) == len(values)
    assert len(dates) >= 2  # at least origin + last point
    # Origin value is 1.0
    assert abs(values[0] - 1.0) < 1e-9


def test_nan_in_summary_handled(tmp_path):
    """NaN values in summary.json (from json.dumps allow_nan) are handled gracefully."""
    strat_dir = _make_strategy(
        tmp_path,
        "nan_strat",
        summary_override={"metrics": {"sharpe": float("nan")}},
    )
    # The summary file for nan_strat has NaN in sharpe; overwrite it to use bare NaN token
    arts = strat_dir / "artifacts"
    summary_path = arts / "nan_strat_summary.json"
    # Write a JSON file with bare NaN (as json.dumps(allow_nan=True) would produce)
    summary_path.write_text(
        '{"as_of": "2024-12-31", "metrics": {"sharpe": NaN, "n_obs": 100}, '
        '"verdict": {"score": 60.0, "verdict": "MARGINAL"}}',
        encoding="utf-8",
    )

    # Should not raise
    snap = build_strategy_snapshot(strat_dir)
    assert snap["slug"] == "nan_strat"
    # sharpe should be None or missing (NaN converted to null → None)
    assert snap["metrics"].get("sharpe") is None
