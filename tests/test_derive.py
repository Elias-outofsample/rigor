"""Hermetic tests for rigor.project.derive.

All tests use ``tmp_path`` and synthetic returns data.  No real strategies/
directory is touched; no network or API key needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rigor.project.derive import (
    derive_per_year,
    derive_risk,
    derive_strategy,
    derive_walk_forward,
)

# --------------------------------------------------------------------------- #
# Helpers / fixtures                                                           #
# --------------------------------------------------------------------------- #

_RNG = np.random.default_rng(0)
_N = 252 * 4  # 4 years of daily returns


def _make_returns() -> pd.Series:
    """Deterministic daily returns: ~+6% CAGR, realistic vol."""
    daily = _RNG.normal(loc=0.00025, scale=0.01, size=_N)
    dates = pd.bdate_range("2020-01-02", periods=_N, freq="B")
    return pd.Series(daily, index=dates, name="test_strategy")


def _write_returns_named(csv_path: Path, returns: pd.Series) -> None:
    """Write ``date,returns`` format (named index column)."""
    df = returns.rename_axis("date").reset_index()
    df.to_csv(csv_path, index=False)


def _write_returns_unnamed(csv_path: Path, returns: pd.Series) -> None:
    """Write ``,returns`` format (pandas default: unnamed leading index)."""
    df = returns.to_frame(name="returns")
    df.to_csv(csv_path)  # default: writes index with no header name


def _make_strategy_dir(
    tmp_path: Path,
    slug: str,
    returns: pd.Series,
    *,
    fmt: str = "named",
) -> Path:
    """Create a minimal strategy directory skeleton in *tmp_path*."""
    strat_dir = tmp_path / slug
    strat_dir.mkdir()
    # config.json
    (strat_dir / "config.json").write_text(
        json.dumps({"slug": slug, "name": "Test Strategy"}),
        encoding="utf-8",
    )
    # artifacts/
    artifacts = strat_dir / "artifacts"
    artifacts.mkdir()
    csv_path = artifacts / f"{slug}_returns.csv"
    if fmt == "named":
        _write_returns_named(csv_path, returns)
    else:
        _write_returns_unnamed(csv_path, returns)
    return strat_dir


# --------------------------------------------------------------------------- #
# Parametrised fixture: both CSV formats                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture(params=["named", "unnamed"])
def strategy_dir(request, tmp_path: Path) -> tuple[Path, str, pd.Series]:
    """Returns (strategy_dir, slug, returns) for both CSV format variants."""
    fmt: str = request.param
    slug = f"test_{fmt}"
    returns = _make_returns()
    sdir = _make_strategy_dir(tmp_path, slug, returns, fmt=fmt)
    return sdir, slug, returns


# --------------------------------------------------------------------------- #
# Test 1: derive_strategy writes all three JSON files and they parse           #
# --------------------------------------------------------------------------- #

def test_derive_strategy_writes_three_json_files(strategy_dir):
    sdir, slug, _ = strategy_dir
    result = derive_strategy(sdir)

    assert not result["skipped"], f"Expected not skipped, got reason={result['reason']!r}"
    assert result["slug"] == slug
    assert len(result["written"]) == 3

    extra_dir = sdir / "extra"
    for suffix in ("_wf.json", "_risk.json", "_peryear.json"):
        p = extra_dir / f"{slug}{suffix}"
        assert p.exists(), f"{p.name} not found"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{p.name} did not deserialise to a dict"


# --------------------------------------------------------------------------- #
# Test 2: risk JSON contains var_95 and cvar_95                               #
# --------------------------------------------------------------------------- #

def test_risk_json_contains_var_and_cvar(strategy_dir):
    sdir, slug, _ = strategy_dir
    derive_strategy(sdir)

    risk_path = sdir / "extra" / f"{slug}_risk.json"
    data = json.loads(risk_path.read_text(encoding="utf-8"))
    assert "var_95" in data, "var_95 missing from risk JSON"
    assert "cvar_95" in data, "cvar_95 missing from risk JSON"
    # Both should be negative numbers (loss-side VaR convention)
    assert data["var_95"] is not None
    assert data["cvar_95"] is not None


# --------------------------------------------------------------------------- #
# Test 3: peryear JSON has one entry per calendar year                        #
# --------------------------------------------------------------------------- #

def test_peryear_json_has_entry_per_year(strategy_dir):
    sdir, slug, returns = strategy_dir
    derive_strategy(sdir)

    peryear_path = sdir / "extra" / f"{slug}_peryear.json"
    data = json.loads(peryear_path.read_text(encoding="utf-8"))
    years_in_data = data["years"]

    # Determine expected years from the returns fixture.
    expected_years = {str(y) for y in sorted(returns.index.year.unique())}
    actual_years = set(years_in_data.keys())
    assert actual_years == expected_years, (
        f"Year mismatch: expected {expected_years}, got {actual_years}"
    )
    # Each year entry must have at least sharpe and return keys.
    for yr, entry in years_in_data.items():
        assert "sharpe" in entry, f"Year {yr} missing sharpe"
        assert "return" in entry, f"Year {yr} missing return"


# --------------------------------------------------------------------------- #
# Test 4: wf JSON has fold_sharpes list + efficiency key                      #
# --------------------------------------------------------------------------- #

def test_wf_json_has_fold_sharpes_and_efficiency(strategy_dir):
    sdir, slug, _ = strategy_dir
    derive_strategy(sdir)

    wf_path = sdir / "extra" / f"{slug}_wf.json"
    data = json.loads(wf_path.read_text(encoding="utf-8"))
    assert "fold_sharpes" in data, "fold_sharpes missing from wf JSON"
    assert "efficiency" in data, "efficiency missing from wf JSON"
    assert isinstance(data["fold_sharpes"], list)
    assert len(data["fold_sharpes"]) > 0, "fold_sharpes must be non-empty"
    assert isinstance(data["efficiency"], (int, float, type(None)))


# --------------------------------------------------------------------------- #
# Test 5: unnamed index CSV format parses correctly (explicit)                #
# --------------------------------------------------------------------------- #

def test_unnamed_index_format_parses(tmp_path: Path):
    slug = "unnamed_test"
    returns = _make_returns()
    sdir = _make_strategy_dir(tmp_path, slug, returns, fmt="unnamed")
    result = derive_strategy(sdir)
    assert not result["skipped"]
    # Confirm per-year years match.
    peryear = json.loads(
        (sdir / "extra" / f"{slug}_peryear.json").read_text(encoding="utf-8")
    )
    expected = {str(y) for y in sorted(returns.index.year.unique())}
    assert set(peryear["years"].keys()) == expected


# --------------------------------------------------------------------------- #
# Test 6: missing returns.csv → skipped=True, no raise                       #
# --------------------------------------------------------------------------- #

def test_no_returns_csv_skips_gracefully(tmp_path: Path):
    slug = "no_returns"
    sdir = tmp_path / slug
    sdir.mkdir()
    (sdir / "config.json").write_text(json.dumps({"slug": slug}), encoding="utf-8")
    # Deliberately do NOT create artifacts/ or the returns CSV.
    result = derive_strategy(sdir)
    assert result["skipped"] is True
    assert result["written"] == []
    # No artefact files written.
    assert not (sdir / "extra" / f"{slug}_wf.json").exists()


# --------------------------------------------------------------------------- #
# Unit tests for the three derive functions                                    #
# --------------------------------------------------------------------------- #

def test_derive_walk_forward_returns_expected_keys():
    returns = _make_returns()
    wf = derive_walk_forward(returns, n_folds=5)
    for key in ("n_folds", "fold_sharpes", "efficiency", "consistency", "cv", "verdict",
                "fold_boundaries"):
        assert key in wf, f"Missing key: {key}"
    assert wf["n_folds"] == 5
    assert len(wf["fold_sharpes"]) == 5
    assert len(wf["fold_boundaries"]) == 5


def test_derive_per_year_augmented_keys():
    returns = _make_returns()
    py = derive_per_year(returns)
    assert "years" in py
    assert "n_years" in py
    for entry in py["years"].values():
        assert "max_drawdown" in entry
        assert "n_obs" in entry


def test_derive_risk_json_safe():
    """All values in derive_risk must be JSON-serialisable."""
    returns = _make_returns()
    risk = derive_risk(returns)
    # Should not raise.
    json.dumps(risk)
    assert "var_95" in risk
    assert "cvar_95" in risk
    assert "gpd_tail" in risk
    assert "kupiec_pof" in risk


def test_derive_risk_nan_becomes_none():
    """derive_risk must not emit raw NaN — they should map to None."""
    returns = _make_returns()
    risk = derive_risk(returns)
    serialised = json.dumps(risk, allow_nan=False)  # strict: NaN would raise
    assert serialised  # just needs to not raise


def test_derive_walk_forward_short_series():
    """A very short series should not crash; n_folds may be 0."""
    short = _make_returns().iloc[:5]
    wf = derive_walk_forward(short, n_folds=5)
    assert "fold_sharpes" in wf
    assert "efficiency" in wf
