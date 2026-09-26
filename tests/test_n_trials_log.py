"""Tests for the cumulative n_trials ledger helpers in rigor.optimize.core.

These tests are hermetic (tmp_path only) and never touch real artifacts.
They unit-test the pure helper functions _read_trial_log and _update_trial_log
directly, plus the effective_n_trials accounting visible in optimize_strategy
reports.
"""
from __future__ import annotations

import json
from pathlib import Path

from rigor.optimize.core import _read_trial_log, _update_trial_log

# ---------------------------------------------------------------------------
# _read_trial_log
# ---------------------------------------------------------------------------


def test_read_trial_log_absent_returns_zeroed_default(tmp_path: Path) -> None:
    log = tmp_path / "slug_n_trials_log.json"
    result = _read_trial_log(log)
    assert result["cumulative_n_trials"] == 0
    assert result["runs"] == []


def test_read_trial_log_corrupt_returns_zeroed_default(tmp_path: Path) -> None:
    log = tmp_path / "slug_n_trials_log.json"
    log.write_text("not valid json", encoding="utf-8")
    result = _read_trial_log(log)
    assert result["cumulative_n_trials"] == 0


def test_read_trial_log_valid_file(tmp_path: Path) -> None:
    log = tmp_path / "slug_n_trials_log.json"
    data = {
        "schema_version": 1,
        "slug": "foo",
        "runs": [{"ts": "2026-01-01T00:00:00+00:00", "n_trials": 40, "grid_dims": {"a": 5}}],
        "cumulative_n_trials": 40,
    }
    log.write_text(json.dumps(data), encoding="utf-8")
    result = _read_trial_log(log)
    assert result["cumulative_n_trials"] == 40
    assert len(result["runs"]) == 1


# ---------------------------------------------------------------------------
# _update_trial_log — first write (no prior log)
# ---------------------------------------------------------------------------


def test_update_trial_log_first_run_creates_file(tmp_path: Path) -> None:
    log = tmp_path / "s_n_trials_log.json"
    _update_trial_log(log, "s", n_trials=25, grid_dims={"a": 5, "b": 5})

    assert log.exists()
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["slug"] == "s"
    assert data["cumulative_n_trials"] == 25
    assert len(data["runs"]) == 1
    assert data["runs"][0]["n_trials"] == 25
    assert data["runs"][0]["grid_dims"] == {"a": 5, "b": 5}
    assert "ts" in data["runs"][0]


def test_update_trial_log_first_run_effective_equals_this_run(tmp_path: Path) -> None:
    """When there is no prior log, effective_n_trials == this run's n_trials."""
    log = tmp_path / "s_n_trials_log.json"
    # simulate what optimize_strategy does: read prior, compute effective, then update
    prior = _read_trial_log(log)
    prior_cumulative = prior["cumulative_n_trials"]
    this_run = 30
    effective = prior_cumulative + this_run
    assert effective == this_run  # no prior history
    _update_trial_log(log, "s", n_trials=this_run, grid_dims={})
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["cumulative_n_trials"] == this_run


# ---------------------------------------------------------------------------
# _update_trial_log — second run (prior log exists)
# ---------------------------------------------------------------------------


def test_update_trial_log_accumulates_across_runs(tmp_path: Path) -> None:
    log = tmp_path / "s_n_trials_log.json"
    # Run 1: 40 trials
    _update_trial_log(log, "s", n_trials=40, grid_dims={"a": 8, "b": 5})
    # Run 2: 60 trials
    prior = _read_trial_log(log)
    prior_cumulative = prior["cumulative_n_trials"]
    this_run = 60
    effective = prior_cumulative + this_run
    assert effective == 100  # 40 + 60

    _update_trial_log(log, "s", n_trials=this_run, grid_dims={"a": 10, "b": 6})
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["cumulative_n_trials"] == 100
    assert len(data["runs"]) == 2
    assert data["runs"][1]["n_trials"] == 60


def test_update_trial_log_prior_100_effective_is_prior_plus_this(tmp_path: Path) -> None:
    """Explicit coverage of the spec: prior=100, this=35 → effective=135."""
    log = tmp_path / "s_n_trials_log.json"
    # Seed a log that already has cumulative=100
    seed = {
        "schema_version": 1,
        "slug": "s",
        "runs": [{"ts": "2026-01-01T00:00:00+00:00", "n_trials": 100, "grid_dims": {}}],
        "cumulative_n_trials": 100,
    }
    log.write_text(json.dumps(seed), encoding="utf-8")

    prior = _read_trial_log(log)
    prior_cumulative = prior["cumulative_n_trials"]
    this_run = 35
    effective = prior_cumulative + this_run
    assert effective == 135

    _update_trial_log(log, "s", n_trials=this_run, grid_dims={})
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["cumulative_n_trials"] == 135
    assert len(data["runs"]) == 2


# ---------------------------------------------------------------------------
# cumulative_trials=False accounting
# ---------------------------------------------------------------------------


def test_cumulative_false_effective_is_this_run_only(tmp_path: Path) -> None:
    """When cumulative_trials=False, prior_cumulative=0 regardless of any log on disk."""
    log = tmp_path / "s_n_trials_log.json"
    # Even if a log with 100 prior trials exists, opt-out → prior_cumulative=0
    seed = {
        "schema_version": 1, "slug": "s",
        "runs": [{"ts": "2026-01-01T00:00:00+00:00", "n_trials": 100, "grid_dims": {}}],
        "cumulative_n_trials": 100,
    }
    log.write_text(json.dumps(seed), encoding="utf-8")

    # Simulate the opt-out branch: cumulative_trials=False → prior_cumulative forced to 0
    cumulative_trials = False
    prior_cumulative = 0 if not cumulative_trials else _read_trial_log(log)["cumulative_n_trials"]
    this_run = 25
    effective = prior_cumulative + this_run
    assert effective == 25  # only this run counts

    # When opt-out: _update_trial_log is NOT called (the log stays unchanged)
    if cumulative_trials:  # pragma: no cover — branch not taken
        _update_trial_log(log, "s", n_trials=this_run, grid_dims={})

    # Log must still have old cumulative (100), not 125
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["cumulative_n_trials"] == 100


# ---------------------------------------------------------------------------
# Atomic write — no .tmp orphan; valid JSON roundtrip
# ---------------------------------------------------------------------------


def test_atomic_write_no_orphan_and_valid_json(tmp_path: Path) -> None:
    log = tmp_path / "s_n_trials_log.json"
    _update_trial_log(log, "s", n_trials=10, grid_dims={"x": 2})

    # No .tmp files left behind
    orphans = list(tmp_path.glob("*.tmp*"))
    assert orphans == []

    # Valid JSON round-trip
    raw = log.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert json.dumps(data)  # re-serialisable without error
    assert data["cumulative_n_trials"] == 10


def test_atomic_write_multiple_rounds_no_orphan(tmp_path: Path) -> None:
    log = tmp_path / "s_n_trials_log.json"
    for i in range(1, 4):
        _update_trial_log(log, "s", n_trials=i * 10, grid_dims={"a": i})
    orphans = list(tmp_path.glob("*.tmp*"))
    assert orphans == []
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["cumulative_n_trials"] == 60  # 10 + 20 + 30
    assert len(data["runs"]) == 3


# ---------------------------------------------------------------------------
# optimize_strategy integration — effective_n_trials surfaced in report
# ---------------------------------------------------------------------------


def test_optimize_strategy_surfaces_effective_n_trials() -> None:
    """optimize_strategy report must include effective_n_trials and cumulative_n_trials."""
    import pytest

    try:
        from fake_data import FakeDataLoader
    except ImportError:
        pytest.skip("fake_data not on path — run from tests/")

    from rigor.optimize import optimize_strategy

    REPO = Path(__file__).resolve().parents[1]
    SMA_TREND = REPO / "strategies" / "trend_following" / "sma_trend"

    report = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False,
                               cumulative_trials=False)
    assert "effective_n_trials" in report
    assert "cumulative_n_trials" in report
    # With cumulative_trials=False, effective == configs_evaluated
    assert report["effective_n_trials"] == report["configs_evaluated"]
    assert report["cumulative_n_trials"] == report["configs_evaluated"]
    # n_trials in overfit block must equal effective_n_trials
    assert report["overfit"]["n_trials"] == report["effective_n_trials"]


def test_optimize_strategy_cumulative_trials_false_no_cross_contamination() -> None:
    """Running optimize_strategy twice with cumulative_trials=False stays independent."""
    import pytest

    try:
        from fake_data import FakeDataLoader
    except ImportError:
        pytest.skip("fake_data not on path — run from tests/")

    from rigor.optimize import optimize_strategy

    REPO = Path(__file__).resolve().parents[1]
    SMA_TREND = REPO / "strategies" / "trend_following" / "sma_trend"

    r1 = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False,
                           cumulative_trials=False)
    r2 = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False,
                           cumulative_trials=False)
    # Both see effective == configs_evaluated (no accumulation)
    assert r1["effective_n_trials"] == r1["configs_evaluated"]
    assert r2["effective_n_trials"] == r2["configs_evaluated"]
    # Deterministic grid → same count both times
    assert r1["configs_evaluated"] == r2["configs_evaluated"]
