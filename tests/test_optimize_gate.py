"""Tests for rigor.project.optimize_gate.

All tests are hermetic — they build strategy folders under tmp_path and never
touch the real repo. The gate enforces ONE rule: every production strategy must
declare a sweepable ``param_grid()`` (>=1 axis with >=2 type-consistent values),
unless it is grandfathered (accepted backfill debt) or explicitly exempt.
"""

from __future__ import annotations

import json
from pathlib import Path

from rigor.project import optimize_gate as og
from rigor.project.optimize_gate import (
    check_all,
    check_strategy,
    classify_param_grid,
    load_baseline,
    main,
    stale_baseline_entries,
    write_baseline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GRIDS = {
    "multi": "        return {'k': [2.0, 1.5], 'w': [10]}",
    "empty": "        return {}",
    "single": "        return {'k': [5]}",
    "inconsistent": "        return {'k': [1, 'a']}",
    "dynamic": "        return {'k': _compute()}",
    "nested_helper": (
        "        def axis(v, opts):\n"
        "            return [v] + opts\n"
        "        return {'k': axis(1, [2, 3])}"
    ),
    "branchy_multi": (
        "        if full:\n"
        "            return {'k': _compute()}\n"
        "        return {'k': [1, 2, 3]}"
    ),
}


def _make_strategy(
    root: Path,
    slug: str,
    *,
    grid: str | None = "multi",
    exempt_reason: str | None = None,
    category: str = "test_cat",
    baseline: bool = False,
    strategy_py: bool = True,
) -> Path:
    """Scaffold a minimal strategy folder; ``grid`` is a key of ``_GRIDS`` or None."""
    parts = ["strategies"]
    if baseline:
        parts.append("Baseline")
    parts += [category, slug]
    strat_dir = root.joinpath(*parts)
    strat_dir.mkdir(parents=True)

    cfg: dict = {"slug": slug, "status": "idle"}
    if exempt_reason is not None:
        cfg["optimize_exempt_reason"] = exempt_reason
    (strat_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    if strategy_py:
        if grid is None:
            body = "class Strategy:\n    def build_cache(self):\n        return {}\n"
        else:
            body = "class Strategy:\n    def param_grid(self, full=True):\n" + _GRIDS[grid] + "\n"
        (strat_dir / "strategy.py").write_text(body, encoding="utf-8")

    return strat_dir


# ---------------------------------------------------------------------------
# classify_param_grid
# ---------------------------------------------------------------------------

def test_classify_multi(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid="multi")
    assert classify_param_grid(d / "strategy.py")[0] == "MULTI"


def test_classify_empty(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid="empty")
    assert classify_param_grid(d / "strategy.py")[0] == "EMPTY"


def test_classify_single_value(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid="single")
    assert classify_param_grid(d / "strategy.py")[0] == "SINGLE"


def test_classify_inconsistent_types(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid="inconsistent")
    assert classify_param_grid(d / "strategy.py")[0] == "INCONSISTENT"


def test_classify_no_method(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid=None)
    assert classify_param_grid(d / "strategy.py")[0] == "NO_METHOD"


def test_classify_dynamic(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "s", grid="dynamic")
    assert classify_param_grid(d / "strategy.py")[0] == "DYNAMIC"


def test_classify_nested_helper_not_mistaken_for_grid(tmp_path: Path) -> None:
    # The nested `axis` helper's `return [v] + opts` must NOT be read as the grid;
    # the real return is a computed call -> DYNAMIC, not MULTI.
    d = _make_strategy(tmp_path, "s", grid="nested_helper")
    assert classify_param_grid(d / "strategy.py")[0] == "DYNAMIC"


def test_classify_branchy_multi(tmp_path: Path) -> None:
    # One literal branch is sweepable -> MULTI wins even with a computed branch.
    d = _make_strategy(tmp_path, "s", grid="branchy_multi")
    assert classify_param_grid(d / "strategy.py")[0] == "MULTI"


def test_classify_missing_file(tmp_path: Path) -> None:
    assert classify_param_grid(tmp_path / "nope.py")[0] == "NO_FILE"


def test_classify_follows_shared_engine_reexport(tmp_path: Path) -> None:
    # A thin re-export inherits param_grid from the shared engine; the gate must
    # follow `rigor.strategies.<engine>` to the engine file and classify its grid.
    eng_dir = tmp_path / "src" / "rigor" / "strategies"
    eng_dir.mkdir(parents=True)
    (eng_dir / "demo_engine.py").write_text(
        "class Strategy:\n"
        "    def param_grid(self):\n"
        "        return {'k': [1, 2, 3]}\n", encoding="utf-8")
    strat = tmp_path / "strategies" / "cat" / "s"
    strat.mkdir(parents=True)
    reexport = strat / "strategy.py"
    reexport.write_text("from rigor.strategies.demo_engine import Strategy\n", encoding="utf-8")
    status, msg = classify_param_grid(reexport)
    assert status == "MULTI" and "shared engine" in msg


# ---------------------------------------------------------------------------
# check_strategy — levels
# ---------------------------------------------------------------------------

def test_multi_passes(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "good", grid="multi")
    assert check_strategy(d, baseline=set()).level == "PASS"


def test_new_violation_fails(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "bad", grid="empty")
    r = check_strategy(d, baseline=set())
    assert r.level == "FAIL" and r.failed


def test_grandfathered_violation_is_debt_not_failure(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "old", grid="empty")
    r = check_strategy(d, baseline={"old"})
    assert r.level == "WARN" and r.debt and not r.failed


def test_exempt_reason_exempts(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "cal", grid="empty", exempt_reason="fixed calendar, no knobs")
    r = check_strategy(d, baseline=set())
    assert r.level == "EXEMPT" and not r.failed


def test_whitespace_exempt_reason_does_not_exempt(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "cal", grid="empty", exempt_reason="   ")
    assert check_strategy(d, baseline=set()).level == "FAIL"


def test_dynamic_is_unverifiable_not_failure(tmp_path: Path) -> None:
    d = _make_strategy(tmp_path, "dyn", grid="dynamic")
    r = check_strategy(d, baseline=set())
    assert r.level == "UNVERIFIABLE" and not r.failed


# ---------------------------------------------------------------------------
# check_all — enumeration
# ---------------------------------------------------------------------------

def test_check_all_skips_baseline_subtree(tmp_path: Path) -> None:
    _make_strategy(tmp_path, "prod", grid="multi")
    _make_strategy(tmp_path, "raw", grid="empty", baseline=True)
    slugs = {r.slug for r in check_all(root=tmp_path, baseline=set())}
    assert "prod" in slugs and "raw" not in slugs


# ---------------------------------------------------------------------------
# Baseline round-trip + ratchet
# ---------------------------------------------------------------------------

def test_baseline_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    write_baseline({"b", "a", "c"}, path=p)
    assert load_baseline(p) == {"a", "b", "c"}
    # sorted + counted on disk
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["count"] == 3 and data["slugs"] == ["a", "b", "c"]


def test_stale_baseline_entries_flags_now_compliant(tmp_path: Path) -> None:
    d1 = _make_strategy(tmp_path, "fixed", grid="multi")   # now compliant
    d2 = _make_strategy(tmp_path, "still", grid="empty")   # still a violation
    results = [check_strategy(d1, {"fixed", "still", "gone"}),
               check_strategy(d2, {"fixed", "still", "gone"})]
    stale = stale_baseline_entries(results, {"fixed", "still", "gone"})
    assert "fixed" in stale       # backfilled -> removable
    assert "gone" in stale        # no longer exists -> removable
    assert "still" not in stale   # genuine remaining debt


# ---------------------------------------------------------------------------
# main() — exit codes
# ---------------------------------------------------------------------------

def _point_baseline(monkeypatch, tmp_path: Path, slugs: set[str]) -> None:
    p = tmp_path / "baseline.json"
    write_baseline(slugs, path=p)
    monkeypatch.setattr(og, "_BASELINE_PATH", p)


def test_main_exit_zero_when_only_grandfathered(tmp_path, monkeypatch, capsys) -> None:
    _make_strategy(tmp_path, "old", grid="empty")
    _make_strategy(tmp_path, "good", grid="multi")
    _point_baseline(monkeypatch, tmp_path, {"old"})
    assert main(["--root", str(tmp_path)]) == 0


def test_main_exit_one_on_new_violation(tmp_path, monkeypatch, capsys) -> None:
    _make_strategy(tmp_path, "fresh_bad", grid="empty")
    _point_baseline(monkeypatch, tmp_path, set())
    assert main(["--root", str(tmp_path)]) == 1


def test_main_strict_fails_on_debt(tmp_path, monkeypatch, capsys) -> None:
    _make_strategy(tmp_path, "old", grid="empty")
    _point_baseline(monkeypatch, tmp_path, {"old"})
    assert main(["--root", str(tmp_path)]) == 0            # lenient: debt is OK
    assert main(["--root", str(tmp_path), "--strict"]) == 1  # strict: debt fails


def test_main_update_baseline_snapshots_violations(tmp_path, monkeypatch, capsys) -> None:
    _make_strategy(tmp_path, "v1", grid="empty")
    _make_strategy(tmp_path, "v2", grid="single")
    _make_strategy(tmp_path, "ok", grid="multi")
    p = tmp_path / "baseline.json"
    monkeypatch.setattr(og, "_BASELINE_PATH", p)
    assert main(["--root", str(tmp_path), "--update-baseline"]) == 0
    assert load_baseline(p) == {"v1", "v2"}


def test_main_json_output(tmp_path, monkeypatch, capsys) -> None:
    _make_strategy(tmp_path, "good", grid="multi")
    _point_baseline(monkeypatch, tmp_path, set())
    main(["--root", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, list) and out[0]["slug"] == "good"
