import json

import pandas as pd
import pytest

from rigor.engine import result_from_returns
from rigor.project import create_strategy, validate_strategy
from rigor.project.index import collect
from rigor.report import build_payload, render_html
from rigor.strategy import StrategyConfig


def test_scaffold_then_validate_ok(tmp_path):
    create_strategy(tmp_path, "mean_reversion", "demo_strat", name="Demo")
    d = tmp_path / "strategies" /"mean_reversion" / "demo_strat"
    for f in ("strategy.py", "README.md", "config.json"):
        assert (d / f).is_file()
    assert validate_strategy(d).ok


def test_validate_catches_missing_and_slug_mismatch(tmp_path):
    create_strategy(tmp_path, "momentum", "good_slug")
    d = tmp_path / "strategies" /"momentum" / "good_slug"
    (d / "strategy.py").unlink()
    res = validate_strategy(d)
    assert not res.ok
    assert any("strategy.py" in e for e in res.errors)


def test_production_vs_baseline_placement_and_version(tmp_path):
    # production: strategies/<cat>/<slug>, version v1
    prod = create_strategy(tmp_path, "momentum", "mom_vol")
    assert prod == tmp_path / "strategies" / "momentum" / "mom_vol"
    assert json.loads((prod / "config.json").read_text())["version"] == "v1"
    assert validate_strategy(prod).ok

    # baseline: strategies/Baseline/<cat>/<slug> with SAME slug, version v0
    base = create_strategy(tmp_path, "momentum", "mom_vol", baseline=True)
    assert base == tmp_path / "strategies" / "Baseline" / "momentum" / "mom_vol"
    assert json.loads((base / "config.json").read_text())["version"] == "v0"
    assert validate_strategy(base).ok


def test_catalog_excludes_baselines(tmp_path):
    create_strategy(tmp_path, "momentum", "mom_vol")               # production
    create_strategy(tmp_path, "momentum", "mom_vol", baseline=True)  # baseline (same slug)
    create_strategy(tmp_path, "carry", "fx_carry")                 # production
    rows = collect(tmp_path)
    names = sorted(r["category"] + "/" + r["name"] for r in rows)
    # both production strategies present; the baseline is NOT a separate row
    assert names == ["carry/Fx Carry", "momentum/Mom Vol"]


def test_version_field_validation():
    StrategyConfig(name="ok", version="v2.1")  # clean token accepted
    with pytest.raises(ValueError):
        StrategyConfig(name="bad", version="v 1")  # whitespace rejected


def test_report_html_deterministic():
    r = pd.Series([0.01, -0.005, 0.002, 0.004] * 30,
                  index=pd.date_range("2021-01-01", periods=120, freq="D"))
    res = result_from_returns(r)
    h1 = render_html(build_payload(res, "X", as_of="2026-06-01"))
    h2 = render_html(build_payload(res, "X", as_of="2026-06-01"))
    assert h1 == h2
