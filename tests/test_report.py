"""Hermetic unit tests for ``rigor.report`` — the standardized HTML/JSON report card.

No network, no API key, no wall clock: every fixture is a deterministic returns
series fed through ``rigor.engine.result_from_returns``. We assert the public
contract of ``build_payload`` / ``render_html`` / ``write_report_card``:

  * files are written and non-empty;
  * the embedded JSON payload has the expected shape and finite numbers;
  * headline metrics and key sections appear in the HTML;
  * generating twice is byte-identical (determinism — no timestamps in bytes);
  * degenerate / empty returns do not raise.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from rigor.engine import result_from_returns
from rigor.report import build_payload, render_html, write_report_card


# --------------------------------------------------------------------------
# Fixtures — deterministic returns -> BacktestResult
# --------------------------------------------------------------------------
def _dated_returns(n: int = 400, seed: int = 7) -> pd.Series:
    """A fixed pseudo-random daily series with a DatetimeIndex.

    Seeded RNG => byte-identical across runs/machines; long enough (>1y) to
    exercise the annual/monthly resample and the rolling-Sharpe window.
    """
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0004, 0.011, n)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.Series(vals, index=idx, name="returns")


def _result(n: int = 400, seed: int = 7):
    return result_from_returns(_dated_returns(n, seed))


def _sample_validation() -> dict:
    """A minimal but shape-correct validation dict (as produced by the
    validation runner: ``{"verdict": Verdict.as_dict(), "overfit": {...}}``)."""
    return {
        "verdict": {
            "verdict": "CONDITIONAL",
            "grade": "MODERATE",
            "score": 71.5,
            "gates": [
                {"name": "sharpe", "passed": True, "value": 1.23,
                 "threshold": 0.5, "weight": 1.0, "critical": True, "neutral": False},
                {"name": "max_drawdown", "passed": False, "value": -0.31,
                 "threshold": -0.25, "weight": 1.0, "critical": False, "neutral": False},
                {"name": "regime_breadth", "passed": False, "value": "n/a",
                 "threshold": None, "weight": 0.0, "critical": False, "neutral": True},
            ],
        },
        "overfit": {
            "psr": 0.81, "dsr": 0.66, "harvey_t": 2.4, "haircut_pct": 0.18,
            "n_trials": 12, "newey_west": {"nw_sharpe": 1.05},
        },
    }


# --------------------------------------------------------------------------
# build_payload — shape + finiteness
# --------------------------------------------------------------------------
def test_build_payload_shape_and_keys():
    payload = build_payload(_result(), "Alpha One", as_of="2026-06-01")
    expected = {
        "name", "as_of", "diagnostics_html", "diagnostics_summary", "start", "end",
        "ppy", "metrics", "equity", "drawdown", "rolling_sharpe", "annual",
        "monthly", "distribution", "validation",
    }
    assert expected <= set(payload)
    assert payload["name"] == "Alpha One"
    assert payload["as_of"] == "2026-06-01"
    assert payload["validation"] is None
    assert payload["ppy"] > 0
    # Dated series -> ISO labels and resampled annual/monthly buckets present.
    assert payload["start"] == "2021-01-04"
    assert payload["annual"]["years"]  # at least one calendar year
    assert payload["monthly"]["table"]


def test_payload_series_lengths_aligned():
    res = _result()
    payload = build_payload(res, "X")
    n = len(res.returns)
    for section in ("equity", "drawdown", "rolling_sharpe"):
        assert len(payload[section]["labels"]) == n
        assert len(payload[section]["values"]) == n
    # Histogram: one count per bin, one center per bin.
    dist = payload["distribution"]
    assert len(dist["centers"]) == len(dist["counts"])
    assert sum(dist["counts"]) == n  # every observation lands in a bin


def test_payload_numbers_are_finite_and_json_safe():
    payload = build_payload(_result(), "X")
    # The payload is embedded as JSON in the HTML, so it must serialize.
    blob = json.dumps(payload, allow_nan=False, default=str)
    assert blob  # round-trips with no NaN/Inf leaking through

    # Equity/drawdown carry no NaN/Inf (rolling_sharpe legitimately uses None
    # for the warm-up window, which is JSON null — explicitly allowed).
    for section in ("equity", "drawdown"):
        for v in payload[section]["values"]:
            assert v is not None and math.isfinite(v)
    for v in payload["rolling_sharpe"]["values"]:
        assert v is None or math.isfinite(v)

    m = payload["metrics"]
    for key in ("sharpe", "cagr", "max_drawdown", "volatility", "sortino", "calmar"):
        assert math.isfinite(m[key])


def test_drawdown_is_non_positive():
    payload = build_payload(_result(), "X")
    # Underwater curve = equity/cummax - 1, always <= 0.
    assert all(v <= 1e-9 for v in payload["drawdown"]["values"])


def test_integer_index_has_no_annual_or_monthly():
    """A returns series with a non-datetime index renders, but the dated-only
    sections (annual / monthly) are empty and labels are stringified positions."""
    res = result_from_returns(pd.Series(np.linspace(-0.01, 0.01, 50)))
    payload = build_payload(res, "Plain")
    assert payload["annual"]["years"] == []
    assert payload["monthly"]["table"] == {}
    assert payload["start"] == "0"
    # Still renders without raising.
    assert render_html(payload)


# --------------------------------------------------------------------------
# render_html — sections + headline metrics
# --------------------------------------------------------------------------
def test_render_html_contains_sections_and_metrics():
    res = _result()
    payload = build_payload(res, "Headline Strat", as_of="2026-06-09")
    html = render_html(payload)

    assert html.startswith("<!doctype html>")
    assert "Headline Strat" in html
    assert "data as-of 2026-06-09" in html

    # Headline KPI labels.
    for label in ("Sharpe", "CAGR", "Max Drawdown", "Volatility", "Sortino",
                  "Calmar", "Win Rate", "Total Return"):
        assert label in html

    # Section headings.
    for section in ("Key Metrics", "Equity Curve", "Drawdown", "Rolling Sharpe",
                    "Annual Returns", "Monthly Returns", "Return Distribution"):
        assert section in html

    # The embedded JS payload that drives Chart.js.
    assert "window.Rigor=" in html
    # Chart canvases by id.
    for cid in ("eq", "dd", "rs", "annual", "dist"):
        assert f'id="{cid}"' in html


def test_embedded_json_payload_is_parseable_and_excludes_diagnostics_html():
    payload = build_payload(_result(), "Embed")
    html = render_html(payload)
    marker = "window.Rigor="
    start = html.index(marker) + len(marker)
    end = html.index("};", start) + 1
    embedded = json.loads(html[start:end])
    # diagnostics_html is intentionally kept OUT of the JS payload to save bytes.
    assert "diagnostics_html" not in embedded
    assert embedded["name"] == "Embed"
    assert embedded["metrics"]["n_obs"] == len(_result().returns)


def test_validation_section_rendered_when_present():
    payload = build_payload(_result(), "Validated", validation=_sample_validation())
    html = render_html(payload)
    assert "Validation" in html and "Verdict" in html
    assert "CONDITIONAL" in html          # verdict badge
    assert "grade MODERATE" in html
    assert "PSR" in html and "DSR" in html  # overfit cards
    assert "PASS" in html and "FAIL" in html and "n/a" in html  # gate marks


def test_no_validation_section_when_absent():
    html = render_html(build_payload(_result(), "Bare"))
    assert "Validation &amp; Verdict" not in html


# --------------------------------------------------------------------------
# write_report_card — files on disk
# --------------------------------------------------------------------------
def test_write_report_card_creates_nonempty_files(tmp_path):
    res = _result()
    paths = write_report_card(res, tmp_path, "my_strat", name="My Strat",
                              as_of="2026-06-01")
    assert set(paths) == {"html", "json"}
    html_path, json_path = paths["html"], paths["json"]

    assert html_path.name == "my_strat_report.html"
    assert json_path.name == "my_strat_summary.json"
    assert html_path.is_file() and json_path.is_file()
    assert html_path.stat().st_size > 0
    assert json_path.stat().st_size > 0

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["name"] == "My Strat"
    assert summary["as_of"] == "2026-06-01"
    assert summary["start"] == "2021-01-04"
    assert "metrics" in summary and math.isfinite(summary["metrics"]["sharpe"])
    # No verdict key unless validation was supplied.
    assert "verdict" not in summary


def test_write_report_card_creates_output_dir(tmp_path):
    nested = tmp_path / "deep" / "out"
    assert not nested.exists()
    paths = write_report_card(_result(), nested, "s")
    assert nested.is_dir()
    assert paths["html"].parent == nested


def test_write_report_card_summary_includes_verdict_with_validation(tmp_path):
    paths = write_report_card(_result(), tmp_path, "v_strat",
                              validation=_sample_validation())
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert summary["verdict"]["verdict"] == "CONDITIONAL"


# --------------------------------------------------------------------------
# Determinism — byte-identical bytes, no wall clock
# --------------------------------------------------------------------------
def test_render_html_byte_identical_twice():
    res = _result()
    a = render_html(build_payload(res, "Det", as_of="2026-06-01"))
    b = render_html(build_payload(res, "Det", as_of="2026-06-01"))
    assert a == b


def test_write_report_card_byte_identical_across_dirs(tmp_path):
    res = _result()
    d1, d2 = tmp_path / "a", tmp_path / "b"
    p1 = write_report_card(res, d1, "s", name="S", as_of="2026-06-01")
    p2 = write_report_card(res, d2, "s", name="S", as_of="2026-06-01")
    assert p1["html"].read_bytes() == p2["html"].read_bytes()
    assert p1["json"].read_bytes() == p2["json"].read_bytes()


def test_no_wallclock_year_leaks_into_bytes():
    """Determinism guard: the only years in the output must come from the
    fixture's data window, never the current wall-clock year."""
    import datetime as _dt

    res = _result()
    html = render_html(build_payload(res, "NoClock"))
    this_year = str(_dt.datetime.now().year)
    # The fixture data ends well before 2024; the current year (2026) must not
    # appear in the rendered bytes unless it happens to be a data year.
    data_years = {str(y) for y in build_payload(res, "NoClock")["annual"]["years"]}
    if this_year not in data_years:
        assert this_year not in html


# --------------------------------------------------------------------------
# Edge cases — degenerate inputs must not raise
# --------------------------------------------------------------------------
def test_empty_returns_does_not_raise(tmp_path):
    res = result_from_returns(pd.Series([], dtype="float64"))
    payload = build_payload(res, "Empty")
    assert payload["start"] is None
    assert payload["end"] is None
    html = render_html(payload)
    assert html  # renders something
    paths = write_report_card(res, tmp_path, "empty")
    assert paths["html"].is_file()
    assert paths["json"].is_file()


def test_single_observation_does_not_raise(tmp_path):
    res = result_from_returns(pd.Series([0.01]))
    html = render_html(build_payload(res, "One"))
    assert html
    write_report_card(res, tmp_path, "one")


def test_constant_zero_returns_does_not_raise(tmp_path):
    """Degenerate: zero variance => rolling Sharpe is 0/0; must coerce to
    JSON-null, never crash or emit NaN/Inf into the payload."""
    res = result_from_returns(
        pd.Series([0.0] * 60, index=pd.date_range("2022-01-03", periods=60, freq="B"))
    )
    payload = build_payload(res, "Flat")
    # Embeddable as strict JSON (no NaN/Inf).
    json.dumps(payload, allow_nan=False, default=str)
    html = render_html(payload)
    assert html
    paths = write_report_card(res, tmp_path, "flat")
    assert paths["html"].stat().st_size > 0


def test_all_negative_returns_renders(tmp_path):
    res = result_from_returns(
        pd.Series([-0.002] * 80, index=pd.date_range("2022-01-03", periods=80, freq="B"))
    )
    html = render_html(build_payload(res, "Down"))
    assert "Down" in html
    write_report_card(res, tmp_path, "down")


@pytest.mark.parametrize("n", [1, 5, 25, 300])
def test_various_lengths_render(n, tmp_path):
    res = _result(n=n)
    html = render_html(build_payload(res, f"len{n}"))
    assert html.startswith("<!doctype html>")
