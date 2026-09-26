"""Tests for rigor.project.governance — hermetic, uses tmp_path only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigor.project.governance import (
    ASSET_CLASSES,
    EDGE_TYPES,
    LEGACY_STATUS_TO_LIFECYCLE,
    LIFECYCLE,
    GovIssue,
    check_all,
    enrich,
    format_report,
    main,
    suggest_asset_class,
    summarize,
    validate_governance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    tmp_root: Path,
    category: str,
    slug: str,
    *,
    status: str = "idle",
    governance: dict | None = None,
    baseline: bool = False,
) -> dict:
    """Write a minimal config.json under *tmp_root*/Strategy[/Baseline]/<cat>/<slug>/."""
    if baseline:
        d = tmp_root / "strategies" / "Baseline" / category / slug
    else:
        d = tmp_root / "strategies" / category / slug
    d.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "name": slug.replace("_", " ").title(),
        "slug": slug,
        "category": category,
        "status": status,
        "version": "v1",
    }
    if governance is not None:
        cfg["governance"] = governance
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# 1. Valid governance block → no ERROR
# ---------------------------------------------------------------------------

def test_valid_governance_no_error():
    cfg = {
        "slug": "test_strat",
        "category": "momentum",
        "status": "idle",
        "governance": {
            "lifecycle": "backtested",
            "asset_class": "equity",
            "edge_type": "momentum",
        },
    }
    issues = validate_governance(cfg)
    errors = [i for i in issues if i.level == "ERROR"]
    assert not errors, f"Expected no ERRORs, got: {errors}"


# ---------------------------------------------------------------------------
# 2. Out-of-vocab edge_type → INVALID_VOCAB ERROR
# ---------------------------------------------------------------------------

def test_invalid_vocab_edge_type():
    cfg = {
        "slug": "bad_strat",
        "category": "crypto",
        "status": "live",
        "governance": {
            "lifecycle": "live",
            "asset_class": "crypto",
            "edge_type": "totally_made_up",   # not in EDGE_TYPES
        },
    }
    issues = validate_governance(cfg)
    errors = [i for i in issues if i.level == "ERROR" and i.code == "INVALID_VOCAB"]
    assert len(errors) == 1
    assert errors[0].field == "edge_type"


def test_invalid_vocab_lifecycle():
    cfg = {
        "slug": "x",
        "category": "macro",
        "status": "paper",
        "governance": {
            "lifecycle": "unknown_phase",
            "asset_class": "macro",
            "edge_type": "trend",
        },
    }
    issues = validate_governance(cfg)
    error_fields = {i.field for i in issues if i.level == "ERROR"}
    assert "lifecycle" in error_fields


# ---------------------------------------------------------------------------
# 3. No governance block → only WARN/INFO, never ERROR
# ---------------------------------------------------------------------------

def test_no_governance_block_advisory_only():
    cfg = {
        "slug": "no_gov",
        "category": "seasonal",
        "status": "idle",
    }
    issues = validate_governance(cfg)
    errors = [i for i in issues if i.level == "ERROR"]
    assert not errors, f"Expected no ERRORs for missing governance, got: {errors}"
    # Should have at least one WARN advisory.
    warns = [i for i in issues if i.level == "WARN"]
    assert warns


# ---------------------------------------------------------------------------
# 4. enrich — adds governance, respects legacy status + category, keeps keys
# ---------------------------------------------------------------------------

def test_enrich_adds_governance_defaults():
    cfg = {
        "slug": "my_strat",
        "category": "crypto",
        "status": "paper",
        "extra": {"foo": 42},
    }
    result = enrich(cfg)

    # governance sub-key added
    assert "governance" in result
    gov = result["governance"]

    # lifecycle derived from legacy status mapping
    assert gov["lifecycle"] == LEGACY_STATUS_TO_LIFECYCLE["paper"]  # "paper"

    # asset_class derived from category
    assert gov["asset_class"] == suggest_asset_class("crypto")       # "crypto"

    # all existing keys untouched
    assert result["slug"] == "my_strat"
    assert result["category"] == "crypto"
    assert result["status"] == "paper"
    assert result["extra"] == {"foo": 42}


def test_enrich_explicit_overrides():
    cfg = {"slug": "s", "category": "momentum", "status": "live"}
    result = enrich(
        cfg,
        lifecycle="validated",
        asset_class="fx",
        edge_type="trend",
        linked_thesis="thesis_s.md",
    )
    gov = result["governance"]
    assert gov["lifecycle"] == "validated"
    assert gov["asset_class"] == "fx"
    assert gov["edge_type"] == "trend"
    assert gov["linked_thesis"] == "thesis_s.md"


def test_enrich_does_not_mutate_original():
    cfg = {"slug": "s", "category": "macro", "status": "idle"}
    _ = enrich(cfg)
    assert "governance" not in cfg


def test_enrich_merges_existing_governance():
    cfg = {
        "slug": "s",
        "category": "carry",
        "status": "idle",
        "governance": {"edge_type": "carry", "extra_custom": "keep_me"},
    }
    result = enrich(cfg)
    gov = result["governance"]
    assert gov["edge_type"] == "carry"         # preserved from original
    assert gov["extra_custom"] == "keep_me"    # arbitrary extra key preserved
    assert "lifecycle" in gov                  # default added
    assert "asset_class" in gov               # default added


# ---------------------------------------------------------------------------
# 5. suggest_asset_class — known category mappings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,expected", [
    ("crypto", "crypto"),
    ("macro", "macro"),
    ("carry", "credit"),
    ("momentum", "equity"),
    ("mean_reversion", "equity"),
    ("trend_following", "equity"),
    ("seasonal", "equity"),
    ("event_driven", "equity"),
    ("breakout", "equity"),
    ("intraday", "equity"),
    ("unknown_cat", "equity"),   # fallback
])
def test_suggest_asset_class(category, expected):
    assert suggest_asset_class(category) == expected


# ---------------------------------------------------------------------------
# 6. summarize + main over a tmp root
# ---------------------------------------------------------------------------

def _build_tmp_root(tmp_path: Path) -> Path:
    """Create a small fake strategy tree for testing."""
    _make_config(tmp_path, "momentum", "strat_a", status="idle")
    _make_config(
        tmp_path, "crypto", "strat_b", status="live",
        governance={"lifecycle": "live", "asset_class": "crypto", "edge_type": "carry"},
    )
    _make_config(
        tmp_path, "carry", "strat_c", status="paper",
        governance={"lifecycle": "paper", "asset_class": "credit", "edge_type": "carry"},
    )
    # A baseline — should be excluded from the catalog.
    _make_config(tmp_path, "momentum", "strat_a", status="idle", baseline=True)
    return tmp_path


def test_summarize_counts(tmp_path):
    _build_tmp_root(tmp_path)
    s = summarize(tmp_path)

    assert s["n_total"] == 3               # 3 production, not the baseline
    assert s["n_with_explicit_governance"] == 2   # strat_b and strat_c have gov blocks

    # strat_a has no gov block → falls back to idle→backtested lifecycle
    assert s["by_lifecycle"].get("backtested", 0) >= 1
    assert s["by_lifecycle"].get("live", 0) == 1
    assert s["by_lifecycle"].get("paper", 0) == 1

    # asset class: strat_a (momentum→equity), strat_b (crypto), strat_c (credit)
    assert s["by_asset_class"].get("equity", 0) >= 1
    assert s["by_asset_class"].get("crypto", 0) == 1

    # No out-of-vocab errors in this clean tree.
    assert s["vocab_error_slugs"] == []


def test_summarize_with_vocab_error(tmp_path):
    _make_config(
        tmp_path, "momentum", "bad_strat", status="idle",
        governance={"lifecycle": "invented_phase", "asset_class": "equity", "edge_type": "trend"},
    )
    s = summarize(tmp_path)
    assert "bad_strat" in s["vocab_error_slugs"]


def test_main_advisory_returns_zero(tmp_path):
    _build_tmp_root(tmp_path)
    ret = main(["--root", str(tmp_path)])
    assert ret == 0


def test_main_strict_returns_one_on_vocab_error(tmp_path):
    _make_config(
        tmp_path, "momentum", "bad_strat", status="idle",
        governance={"lifecycle": "invented_phase", "asset_class": "equity", "edge_type": "trend"},
    )
    ret = main(["--root", str(tmp_path), "--strict"])
    assert ret == 1


def test_main_strict_clean_returns_zero(tmp_path):
    _build_tmp_root(tmp_path)
    ret = main(["--root", str(tmp_path), "--strict"])
    assert ret == 0


def test_main_json_output(tmp_path, capsys):
    _build_tmp_root(tmp_path)
    ret = main(["--root", str(tmp_path), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "summary" in data
    assert "issues" in data


# ---------------------------------------------------------------------------
# 7. Vocabulary constants sanity checks
# ---------------------------------------------------------------------------

def test_vocab_constants_are_tuples_of_strings():
    for vocab in (LIFECYCLE, ASSET_CLASSES, EDGE_TYPES):
        assert isinstance(vocab, tuple)
        assert all(isinstance(v, str) for v in vocab)


def test_legacy_status_mapping_values_in_lifecycle():
    for v in LEGACY_STATUS_TO_LIFECYCLE.values():
        assert v in LIFECYCLE


# ---------------------------------------------------------------------------
# 8. check_all and format_report smoke
# ---------------------------------------------------------------------------

def test_check_all_and_format_report(tmp_path):
    _build_tmp_root(tmp_path)
    result = check_all(tmp_path)
    assert "summary" in result
    assert "issues" in result
    report = format_report(result)
    assert "Governance Report" in report
    assert "Production strategies" in report


def test_validate_governance_linked_research_bad_type():
    cfg = {
        "slug": "s",
        "category": "macro",
        "status": "idle",
        "governance": {
            "lifecycle": "backtested",
            "asset_class": "macro",
            "edge_type": "trend",
            "linked_research": 42,   # should be str or list[str]
        },
    }
    issues = validate_governance(cfg)
    errors = [i for i in issues if i.level == "ERROR" and i.field == "linked_research"]
    assert errors


def test_validate_governance_linked_research_list_ok():
    cfg = {
        "slug": "s",
        "category": "macro",
        "status": "idle",
        "governance": {
            "lifecycle": "backtested",
            "asset_class": "macro",
            "edge_type": "trend",
            "linked_research": ["paper_a.md", "paper_b.md"],
        },
    }
    issues = validate_governance(cfg)
    errors = [i for i in issues if i.level == "ERROR"]
    assert not errors


def test_govissue_is_dataclass():
    issue = GovIssue(code="C", level="WARN", field="f", message="m")
    assert issue.code == "C"
    assert issue.level == "WARN"
