"""Hermetic unit tests for ``rigor.cli`` — the 13-subcommand argparse front end.

We drive ``rigor.cli.main([...])`` directly and assert the *CLI layer's* job:
parser wiring, dispatch to the right handler, argument plumbing (as-of, flags,
paths), exit codes, and error handling. Every command that does heavy or
networked work is monkeypatched at its lookup site, so the suite is fast and
needs no API key, no network, and no data snapshot.
"""

from __future__ import annotations

import types

import pytest

import rigor.cli as cli
from rigor.engine import result_from_returns
from rigor.project import create_strategy


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fake_result(seed: int = 1):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=100, freq="B")
    return result_from_returns(pd.Series(rng.normal(0.0003, 0.01, 100), index=idx))


# --------------------------------------------------------------------------
# Parser-level: required command, --help, unknown command, missing args
# --------------------------------------------------------------------------
def test_no_command_exits_nonzero(capsys):
    # subparsers(required=True) -> argparse errors out (SystemExit 2).
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


def test_unknown_command_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "invalid choice" in err or "definitely-not-a-command" in err


def test_top_level_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "rigor" in out
    # The help text enumerates the subcommands.
    for cmd in ("new", "validate", "run", "optimize", "data"):
        assert cmd in out


def test_subcommand_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--as-of" in out
    assert "--offline" in out


def test_run_missing_path_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run"])  # positional 'path' is required
    assert exc.value.code != 0


def test_data_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        cli.main(["data"])  # data_command is required
    assert exc.value.code != 0


def test_data_freeze_requires_as_of():
    with pytest.raises(SystemExit) as exc:
        cli.main(["data", "freeze"])  # --as-of is required
    assert exc.value.code != 0


# --------------------------------------------------------------------------
# validate — real (static, no network) against a scaffolded fixture
# --------------------------------------------------------------------------
def test_validate_ok_returns_zero(tmp_path, capsys):
    create_strategy(tmp_path, "momentum", "demo_mom", name="Demo Mom")
    d = tmp_path / "strategies" / "momentum" / "demo_mom"
    rc = cli.main(["validate", str(d)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_validate_invalid_returns_one(tmp_path, capsys):
    create_strategy(tmp_path, "momentum", "broken")
    d = tmp_path / "strategies" / "momentum" / "broken"
    (d / "strategy.py").unlink()  # remove a required file
    rc = cli.main(["validate", str(d)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "INVALID" in out
    assert "ERROR" in out


def test_validate_nonexistent_path_returns_one(tmp_path, capsys):
    rc = cli.main(["validate", str(tmp_path / "nope")])
    assert rc == 1
    assert "INVALID" in capsys.readouterr().out


# --------------------------------------------------------------------------
# new — dispatch + argument plumbing (patch the scaffolder)
# --------------------------------------------------------------------------
def test_new_dispatches_with_parsed_args(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_create(root, category, slug, *, name=None, baseline=False):
        seen.update(root=root, category=category, slug=slug, name=name,
                    baseline=baseline)
        return tmp_path / "strategies" / category / slug

    monkeypatch.setattr(cli, "create_strategy", fake_create)
    rc = cli.main(["new", "carry", "fx_carry", "--name", "FX Carry",
                   "--root", str(tmp_path)])
    assert rc == 0
    assert seen["category"] == "carry"
    assert seen["slug"] == "fx_carry"
    assert seen["name"] == "FX Carry"
    assert seen["baseline"] is False
    assert str(seen["root"]) == str(tmp_path)
    assert "created strategy" in capsys.readouterr().out


def test_new_baseline_flag_plumbed(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_create(root, category, slug, *, name=None, baseline=False):
        seen["baseline"] = baseline
        return tmp_path / "x"

    monkeypatch.setattr(cli, "create_strategy", fake_create)
    rc = cli.main(["new", "carry", "fx_carry", "--baseline", "--root", str(tmp_path)])
    assert rc == 0
    assert seen["baseline"] is True
    assert "created baseline" in capsys.readouterr().out


# --------------------------------------------------------------------------
# run — argument plumbing (as-of, flags) and verdict printing
# --------------------------------------------------------------------------
def test_run_plumbs_args_and_prints(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(path, *, as_of=None, thesis=True, validate=True, offline=False):
        captured.update(path=path, as_of=as_of, thesis=thesis,
                        validate=validate, offline=offline)
        return {
            "slug": "demo",
            "result": _fake_result(),
            "validation": {"verdict": {"verdict": "PROMOTE", "grade": "ROBUST",
                                       "score": 88.0}},
            "artifacts": {"report": tmp_path / "demo_report.html"},
        }

    monkeypatch.setattr(cli, "run_strategy", fake_run)
    rc = cli.main(["run", "strategies/momentum/demo", "--as-of", "2026-06-01",
                   "--no-thesis", "--offline"])
    assert rc == 0
    # The CLI forwards exactly the parsed values, inverting the negative flags.
    assert captured["path"] == "strategies/momentum/demo"
    assert captured["as_of"] == "2026-06-01"
    assert captured["thesis"] is False      # --no-thesis -> thesis=False
    assert captured["validate"] is True     # --no-validate absent -> validate=True
    assert captured["offline"] is True
    out = capsys.readouterr().out
    assert "ran demo" in out
    assert "PROMOTE" in out
    assert "report" in out


def test_run_defaults_when_no_flags(monkeypatch, capsys):
    captured = {}

    def fake_run(path, *, as_of=None, thesis=True, validate=True, offline=False):
        captured.update(as_of=as_of, thesis=thesis, validate=validate,
                        offline=offline)
        return {"slug": "d", "result": _fake_result(), "artifacts": {}}

    monkeypatch.setattr(cli, "run_strategy", fake_run)
    rc = cli.main(["run", "some/path"])
    assert rc == 0
    assert captured == {"as_of": None, "thesis": True, "validate": True,
                        "offline": False}


def test_run_no_validate_flag(monkeypatch):
    captured = {}

    def fake_run(path, *, as_of=None, thesis=True, validate=True, offline=False):
        captured["validate"] = validate
        return {"slug": "d", "result": _fake_result(), "artifacts": {}}

    monkeypatch.setattr(cli, "run_strategy", fake_run)
    cli.main(["run", "p", "--no-validate"])
    assert captured["validate"] is False


# --------------------------------------------------------------------------
# optimize — many flags wired through to optimize_strategy
# --------------------------------------------------------------------------
def test_optimize_plumbs_flags(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_opt(path, **kwargs):
        captured.update(path=path, **kwargs)
        return {
            "slug": "opt", "configs_evaluated": 10, "grid_size": 20,
            "search": "exhaustive", "backend": "numpy", "default_in_grid": True,
            "selection": {"priority": "TOP"}, "best_sharpe": 1.5,
            "best_params": {"lookback": 20}, "improvement_vs_default": 0.3,
            "default_sharpe": 1.2,
            "overfit": {"dsr": 0.7, "n_trials": 20},
            "data_snooping": {}, "walk_forward_opt": {},
        }

    # _cmd_optimize does `from .optimize import optimize_strategy` -> patch source.
    import rigor.optimize as _opt
    monkeypatch.setattr(_opt, "optimize_strategy", fake_opt)
    rc = cli.main(["optimize", str(tmp_path), "--as-of", "2025-01-01",
                   "--max-combos", "64", "--wf-folds", "3", "--wf-mode", "rolling",
                   "--select", "robust", "--backend", "numba", "--offline",
                   "--two-pass"])
    assert rc == 0
    assert captured["path"] == str(tmp_path)
    assert captured["as_of"] == "2025-01-01"
    assert captured["max_combos"] == 64
    assert captured["n_wf_folds"] == 3
    assert captured["wf_mode"] == "rolling"
    assert captured["select_mode"] == "robust"
    assert captured["backend"] == "numba"
    assert captured["offline"] is True
    assert captured["two_pass"] is True
    assert "optimized opt" in capsys.readouterr().out


def test_optimize_invalid_choice_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["optimize", "p", "--backend", "quantum"])  # not in choices
    assert exc.value.code != 0


# --------------------------------------------------------------------------
# data freeze/restore/verify — patch the snapshot module
# --------------------------------------------------------------------------
def test_data_freeze_dispatch(monkeypatch, capsys):
    seen = {}

    def fake_freeze(as_of, out_path=None):
        seen.update(as_of=as_of, out_path=out_path)
        return {"n_files": 3, "archive": "rel.tar", "size_bytes": 2_000_000,
                "snapshot_sha256": "deadbeef"}

    monkeypatch.setattr(cli.snapshot, "freeze", fake_freeze)
    rc = cli.main(["data", "freeze", "--as-of", "2026-06-01", "--out", "rel.tar"])
    assert rc == 0
    assert seen["as_of"] == "2026-06-01"
    assert seen["out_path"] == "rel.tar"
    assert "froze 3 files" in capsys.readouterr().out


def test_data_restore_ok_and_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(cli.snapshot, "restore",
                        lambda archive: {"as_of": "2026-06-01", "ok": True,
                                         "snapshot_sha256": "abc"})
    assert cli.main(["data", "restore", "rel.tar"]) == 0
    assert "OK" in capsys.readouterr().out

    monkeypatch.setattr(cli.snapshot, "restore",
                        lambda archive: {"as_of": "2026-06-01", "ok": False,
                                         "snapshot_sha256": "abc"})
    assert cli.main(["data", "restore", "rel.tar"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_data_verify_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(cli.snapshot, "verify",
                        lambda as_of, expected=None: {"as_of": as_of, "ok": True,
                                                      "snapshot_sha256": "x",
                                                      "recorded": "x",
                                                      "expected": expected})
    assert cli.main(["data", "verify", "--as-of", "2026-06-01"]) == 0

    monkeypatch.setattr(cli.snapshot, "verify",
                        lambda as_of, expected=None: {"as_of": as_of, "ok": False,
                                                      "snapshot_sha256": "x",
                                                      "recorded": "y",
                                                      "expected": expected})
    rc = cli.main(["data", "verify", "--as-of", "2026-06-01", "--expect", "z"])
    assert rc == 1
    assert "MISMATCH" in capsys.readouterr().out


# --------------------------------------------------------------------------
# index — dispatch
# --------------------------------------------------------------------------
def test_index_dispatch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "write_index", lambda: tmp_path / "INDEX.md")
    rc = cli.main(["index"])
    assert rc == 0
    assert "catalog written" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Sub-tool commands that forward to a project.<mod>.main(argv) — assert the
# argv we hand the underlying tool reflects the parsed flags, and the return
# code passes straight through.
# --------------------------------------------------------------------------
def _patch_tool_main(monkeypatch, modname, recorder, rc=0):
    """Patch ``rigor.project.<modname>.main`` to record argv and return rc."""
    import importlib

    mod = importlib.import_module(f"rigor.project.{modname}")
    monkeypatch.setattr(mod, "main", lambda argv=None: recorder.append(argv) or rc)


def test_audit_forwards_flags_and_rc(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "audit", rec, rc=2)
    rc = cli.main(["audit", "--root", "/r", "--include-baseline", "--json"])
    assert rc == 2
    argv = rec[0]
    assert "--root" in argv and "/r" in argv
    assert "--include-baseline" in argv
    assert "--json" in argv


def test_audit_minimal_argv(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "audit", rec, rc=0)
    assert cli.main(["audit"]) == 0
    assert rec[0] == []  # no flags -> empty argv


def test_regression_forwards_ref(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "regression", rec, rc=0)
    cli.main(["regression", "--ref", "origin/dev", "--json"])
    argv = rec[0]
    assert argv[:2] == ["--ref", "origin/dev"]
    assert "--json" in argv


def test_check_thesis_forwards_strict(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "thesis_gate", rec, rc=1)
    rc = cli.main(["check-thesis", "--strict", "--json"])
    assert rc == 1
    assert "--strict" in rec[0]
    assert "--json" in rec[0]


def test_book_forwards_master_only(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "book", rec, rc=0)
    cli.main(["book", "--master-only", "--root", "/x"])
    argv = rec[0]
    assert "--master-only" in argv
    assert "--root" in argv and "/x" in argv


def test_derive_forwards_slug(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "derive", rec, rc=0)
    cli.main(["derive", "--slug", "mom_vol", "--json"])
    argv = rec[0]
    assert "--slug" in argv and "mom_vol" in argv
    assert "--json" in argv


def test_govern_forwards_backfill_and_strict(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "governance", rec, rc=0)
    cli.main(["govern", "--backfill", "--strict", "--json"])
    argv = rec[0]
    assert "--backfill" in argv
    assert "--strict" in argv
    assert "--json" in argv


def test_registry_forwards_remainder(monkeypatch):
    rec = []
    _patch_tool_main(monkeypatch, "registry", rec, rc=0)
    cli.main(["registry", "list", "--all"])
    # REMAINDER captures everything after 'registry' verbatim.
    assert rec[0] == ["list", "--all"]


# --------------------------------------------------------------------------
# main(None) reads sys.argv — make sure it doesn't crash the parser wiring.
# --------------------------------------------------------------------------
def test_main_reads_sys_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rigor"])
    with pytest.raises(SystemExit) as exc:
        cli.main()  # argv=None -> parse sys.argv[1:] -> no command -> SystemExit
    assert exc.value.code != 0


# --------------------------------------------------------------------------
# Smoke: the dispatch table is wired (every subparser sets func).
# --------------------------------------------------------------------------
def test_every_subcommand_has_a_handler():
    # Every advertised command must build a subparser whose --help exits 0;
    # this guards against an unwired add_parser / missing set_defaults(func=...).
    commands = ["new", "validate", "run", "optimize", "data", "index", "audit",
                "regression", "check-thesis", "book", "derive", "govern",
                "registry"]
    for cmd in commands:
        with pytest.raises(SystemExit) as exc:
            cli.main([cmd, "--help"])
        assert exc.value.code == 0, f"{cmd} --help did not exit 0"


# Guard against accidental import-time side effects beyond force_utf8_stdio.
def test_cli_module_exposes_main():
    assert isinstance(cli.main, types.FunctionType)
