"""Rigor command line: scaffold, validate, and run strategies to the standard.

    python -m rigor new <category> <slug> [--name "..."]
    python -m rigor validate strategies/<category>/<slug>
    python -m rigor run      strategies/<category>/<slug> [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .console import force_utf8_stdio
from .data import snapshot
from .project import create_strategy, run_strategy, validate_strategy, write_index


def _cmd_new(args: argparse.Namespace) -> int:
    path = create_strategy(
        Path(args.root), args.category, args.slug, name=args.name, baseline=args.baseline
    )
    kind = "baseline" if args.baseline else "strategy"
    print(f"created {kind} {path}")
    print("  edit strategy.py + config.json, then:")
    print(f"  python -m rigor run {path.relative_to(args.root)} --as-of <YYYY-MM-DD>")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    res = validate_strategy(args.path)
    for w in res.warnings:
        print(f"  warn:  {w}")
    for e in res.errors:
        print(f"  ERROR: {e}")
    print(f"{'OK' if res.ok else 'INVALID'}: {res.path}")
    return 0 if res.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    out = run_strategy(args.path, as_of=args.as_of, thesis=not args.no_thesis,
                       validate=not args.no_validate, offline=args.offline)
    m = out["result"].metrics
    print(f"ran {out['slug']}: sharpe={m['sharpe']:.2f} cagr={m['cagr']:.2%} "
          f"maxdd={m['max_drawdown']:.2%} (n={m['n_obs']})")
    v = out.get("validation")
    rc = 0
    if v:
        vd = v["verdict"]
        print(f"  verdict : {vd['verdict']}  (grade {vd['grade']}, score {vd['score']}/100)")
        # Opt-in promotion gate: with --gate, a REJECT verdict fails the command so
        # CI/pre-commit hooks can block shipping an overfit strategy. Default off —
        # behaviour is unchanged unless --gate is passed.
        if args.gate and vd["verdict"] == "REJECT":
            print("  GATE   : REJECT verdict — failing (run without --gate to ignore)")
            rc = 1
    for label, p in out["artifacts"].items():
        print(f"  {label:8}: {p}")
    return rc


def _cmd_optimize(args: argparse.Namespace) -> int:
    # --profile runs the unified pipeline (research -> selection -> deployment-
    # readiness) under a named calibration profile. Default (None) preserves the
    # legacy per-stage report exactly — nothing about the old path changes.
    if args.profile:
        return _cmd_optimize_pipeline(args)
    from .optimize import optimize_strategy
    r = optimize_strategy(
        args.path, as_of=args.as_of, max_combos=args.max_combos, n_wf_folds=args.wf_folds,
        wf_mode=args.wf_mode, select_mode=args.select, two_pass=args.two_pass,
        ensemble=args.ensemble, full=args.full, enrich=args.enrich, holdout=args.holdout,
        falsify=args.falsify, score=args.score, regime=args.regime, cluster=args.cluster,
        backend=args.backend, offline=args.offline)
    print(f"optimized {r['slug']}: {r['configs_evaluated']}/{r['grid_size']} configs "
          f"({r['search']} search, {r['backend']} backend)")
    if not r.get("default_in_grid", True):
        print("  WARNING: default_params is not a cell in param_grid() — "
              "default comparison is unavailable (add the default values to the grid)")
    sel = r["selection"]
    if sel["priority"] == "REJECTED":
        print(f"  select : REJECTED ({sel.get('reason', '')})")
    print(f"  best   : sharpe={r['best_sharpe']:.2f}  {r['best_params']}")
    imp = r["improvement_vs_default"]
    print(f"  default: sharpe={r['default_sharpe']:.2f}  (improvement {imp:+.2f})")
    o = r["overfit"]
    print(f"  overfit: deflated_sharpe={o['dsr']:.3f} over n_trials={o['n_trials']}")
    if r.get("cpcv"):
        print(f"  cpcv   : PBO={r['cpcv']['pbo']:.2f} ({r['cpcv']['verdict']})")
    snoop = r["data_snooping"]
    if "white_reality_check" in snoop:
        wrc = snoop["white_reality_check"]["p_value"]
        spa = snoop["hansen_spa"]["p_consistent"]
        rw = snoop["romano_wolf_stepm"]
        print(f"  snoop  : White p={wrc:.3f}  Hansen-SPA p={spa:.3f}  "
              f"StepM beats {rw['n_rejected']}/{rw['n_total']}")
    wf = r["walk_forward_opt"]
    if wf.get("n_folds"):
        print(f"  walk-fwd: IS={wf['is_sharpe_mean']:.2f} -> OOS={wf['oos_sharpe_mean']:.2f} "
              f"({wf['mode']}, degradation {wf['is_to_oos_degradation']:+.2f}, "
              f"{wf['oos_positive_folds']}/{wf['n_folds']} folds positive)")
    if r.get("ensemble") and r["ensemble"].get("members"):
        e = r["ensemble"]
        print(f"  ensemble: {e['n_members']} configs  sharpe={e['ensemble_sharpe']:.2f} "
              f"(vs best {e['ensemble_vs_best']:+.2f})")
    if r.get("holdout") and "holdout_sharpe" in r["holdout"]:
        h = r["holdout"]
        flag = "DEGRADED" if h["degraded"] else "ok"
        print(f"  holdout: IS={h['is_sharpe']:.2f} OOS={h['oos_sharpe']:.2f} "
              f"hold={h['holdout_sharpe']:.2f} ({flag})")
    if r.get("falsification", {}).get("noise_injection"):
        ni = r["falsification"]["noise_injection"]
        print(f"  falsify: noise fragility={ni.get('fragility')}  "
              f"suspect={r['falsification'].get('suspect')}")
    if r.get("selection_score_5pillar", {}).get("best_composite") is not None:
        sc = r["selection_score_5pillar"]
        print(f"  5-pillar: composite={sc['best_composite']:.2f}  {sc['best_params']}")
    if r.get("regime", {}).get("consensus_params"):
        rg = r["regime"]
        print(f"  regime : {rg['n_regimes']} regimes, consensus "
              f"({rg['consensus_votes']} votes) {rg['consensus_params']}")
    if r.get("clustering", {}).get("n_clusters"):
        print(f"  cluster: {r['clustering']['n_clusters']} parameter clusters")
    print(f"  report : {Path(args.path) / 'artifacts' / (r['slug'] + '_optimization.json')}")
    return 0


def _cmd_optimize_pipeline(args: argparse.Namespace) -> int:
    """``rigor optimize --profile {discovery|deployment}`` — the unified pipeline.

    One command: search -> validate -> select under the profile's floors -> gate
    the winner for deployment-readiness against the shared promotion verdict.
    With ``deployment`` the command exits non-zero on a REJECT outcome so a CI /
    pre-push hook can block shipping a winner that is not deployment-ready.
    """
    from .optimize import run_pipeline
    res = run_pipeline(
        args.path, profile=args.profile, as_of=args.as_of, max_combos=args.max_combos,
        offline=args.offline, write=True)
    s = res.stages
    print(f"optimized {res.slug}: {s['configs_evaluated']}/{s['grid_size']} configs "
          f"(profile={res.profile})")
    if not s.get("default_in_grid", True):
        print("  WARNING: default_params is not a cell in param_grid() — "
              "default comparison is unavailable (add the default values to the grid)")
    sel = s["selection"]
    if sel["priority"] == "REJECTED":
        print(f"  select : REJECTED ({sel.get('reason', '')})")
    else:
        print(f"  select : {sel['priority']}  sharpe={s['best_sharpe']:.2f}  {s['best_params']}")
    o = s["overfit"]
    print(f"  overfit: deflated_sharpe={o['dsr']:.3f} over n_trials={o['n_trials']}")
    if s.get("cpcv"):
        print(f"  cpcv   : PBO={s['cpcv']['pbo']:.2f} ({s['cpcv']['verdict']})")
    vd = s["verdict"]
    print(f"  verdict: {vd['verdict']}  (grade {vd['grade']}, score {vd['score']}/100)")
    for c in res.gate["checks"]:
        mark = "·" if c.get("neutral") else "ok " if c.get("passed") else "FAIL"
        extra = c.get("detail") or f"{c.get('value')} vs {c.get('threshold')}"
        print(f"  gate   : [{mark}] {c['name']:14} {extra}")
    print(f"  OUTCOME: {res.outcome}  (deploy_ready={res.deploy_ready})")
    report_name = f"{res.slug}_pipeline_{res.profile}.json"
    print(f"  report : {Path(args.path) / 'artifacts' / report_name}")
    # deployment is a gate: a non-deploy-ready winner exits non-zero so CI can
    # block. discovery is advisory and always exits 0.
    return 0 if (res.deploy_ready or res.profile != "deployment") else 1


def _cmd_data_freeze(args: argparse.Namespace) -> int:
    info = snapshot.freeze(args.as_of, out_path=args.out)
    mb = info["size_bytes"] / 1e6
    print(f"froze {info['n_files']} files -> {info['archive']} ({mb:.1f} MB)")
    print(f"  snapshot sha256: {info['snapshot_sha256']}")
    print("  share the archive + this hash; others: rigor data restore <archive>,"
          " then run --offline")
    return 0


def _cmd_data_restore(args: argparse.Namespace) -> int:
    res = snapshot.restore(args.archive)
    print(f"restored snapshot {res['as_of']}  (integrity {'OK' if res['ok'] else 'MISMATCH'})")
    print(f"  snapshot sha256: {res['snapshot_sha256']}")
    return 0 if res["ok"] else 1


def _cmd_data_verify(args: argparse.Namespace) -> int:
    res = snapshot.verify(args.as_of, expected=args.expect)
    status = "OK" if res["ok"] else "MISMATCH"
    print(f"snapshot {res['as_of']}: {status}")
    print(f"  computed: {res['snapshot_sha256']}")
    if res["recorded"]:
        print(f"  recorded: {res['recorded']}")
    if res["expected"]:
        print(f"  expected: {res['expected']}")
    return 0 if res["ok"] else 1


def _cmd_index(args: argparse.Namespace) -> int:
    path = write_index()
    print(f"strategy catalog written -> {path}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .project import audit
    argv: list[str] = []
    if args.root:
        argv += ["--root", args.root]
    if args.include_baseline:
        argv += ["--include-baseline"]
    if args.json:
        argv += ["--json"]
    return audit.main(argv)


def _cmd_regression(args: argparse.Namespace) -> int:
    from .project import regression
    argv = ["--ref", args.ref]
    if args.root:
        argv += ["--root", args.root]
    if args.json:
        argv += ["--json"]
    return regression.main(argv)


def _cmd_check_thesis(args: argparse.Namespace) -> int:
    from .project import thesis_gate
    argv: list[str] = []
    if args.root:
        argv += ["--root", args.root]
    if args.strict:
        argv += ["--strict"]
    if args.json:
        argv += ["--json"]
    return thesis_gate.main(argv)


def _cmd_book(args: argparse.Namespace) -> int:
    from .project import book
    argv: list[str] = []
    if args.root:
        argv += ["--root", args.root]
    if args.master_only:
        argv += ["--master-only"]
    if args.json:
        argv += ["--json"]
    return book.main(argv)


def _cmd_derive(args: argparse.Namespace) -> int:
    from .project import derive
    argv: list[str] = []
    if args.root:
        argv += ["--root", args.root]
    if args.slug:
        argv += ["--slug", args.slug]
    if args.json:
        argv += ["--json"]
    return derive.main(argv)


def _cmd_govern(args: argparse.Namespace) -> int:
    from .project import governance
    argv: list[str] = []
    if args.root:
        argv += ["--root", args.root]
    if args.backfill:
        argv += ["--backfill"]
    if args.strict:
        argv += ["--strict"]
    if args.json:
        argv += ["--json"]
    return governance.main(argv)


def _cmd_registry(args: argparse.Namespace) -> int:
    from .project import registry
    return registry.main(args.registry_args)


def _cmd_demo(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .demo import run_demo
    return run_demo(Path(args.out))


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor",
        description="rigor: reproducible strategy research (backtest, report, thesis, verdict)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="offline tour: validate and optimise a strategy on pure noise"
    )
    p_demo.add_argument(
        "--out", default="rigor-demo", help="where to write the report card and thesis"
    )
    p_demo.set_defaults(func=_cmd_demo)

    p_new = sub.add_parser("new", help="scaffold a new strategy folder")
    p_new.add_argument("category")
    p_new.add_argument("slug")
    p_new.add_argument("--name", default=None)
    p_new.add_argument("--baseline", action="store_true",
                       help="scaffold the raw version under strategies/Baseline/<category>/")
    p_new.add_argument("--root", default=".")
    p_new.set_defaults(func=_cmd_new)

    p_val = sub.add_parser("validate", help="check a strategy folder conforms")
    p_val.add_argument("path")
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="backtest a strategy and write artifacts")
    p_run.add_argument("path")
    p_run.add_argument("--as-of", default=None, help="pin the data as-of date (YYYY-MM-DD)")
    p_run.add_argument("--no-thesis", action="store_true", help="skip thesis PDF generation")
    p_run.add_argument("--no-validate", action="store_true", help="skip the validation battery")
    p_run.add_argument("--offline", action="store_true",
                       help="read only the frozen snapshot; never fetch live (reproducible)")
    p_run.add_argument("--gate", "--strict", action="store_true", dest="gate",
                       help="exit non-zero if the validation verdict is REJECT (opt-in)")
    p_run.set_defaults(func=_cmd_run)

    p_opt = sub.add_parser("optimize", help="grid-search a strategy's param_grid (opt-in)")
    p_opt.add_argument("path")
    p_opt.add_argument("--as-of", default=None, help="pin the data as-of date (YYYY-MM-DD)")
    p_opt.add_argument("--profile", choices=("discovery", "deployment"), default=None,
                       help="run the unified pipeline under a calibration profile: 'discovery' "
                            "(permissive research floors, advisory) or 'deployment' (strict "
                            "production floors; exits non-zero unless the winner is "
                            "deployment-ready). Default (omitted) preserves the legacy report.")
    p_opt.add_argument("--max-combos", type=int, default=512,
                       help="cap the search; larger grids are sampled deterministically")
    p_opt.add_argument("--wf-folds", type=int, default=5, help="walk-forward optimisation folds")
    p_opt.add_argument("--wf-mode", choices=("expanding", "rolling"), default="expanding",
                       help="walk-forward window: expanding (anchored) or rolling")
    p_opt.add_argument("--select", choices=("sharpe", "robust"), default="sharpe",
                       help="winner selection: top Sharpe, or robustness floors then performance")
    p_opt.add_argument("--two-pass", action="store_true",
                       help="coarse->fine search (visits a fraction of a big grid; weak hardware)")
    p_opt.add_argument("--ensemble", type=int, default=None, metavar="N",
                       help="also report an equal-weight ensemble of N diverse top configs")
    p_opt.add_argument("--full", action="store_true",
                       help="run every stage (enrich, holdout, falsify, 5-pillar score, regime, "
                            "cluster) — the full research pipeline")
    p_opt.add_argument("--enrich", action="store_true",
                       help="stage-3 robustness scalars (ROC310, PSR, UPI, ...) on the winner")
    p_opt.add_argument("--holdout", action="store_true",
                       help="final-20%% holdout validation of the winner")
    p_opt.add_argument("--falsify", action="store_true",
                       help="falsification: noise-injection fragility on the winner")
    p_opt.add_argument("--score", action="store_true",
                       help="5-pillar composite selection score across the grid")
    p_opt.add_argument("--regime", action="store_true",
                       help="per-regime best config + consensus (volatility terciles)")
    p_opt.add_argument("--cluster", action="store_true",
                       help="cluster the top configs in parameter space")
    p_opt.add_argument("--backend", choices=("auto", "numpy", "numba", "cuda"), default="auto",
                       help="compute backend for the bootstrap stages (auto picks the fastest "
                            "available; results are identical across backends)")
    p_opt.add_argument("--offline", action="store_true",
                       help="read only the frozen snapshot; never fetch live")
    p_opt.set_defaults(func=_cmd_optimize)

    p_data = sub.add_parser("data", help="manage frozen data snapshots (releases)")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_freeze = data_sub.add_parser("freeze", help="package data_cache/<as-of> into a release")
    p_freeze.add_argument("--as-of", required=True)
    p_freeze.add_argument("--out", default=None)
    p_freeze.set_defaults(func=_cmd_data_freeze)
    p_restore = data_sub.add_parser("restore", help="extract a release archive into data_cache/")
    p_restore.add_argument("archive")
    p_restore.set_defaults(func=_cmd_data_restore)
    p_verify = data_sub.add_parser("verify", help="recompute and check a snapshot's hash")
    p_verify.add_argument("--as-of", required=True)
    p_verify.add_argument("--expect", default=None, help="hash it must match")
    p_verify.set_defaults(func=_cmd_data_verify)

    p_index = sub.add_parser("index", help="regenerate the shared strategy catalog")
    p_index.set_defaults(func=_cmd_index)

    p_audit = sub.add_parser(
        "audit", help="recompute metrics from committed returns and flag mismatches")
    p_audit.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_audit.add_argument("--include-baseline", action="store_true",
                         help="also audit strategies/Baseline/")
    p_audit.add_argument("--json", action="store_true", help="machine-readable output")
    p_audit.set_defaults(func=_cmd_audit)

    p_reg = sub.add_parser(
        "regression", help="diff committed metrics against a git baseline ref")
    p_reg.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_reg.add_argument("--ref", default="origin/main", help="git baseline ref")
    p_reg.add_argument("--json", action="store_true", help="machine-readable output")
    p_reg.set_defaults(func=_cmd_regression)

    p_thesis = sub.add_parser(
        "check-thesis", help="check README headline numbers against committed data (advisory)")
    p_thesis.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_thesis.add_argument("--strict", action="store_true", help="treat WARN findings as failures")
    p_thesis.add_argument("--json", action="store_true", help="machine-readable output")
    p_thesis.set_defaults(func=_cmd_check_thesis)

    p_book = sub.add_parser(
        "book", help="aggregate committed artifacts into the book-wide MASTER table + snapshots")
    p_book.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_book.add_argument("--master-only", action="store_true",
                        help="write only MASTER_SNAPSHOT.parquet (skip per-strategy snapshots)")
    p_book.add_argument("--json", action="store_true", help="machine-readable output")
    p_book.set_defaults(func=_cmd_book)

    p_derive = sub.add_parser(
        "derive", help="compute wf/risk/per-year artifacts from committed returns (offline)")
    p_derive.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_derive.add_argument("--slug", default=None, help="derive a single strategy by slug")
    p_derive.add_argument("--json", action="store_true", help="machine-readable output")
    p_derive.set_defaults(func=_cmd_derive)

    p_govern = sub.add_parser(
        "govern", help="controlled-vocabulary governance summary / validation (advisory)")
    p_govern.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    p_govern.add_argument("--backfill", action="store_true",
                          help="add a default governance block to configs that lack one")
    p_govern.add_argument("--strict", action="store_true", help="exit non-zero on vocab errors")
    p_govern.add_argument("--json", action="store_true", help="machine-readable output")
    p_govern.set_defaults(func=_cmd_govern)

    p_reg2 = sub.add_parser(
        "registry", help="model provenance ledger (list / show <id> / record <dir>)")
    p_reg2.add_argument("registry_args", nargs=argparse.REMAINDER,
                        help="forwarded to the registry CLI")
    p_reg2.set_defaults(func=_cmd_registry)

    args = parser.parse_args(argv)
    return args.func(args)
