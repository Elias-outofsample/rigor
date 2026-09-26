"""``rigor demo`` — the validation battery on a strategy that cannot have an edge.

A moving-average crossover runs on a zero-drift random walk. Its default settings are
backtested and graded; then the optimiser searches 25 parameter combinations on the
same noise. The search always finds a better in-sample Sharpe — the question is whether
the framework is fooled by it. Runs offline in a few seconds and writes the report
card and the thesis PDF of the "optimised" configuration.
"""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path
from typing import Any

from .data.synthetic import SyntheticDataLoader
from .project.run import load_config, load_strategy_module
from .report import write_report_card
from .thesis_card import write_thesis
from .validation import validate_strategy


def _strategy_dir() -> Path:
    return Path(str(resources.files("rigor") / "examples" / "ma_crossover"))


def _fmt_gate(g: dict[str, Any]) -> str:
    mark = "pass" if g["passed"] else ("n/a " if g.get("neutral") else "FAIL")
    crit = "  (critical)" if g.get("critical") else ""
    value = g["value"]
    shown = f"{value:>9.3g}" if isinstance(value, (int, float)) else f"{value!s:>9}"
    return f"      {mark}  {g['name']:<34} {shown}{crit}"


def run_demo(out_dir: Path) -> int:
    t0 = time.perf_counter()
    sdir = _strategy_dir()
    config, raw = load_config(sdir)
    data = SyntheticDataLoader()

    print("rigor demo — a moving-average crossover on a market with no edge\n")
    print("[1/3] Data: a synthetic random walk, 2008-2024, zero drift, 20% vol.")
    print("      Nothing is predictable by construction: any edge found here is luck.\n")

    strategy = load_strategy_module(sdir, raw["slug"]).build(config, data)
    result = strategy.backtest()
    validation = validate_strategy(strategy, result)
    v = validation["verdict"]
    m = result.metrics
    print(f"[2/3] Backtest with default parameters (fast {strategy.fast} / slow {strategy.slow})")
    print(f"      Sharpe {m['sharpe']:+.2f}  CAGR {m['cagr']:+.1%}  MaxDD {m['max_drawdown']:.1%}")
    print(f"      Verdict: {v['verdict']} ({v['grade']}, score {v['score']:.0f}/100)")
    for g in v["gates"]:
        print(_fmt_gate(g))

    from .optimize.core import optimize_strategy

    rep = optimize_strategy(
        sdir, data=data, write=False, cumulative_trials=False  # type: ignore[arg-type]
    )
    best = rep["best_params"]
    wf = rep.get("walk_forward_opt") or {}
    snoop = rep.get("data_snooping") or {}
    wrc = snoop.get("white_reality_check")
    wrc_p = wrc.get("p_value") if isinstance(wrc, dict) else wrc
    print(f"\n[3/3] Optimise: {rep['configs_evaluated']} parameter combinations on the same noise")
    print(f"      best in-sample Sharpe {rep['best_sharpe']:+.2f} (fast {best['fast']} / slow "
          f"{best['slow']}) vs default {rep['default_sharpe']:+.2f}  <- looks like an improvement")
    print(f"      deflated Sharpe P(skill)      {rep['overfit']['dsr']:.2f}   (needs > 0.50)")
    if rep.get("cpcv"):
        print(f"      CPCV prob. of overfitting     {rep['cpcv']['pbo']:.2f}")
    if wrc_p is not None:
        print(f"      White reality check p-value   {wrc_p:.2f}   (vs the default config)")
    if wf:
        print(f"      walk-forward: IS Sharpe {wf.get('is_sharpe_mean', float('nan')):+.2f} -> "
              f"OOS {wf.get('oos_sharpe_mean', float('nan')):+.2f}, positive OOS folds "
              f"{wf.get('oos_positive_folds')}/{wf.get('n_folds')}")
    print(f"      Verdict on the winner: {rep['verdict']['verdict']} "
          f"(score {rep['verdict']['score']:.0f}/100)")
    fooled = rep["verdict"]["verdict"] == "PROMOTE"
    print("\n      " + ("The search was promoted: investigate before trusting it." if fooled else
                        "The search found luck, not skill — nothing to adopt."))

    out_dir.mkdir(parents=True, exist_ok=True)
    tuned = strategy.backtest(best)
    tuned_validation = validate_strategy(strategy, tuned, n_trials=rep["configs_evaluated"])
    card = write_report_card(tuned, out_dir, "ma_crossover", name=config.name,
                             as_of=None, validation=tuned_validation)
    thesis = write_thesis(tuned, out_dir, "ma_crossover", name=config.name,
                          category=raw.get("category", "other"))
    print(f"\nArtifacts for the optimised configuration ({time.perf_counter() - t0:.1f}s):")
    for path in [*card.values(), thesis]:
        print(f"  {path}")
    return 0
