"""The Rigor strategy folder standard — one shape every strategy follows.

Strategies are category-grouped with snake_case slugs. There are two homes:

    strategies/<category>/<slug>/              production / evolved (filters + optimization)
    strategies/Baseline/<category>/<slug>/     the RAW version, before any filter/optimization

Both use the identical folder shape:

    <slug>/
    ├── strategy.py     # StrategyBase subclass + build(config, data) factory
    ├── README.md       # thesis, parameters, notes
    ├── config.json     # declarative StrategyConfig (name, dates, costs, version, ...)
    ├── artifacts/       # CORE generated outputs (report card + committed evidence)
    │   ├── <slug>_returns.csv
    │   ├── <slug>_summary.json
    │   ├── <slug>_report.html
    │   ├── <slug>_positions.csv
    │   ├── <slug>_thesis.pdf
    │   └── <slug>_optimization.json
    └── extra/           # AUXILIARY / diagnostic files (regenerable; not the report card)
        ├── <slug>_n_trials_log.json   # cumulative optimize trial ledger
        ├── <slug>_wf.json / _risk.json / _peryear.json   # `rigor derive`
        └── <slug>_snapshot.json.gz    # `rigor book`

Classification rules:
  - The **slug** is the single identifier: it names the folder AND prefixes every
    artifact, so a strategy is grep-able end to end.
  - A change to the **core signal** (e.g. rank by ATR instead of vol) is a
    DIFFERENT strategy -> a different slug -> a different folder.
  - Adding a **filter or optimization** is a new *version* of the same strategy,
    tracked by the ``version`` field in config.json (git holds the trail).
  - A strategy's raw form lives at ``strategies/Baseline/<category>/<slug>/`` under
    the **same slug** as its evolved counterpart at ``strategies/<category>/<slug>/``.

This module is the one place those rules live; scaffold / validate / run / index
all read from here.
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical edge families. Free-form categories are allowed but the validator
# nudges toward these for consistent grouping across the book.
CANONICAL_CATEGORIES = (
    "trend_following", "momentum", "mean_reversion", "breakout", "carry",
    "seasonal", "volatility", "stat_arb", "macro", "event_driven", "crypto",
    "intraday", "other",
)

STRATEGIES_ROOT = "strategies"           # strategy folders live under <root>/Strategy/
BASELINE_ROOT = "Baseline"             # raw versions under <root>/Strategy/Baseline/<category>/
STRATEGY_LIST_ROOT = "catalog"   # the shared catalog folder (committed)
INDEX_FILENAME = "INDEX.md"

# The required files, in canonical order (used by scaffold + validate output).
REQUIRED_FILES = ("strategy.py", "README.md", "config.json")
ARTIFACTS_DIR = "artifacts"   # CORE report-card outputs (committed)
EXTRA_DIR = "extra"           # auxiliary / diagnostic files (regenerable; see extra_dir)


def strategies_root(root: Path) -> Path:
    return Path(root) / STRATEGIES_ROOT


def baseline_root(root: Path) -> Path:
    return Path(root) / STRATEGIES_ROOT / BASELINE_ROOT


def index_path(root: Path) -> Path:
    return Path(root) / STRATEGY_LIST_ROOT / INDEX_FILENAME


def is_baseline_rel(rel_to_strategies: Path) -> bool:
    """True if a path (relative to strategies/) is inside the Baseline subtree."""
    parts = Path(rel_to_strategies).parts
    return bool(parts) and parts[0] == BASELINE_ROOT

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug))


def is_valid_category(category: str) -> bool:
    return bool(_NAME_RE.match(category))


def strategy_dir(root: Path, category: str, slug: str, *, baseline: bool = False) -> Path:
    if baseline:
        return Path(root) / STRATEGIES_ROOT / BASELINE_ROOT / category / slug
    return Path(root) / STRATEGIES_ROOT / category / slug


def baseline_dir(root: Path, category: str, slug: str) -> Path:
    return strategy_dir(root, category, slug, baseline=True)


def artifacts_dir(strategy_path: Path) -> Path:
    return Path(strategy_path) / ARTIFACTS_DIR


def extra_dir(strategy_path: Path) -> Path:
    """The strategy's ``extra/`` folder — the canonical home for AUXILIARY files.

    Anything a strategy/run generates that is NOT part of the core report card
    (returns/summary/report/positions/thesis/optimization.json) belongs here:
    the cumulative optimize trial ledger, ``rigor derive`` wf/risk/peryear, the
    ``rigor book`` snapshot, and any future per-strategy scratch output. Created
    lazily by whatever writes into it.
    """
    return Path(strategy_path) / EXTRA_DIR


def artifact_paths(strategy_path: Path, slug: str) -> dict[str, Path]:
    a = artifacts_dir(strategy_path)
    return {
        "returns": a / f"{slug}_returns.csv",
        "summary": a / f"{slug}_summary.json",
        "report": a / f"{slug}_report.html",
        "positions": a / f"{slug}_positions.csv",
    }
