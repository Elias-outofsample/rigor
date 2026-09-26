"""
Repo-relative path resolution.
==============================
Locates optional shared resources (a ``Data/`` folder with factor or benchmark
files) relative to this package, so the thesis generator works after a fresh clone
on any machine/OS — no hardcoded, user-specific, or platform-specific paths.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches the repo filename convention <ASSET>__<strategy>__<start>_to_<end>.parquet
_DATE_RANGE_RE = re.compile(r"__(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")


def _longest_coverage(paths: list[Path]) -> Path | None:
    """Pick the file whose filename date range spans the most days.

    Falls back to the first path when no parseable date range is present.
    """
    best, best_span = None, -1
    for p in paths:
        m = _DATE_RANGE_RE.search(p.name)
        span = 0
        if m:
            from datetime import date
            s = date.fromisoformat(m.group(1))
            e = date.fromisoformat(m.group(2))
            span = (e - s).days
        if span > best_span:
            best, best_span = p, span
    return best

# <root>/Thesis/thesis_gen/paths.py  ->  parents[2] == <root>
PACKAGE_DIR: Path = Path(__file__).resolve().parent          # <root>/Thesis/thesis_gen
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "Data"
OUTPUT_DIR: Path = PACKAGE_DIR / "output"                    # generalist (non-strategy) theses


def slugify(text: str, default: str = "thesis") -> str:
    """Filesystem-safe slug from a title, e.g. 'My VRP Study' -> 'my_vrp_study'."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s or default


def ensure_repo_on_path() -> None:
    """Put the repo root on sys.path so repo-level modules resolve.

    The repo's root conftest.py / sitecustomize.py already do this when running
    under the repo, but doing it here makes the package robust to any cwd.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ff_factors_path() -> Path | None:
    """Path to the local Fama-French factors parquet, or None if absent.

    Expected columns: [mkt_rf, smb, hml, rf, mom], DatetimeIndex named 'Date'.
    """
    factors_dir = DATA_DIR / "factors"
    if not factors_dir.is_dir():
        return None
    matches = sorted(factors_dir.glob("*FACTORS*.parquet")) or \
        sorted(factors_dir.glob("*.parquet"))
    return matches[0] if matches else None


def benchmark_path(ticker: str = "SPY") -> Path | None:
    """Path to a local benchmark price parquet for ``ticker`` (e.g. SPY), or None.

    Looks under Data/equities/us/etfs first (price files use the
    ``<TICKER>__<strategy>__...parquet`` convention with an OHLCV ``close``
    column — the ``__`` excludes derived files like ``SPY_DIVYIELD_...``),
    then falls back to the S&P 500 index level in Data/norgate.
    """
    etfs_dir = DATA_DIR / "equities" / "us" / "etfs"
    if etfs_dir.is_dir():
        matches = sorted(etfs_dir.glob(f"{ticker}__*.parquet"))
        if matches:
            return _longest_coverage(matches)
    norgate_dir = DATA_DIR / "norgate"
    if norgate_dir.is_dir():
        if ticker.upper() in ("SPY", "SPX", "^GSPC", "SP500"):
            spx = sorted(norgate_dir.glob("spx_index*.parquet"))
            if spx:
                return _longest_coverage(spx)
        matches = sorted(norgate_dir.glob(f"{ticker.lower()}*.parquet"))
        if matches:
            return _longest_coverage(matches)
    return None
