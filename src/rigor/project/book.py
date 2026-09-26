"""Strategy book utilities — equity snapshots, per-strategy snapshots, and the master table.

Three main workflows:

  1. ``build_strategy_snapshot(strategy_dir)``
     Reads a strategy's returns CSV + summary JSON, builds a down-sampled equity curve,
     and writes ``extra/<slug>_snapshot.json.gz`` so dashboards can load it cheaply.

  2. ``build_master(root)``
     Iterates every PRODUCTION strategy, calls ``build_strategy_snapshot``, and writes
     ``catalog/MASTER_SNAPSHOT.parquet`` — one row per strategy.

  3. ``load_master(root)``
     Read-only helper that returns the parquet as a DataFrame (empty if absent).

CLI: ``python -m rigor.project.book [--root DIR] [--master-only] [--json]``
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rigor.data.config import find_repo_root
from rigor.project import layout

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAN_SAFE_SENTINEL = object()

# Mirrors json.decoder.WHITESPACE (not exported in typeshed stubs); used as the
# default for the decode() override so its signature matches JSONDecoder.decode.
_WHITESPACE_MATCH = re.compile(r"[ \t\n\r]*", re.VERBOSE | re.MULTILINE | re.DOTALL).match


class _NaNDecoder(json.JSONDecoder):
    """JSON decoder that converts bare ``NaN`` tokens to ``None``.

    Standard ``json`` rejects NaN, but ``json.dumps`` with ``allow_nan=True``
    writes bare ``NaN`` tokens that re-``loads`` would error on. This decoder
    pre-processes the string to replace them with ``null``.
    """

    def decode(  # noqa: D401
        self, s: str, _w: Callable[[str, int], Any] = _WHITESPACE_MATCH
    ) -> Any:
        s = _replace_nan_tokens(s)
        return super().decode(s, _w)


def _replace_nan_tokens(text: str) -> str:
    """Replace bare JSON-invalid ``NaN`` / ``Infinity`` tokens with ``null``."""
    import re

    # Replace -Infinity, Infinity, NaN (bare, not inside strings)
    return re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "null", text)


def _load_json_safe(path: Path) -> dict:
    """Load JSON from *path*, tolerating bare NaN tokens produced by json.dumps(allow_nan=True)."""
    raw = path.read_text(encoding="utf-8")
    return json.loads(_replace_nan_tokens(raw))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decimate_equity(
    returns: pd.Series | np.ndarray,
    n: int = 800,
) -> tuple[list[str], list[float]]:
    """Build an equity curve and downsample it to at most *n* points.

    The first and last points are always preserved.  If the returns carry a
    ``DatetimeIndex`` the labels are ISO-8601 date strings; otherwise they are
    string representations of the integer position (``"0"``, ``"1"``, …).

    Parameters
    ----------
    returns:
        Daily return series (not cumulative).  Values may be a pandas Series
        or a plain numpy array / list.
    n:
        Maximum number of output points.  Defaults to 800.

    Returns
    -------
    dates:
        List of label strings (ISO date or position index).
    values:
        List of equity float values (starting from 1.0).
    """
    arr = np.asarray(returns, dtype=np.float64)
    # Replace non-finite values with 0 to avoid corrupting the equity curve
    arr = np.where(np.isfinite(arr), arr, 0.0)

    equity: np.ndarray = np.cumprod(1.0 + arr)
    # Prepend the origin (equity = 1.0 before first return)
    equity = np.concatenate([[1.0], equity])

    has_datetime_index = (
        isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex)
    )

    if has_datetime_index:
        dates_arr = pd.Series(returns).index  # type: ignore[arg-type]
        # Prepend a synthetic origin label one day before first date
        first_dt = dates_arr[0] - pd.Timedelta(days=1)
        labels: list[str] = [first_dt.strftime("%Y-%m-%d")] + [
            d.strftime("%Y-%m-%d") for d in dates_arr
        ]
    else:
        labels = [str(i) for i in range(len(equity))]

    total = len(equity)
    if total <= n:
        return labels, equity.tolist()

    # Always include index 0 (origin) and index total-1 (last)
    # Pick n-2 evenly spaced indices in between, then add endpoints
    inner_count = n - 2
    inner_idx = np.linspace(1, total - 2, inner_count, dtype=int)
    # Remove duplicates that linspace might create at boundaries
    inner_idx = np.unique(inner_idx)
    idx = np.concatenate([[0], inner_idx, [total - 1]])
    idx = np.unique(idx)

    return [labels[i] for i in idx], equity[idx].tolist()


def build_strategy_snapshot(strategy_dir: Path | str) -> dict:
    """Assemble a strategy snapshot dict and write it to ``extra/<slug>_snapshot.json.gz``.

    The snapshot contains the slug, category, name, as_of date, metrics dict,
    verdict string, and a down-sampled equity curve (≤ 800 points).

    Parameters
    ----------
    strategy_dir:
        Path to the strategy folder (the directory containing ``config.json``).

    Returns
    -------
    dict with keys:
        ``slug``, ``category``, ``name``, ``as_of``, ``metrics``, ``verdict``,
        ``equity`` (sub-dict with ``dates`` and ``values``).

    Raises
    ------
    FileNotFoundError
        If ``config.json``, ``<slug>_summary.json``, or ``<slug>_returns.csv``
        are missing.
    """
    strategy_dir = Path(strategy_dir)
    cfg_path = strategy_dir / "config.json"
    cfg = _load_json_safe(cfg_path)

    slug: str = cfg.get("slug", strategy_dir.name)
    category: str = cfg.get("category", strategy_dir.parent.name)
    name: str = cfg.get("name", slug)

    arts = layout.artifacts_dir(strategy_dir)
    summary_path = arts / f"{slug}_summary.json"
    returns_path = arts / f"{slug}_returns.csv"

    # -- summary -----------------------------------------------------------
    summary: dict = {}
    if summary_path.exists():
        summary = _load_json_safe(summary_path)

    metrics: dict = summary.get("metrics", {})
    verdict_block: dict = summary.get("verdict", {})
    verdict: str | None = verdict_block.get("verdict") if isinstance(verdict_block, dict) else None
    as_of: str | None = summary.get("as_of")

    # -- returns CSV -------------------------------------------------------
    returns_path_resolved = returns_path  # may raise FileNotFoundError on read
    df = _read_returns_csv(returns_path_resolved)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # unnamed format: integer positional index — keep as-is
        pass

    ret_series = df["returns"] if "returns" in df.columns else df.iloc[:, 0]

    dates, values = decimate_equity(ret_series)

    snapshot: dict = {
        "slug": slug,
        "category": category,
        "name": name,
        "as_of": as_of,
        "metrics": metrics,
        "verdict": verdict,
        "equity": {"dates": dates, "values": values},
    }

    # -- write artifact (auxiliary aggregate -> extra/) --------------------
    extra = layout.extra_dir(strategy_dir)
    extra.mkdir(parents=True, exist_ok=True)
    out_path = extra / f"{slug}_snapshot.json.gz"
    payload = json.dumps(snapshot, allow_nan=False).encode("utf-8")
    with gzip.open(out_path, "wb") as fh:
        fh.write(payload)

    logger.debug("Wrote snapshot: %s", out_path)
    return snapshot


def build_master(root: Path | None = None) -> pd.DataFrame:
    """Build a one-row-per-strategy DataFrame and write ``catalog/MASTER_SNAPSHOT.parquet``.

    Columns: ``slug``, ``name``, ``category``, ``status``, ``version``,
    ``sharpe``, ``cagr``, ``max_drawdown``, ``volatility``, ``sortino``,
    ``calmar``, ``n_obs``, ``verdict``, ``score``, ``as_of``.

    Strategies that are missing required files are skipped (a warning is logged).

    Parameters
    ----------
    root:
        Workspace root (where ``strategies/`` lives).  Defaults to auto-detected
        repo root via :func:`rigor.data.config.find_repo_root`.

    Returns
    -------
    pandas DataFrame with the columns listed above.
    """
    root = Path(root) if root is not None else find_repo_root()
    strat_root = layout.strategies_root(root)

    rows: list[dict] = []
    if not strat_root.is_dir():
        logger.warning("Strategy root not found: %s", strat_root)
        return _empty_master()

    for cfg_path in sorted(strat_root.glob("*/*/config.json")):
        # Skip Baseline subtree
        rel = cfg_path.parent.relative_to(strat_root)
        if layout.is_baseline_rel(rel):
            continue

        strategy_dir = cfg_path.parent
        try:
            row = _master_row_from_dir(strategy_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", strategy_dir.name, exc)
            continue
        rows.append(row)

    df = pd.DataFrame(rows, columns=_MASTER_COLS) if rows else _empty_master()

    out_dir = root / layout.STRATEGY_LIST_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "MASTER_SNAPSHOT.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote master snapshot: %s (%d strategies)", out_path, len(df))
    return df


def load_master(root: Path | None = None) -> pd.DataFrame:
    """Read ``catalog/MASTER_SNAPSHOT.parquet``; return empty DataFrame if absent.

    Parameters
    ----------
    root:
        Workspace root.  Defaults to auto-detected repo root.

    Returns
    -------
    pandas DataFrame (may be empty if the parquet has not been built yet).
    """
    root = Path(root) if root is not None else find_repo_root()
    path = root / layout.STRATEGY_LIST_ROOT / "MASTER_SNAPSHOT.parquet"
    if not path.exists():
        return _empty_master()
    return pd.read_parquet(path)


def build_all(root: Path | None = None) -> dict:
    """Run ``build_master`` and per-strategy snapshots.

    Strategies missing required files are skipped (same as ``build_master``).

    Parameters
    ----------
    root:
        Workspace root.  Defaults to auto-detected repo root.

    Returns
    -------
    dict with keys:
        ``n_strategies`` (int), ``master_path`` (str), ``snapshots`` (list of slugs).
    """
    root = Path(root) if root is not None else find_repo_root()
    master_df = build_master(root)
    master_path = str(root / layout.STRATEGY_LIST_ROOT / "MASTER_SNAPSHOT.parquet")

    strat_root = layout.strategies_root(root)
    snapshot_slugs: list[str] = []

    if strat_root.is_dir():
        for cfg_path in sorted(strat_root.glob("*/*/config.json")):
            rel = cfg_path.parent.relative_to(strat_root)
            if layout.is_baseline_rel(rel):
                continue
            strategy_dir = cfg_path.parent
            try:
                snap = build_strategy_snapshot(strategy_dir)
                snapshot_slugs.append(snap["slug"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Snapshot skipped for %s: %s", strategy_dir.name, exc)

    return {
        "n_strategies": len(master_df),
        "master_path": master_path,
        "snapshots": snapshot_slugs,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the book builder.

    Arguments
    ---------
    --root DIR
        Workspace root directory.  Defaults to auto-detected repo root.
    --master-only
        Build only the master parquet; skip per-strategy snapshots.
    --json
        Emit a JSON summary to stdout on success.

    Returns
    -------
    Exit code (0 = success, 1 = error).
    """
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-book",
        description="Build the strategy book (master parquet + per-strategy snapshots).",
    )
    parser.add_argument("--root", default=None, help="Workspace root directory.")
    parser.add_argument(
        "--master-only",
        action="store_true",
        help="Build only the master parquet; skip per-strategy snapshots.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary to stdout.",
    )
    args = parser.parse_args(argv)

    root: Path | None = Path(args.root) if args.root else None

    try:
        if args.master_only:
            df = build_master(root)
            result: dict = {
                "n_strategies": len(df),
                "master_path": str(
                    (root or find_repo_root())
                    / layout.STRATEGY_LIST_ROOT
                    / "MASTER_SNAPSHOT.parquet"
                ),
                "snapshots": [],
            }
        else:
            result = build_all(root)
    except Exception as exc:  # noqa: BLE001
        logger.error("book build failed: %s", exc)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Done. {result['n_strategies']} strategies | "
            f"master -> {result['master_path']}"
        )
    return 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_MASTER_COLS: list[str] = [
    "slug",
    "name",
    "category",
    "status",
    "version",
    "sharpe",
    "cagr",
    "max_drawdown",
    "volatility",
    "sortino",
    "calmar",
    "n_obs",
    "verdict",
    "score",
    "as_of",
]


def _empty_master() -> pd.DataFrame:
    return pd.DataFrame(columns=_MASTER_COLS)


def _read_returns_csv(path: Path) -> pd.DataFrame:
    """Read a returns CSV that may use either named or unnamed (positional) date column."""
    df = pd.read_csv(path, index_col=0)
    # If the first column is named "date" after reset, detect via columns
    df = df.reset_index()
    # After reset_index the old index becomes a column; rename it deterministically
    first_col = df.columns[0]
    if first_col not in ("date",):
        df = df.rename(columns={first_col: "__idx__"})
    return df


def _master_row_from_dir(strategy_dir: Path) -> dict:
    """Build one master-row dict from a strategy directory.

    Raises on any missing file or parse error so the caller can log and skip.
    """
    cfg_path = strategy_dir / "config.json"
    cfg = _load_json_safe(cfg_path)

    slug: str = cfg.get("slug", strategy_dir.name)
    name: str = cfg.get("name", slug)
    category: str = cfg.get("category", strategy_dir.parent.name)
    status: str = cfg.get("status", "idle")
    version: str = cfg.get("version", "v1")

    arts = layout.artifacts_dir(strategy_dir)
    summary_path = arts / f"{slug}_summary.json"
    returns_path = arts / f"{slug}_returns.csv"

    # returns.csv must exist for a valid master row
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing returns CSV: {returns_path}")

    summary: dict = {}
    if summary_path.exists():
        summary = _load_json_safe(summary_path)

    metrics: dict = summary.get("metrics", {})
    verdict_block = summary.get("verdict", {})
    verdict: str | None = (
        verdict_block.get("verdict") if isinstance(verdict_block, dict) else None
    )
    score: float | None = (
        verdict_block.get("score") if isinstance(verdict_block, dict) else None
    )
    as_of: str | None = summary.get("as_of")

    def _safe(key: str) -> float | None:
        val = metrics.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    return {
        "slug": slug,
        "name": name,
        "category": category,
        "status": status,
        "version": version,
        "sharpe": _safe("sharpe"),
        "cagr": _safe("cagr"),
        "max_drawdown": _safe("max_drawdown"),
        "volatility": _safe("volatility"),
        "sortino": _safe("sortino"),
        "calmar": _safe("calmar"),
        "n_obs": metrics.get("n_obs"),
        "verdict": verdict,
        "score": score,
        "as_of": as_of,
    }


if __name__ == "__main__":
    sys.exit(main())
