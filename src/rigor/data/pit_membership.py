"""Deep-history point-in-time index membership (Sharadar) mapped to EODHD prices.

EODHD's own constituents feed is shallow before ~2015 (it re-introduces
survivorship bias for older backtests). This module instead uses a *frozen*
Sharadar membership reference (an add/remove event log back to 1957, 1192
ever-members) and maps those tickers onto EODHD price symbols.

Provenance/reproducibility: the membership file is a static, frozen reference
(``<reference dir>/sp500/sp500_membership.json``) -- not a live dependency -- so every
machine resolves identical membership. Prices remain single-source EODHD.

Licensing: S&P 500 and Nasdaq-100 histories come from a licensed vendor and are **not
shipped** with this repository. Point ``RIGOR_REFERENCE_DIR`` at a directory holding
``<index>/<index>_membership.json`` (same event-log format as the bundled CAC 40,
DAX 40 and EURO STOXX 50 files, which are compiled from public sources).

Mapping coverage (validated): active 100%, delisted ~95%. The unmapped tail is
bankruptcy "Q" tickers that differ between vendors; extend ``ALIASES`` as needed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def reference_dirs() -> list[Path]:
    """Where membership files are looked up: ``$RIGOR_REFERENCE_DIR`` first, then the package."""
    dirs = []
    env = os.environ.get("RIGOR_REFERENCE_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(REFERENCE_DIR)
    return dirs


def membership_path(index: str) -> Path:
    """Resolve ``<index>/<index>_membership.json`` or explain how to provide it."""
    for base in reference_dirs():
        path = base / index / f"{index}_membership.json"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"membership reference not found for '{index}'. S&P 500 / Nasdaq-100 histories are "
        f"licensed vendor data and are not shipped: place {index}_membership.json under "
        f"$RIGOR_REFERENCE_DIR/{index}/ (event-log format, see docs/data.md)."
    )

# Manual Sharadar-ticker -> EODHD-code overrides for the ~5% the heuristic misses
# (bankruptcy "Q" tickers whose roots EODHD reused for other companies).
# ONLY add entries verified to end at the known failure date -- never map to a
# reused ticker showing a *current* company's prices (e.g. DYN, AMR), which would
# silently inject the wrong instrument. Each entry below was confirmed by its
# price history ending at the real delisting date.
ALIASES: dict[str, str] = {
    "LEHMQ": "LEH",      # Lehman Brothers   -> EODHD LEH (ends 2008-09-17)
    "MTLQQ": "GM_old",   # Motors Liquidation (old GM) -> GM_old (ends 2011-03-31)
    "RSHCQ": "RSH",      # RadioShack        -> EODHD RSH (ends 2015-02-02)
}


# --------------------------------------------------------------------------
# Membership (event-log replay)
# --------------------------------------------------------------------------
class SharadarMembership:
    """Point-in-time membership from a frozen Sharadar add/remove event log."""

    def __init__(self, index: str = "sp500"):
        path = membership_path(index)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.intervals = self._build_intervals(payload.get("events", []))

    @staticmethod
    def _build_intervals(events: list[dict]) -> dict[str, list[tuple[str, str | None]]]:
        """Replay add/remove events into per-ticker [(start, end|None), ...] intervals."""
        by_ticker: dict[str, list[tuple[str, str]]] = {}
        for ev in events:
            t, d, a = ev.get("ticker"), ev.get("date"), ev.get("action")
            if t and d and a:
                by_ticker.setdefault(t, []).append((d, a))

        intervals: dict[str, list[tuple[str, str | None]]] = {}
        for ticker, evs in by_ticker.items():
            evs.sort(key=lambda x: x[0])
            spans: list[tuple[str, str | None]] = []
            open_start: str | None = None
            for d, a in evs:
                if a == "added" and open_start is None:
                    open_start = d
                elif a == "removed" and open_start is not None:
                    spans.append((open_start, d))
                    open_start = None
            if open_start is not None:
                spans.append((open_start, None))  # still a member
            if spans:
                intervals[ticker] = spans
        return intervals

    def members_on(self, day: str) -> set[str]:
        """Sharadar tickers that were index members on ``day`` (YYYY-MM-DD)."""
        out = set()
        for ticker, spans in self.intervals.items():
            for start, end in spans:
                if start <= day and (end is None or day <= end):
                    out.add(ticker)
                    break
        return out

    def all_tickers(self) -> set[str]:
        return set(self.intervals)


# --------------------------------------------------------------------------
# Ticker mapping (Sharadar -> EODHD)
# --------------------------------------------------------------------------
def _candidates(ticker: str) -> list[str]:
    """Plausible EODHD codes for a Sharadar ticker, in DETERMINISTIC priority order.

    MUST return an ordered list (not a set): the mapper returns the first
    candidate present in the EODHD code set, and Python set-iteration order for
    strings is randomised per process (PYTHONHASHSEED). A set here made the
    universe — and therefore every backtest — non-reproducible across machines.

    Priority: exact ticker, then digit-stripped / share-class variants, then the
    ``_old`` reused-ticker fallbacks (so active members resolve to their live
    code and only delisted/reused names fall back to ``_old``).
    """
    bases: list[str] = [ticker]
    stripped = re.sub(r"\d+$", "", ticker)  # BSC1 -> BSC, BEAM2 -> BEAM
    if stripped and stripped != ticker:
        bases.append(stripped)
    if "." in ticker:  # share class: AFS.A -> AFS-A / AFSA / AFS
        root, _, cls = ticker.partition(".")
        bases += [ticker.replace(".", "-"), root + cls, root]
        s2 = re.sub(r"\d+$", "", root)
        bases += [s2, f"{s2}-{cls}"]
    seen: list[str] = []
    for b in bases:
        if b and b not in seen:
            seen.append(b)
    ordered = seen + [f"{b}_old" for b in seen]  # _old as a fallback, in fixed order
    out: list[str] = []
    for c in ordered:
        if c not in out:
            out.append(c)
    return out


class TickerMapper:
    """Resolve Sharadar tickers to EODHD price symbols (CODE.US)."""

    def __init__(self, eodhd_codes: set[str], exchange: str = "US", aliases: dict | None = None):
        self.codes = eodhd_codes
        self.exchange = exchange
        self.aliases = {**ALIASES, **(aliases or {})}

    def map(self, ticker: str) -> str | None:
        if ticker in self.aliases:
            return f"{self.aliases[ticker]}.{self.exchange}"
        for c in _candidates(ticker):
            if c in self.codes:
                return f"{c}.{self.exchange}"
        return None

    def map_many(self, tickers) -> tuple[dict[str, str], list[str]]:
        """Return ({sharadar_ticker: eodhd_symbol}, [unmapped_tickers])."""
        mapped: dict[str, str] = {}
        unmapped: list[str] = []
        for t in tickers:
            sym = self.map(t)
            if sym:
                mapped[t] = sym
            else:
                unmapped.append(t)
        return mapped, unmapped
