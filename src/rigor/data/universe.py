"""Point-in-time index membership from EODHD's historical constituents feed.

The feed is a dict of records, each with Code / StartDate / EndDate /
IsActiveNow / IsDelisted. A ticker may appear in several records (added,
removed, re-added), so we collect *all* intervals per code. This is what makes
backtests survivorship-bias-free: ``members_on("2008-06-30")`` returns the index
as it was then -- including names later delisted -- not today's survivors.
"""

from __future__ import annotations

from dataclasses import dataclass

_OPEN_ENDED = {"", "0000-00-00", None}


@dataclass(frozen=True)
class MembershipInterval:
    code: str
    start: str  # YYYY-MM-DD
    end: str | None  # None == still a member
    is_delisted: bool

    def active_on(self, day: str) -> bool:
        if day < self.start:
            return False
        return self.end is None or day <= self.end


def parse_constituents(raw: dict) -> list[MembershipInterval]:
    """Flatten the raw feed into a list of membership intervals."""
    intervals: list[MembershipInterval] = []
    for rec in raw.values():
        if not isinstance(rec, dict):
            continue
        code = rec.get("Code")
        start = rec.get("StartDate")
        if not code or not start:
            continue
        end = rec.get("EndDate")
        if end in _OPEN_ENDED:
            end = None
        delisted = str(rec.get("IsDelisted")) in ("1", "True", "true")
        intervals.append(MembershipInterval(code, start, end, delisted))
    return intervals


def members_on(raw: dict, day: str, *, exchange: str = "US") -> list[str]:
    """Sorted list of symbols (with exchange suffix) that were index members on ``day``."""
    seen = {
        iv.code for iv in parse_constituents(raw) if iv.active_on(day)
    }
    return sorted(f"{c}.{exchange}" if "." not in c else c for c in seen)


def all_symbols_ever(raw: dict, *, exchange: str = "US") -> list[str]:
    """Every symbol that was ever a member (the survivorship-free superset)."""
    seen = {iv.code for iv in parse_constituents(raw)}
    return sorted(f"{c}.{exchange}" if "." not in c else c for c in seen)
