"""Strategy configuration — the declarative header every Rigor strategy carries.

Adapted from an earlier in-house library's StrategyConfig, trimmed to what v1 needs
and tied to the Rigor data layer (EODHD, as-of dates). Costs are expressed in basis
points to match how retail-broker commissions/slippage are usually quoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LONG_SHORT = ("long", "short", "long_short")
_ROLES = ("alpha", "hedge", "defensive")
_STATUS = ("live", "paper", "idle")
# A short, clean version token (e.g. v1, v2, v1.1) — the current version of the
# strategy this folder holds. Variants (different core signal) get a new folder,
# not a new version; filters/optimization bump the version. Baselines stay "v0".
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class StrategyConfig:
    name: str
    start_date: str = "2000-01-01"
    end_date: str | None = None
    initial_capital: float = 200_000.0
    commission_bps: float = 10.0          # round-trip cost charged on turnover
    long_short: str = "long"
    max_positions: int = 10
    rebalance_freq: str = "monthly"       # daily | weekly | monthly
    role: str = "alpha"                   # alpha | hedge | defensive
    status: str = "idle"                  # live | paper | idle (deployment state)
    version: str = "v1"                   # current version this folder holds (v0 = baseline)
    ledger_exempt: bool = False            # True → skip MISSING_LEDGER / DEGENERATE_LEDGER audit
    #   Set this for strategies where per-trade MAE/MFE is semantically meaningless
    #   (e.g. delta-neutral carry, always-invested strategies, or returns-only wrappers).
    optimize_exempt_reason: str | None = None   # non-empty → EXEMPT from the optimize_gate
    #   The optimize_gate requires every strategy to declare a real multi-value param_grid()
    #   (so it is optimisable here AND by an external optimiser). Set this to a short
    #   justification ONLY for a strategy with a genuinely empty tunable surface (e.g. a fixed
    #   calendar/seasonal pattern with no knobs). An empty/whitespace value does NOT exempt.
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.long_short not in _LONG_SHORT:
            raise ValueError(f"long_short must be one of {_LONG_SHORT}")
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}")
        if self.status not in _STATUS:
            raise ValueError(f"status must be one of {_STATUS}")
        if not _VERSION_RE.match(str(self.version)):
            raise ValueError(
                f"version {self.version!r} must be a short clean token (e.g. v1, v2, v1.1)"
            )
