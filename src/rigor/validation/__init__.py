"""Validation layer: is a strategy trustworthy, or overfit?

Turns a BacktestResult into anti-overfit statistics (PSR, DSR, haircut,
Newey-West, min track record), temporal robustness (walk-forward fold
consistency, per-year, bootstrap CI), optional CPCV/PBO (for strategies with a
param grid), and a single promotion verdict (PROMOTE / CONDITIONAL / REJECT).

    from rigor.validation import validate, validate_strategy
    report = validate_strategy(strategy)      # includes CPCV if a grid exists
    report = validate(backtest_result)        # returns-only battery
"""

from .permutation import (
    hansen_spa_test,
    multiple_testing_adjustment,
    sign_permutation_test,
    stepm_test,
    white_reality_check,
)
from .runner import validate, validate_strategy
from .verdict import Gate, Verdict, compute_verdict

__all__ = [
    "validate", "validate_strategy", "compute_verdict", "Verdict", "Gate",
    "white_reality_check", "hansen_spa_test", "stepm_test",
    "multiple_testing_adjustment", "sign_permutation_test",
]
