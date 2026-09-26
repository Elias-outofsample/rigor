"""Project standard: folder/naming convention, scaffolding, validation, running."""

from .index import write_index
from .optimize import optimize_strategy
from .run import load_config, run_strategy
from .scaffold import create_strategy
from .validate import ValidationResult, validate_strategy

__all__ = [
    "create_strategy",
    "validate_strategy",
    "ValidationResult",
    "run_strategy",
    "optimize_strategy",
    "load_config",
    "write_index",
]
