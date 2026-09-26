# paper_types package
from .base_paper_type import BasePaperType
from .empirical_study import EmpiricalStudyPaperType
from .factor_anomaly import FactorAnomalyPaperType
from .market_phenomenon import MarketPhenomenonPaperType
from .strategy_backtest import StrategyBacktestPaperType

# Registry values are concrete (non-abstract) BasePaperType subclasses. The
# annotation uses the concrete-subclass form so mypy permits instantiation in
# the factory below.
_REGISTRY: dict[str, type[BasePaperType]] = {
    "market_phenomenon": MarketPhenomenonPaperType,
    "factor_anomaly": FactorAnomalyPaperType,
    "empirical_study": EmpiricalStudyPaperType,
    "strategy_backtest": StrategyBacktestPaperType,
}


def get_paper_type(name: str) -> BasePaperType:
    """Factory: return instantiated PaperType for the given name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown paper_type {name!r}. Valid: {valid}")
    return cls()
