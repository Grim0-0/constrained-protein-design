"""constrained-protein-design: is a hard biophysical constraint better treated as a constraint?

Public surface::

    from cpd import Variant, Mutation
    from cpd import SyntheticLandscape, ExactLandscape, BudgetedOracle, Measurement
    from cpd import Scalarized, FixedPenalty, Lagrangian, BindingOnly
    from cpd import SearchConfig, evolutionary_search
    from cpd import default_methods, summarize

Sign conventions, stated once and used everywhere: both ``ddg_binding`` and
``ddg_folding`` are minimized, positive means worse, and the hard constraint
is ``ddg_folding <= tau``.
"""

from .landscape import (
    BudgetedOracle,
    BudgetExhausted,
    ExactLandscape,
    Landscape,
    Measurement,
    MeasuredAdditiveLandscape,
    SyntheticLandscape,
)
from .methods import DesignRun, default_methods
from .metrics import (
    DesignSetReport,
    feasibility_rate,
    feasible_best,
    feasible_best_k,
    hypervolume,
    pareto_mask,
    summarize,
    violation_severity,
)
from .objectives import BindingOnly, FixedPenalty, Lagrangian, Objective, Scalarized
from .search import SearchConfig, SearchResult, evolutionary_search
from .variants import AMINO_ACIDS, Mutation, Variant

__version__ = "0.1.0"

__all__ = [
    "AMINO_ACIDS",
    "BindingOnly",
    "BudgetExhausted",
    "BudgetedOracle",
    "DesignRun",
    "DesignSetReport",
    "ExactLandscape",
    "FixedPenalty",
    "Lagrangian",
    "Landscape",
    "MeasuredAdditiveLandscape",
    "Measurement",
    "Mutation",
    "Objective",
    "Scalarized",
    "SearchConfig",
    "SearchResult",
    "SyntheticLandscape",
    "Variant",
    "default_methods",
    "evolutionary_search",
    "feasibility_rate",
    "feasible_best",
    "feasible_best_k",
    "hypervolume",
    "pareto_mask",
    "summarize",
    "violation_severity",
    "__version__",
]
