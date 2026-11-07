"""Wiring: search engine x constraint-handling rule.

A *method* is a constraint-handling rule plus the policy it uses to pick which
designs to return. Both halves matter. Optimize-then-reject and plain binding
maximization search identically and differ only in the selection step, which
is exactly what "reject at the end" means.

Each method returns its designs from its full evaluation trace, ranked by the
only information that method has:

======================  ==================================================
Rule                    Selection policy
======================  ==================================================
Scalarization           lowest weighted sum (the rule never sees ``tau``)
Fixed penalty           lowest penalized score
Lagrangian              lowest Lagrangian at the final multiplier
Optimize-then-reject    discard infeasible, then lowest ``ddg_binding``
======================  ==================================================

Giving every method a feasibility-aware selection step would hide the
difference being measured: a rule that ignores the constraint during search
must also ignore it when it hands over designs, or it is not the rule
practitioners actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .landscape import BudgetedOracle, Landscape, Measurement
from .objectives import (
    NSGA2,
    BindingOnly,
    FixedPenalty,
    Lagrangian,
    Minimize,
    Objective,
    Scalarized,
    Target,
)
from .search import SearchConfig, SearchResult, evolutionary_search
from .variants import Variant

__all__ = [
    "DesignRun",
    "Method",
    "run_scalarization",
    "run_scalarization_sweep",
    "run_fixed_penalty",
    "run_lagrangian",
    "run_optimize_then_reject",
    "run_nsga2",
    "default_methods",
    "DEFAULT_WEIGHTS",
]

#: Weights swept for the scalarization baseline, spanning folding-dominated to
#: binding-dominated. A practitioner picks one of these and hopes.
DEFAULT_WEIGHTS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass
class DesignRun:
    """One method's output on one problem instance."""

    method: str
    tau: float
    designs: list[Variant]
    measurements: list[Measurement]
    """Oracle readings for ``designs``, in the same order."""
    n_evaluations: int
    """Distinct variants scored -- the budget actually spent."""
    n_requested: int
    """How many designs were asked for. Fewer may be returned; see ``designs``."""
    trace: list[tuple[Variant, Measurement]] = field(default_factory=list, repr=False)
    objective_log: list[dict[str, float]] = field(default_factory=list, repr=False)
    stopped_early: bool = False
    target: Target = field(default_factory=Minimize)
    """What the run was aiming at. Ranking inside :meth:`filtered` follows it,
    so the uniform readout means the same thing under either regime."""

    @property
    def n_returned(self) -> int:
        return len(self.designs)

    @property
    def trace_measurements(self) -> list[Measurement]:
        """Every design the search actually scored -- where the budget went."""
        return [m for _, m in self.trace]

    def filtered(self, n: int | None = None) -> list[Measurement]:
        """Re-read the same run under a uniform feasibility filter.

        Selection is post-hoc over a trace that is already paid for, so both
        readouts come from one run at no extra oracle cost:

        * ``measurements`` -- what the rule hands over on its own terms. The
          feasibility rate of *that* set is the metric nobody reports.
        * ``filtered(n)`` -- every rule made to discard infeasible designs and
          rank the rest by binding. Selection is then identical across rules,
          so what remains is the effect of the search objective alone.

        Reporting only the first would flatter rules that quietly return
        unusable designs; reporting only the second would hide that they do.
        """
        n = self.n_requested if n is None else n
        feasible = [pair for pair in self.trace if pair[1].is_feasible(self.tau)]
        feasible.sort(key=lambda pair: self.target.cost(pair[1]))
        return [m for _, m in feasible[:n]]


def _select_by_objective(
    trace: Sequence[tuple[Variant, Measurement]],
    objective: Objective,
    n: int,
) -> list[tuple[Variant, Measurement]]:
    return sorted(trace, key=lambda pair: objective.score(pair[1]))[:n]


def _select_by_binding(
    trace: Sequence[tuple[Variant, Measurement]],
    n: int,
    target: Target,
) -> list[tuple[Variant, Measurement]]:
    """Rank by the target, which under ``Minimize`` is binding alone."""
    return sorted(trace, key=lambda pair: target.cost(pair[1]))[:n]


def _select_reject_then_binding(
    trace: Sequence[tuple[Variant, Measurement]],
    tau: float,
    n: int,
    target: Target,
) -> list[tuple[Variant, Measurement]]:
    """Discard everything that breaks the constraint, then take the best binders.

    The filter runs over the whole trace, not just the final population. That
    is the strongest honest form of optimize-then-reject: filtering only the
    final population would be a straw man, and a weak null hypothesis would
    make the headline result worthless.
    """
    feasible = [pair for pair in trace if pair[1].is_feasible(tau)]
    return sorted(feasible, key=lambda pair: target.cost(pair[1]))[:n]


def _finish(
    name: str,
    tau: float,
    selected: Sequence[tuple[Variant, Measurement]],
    result: SearchResult,
    oracle: BudgetedOracle,
    n_return: int,
    target: Target,
) -> DesignRun:
    return DesignRun(
        method=name,
        tau=tau,
        designs=[v for v, _ in selected],
        measurements=[m for _, m in selected],
        n_evaluations=oracle.n_unique,
        n_requested=n_return,
        trace=list(oracle.history),
        objective_log=result.objective_log,
        stopped_early=result.stopped_early,
        target=target,
    )


def _oracle(landscape: Landscape, budget: int) -> BudgetedOracle:
    return BudgetedOracle(landscape=landscape, budget=budget)


# --------------------------------------------------------------------------
# Rule 1 -- scalarization
# --------------------------------------------------------------------------


def run_scalarization(
    landscape: Landscape,
    tau: float,
    budget: int,
    weight: float = 0.5,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 1 at a single fixed weight. ``tau`` is recorded but never used."""
    target = target or Minimize()
    objective = Scalarized(weight=weight, target=target)
    oracle = _oracle(landscape, budget)
    result = evolutionary_search(oracle, objective, config)
    selected = _select_by_objective(result.trace, objective, n_return)
    return _finish(objective.name, tau, selected, result, oracle, n_return, target)


def run_scalarization_sweep(
    landscape: Landscape,
    tau: float,
    budget: int,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 1 as practitioners run it: sweep the weight, pool, submit the best binders.

    The sweep gets one budget in total, split evenly between its weights, and
    one shared oracle cache. Charging it a full budget per weight would be the
    easy mistake in the other direction: sweeping five weights really does cost
    five times as many assays, and pretending otherwise would make every other
    method look worse than it is.

    The split is enforced by raising the oracle's cap one slice at a time, so a
    weight that converges early leaves its slack to the next one rather than
    letting the first weight eat the whole budget.

    Pooled designs are ranked by binding alone: weighted sums computed at
    different weights are not comparable to each other, and a practitioner with
    no feasibility criterion submits their best binders.
    """
    if not weights:
        raise ValueError("need at least one weight")
    target = target or Minimize()
    base = config or SearchConfig()
    per_weight = max(1, base.n_generations // len(weights))
    oracle = _oracle(landscape, budget)

    merged_log: list[dict[str, float]] = []
    stopped_early = False
    last_result: SearchResult | None = None
    for i, weight in enumerate(weights):
        # Each weight may spend up to its own slice of the shared budget.
        oracle.budget = round(budget * (i + 1) / len(weights))
        sub_config = SearchConfig(
            **{**base.__dict__, "n_generations": per_weight, "seed": base.seed + i}
        )
        objective = Scalarized(weight=weight, target=target)
        last_result = evolutionary_search(oracle, objective, sub_config)
        merged_log.extend(last_result.objective_log)
        stopped_early = stopped_early or last_result.stopped_early
    oracle.budget = budget

    assert last_result is not None
    last_result.stopped_early = stopped_early
    last_result.objective_log = merged_log
    selected = _select_by_binding(oracle.history, n_return, target)
    return _finish("scalarization_sweep", tau, selected, last_result, oracle, n_return, target)


# --------------------------------------------------------------------------
# Rule 2 -- fixed penalty
# --------------------------------------------------------------------------


def run_fixed_penalty(
    landscape: Landscape,
    tau: float,
    budget: int,
    rho: float = 10.0,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 2. ``rho`` is fixed in advance and never adapts."""
    target = target or Minimize()
    objective = FixedPenalty(tau=tau, rho=rho, target=target)
    oracle = _oracle(landscape, budget)
    result = evolutionary_search(oracle, objective, config)
    selected = _select_by_objective(result.trace, objective, n_return)
    return _finish(objective.name, tau, selected, result, oracle, n_return, target)


# --------------------------------------------------------------------------
# Rule 3 -- Lagrangian
# --------------------------------------------------------------------------


def run_lagrangian(
    landscape: Landscape,
    tau: float,
    budget: int,
    lr: float = 0.5,
    lam0: float = 1.0,
    lam_max: float = 100.0,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 3 -- the proposed method."""
    target = target or Minimize()
    objective = Lagrangian(tau=tau, lr=lr, lam=lam0, lam_max=lam_max, target=target)
    oracle = _oracle(landscape, budget)
    result = evolutionary_search(oracle, objective, config)
    selected = _select_by_objective(result.trace, objective, n_return)
    return _finish(objective.name, tau, selected, result, oracle, n_return, target)


# --------------------------------------------------------------------------
# Rule 4 -- optimize then reject
# --------------------------------------------------------------------------


def run_optimize_then_reject(
    landscape: Landscape,
    tau: float,
    budget: int,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 4 -- the null hypothesis. Chase the target, discard infeasible at the end."""
    target = target or Minimize()
    objective = BindingOnly(target=target)
    oracle = _oracle(landscape, budget)
    result = evolutionary_search(oracle, objective, config)
    selected = _select_reject_then_binding(result.trace, tau, n_return, target)
    return _finish("optimize_then_reject", tau, selected, result, oracle, n_return, target)


# --------------------------------------------------------------------------
# Rule 5 -- NSGA-II
# --------------------------------------------------------------------------


def run_nsga2(
    landscape: Landscape,
    tau: float,
    budget: int,
    n_return: int = 20,
    config: SearchConfig | None = None,
    target: Target | None = None,
) -> DesignRun:
    """Rule 5 -- non-dominated sorting, the framing this benchmark argues with.

    Folding is a second objective rather than a constraint, so like
    scalarization this rule never sees ``tau``. Designs are returned by the
    same crowded-comparison ordering that drove the search: non-dominated
    front first, then crowding distance.

    That selection is the honest one for this rule. Filtering its output by
    feasibility would make it a different method -- a constraint-aware one --
    and would hide the thing worth measuring, which is what a Pareto front
    hands you when one of its axes is not negotiable.
    """
    target = target or Minimize()
    objective = NSGA2(target=target)
    oracle = _oracle(landscape, budget)
    result = evolutionary_search(oracle, objective, config)
    trace = list(result.trace)
    order = objective.population_order([m for _, m in trace]) or []
    selected = [trace[i] for i in order[:n_return]]
    return _finish("nsga2", tau, selected, result, oracle, n_return, target)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

Method = Callable[..., DesignRun]


def default_methods(
    include_weight_sweep: bool = True, target: Target | None = None
) -> dict[str, Method]:
    """The comparison set: one entry per constraint-handling rule.

    Signatures are uniform -- ``(landscape, tau, budget, n_return, config)`` --
    so an experiment can loop over them without special-casing.

    ``target`` sets the regime for every rule at once. It defaults to
    :class:`~cpd.objectives.Minimize`, so existing callers are unaffected;
    passing a :class:`~cpd.objectives.Window` runs the identical comparison
    against a target range instead of an unbounded one. Rule names do not
    change with the regime -- the regime is recorded as its own column, so the
    two runs stay directly comparable row by row.
    """
    target = target or Minimize()
    methods: dict[str, Method] = {
        "scalarization_w0.5": lambda landscape, tau, budget, n_return=20, config=None: (
            run_scalarization(landscape, tau, budget, 0.5, n_return, config, target)
        ),
        "fixed_penalty": lambda landscape, tau, budget, n_return=20, config=None: (
            run_fixed_penalty(landscape, tau, budget, 10.0, n_return, config, target)
        ),
        "lagrangian": lambda landscape, tau, budget, n_return=20, config=None: (
            run_lagrangian(landscape, tau, budget, 0.5, 1.0, 100.0, n_return, config, target)
        ),
        "optimize_then_reject": lambda landscape, tau, budget, n_return=20, config=None: (
            run_optimize_then_reject(landscape, tau, budget, n_return, config, target)
        ),
        "nsga2": lambda landscape, tau, budget, n_return=20, config=None: (
            run_nsga2(landscape, tau, budget, n_return, config, target)
        ),
    }
    if include_weight_sweep:
        methods["scalarization_sweep"] = (
            lambda landscape, tau, budget, n_return=20, config=None: run_scalarization_sweep(
                landscape, tau, budget, DEFAULT_WEIGHTS, n_return, config, target
            )
        )
    return methods
