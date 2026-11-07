"""Baselines that do no search, and the exact optimum where one can be had.

A method comparison means nothing until you know the problem is hard. These
two checks establish that, and on this benchmark they change the conclusion.

:func:`greedy_feasible` is the dumbest thing that could work: measure every
single mutant once, sort them by the objective, then add substitutions one at a
time while the constraint still holds. It has no population, no stochasticity
and no hyperparameter, and it costs the singles plus at most ``max_order``
composition calls. If it ties a budgeted evolutionary search, the search is not
what is producing the result.

:func:`exact_optimum` solves the constrained problem outright -- but only for an
additive landscape, where both objectives are sums of per-substitution
constants and the problem is therefore an integer program rather than a search
over a rugged landscape. That restriction is the finding, not a limitation of
the implementation: see :mod:`cpd.landscape` on what the additive oracle
assumes. The optimum turns a relative ranking into an absolute one -- how much
of the attainable objective did each rule actually capture.

See :mod:`experiments.baselines` for the comparison these are used in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .landscape import BudgetedOracle, Landscape, MeasuredAdditiveLandscape, Measurement
from .variants import Mutation, Variant

__all__ = ["GreedyResult", "greedy_feasible", "exact_optimum", "pareto_prune"]


@dataclass(frozen=True)
class GreedyResult:
    """What the greedy baseline produced, and what it cost."""

    variant: Variant
    measurement: Measurement | None
    """None when nothing feasible was found, or when the delivered design
    violated the constraint its additive plan predicted it would satisfy."""
    n_evaluations: int
    """Distinct oracle calls spent, counted the way the search counts them."""
    order: int

    @property
    def ddg_binding(self) -> float:
        return float("nan") if self.measurement is None else self.measurement.ddg_binding

    @property
    def ddg_folding(self) -> float:
        return float("nan") if self.measurement is None else self.measurement.ddg_folding


def greedy_feasible(
    landscape: Landscape,
    tau: float,
    max_order: int | None = None,
    rank: Callable[[Measurement], float] | None = None,
    budget: int | None = None,
) -> GreedyResult:
    """Rank the singles once, then add substitutions while the constraint holds.

    The constraint is checked against the running sum of single-mutant folding
    effects, which is an additive *plan*. On a landscape with epistasis the
    delivered design can miss the folding cost that plan predicted, so the
    design is finally checked against its true measurement and rejected if it
    violates ``tau`` -- the miss shows up instead of being assumed away.

    ``rank`` defaults to ranking by binding alone, which is the version worth
    beating: it is what a practitioner does when they hold per-residue
    predictions and a stability cutoff.
    """
    max_order = landscape.max_order if max_order is None else max_order
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")
    rank = rank or (lambda m: m.ddg_binding)
    oracle = BudgetedOracle(landscape, budget=budget)

    scored: list[tuple[float, Mutation, Measurement]] = []
    for position in landscape.positions:
        for mutation in landscape.candidate_mutations(position):
            single = Variant((mutation,))
            if not landscape.contains(single):
                continue
            measurement = oracle(single)
            scored.append((rank(measurement), mutation, measurement))
    if not scored:
        raise ValueError("landscape exposes no single mutants to rank")
    scored.sort(key=lambda row: row[0])

    chosen: list[Mutation] = []
    used: set[int] = set()
    planned_folding = 0.0
    for _, mutation, measurement in scored:
        if len(chosen) >= max_order:
            break
        if mutation.position in used:
            continue
        if planned_folding + measurement.ddg_folding <= tau:
            candidate = Variant.of(chosen + [mutation])
            if not landscape.contains(candidate):
                continue
            chosen.append(mutation)
            used.add(mutation.position)
            planned_folding += measurement.ddg_folding

    if not chosen:
        return GreedyResult(Variant(), None, oracle.n_unique, 0)

    variant = Variant.of(chosen)
    delivered = oracle(variant)
    if delivered.ddg_folding > tau:
        return GreedyResult(variant, None, oracle.n_unique, variant.order)
    return GreedyResult(variant, delivered, oracle.n_unique, variant.order)


def pareto_prune(front: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only the non-dominated ``(folding, binding)`` pairs; both minimized."""
    if not front:
        return []
    out: list[tuple[float, float]] = []
    best_binding = float("inf")
    for folding, binding in sorted(front):
        if binding < best_binding - 1e-12:
            out.append((folding, binding))
        best_binding = min(best_binding, binding)
    return out


def exact_optimum(
    landscape: MeasuredAdditiveLandscape,
    taus: Sequence[float],
    max_order: int | None = None,
) -> dict[float, float]:
    """Lowest attainable ``ddg_binding`` subject to ``ddg_folding <= tau``.

    Exact, by dynamic programming over positions. ``state[k]`` holds the Pareto
    front of ``(cumulative_folding, cumulative_binding)`` reachable using
    exactly ``k`` substitutions among the positions processed so far.

    Pareto pruning is lossless for the query ``min binding subject to
    folding <= tau``: if one partial solution is dominated by another, then
    every extension of the first is dominated by the same extension of the
    second, because the landscape adds the extension's effect to both alike.
    Carrying ``k`` in the state makes the order cap exact, and processing one
    position at a time makes "at most one substitution per position" exact.

    That argument rests on the objectives being *separable*, which holds for
    :class:`~cpd.landscape.MeasuredAdditiveLandscape` and fails for any
    landscape with epistasis -- hence the type restriction.
    """
    if not isinstance(landscape, MeasuredAdditiveLandscape):
        raise TypeError(
            "exact_optimum is only valid for an additive landscape, where both "
            "objectives are sums of per-substitution constants. Got "
            f"{type(landscape).__name__}, which is not separable, so the dynamic "
            "program would not be exact."
        )
    max_order = landscape.max_order if max_order is None else max_order
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")

    state: list[list[tuple[float, float]]] = [[] for _ in range(max_order + 1)]
    state[0] = [(0.0, 0.0)]
    for position in landscape.positions:
        options: list[tuple[float, float]] = []
        for mutation in landscape.candidate_mutations(position):
            m = landscape.evaluate(Variant((mutation,)))
            options.append((m.ddg_folding, m.ddg_binding))
        if not options:
            continue
        nxt = [list(level) for level in state]
        for k in range(max_order, 0, -1):
            for folding, binding in state[k - 1]:
                for d_folding, d_binding in options:
                    nxt[k].append((folding + d_folding, binding + d_binding))
        state = [pareto_prune(level) for level in nxt]

    pooled = pareto_prune([point for level in state for point in level])
    out: dict[float, float] = {}
    for tau in taus:
        feasible = [b for f, b in pooled if f <= tau]
        out[float(tau)] = min(feasible) if feasible else float("nan")
    return out
