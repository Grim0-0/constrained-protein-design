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
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from .landscape import (
    BudgetedOracle,
    BudgetExhausted,
    Landscape,
    MeasuredAdditiveLandscape,
    Measurement,
)
from .variants import Mutation, Variant

__all__ = [
    "GreedyResult",
    "greedy_feasible",
    "greedy_forward",
    "greedy_window",
    "exact_optimum",
    "exact_optimum_window",
    "pareto_prune",
]

if TYPE_CHECKING:  # pragma: no cover - import only for the annotation
    from .objectives import Window


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


def greedy_forward(
    landscape: Landscape,
    tau: float,
    max_order: int | None = None,
    budget: int | None = None,
) -> GreedyResult:
    """Forward selection against the true oracle: the strong baseline.

    At each step, score every legal single addition on the real oracle and keep
    the feasible one that improves binding most. Stop when nothing feasible
    improves, at the order cap, or when the budget runs out.

    Unlike :func:`greedy_feasible` this never trusts an additive plan, so
    epistasis cannot mislead it -- it pays for the truth at every step. It is
    also myopic and much more expensive, costing up to ``max_order`` passes
    over the whole single-mutant library. Both properties matter for the
    comparison it is used in: if a budgeted search cannot beat even this, the
    search is not what produces the result.
    """
    max_order = landscape.max_order if max_order is None else max_order
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")
    oracle = BudgetedOracle(landscape, budget=budget)

    chosen: list[Mutation] = []
    used: set[int] = set()
    best: Measurement | None = None

    while len(chosen) < max_order:
        candidate_best: tuple[Mutation, Measurement, int] | None = None
        exhausted = False
        for position in landscape.positions:
            if position in used:
                continue
            for mutation in landscape.candidate_mutations(position):
                trial = Variant.of(chosen + [mutation])
                if not landscape.contains(trial):
                    continue
                try:
                    m = oracle(trial)
                except BudgetExhausted:
                    exhausted = True
                    break
                if m.ddg_folding > tau:
                    continue
                if best is not None and m.ddg_binding >= best.ddg_binding:
                    continue
                if candidate_best is None or m.ddg_binding < candidate_best[1].ddg_binding:
                    candidate_best = (mutation, m, position)
            if exhausted:
                break
        if candidate_best is None:
            break
        chosen.append(candidate_best[0])
        used.add(candidate_best[2])
        best = candidate_best[1]
        if exhausted:
            break

    if best is None:
        return GreedyResult(Variant(), None, oracle.n_unique, 0)
    return GreedyResult(Variant.of(chosen), best, oracle.n_unique, len(chosen))


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


# --------------------------------------------------------------------------
# The window regime
# --------------------------------------------------------------------------


def greedy_window(
    landscape: Landscape,
    window: "Window",
    tau: float,
    max_order: int | None = None,
    budget: int | None = None,
) -> GreedyResult:
    """Forward selection toward a target window, scored on the true oracle.

    The rule: at each step score every legal single addition, discard the ones
    that break the folding constraint, and keep whichever brings cumulative
    binding closest to the window. Stop on entering the window, when no
    addition gets closer, at the order cap, or when the budget runs out.

    This is the fair window-aware baseline, not a handicapped one. Ranking
    singles by binding -- which is what :func:`greedy_feasible` does and what
    nearly solves the unbounded problem -- is meaningless against a window,
    because overshooting costs as much as undershooting. Giving greedy that
    rule here and then reporting that it fails would be a straw man. This
    version gets the best myopic rule available: full knowledge of the true
    objective at every step. If it still fails, the failure is structural.

    And it should be expected to: hitting a cumulative sum inside a range is
    subset-sum, where the locally closest step routinely rules out the
    combination that would have landed inside.
    """
    max_order = landscape.max_order if max_order is None else max_order
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")
    oracle = BudgetedOracle(landscape, budget=budget)

    chosen: list[Mutation] = []
    used: set[int] = set()
    best: Measurement | None = None
    best_distance = float("inf")

    while len(chosen) < max_order:
        candidate: tuple[Mutation, Measurement, int, float] | None = None
        exhausted = False
        for position in landscape.positions:
            if position in used:
                continue
            for mutation in landscape.candidate_mutations(position):
                trial = Variant.of(chosen + [mutation])
                if not landscape.contains(trial):
                    continue
                try:
                    m = oracle(trial)
                except BudgetExhausted:
                    exhausted = True
                    break
                if m.ddg_folding > tau:
                    continue
                distance = window.distance(m.ddg_binding)
                if distance >= best_distance:
                    continue
                if candidate is None or distance < candidate[3]:
                    candidate = (mutation, m, position, distance)
            if exhausted:
                break
        if candidate is None:
            break
        chosen.append(candidate[0])
        used.add(candidate[2])
        best, best_distance = candidate[1], candidate[3]
        if best_distance <= 0.0 or exhausted:
            break

    if best is None:
        return GreedyResult(Variant(), None, oracle.n_unique, 0)
    return GreedyResult(Variant.of(chosen), best, oracle.n_unique, len(chosen))


def _shift(row: np.ndarray, offset: int) -> np.ndarray:
    """Move ``row`` right by ``offset`` bins, filling vacated bins with inf."""
    out = np.full_like(row, np.inf)
    n = row.size
    if offset == 0:
        out[:] = row
    elif 0 < offset < n:
        out[offset:] = row[: n - offset]
    elif -n < offset < 0:
        out[: n + offset] = row[-offset:]
    return out


def exact_optimum_window(
    landscape: MeasuredAdditiveLandscape,
    window: "Window",
    taus: Sequence[float],
    max_order: int | None = None,
    bin_width: float = 0.01,
) -> dict[float, float]:
    """Smallest achievable distance to ``window`` subject to ``folding <= tau``.

    **The Pareto argument used by :func:`exact_optimum` does not hold here.**
    That one prunes any point another dominates in both objectives, which is
    lossless only because lower binding is always better. Against a window a
    point with *higher* binding can be the one that lands inside, so pruning
    it can discard the optimum outright. Reusing that dynamic program here
    would silently report a worse answer than the true one.

    So this is a different algorithm: discretize cumulative binding into bins
    and store, for each ``(k, bin)``, the minimum cumulative folding reachable
    with exactly ``k`` substitutions. Processing one position at a time keeps
    "at most one substitution per position" exact, and the ``k`` index keeps
    the order cap exact. A tau query scans the bins whose minimum folding
    clears it and takes the smallest distance to the window.

    **Discretization error.** Each step's binding effect is snapped to a
    multiple of ``bin_width``, so a design of order ``k`` can be misplaced by
    at most ``k * bin_width / 2``. At the default 0.01 kcal/mol and an order
    cap of 4 that is 0.02 kcal/mol -- an order of magnitude below the gaps
    this benchmark reports, and it is the reason the default is not coarser.
    Pass a smaller ``bin_width`` to shrink it at quadratic cost in memory.
    """
    if not isinstance(landscape, MeasuredAdditiveLandscape):
        raise TypeError(
            "exact_optimum_window is only valid for an additive landscape, where "
            f"both objectives are sums of per-substitution constants. Got "
            f"{type(landscape).__name__}, which is not separable."
        )
    max_order = landscape.max_order if max_order is None else max_order
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    options: list[list[tuple[float, float]]] = []
    for position in landscape.positions:
        per_position = []
        for mutation in landscape.candidate_mutations(position):
            m = landscape.evaluate(Variant((mutation,)))
            per_position.append((m.ddg_binding, m.ddg_folding))
        if per_position:
            options.append(per_position)
    if not options:
        raise ValueError("landscape exposes no single mutants")

    every_binding = [b for group in options for b, _ in group]
    worst_down = sorted(b for b in every_binding if b < 0)[:max_order]
    worst_up = sorted((b for b in every_binding if b > 0), reverse=True)[:max_order]
    low = sum(worst_down) - bin_width
    high = sum(worst_up) + bin_width
    n_bins = int(np.ceil((high - low) / bin_width)) + 1
    grid = low + bin_width * np.arange(n_bins)

    #: dp[k, bin] = least cumulative folding using exactly k substitutions
    dp = np.full((max_order + 1, n_bins), np.inf)
    dp[0, int(round((0.0 - low) / bin_width))] = 0.0

    for group in options:
        nxt = dp.copy()
        for d_binding, d_folding in group:
            offset = int(round(d_binding / bin_width))
            for k in range(max_order, 0, -1):
                nxt[k] = np.minimum(nxt[k], _shift(dp[k - 1], offset) + d_folding)
        dp = nxt

    distance = np.maximum(window.lo - grid, 0.0) + np.maximum(grid - window.hi, 0.0)
    reachable = dp.min(axis=0)
    out: dict[float, float] = {}
    for tau in taus:
        feasible = reachable <= tau
        out[float(tau)] = float(distance[feasible].min()) if feasible.any() else float("nan")
    return out
