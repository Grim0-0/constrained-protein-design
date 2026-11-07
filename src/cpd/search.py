"""The shared evolutionary search engine.

There is exactly one search engine in this package and every method runs
through it. The objective function is injected; nothing else about the search
changes between methods. Comparing, say, a policy-gradient agent against
NSGA-II would confound the search algorithm with the constraint-handling
mechanism, and the resulting comparison would say nothing about constraint
handling at all.

The engine is a (mu + lambda) evolutionary algorithm over variable-order
variants, with tournament selection and three move types: add a substitution,
revert one to wild type, or change one to a different residue. Moves are
rejection-sampled against :meth:`~cpd.landscape.Landscape.contains`, so the
same engine works over a dense synthetic landscape and over the sparse
measured library of Track 1 without modification.

One rule does not fit "score each design independently": NSGA-II decides
survival by non-dominated sorting, so whether a variant lives depends on what
else is in the population. Rather than give it a second engine -- which would
reintroduce exactly the confound this module exists to avoid -- the engine
asks every objective for a set-level ordering via
:meth:`~cpd.objectives.Objective.population_order`. That returns ``None`` for
every other rule, and the engine then takes its original per-design path
unchanged. The opt-in is deliberate: results committed before NSGA-II existed
reproduce byte for byte, which is checked in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .landscape import BudgetedOracle, BudgetExhausted, Landscape, Measurement
from .objectives import Objective
from .variants import Mutation, Variant

__all__ = ["SearchConfig", "SearchResult", "evolutionary_search"]


@dataclass(frozen=True)
class SearchConfig:
    """Search hyperparameters. Held fixed across every method in a comparison."""

    population_size: int = 32
    n_generations: int = 40
    n_offspring: int = 32
    tournament_size: int = 3
    init_order: int = 1
    max_order: int | None = None
    """Cap on variant order. Defaults to the landscape's own ``max_order``."""
    p_add: float = 0.4
    p_remove: float = 0.2
    p_substitute: float = 0.4
    crossover_rate: float = 0.3
    max_resample: int = 64
    """Attempts to find a valid neighbour before giving up and keeping the parent."""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        total = self.p_add + self.p_remove + self.p_substitute
        if not np.isclose(total, 1.0):
            raise ValueError(f"p_add + p_remove + p_substitute must sum to 1, got {total}")


@dataclass
class SearchResult:
    """What one search run produced."""

    population: list[Variant]
    """Final population, best first by the objective's score at the end of the run."""
    trace: list[tuple[Variant, Measurement]]
    """Every distinct variant evaluated, in the order it was evaluated."""
    n_evaluations: int
    """Distinct variants evaluated -- the budget actually spent."""
    generations_run: int
    stopped_early: bool
    """True if the run ended because the oracle budget ran out."""
    objective_log: list[dict[str, float]] = field(default_factory=list)
    """Objective state after each generation, e.g. the Lagrangian multiplier."""

    def measurements(self) -> list[Measurement]:
        return [m for _, m in self.trace]


def _tournament(
    scored: Sequence[tuple[Variant, Measurement]],
    objective: Objective,
    size: int,
    rng: np.random.Generator,
    by_position: bool = False,
) -> Variant:
    """Pick the best of ``size`` random candidates.

    ``by_position`` is for rules that rank a population rather than scoring
    its members, such as NSGA-II. The population is kept best-first, so the
    lowest sampled index is the winner. Every other rule takes the original
    score-based path unchanged.
    """
    k = min(size, len(scored))
    idx = rng.choice(len(scored), size=k, replace=False)
    if by_position:
        return scored[int(min(idx))][0]
    best = min((scored[int(i)] for i in idx), key=lambda pair: objective.score(pair[1]))
    return best[0]


def _mutate(
    variant: Variant,
    landscape: Landscape,
    config: SearchConfig,
    max_order: int,
    rng: np.random.Generator,
) -> Variant:
    """Apply one random move. Returns the parent unchanged if no move is available."""
    mutated_positions = sorted(variant.positions)
    free_positions = [p for p in landscape.positions if p not in variant.positions]

    moves: list[str] = []
    weights: list[float] = []
    if free_positions and variant.order < max_order:
        moves.append("add")
        weights.append(config.p_add)
    if mutated_positions:
        moves.append("remove")
        weights.append(config.p_remove)
        moves.append("substitute")
        weights.append(config.p_substitute)
    if not moves:
        return variant

    move = str(rng.choice(moves, p=np.asarray(weights) / sum(weights)))
    if move == "add":
        position = int(rng.choice(free_positions))
        candidates = landscape.candidate_mutations(position)
        if not candidates:
            return variant
        return variant.with_mutation(candidates[int(rng.integers(len(candidates)))])
    if move == "remove":
        return variant.without_position(int(rng.choice(mutated_positions)))

    position = int(rng.choice(mutated_positions))
    current = next(m.mut for m in variant if m.position == position)
    candidates = [m for m in landscape.candidate_mutations(position) if m.mut != current]
    if not candidates:
        return variant
    return variant.with_mutation(candidates[int(rng.integers(len(candidates)))])


def _crossover(
    parent_a: Variant,
    parent_b: Variant,
    max_order: int,
    rng: np.random.Generator,
) -> Variant:
    """Uniformly inherit each parent's substitutions, keeping one per position."""
    pooled: dict[int, Mutation] = {}
    for mutation in list(parent_a) + list(parent_b):
        # A coin flip per substitution; later duplicates at a position overwrite earlier.
        if rng.random() < 0.5:
            pooled[mutation.position] = mutation
    chosen = list(pooled.values())
    if len(chosen) > max_order:
        keep = rng.choice(len(chosen), size=max_order, replace=False)
        chosen = [chosen[int(i)] for i in keep]
    return Variant.of(chosen)


def _rank_population(
    scored: Sequence[tuple[Variant, Measurement]], objective: Objective
) -> list[tuple[Variant, Measurement]]:
    """Reorder best-first using the objective's own set-level ranking."""
    order = objective.population_order([m for _, m in scored])
    if order is None:
        return list(scored)
    return [scored[i] for i in order]


def _propose(
    parents: Sequence[tuple[Variant, Measurement]],
    landscape: Landscape,
    objective: Objective,
    config: SearchConfig,
    max_order: int,
    rng: np.random.Generator,
    by_position: bool = False,
) -> Variant:
    """Draw one valid offspring, rejection-sampling against the landscape."""
    for _ in range(config.max_resample):
        parent = _tournament(parents, objective, config.tournament_size, rng, by_position)
        if len(parents) > 1 and rng.random() < config.crossover_rate:
            other = _tournament(parents, objective, config.tournament_size, rng, by_position)
            child = _crossover(parent, other, max_order, rng)
        else:
            child = parent
        child = _mutate(child, landscape, config, max_order, rng)
        if child.order <= max_order and landscape.contains(child):
            return child
    # Every proposal landed outside the measured library; fall back to a parent,
    # which is already known-valid and costs nothing to re-evaluate.
    return parents[0][0]


def _random_variant(
    landscape: Landscape,
    order: int,
    max_resample: int,
    rng: np.random.Generator,
) -> Variant:
    """Draw a random valid variant of the requested order, falling back to wild type."""
    positions = list(landscape.positions)
    order = min(order, len(positions))
    for _ in range(max_resample):
        chosen = rng.choice(len(positions), size=order, replace=False)
        mutations = []
        for i in chosen:
            candidates = landscape.candidate_mutations(positions[int(i)])
            if not candidates:
                break
            mutations.append(candidates[int(rng.integers(len(candidates)))])
        else:
            candidate = Variant.of(mutations)
            if landscape.contains(candidate):
                return candidate
    return Variant()


def evolutionary_search(
    oracle: BudgetedOracle,
    objective: Objective,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Run the shared engine with ``objective`` as the thing being minimized.

    The oracle's budget, not ``n_generations``, is usually what ends the run.
    When the budget is spent mid-generation the run stops cleanly and returns
    everything evaluated so far with ``stopped_early`` set.
    """
    config = config or SearchConfig()
    landscape = oracle.landscape
    max_order = config.max_order if config.max_order is not None else landscape.max_order
    rng = np.random.default_rng(config.seed)

    population: list[tuple[Variant, Measurement]] = []
    stopped_early = False

    # --- initialization -------------------------------------------------
    seen: set[Variant] = set()
    for _ in range(config.population_size):
        candidate = _random_variant(landscape, config.init_order, config.max_resample, rng)
        try:
            measurement = oracle(candidate)
        except BudgetExhausted:
            stopped_early = True
            break
        if candidate not in seen:
            seen.add(candidate)
        population.append((candidate, measurement))

    if not population:
        return SearchResult([], list(oracle.history), oracle.n_unique, 0, True, [])

    objective.update([m for _, m in population])
    # A rule that ranks populations rather than scoring designs opts in here.
    # None means "use score", which is every rule but NSGA-II, so the engine
    # behaves exactly as before for them.
    ranks_population = objective.population_order([m for _, m in population]) is not None
    if ranks_population:
        population = _rank_population(population, objective)
    objective_log: list[dict[str, float]] = [dict(objective.state())]

    # --- generations ----------------------------------------------------
    generations_run = 0
    for _ in range(config.n_generations):
        if stopped_early:
            break
        offspring: list[tuple[Variant, Measurement]] = []
        for _ in range(config.n_offspring):
            child = _propose(
                population, landscape, objective, config, max_order, rng, ranks_population
            )
            try:
                offspring.append((child, oracle(child)))
            except BudgetExhausted:
                stopped_early = True
                break
        generations_run += 1

        # Batch-level objective update happens before survivor selection, so a
        # rule whose multiplier just moved re-ranks the whole population.
        if offspring:
            objective.update([m for _, m in offspring])
        objective_log.append(dict(objective.state()))

        combined = population + offspring
        if ranks_population:
            combined = _rank_population(combined, objective)
        else:
            combined.sort(key=lambda pair: objective.score(pair[1]))
        # Deduplicate by variant, keeping the best-ranked copy.
        unique: list[tuple[Variant, Measurement]] = []
        kept: set[Variant] = set()
        for variant, measurement in combined:
            if variant not in kept:
                kept.add(variant)
                unique.append((variant, measurement))
        population = unique[: config.population_size]

    if ranks_population:
        population = _rank_population(population, objective)
    else:
        population.sort(key=lambda pair: objective.score(pair[1]))
    return SearchResult(
        population=[v for v, _ in population],
        trace=list(oracle.history),
        n_evaluations=oracle.n_unique,
        generations_run=generations_run,
        stopped_early=stopped_early,
        objective_log=objective_log,
    )
