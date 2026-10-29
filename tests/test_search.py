import numpy as np
import pytest

from cpd.landscape import BudgetedOracle, ExactLandscape, SyntheticLandscape
from cpd.objectives import BindingOnly, Lagrangian, Scalarized
from cpd.search import SearchConfig, evolutionary_search
from cpd.variants import Variant

CONFIG = SearchConfig(population_size=12, n_generations=8, n_offspring=12, seed=0)


def _landscape() -> SyntheticLandscape:
    return SyntheticLandscape(n_positions=12, n_interface=4, n_core=3, max_order=4, seed=5)


def test_search_config_validates_its_move_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        SearchConfig(p_add=0.5, p_remove=0.3, p_substitute=0.9)
    with pytest.raises(ValueError):
        SearchConfig(population_size=0)


def test_same_seed_gives_an_identical_run():
    def run(seed: int):
        oracle = BudgetedOracle(_landscape(), budget=300)
        result = evolutionary_search(
            oracle, BindingOnly(), SearchConfig(**{**CONFIG.__dict__, "seed": seed})
        )
        return [str(v) for v, _ in result.trace]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_search_never_exceeds_its_budget():
    oracle = BudgetedOracle(_landscape(), budget=50)
    result = evolutionary_search(oracle, BindingOnly(), CONFIG)

    assert oracle.n_unique <= 50
    assert result.n_evaluations == oracle.n_unique
    assert result.stopped_early  # 12 + 8*12 proposals would need far more than 50


def test_search_finishes_without_a_budget_cap():
    oracle = BudgetedOracle(_landscape(), budget=None)
    result = evolutionary_search(oracle, BindingOnly(), CONFIG)
    assert not result.stopped_early
    assert result.generations_run == CONFIG.n_generations


def test_search_stays_inside_the_design_space():
    landscape = _landscape()
    oracle = BudgetedOracle(landscape, budget=400)
    result = evolutionary_search(oracle, BindingOnly(), SearchConfig(**{**CONFIG.__dict__, "max_order": 3}))

    assert result.trace
    for variant, _ in result.trace:
        assert landscape.contains(variant)
        assert variant.order <= 3


def test_search_beats_random_sampling_on_the_objective_it_is_given():
    landscape = _landscape()
    budget = 400
    oracle = BudgetedOracle(landscape, budget=budget)
    result = evolutionary_search(oracle, BindingOnly(), CONFIG)
    searched_best = min(m.ddg_binding for _, m in result.trace)

    # Random variants of comparable order, same budget, same seed discipline.
    rng = np.random.default_rng(0)
    random_best = float("inf")
    for _ in range(budget):
        order = int(rng.integers(1, 5))
        positions = rng.choice(landscape.positions, size=order, replace=False)
        mutations = []
        for p in positions:
            candidates = landscape.candidate_mutations(int(p))
            mutations.append(candidates[int(rng.integers(len(candidates)))])
        random_best = min(random_best, landscape.evaluate(Variant.of(mutations)).ddg_binding)

    assert searched_best < random_best


def test_the_objective_is_updated_once_per_generation():
    oracle = BudgetedOracle(_landscape(), budget=None)
    objective = Lagrangian(tau=1.0, lr=0.3)
    result = evolutionary_search(oracle, objective, CONFIG)

    # One update at initialization, then one per generation.
    assert len(objective.history) == CONFIG.n_generations + 2
    assert len(result.objective_log) == CONFIG.n_generations + 1
    assert all("lam" in entry for entry in result.objective_log)


def test_a_stateful_objective_re_ranks_the_population_after_its_state_moves():
    """Scores are never cached, so a multiplier change reorders survivors."""
    landscape = _landscape()
    oracle = BudgetedOracle(landscape, budget=None)
    objective = Lagrangian(tau=-1.0, lr=2.0, lam=0.0)
    result = evolutionary_search(oracle, objective, CONFIG)

    # tau is tight, so the multiplier must have climbed off its initial zero.
    assert objective.lam > 0.0
    # The returned population is sorted by the objective at its *final* state.
    scores = [objective.score(landscape.evaluate(v)) for v in result.population]
    assert scores == sorted(scores)


def test_search_works_on_a_sparse_measured_library():
    """Track 1: most order-2 combinations were never assayed and must be skipped."""
    records = [("E24K", -1.0, 0.4), ("E24R", 0.2, -0.1), ("L31P", -0.5, 2.0), ("W40A", 0.3, 0.9)]
    records.append(("E24K,L31P", -1.8, 2.6))
    records.append(("E24R,W40A", -0.2, 0.5))
    landscape = ExactLandscape.from_records(records)

    oracle = BudgetedOracle(landscape, budget=None)
    result = evolutionary_search(oracle, BindingOnly(), SearchConfig(population_size=6, n_generations=6, n_offspring=6, seed=1))

    for variant, _ in result.trace:
        assert landscape.contains(variant)
    # The best binder in the library is reachable and should be found.
    assert min(m.ddg_binding for _, m in result.trace) == pytest.approx(-1.8)


def test_a_population_of_one_still_runs():
    oracle = BudgetedOracle(_landscape(), budget=None)
    config = SearchConfig(population_size=1, n_generations=3, n_offspring=2, tournament_size=1, seed=0)
    result = evolutionary_search(oracle, Scalarized(0.5), config)
    assert len(result.population) == 1
