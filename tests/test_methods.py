import math

import pytest

from cpd.landscape import SyntheticLandscape
from cpd.methods import (
    default_methods,
    run_lagrangian,
    run_optimize_then_reject,
    run_scalarization,
    run_scalarization_sweep,
)
from cpd.metrics import summarize
from cpd.search import SearchConfig

CONFIG = SearchConfig(population_size=16, n_generations=10, n_offspring=16, seed=0)
BUDGET = 400
TAU = 0.5


def _landscape() -> SyntheticLandscape:
    return SyntheticLandscape(n_positions=12, n_interface=4, n_core=3, max_order=4, seed=5)


@pytest.mark.parametrize("name", sorted(default_methods()))
def test_every_method_runs_and_respects_its_budget(name):
    method = default_methods()[name]
    run = method(_landscape(), TAU, BUDGET, 20, CONFIG)

    assert run.method
    assert run.tau == pytest.approx(TAU)
    assert run.n_evaluations <= BUDGET
    assert len(run.designs) == len(run.measurements)
    assert run.n_returned <= run.n_requested
    assert run.n_requested == 20


@pytest.mark.parametrize("name", sorted(default_methods()))
def test_every_method_is_reproducible(name):
    method = default_methods()[name]
    first = method(_landscape(), TAU, BUDGET, 20, CONFIG)
    second = method(_landscape(), TAU, BUDGET, 20, CONFIG)
    assert [str(v) for v in first.designs] == [str(v) for v in second.designs]


def test_optimize_then_reject_returns_only_feasible_designs():
    run = run_optimize_then_reject(_landscape(), TAU, BUDGET, n_return=20, config=CONFIG)
    assert all(m.is_feasible(TAU) for m in run.measurements)
    # By construction its feasibility rate is 1.0 -- which is exactly why
    # feasibility rate alone is not the story: it may return almost nothing.
    if run.measurements:
        assert summarize(run.measurements, TAU, (5.0, 5.0)).feasibility_rate == pytest.approx(1.0)


def test_optimize_then_reject_searches_where_it_should_not():
    """The null hypothesis spends its budget without regard for the constraint."""
    run = run_optimize_then_reject(_landscape(), TAU, BUDGET, n_return=20, config=CONFIG)
    evaluated = [m for _, m in run.trace]
    searched_feasibility = sum(m.is_feasible(TAU) for m in evaluated) / len(evaluated)
    assert searched_feasibility < 0.5  # most of the budget went past tau


def test_scalarization_ignores_tau_entirely():
    a = run_scalarization(_landscape(), tau=0.5, budget=BUDGET, weight=0.5, config=CONFIG)
    b = run_scalarization(_landscape(), tau=-3.0, budget=BUDGET, weight=0.5, config=CONFIG)
    # Same search, same designs: only the recorded tau differs.
    assert [str(v) for v in a.designs] == [str(v) for v in b.designs]
    assert a.tau != b.tau


def test_scalarization_weight_changes_what_is_returned():
    binding_heavy = run_scalarization(_landscape(), TAU, BUDGET, weight=0.9, config=CONFIG)
    folding_heavy = run_scalarization(_landscape(), TAU, BUDGET, weight=0.1, config=CONFIG)

    mean_binding = lambda run: sum(m.ddg_binding for m in run.measurements) / len(run.measurements)
    mean_folding = lambda run: sum(m.ddg_folding for m in run.measurements) / len(run.measurements)
    assert mean_binding(binding_heavy) < mean_binding(folding_heavy)
    assert mean_folding(folding_heavy) < mean_folding(binding_heavy)


def test_scalarization_sweep_shares_one_budget_across_its_weights():
    run = run_scalarization_sweep(
        _landscape(), TAU, BUDGET, weights=(0.1, 0.5, 0.9), n_return=20, config=CONFIG
    )
    assert run.n_evaluations <= BUDGET
    assert run.method == "scalarization_sweep"
    # Pooled designs are ranked by binding alone.
    binding = [m.ddg_binding for m in run.measurements]
    assert binding == sorted(binding)

    with pytest.raises(ValueError):
        run_scalarization_sweep(_landscape(), TAU, BUDGET, weights=(), config=CONFIG)


def test_lagrangian_logs_its_multiplier_trajectory():
    run = run_lagrangian(_landscape(), TAU, BUDGET, lr=0.5, lam0=1.0, config=CONFIG)
    trajectory = [entry["lam"] for entry in run.objective_log]
    assert len(trajectory) == CONFIG.n_generations + 1
    assert all(lam >= 0.0 for lam in trajectory)
    assert len(set(trajectory)) > 1  # the multiplier actually moved


def test_a_tight_tau_drives_the_multiplier_higher_than_a_loose_one():
    loose = run_lagrangian(_landscape(), tau=6.0, budget=BUDGET, config=CONFIG)
    tight = run_lagrangian(_landscape(), tau=-2.0, budget=BUDGET, config=CONFIG)
    assert tight.objective_log[-1]["lam"] > loose.objective_log[-1]["lam"]


def test_methods_are_comparable_because_they_spend_the_same_budget():
    """With enough generations the budget binds, so every method spends exactly it.

    This is a precondition for the whole comparison. If the generation count
    runs out first, the methods are not being compared at equal oracle cost and
    the budget curves measure nothing.
    """
    generous = SearchConfig(**{**CONFIG.__dict__, "n_generations": 80})
    spends = {
        name: method(_landscape(), TAU, BUDGET, 20, generous).n_evaluations
        for name, method in default_methods().items()
    }
    assert set(spends.values()) == {BUDGET}, spends


def test_a_short_run_is_limited_by_generations_rather_than_budget():
    """The counterpart: too few generations and the budget is never reached."""
    run = run_lagrangian(_landscape(), TAU, budget=10_000, config=CONFIG)
    assert run.n_evaluations < 10_000
    assert not run.stopped_early


def test_an_impossible_tau_makes_feasible_metrics_nan_rather_than_zero():
    """No method can satisfy tau = -20; the metrics must say so, not report a win."""
    landscape = _landscape()
    for name, method in default_methods().items():
        run = method(landscape, -20.0, BUDGET, 20, CONFIG)
        report = summarize(run.measurements, -20.0, (5.0, 5.0), k=10)
        assert math.isnan(report.feasible_best), name
        assert math.isnan(report.feasible_best_k), name


def test_filtered_readout_returns_only_feasible_designs_ranked_by_binding():
    """Both readouts come from one paid-for trace; see DesignRun.filtered."""
    run = run_scalarization(_landscape(), TAU, BUDGET, weight=0.9, config=CONFIG)
    filtered = run.filtered(10)

    assert all(m.is_feasible(TAU) for m in filtered)
    assert [m.ddg_binding for m in filtered] == sorted(m.ddg_binding for m in filtered)
    assert len(filtered) <= 10
    # A binding-heavy weight ignores tau, so filtering must actually remove things.
    assert any(not m.is_feasible(TAU) for m in run.measurements)


def test_filtered_readout_costs_no_extra_oracle_calls():
    run = run_scalarization(_landscape(), TAU, BUDGET, config=CONFIG)
    before = run.n_evaluations
    run.filtered(20)
    assert run.n_evaluations == before


def test_filtered_readout_can_come_up_short_and_says_so():
    """At an impossible tau there is nothing to return, and that is the finding."""
    run = run_scalarization(_landscape(), -20.0, BUDGET, config=CONFIG)
    assert run.filtered(20) == []


def test_filtered_readout_defaults_to_the_requested_count():
    run = run_scalarization(_landscape(), 5.0, BUDGET, n_return=7, config=CONFIG)
    assert len(run.filtered()) == 7
