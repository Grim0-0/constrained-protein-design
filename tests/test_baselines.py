"""Is the design problem hard enough for a method comparison to mean anything?

These guard the two checks that answer that: a search-free greedy baseline, and
the exact optimum on an additive landscape. The exact optimum is checked against
brute force, because a dynamic program that silently returns something merely
good would make every rule look closer to optimal than it is.
"""

import itertools
import math

import pytest

from cpd.baselines import (
    GreedyResult,
    exact_optimum,
    greedy_feasible,
    greedy_forward,
    pareto_prune,
)
from cpd.landscape import MeasuredAdditiveLandscape, Measurement, SyntheticLandscape
from cpd.variants import Variant

SINGLES = {
    Variant.parse("E24K"): Measurement(-1.0, 0.4),
    Variant.parse("E24R"): Measurement(-0.2, -0.3),
    Variant.parse("L31P"): Measurement(-0.5, 2.0),
    Variant.parse("L31A"): Measurement(-0.4, -0.1),
    Variant.parse("T53G"): Measurement(-0.9, 1.1),
    Variant.parse("T53S"): Measurement(-0.1, -0.5),
}


def _landscape(max_order: int = 3) -> MeasuredAdditiveLandscape:
    return MeasuredAdditiveLandscape(SINGLES, domain="toy", max_order=max_order)


def _brute_force(landscape, tau, max_order):
    """Every legal variant, scored directly. The reference the DP must match."""
    by_position: dict[int, list] = {}
    for position in landscape.positions:
        by_position[position] = list(landscape.candidate_mutations(position))

    best = float("nan")
    positions = list(by_position)
    for order in range(0, max_order + 1):
        for chosen_positions in itertools.combinations(positions, order):
            option_lists = [by_position[p] for p in chosen_positions]
            for combo in itertools.product(*option_lists):
                variant = Variant.of(combo)
                if not landscape.contains(variant):
                    continue
                m = landscape.evaluate(variant)
                if m.ddg_folding <= tau:
                    if math.isnan(best) or m.ddg_binding < best:
                        best = m.ddg_binding
    return best


# --------------------------------------------------------------------------
# exact_optimum
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [3.0, 1.5, 0.5, 0.0, -0.4, -0.8, -2.0])
@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_the_dynamic_program_matches_brute_force(tau, max_order):
    landscape = _landscape(max_order=max_order)
    got = exact_optimum(landscape, [tau], max_order=max_order)[tau]
    want = _brute_force(landscape, tau, max_order)
    if math.isnan(want):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(want)


def test_the_wild_type_is_available_whenever_tau_allows_it():
    """Doing nothing scores zero and is feasible for any non-negative tau."""
    landscape = _landscape()
    assert exact_optimum(landscape, [0.0])[0.0] <= 0.0


def test_no_feasible_design_returns_nan():
    landscape = _landscape()
    assert math.isnan(exact_optimum(landscape, [-99.0])[-99.0])


def test_tightening_tau_never_helps():
    landscape = _landscape()
    taus = [2.0, 1.0, 0.0, -0.5]
    values = exact_optimum(landscape, taus)
    ordered = [values[t] for t in taus]
    assert all(a <= b + 1e-9 for a, b in zip(ordered, ordered[1:])), ordered


def test_raising_the_order_cap_never_hurts():
    tau = 0.5
    loose = exact_optimum(_landscape(max_order=3), [tau], max_order=3)[tau]
    tight = exact_optimum(_landscape(max_order=1), [tau], max_order=1)[tau]
    assert loose <= tight + 1e-9


def test_it_refuses_a_landscape_with_epistasis():
    """The DP is exact only because the objectives are separable; say so loudly."""
    rugged = SyntheticLandscape(n_positions=6, n_interface=2, n_core=2, seed=0)
    with pytest.raises(TypeError, match="separable"):
        exact_optimum(rugged, [0.0])


def test_a_bad_order_cap_is_rejected():
    with pytest.raises(ValueError, match="max_order"):
        exact_optimum(_landscape(), [0.0], max_order=0)


# --------------------------------------------------------------------------
# pareto_prune
# --------------------------------------------------------------------------


def test_pareto_prune_drops_dominated_points():
    front = [(0.0, 0.0), (1.0, -1.0), (2.0, -0.5), (1.0, -0.5), (3.0, -2.0)]
    kept = pareto_prune(front)
    assert kept == [(0.0, 0.0), (1.0, -1.0), (3.0, -2.0)]


def test_pareto_prune_handles_an_empty_front():
    assert pareto_prune([]) == []


# --------------------------------------------------------------------------
# greedy_feasible
# --------------------------------------------------------------------------


def test_greedy_returns_a_feasible_design():
    landscape = _landscape()
    result = greedy_feasible(landscape, tau=0.5)
    assert result.measurement is not None
    assert result.ddg_folding <= 0.5
    assert result.ddg_binding < 0.0


def test_greedy_respects_the_order_cap():
    landscape = _landscape(max_order=3)
    result = greedy_feasible(landscape, tau=10.0, max_order=2)
    assert result.order <= 2


def test_greedy_uses_at_most_one_substitution_per_position():
    landscape = _landscape()
    result = greedy_feasible(landscape, tau=10.0)
    positions = [m.position for m in result.variant]
    assert len(positions) == len(set(positions))


def test_greedy_pays_for_every_single_it_ranks():
    """The cost must be reported honestly: the singles are not free."""
    landscape = _landscape()
    result = greedy_feasible(landscape, tau=0.5)
    assert result.n_evaluations >= len(SINGLES)


def test_greedy_finds_nothing_when_nothing_is_feasible():
    landscape = _landscape()
    result = greedy_feasible(landscape, tau=-99.0)
    assert result.measurement is None
    assert math.isnan(result.ddg_binding)
    assert result.order == 0


def test_greedy_can_be_ranked_by_a_custom_objective():
    landscape = _landscape()
    by_binding = greedy_feasible(landscape, tau=10.0)
    by_folding = greedy_feasible(landscape, tau=10.0, rank=lambda m: m.ddg_folding)
    assert by_binding.variant != by_folding.variant


def test_greedy_reaches_the_optimum_on_an_additive_landscape_at_loose_tau():
    """With the constraint slack, sorting by binding IS the optimal policy.

    This is the point of the baseline: on a separable landscape there is nothing
    for a search to discover, so any search that fails to match this is losing
    to a sort.
    """
    landscape = _landscape()
    tau = 10.0
    greedy = greedy_feasible(landscape, tau=tau)
    assert greedy.ddg_binding == pytest.approx(exact_optimum(landscape, [tau])[tau])


def test_greedy_rejects_a_design_whose_additive_plan_did_not_hold():
    """On a rugged landscape the plan can miss; the result must not claim success."""
    rugged = SyntheticLandscape(
        n_positions=12, n_interface=4, n_core=3, n_coupled_pairs=20,
        coupling_scale=1.5, max_order=5, seed=3,
    )
    # Sweep a range of tau; at least one plan should overshoot once couplings
    # are paid for, and every returned measurement must satisfy its own tau.
    for tau in (-1.0, -0.5, 0.0, 0.5, 1.0):
        result = greedy_feasible(rugged, tau=tau)
        assert isinstance(result, GreedyResult)
        if result.measurement is not None:
            assert result.ddg_folding <= tau


# --------------------------------------------------------------------------
# greedy_forward
# --------------------------------------------------------------------------


def test_forward_selection_returns_a_feasible_design():
    landscape = _landscape()
    result = greedy_forward(landscape, tau=0.5)
    assert result.measurement is not None
    assert result.ddg_folding <= 0.5


def test_forward_selection_is_at_least_as_good_as_the_additive_plan_when_additive():
    """On a separable landscape both greedies see the same truth."""
    landscape = _landscape()
    for tau in (2.0, 0.5, 0.0, -0.3):
        planned = greedy_feasible(landscape, tau=tau)
        forward = greedy_forward(landscape, tau=tau)
        if planned.measurement is None:
            continue
        assert forward.measurement is not None
        assert forward.ddg_binding <= planned.ddg_binding + 1e-9


def test_forward_selection_costs_more_than_ranking_once():
    """It pays for the truth at every step; the cost must show that."""
    landscape = _landscape()
    planned = greedy_feasible(landscape, tau=10.0)
    forward = greedy_forward(landscape, tau=10.0)
    assert forward.n_evaluations > planned.n_evaluations


def test_forward_selection_respects_the_order_cap():
    landscape = _landscape(max_order=3)
    assert greedy_forward(landscape, tau=10.0, max_order=1).order <= 1


def test_forward_selection_stops_when_the_budget_runs_out():
    landscape = _landscape()
    result = greedy_forward(landscape, tau=10.0, budget=3)
    assert result.n_evaluations <= 3


def test_forward_selection_finds_nothing_when_nothing_is_feasible():
    result = greedy_forward(_landscape(), tau=-99.0)
    assert result.measurement is None
    assert result.order == 0


def test_forward_selection_never_returns_an_infeasible_design_on_a_rugged_landscape():
    rugged = SyntheticLandscape(
        n_positions=10, n_interface=3, n_core=3, n_coupled_pairs=16,
        coupling_scale=1.2, max_order=4, seed=7,
    )
    for tau in (-1.0, 0.0, 1.0):
        result = greedy_forward(rugged, tau=tau, budget=900)
        if result.measurement is not None:
            assert result.ddg_folding <= tau


def test_forward_selection_only_keeps_a_step_that_improves_binding():
    landscape = _landscape()
    result = greedy_forward(landscape, tau=10.0)
    # Every prefix of the chosen set must be worse than the final design,
    # otherwise a non-improving step was accepted.
    chosen = list(result.variant)
    for k in range(1, len(chosen)):
        prefix = landscape.evaluate(Variant.of(chosen[:k]))
        assert result.ddg_binding <= prefix.ddg_binding + 1e-9
