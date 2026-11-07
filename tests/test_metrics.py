"""Metrics are the contribution, so every expected value here is hand-computed."""

import math

import numpy as np
import pytest

from cpd.landscape import Measurement
from cpd.metrics import (
    feasibility_rate,
    feasible_best,
    feasible_best_k,
    hypervolume,
    pareto_mask,
    summarize,
    violation_severity,
)

# binding: -1.0,  0.5, -3.0,  2.0
# folding:  1.0,  2.0,  3.0,  4.0
# At tau = 2.5 the first two are feasible and the best binder (-3.0) is not.
DESIGNS = [
    Measurement(-1.0, 1.0),
    Measurement(0.5, 2.0),
    Measurement(-3.0, 3.0),
    Measurement(2.0, 4.0),
]
TAU = 2.5


def test_feasibility_rate():
    assert feasibility_rate(DESIGNS, TAU) == pytest.approx(0.5)
    assert feasibility_rate(DESIGNS, 10.0) == pytest.approx(1.0)
    assert feasibility_rate(DESIGNS, -1.0) == pytest.approx(0.0)


def test_feasibility_rate_of_nothing_is_not_zero_percent():
    # "No designs" and "zero percent feasible" are different claims.
    assert math.isnan(feasibility_rate([], TAU))


def test_constraint_is_inclusive_at_tau():
    exactly_on_the_line = [Measurement(0.0, 2.5)]
    assert feasibility_rate(exactly_on_the_line, TAU) == pytest.approx(1.0)


def test_feasible_best_ignores_the_infeasible_best_binder():
    assert feasible_best(DESIGNS, TAU) == pytest.approx(-1.0)
    # Loosen tau and the -3.0 binder becomes admissible.
    assert feasible_best(DESIGNS, 3.0) == pytest.approx(-3.0)


def test_feasible_best_is_nan_when_nothing_is_feasible():
    assert math.isnan(feasible_best(DESIGNS, -5.0))


def test_feasible_best_k_refuses_to_average_over_too_few_designs():
    # Two feasible designs at tau = 2.5: mean of the best 2 is (-1.0 + 0.5) / 2.
    assert feasible_best_k(DESIGNS, TAU, k=2) == pytest.approx(-0.25)
    # Asking for 3 when only 2 are feasible must not quietly average over 2.
    assert math.isnan(feasible_best_k(DESIGNS, TAU, k=3))
    with pytest.raises(ValueError):
        feasible_best_k(DESIGNS, TAU, k=0)


def test_violation_severity():
    # Excesses over tau = 2.5 are 0, 0, 0.5, 1.5.
    stats = violation_severity(DESIGNS, TAU)
    assert stats.n_violated == 2
    assert stats.mean_violation == pytest.approx(1.0)  # over violators only
    assert stats.max_violation == pytest.approx(1.5)
    assert stats.mean_violation_all == pytest.approx(0.5)  # over all four


def test_violation_severity_when_everything_is_feasible():
    stats = violation_severity(DESIGNS, 10.0)
    assert stats.n_violated == 0
    assert math.isnan(stats.mean_violation)
    assert stats.max_violation == pytest.approx(0.0)


def test_pareto_mask_minimizes_every_column():
    points = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0], [2.0, 3.0]])
    # (3, 4) is dominated by (2, 3). Duplicates of a front point are both kept.
    assert pareto_mask(points).tolist() == [True, True, False, True, True]


def test_hypervolume_of_a_single_point_is_a_rectangle():
    assert hypervolume(np.array([[2.0, 3.0]]), (10.0, 10.0)) == pytest.approx(56.0)


def test_hypervolume_of_two_points_is_their_union():
    # [2,10]x[8,10] = 16, [5,10]x[3,10] = 35, overlap [5,10]x[8,10] = 10.
    points = np.array([[2.0, 8.0], [5.0, 3.0]])
    assert hypervolume(points, (10.0, 10.0)) == pytest.approx(41.0)


def test_hypervolume_ignores_dominated_points():
    front = np.array([[2.0, 8.0], [5.0, 3.0]])
    with_dominated = np.array([[2.0, 8.0], [5.0, 3.0], [6.0, 9.0]])
    assert hypervolume(with_dominated, (10.0, 10.0)) == pytest.approx(
        hypervolume(front, (10.0, 10.0))
    )


def test_hypervolume_ignores_points_worse_than_the_reference():
    points = np.array([[2.0, 3.0], [20.0, 20.0]])
    assert hypervolume(points, (10.0, 10.0)) == pytest.approx(56.0)
    assert hypervolume(np.empty((0, 2)), (10.0, 10.0)) == pytest.approx(0.0)


def test_hypervolume_over_all_designs_exceeds_the_feasible_only_hypervolume():
    # This gap is the argument of the project: the infeasible design (-3, 3)
    # buys hypervolume that no one could use.
    reference = (5.0, 5.0)
    report = summarize(DESIGNS, TAU, reference, k=2)
    assert report.hypervolume_all > report.hypervolume_feasible


def test_summarize_reports_the_unconstrained_number_alongside_the_honest_one():
    report = summarize(DESIGNS, TAU, (5.0, 5.0), k=2)
    assert report.n_designs == 4
    assert report.n_feasible == 2
    assert report.feasibility_rate == pytest.approx(0.5)
    assert report.best_binding_any == pytest.approx(-3.0)  # what you would quote
    assert report.feasible_best == pytest.approx(-1.0)  # what you actually have
    assert report.tau == pytest.approx(TAU)
    assert report.reference == (5.0, 5.0)


def test_summarize_of_an_empty_design_set_does_not_crash():
    report = summarize([], TAU, (5.0, 5.0), k=2)
    assert report.n_designs == 0
    assert report.n_feasible == 0
    assert math.isnan(report.feasible_best)
    assert report.hypervolume_all == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Metrics under a target window
# --------------------------------------------------------------------------


def test_best_feasible_cost_reduces_to_feasible_best_when_unbounded():
    from cpd.metrics import best_feasible_cost, feasible_best
    from cpd.objectives import Minimize

    designs = [Measurement(-3.0, 0.1), Measurement(-1.5, 0.2), Measurement(-9.0, 5.0)]
    assert best_feasible_cost(designs, 1.0, Minimize()) == feasible_best(designs, 1.0)


def test_best_feasible_cost_is_zero_when_something_lands_in_the_window():
    from cpd.metrics import best_feasible_cost
    from cpd.objectives import Window

    designs = [Measurement(-2.4, 0.1), Measurement(-1.0, 0.2)]
    assert best_feasible_cost(designs, 1.0, Window(-2.5, -2.0)) == pytest.approx(0.0)


def test_best_feasible_cost_ignores_an_infeasible_design_that_hits_the_window():
    """Hitting the target while breaking the constraint is not a hit."""
    from cpd.metrics import best_feasible_cost
    from cpd.objectives import Window

    designs = [Measurement(-2.4, 9.0), Measurement(-1.0, 0.2)]
    got = best_feasible_cost(designs, 1.0, Window(-2.5, -2.0))
    assert got == pytest.approx(1.0)


def test_best_feasible_cost_is_nan_when_nothing_is_feasible():
    import math

    from cpd.metrics import best_feasible_cost
    from cpd.objectives import Window

    assert math.isnan(best_feasible_cost([Measurement(-2.4, 9.0)], 1.0, Window(-2.5, -2.0)))


def test_the_hit_rate_counts_only_designs_that_are_feasible_and_on_target():
    from cpd.metrics import target_hit_rate
    from cpd.objectives import Window

    designs = [
        Measurement(-2.4, 0.1),   # feasible and inside  -> counts
        Measurement(-2.4, 9.0),   # inside but infeasible
        Measurement(-9.0, 0.1),   # feasible but overshoots
        Measurement(-1.0, 0.1),   # feasible but undershoots
    ]
    assert target_hit_rate(designs, 1.0, Window(-2.5, -2.0)) == pytest.approx(0.25)


def test_the_hit_rate_is_the_feasibility_rate_when_unbounded():
    from cpd.metrics import feasibility_rate, target_hit_rate
    from cpd.objectives import Minimize

    designs = [Measurement(-3.0, 0.1), Measurement(-1.5, 9.0), Measurement(-2.0, 0.3)]
    assert target_hit_rate(designs, 1.0, Minimize()) == feasibility_rate(designs, 1.0)


def test_the_hit_rate_of_an_empty_set_is_nan():
    import math

    from cpd.metrics import target_hit_rate
    from cpd.objectives import Window

    assert math.isnan(target_hit_rate([], 1.0, Window(-2.5, -2.0)))


# --------------------------------------------------------------------------
# Feasibility under measurement uncertainty
# --------------------------------------------------------------------------


def test_a_design_far_inside_the_bar_is_confidently_feasible():
    m = Measurement(-2.0, -1.0, 0.1, 0.05)
    assert m.feasibility_probability(0.0) == pytest.approx(1.0, abs=1e-6)
    assert m.is_confidently_feasible(0.0, alpha=0.01)


def test_a_design_just_inside_the_bar_is_a_coin_flip():
    """The case the nominal test reports as a clean pass."""
    m = Measurement(-2.0, -0.05, 0.1, 0.30)
    assert m.is_feasible(0.0)
    assert m.feasibility_probability(0.0) == pytest.approx(0.566, abs=0.005)
    assert not m.is_confidently_feasible(0.0, alpha=0.05)


def test_a_design_exactly_on_the_bar_is_half_feasible():
    m = Measurement(-2.0, 0.0, 0.1, 0.2)
    assert m.feasibility_probability(0.0) == pytest.approx(0.5)


def test_with_no_reported_error_the_test_collapses_to_the_nominal_one():
    for folding, want in ((-0.5, 1.0), (0.5, 0.0)):
        m = Measurement(-2.0, folding, None, None)
        assert m.feasibility_probability(0.0) == want


def test_a_bad_alpha_is_rejected():
    m = Measurement(-2.0, -0.5, 0.1, 0.1)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            m.is_confidently_feasible(0.0, alpha=bad)


def test_expected_rate_sits_below_the_nominal_rate_near_the_boundary():
    from cpd.metrics import expected_feasibility_rate, feasibility_rate

    designs = [Measurement(-2.0, -0.05, 0.1, 0.3) for _ in range(4)]
    assert feasibility_rate(designs, 0.0) == pytest.approx(1.0)
    assert expected_feasibility_rate(designs, 0.0) < 0.7


def test_expected_rate_equals_the_nominal_rate_when_errors_are_negligible():
    from cpd.metrics import expected_feasibility_rate, feasibility_rate

    designs = [Measurement(-2.0, -1.0, 0.01, 0.01), Measurement(-1.0, 1.0, 0.01, 0.01)]
    assert expected_feasibility_rate(designs, 0.0) == pytest.approx(
        feasibility_rate(designs, 0.0), abs=1e-6
    )


def test_alpha_one_half_reproduces_the_nominal_feasibility_rate():
    """The nominal test is the chance constraint at alpha = 0.5."""
    from cpd.metrics import confident_feasibility_rate, feasibility_rate

    designs = [
        Measurement(-2.0, -0.05, 0.1, 0.30),
        Measurement(-1.0, 0.40, 0.1, 0.30),
        Measurement(-3.0, -2.00, 0.1, 0.30),
    ]
    assert confident_feasibility_rate(designs, 0.0, alpha=0.5) == pytest.approx(
        feasibility_rate(designs, 0.0)
    )


def test_tightening_alpha_never_raises_the_confident_rate():
    from cpd.metrics import confident_feasibility_rate

    designs = [Measurement(-2.0, -0.1 * i, 0.1, 0.25) for i in range(8)]
    rates = [confident_feasibility_rate(designs, 0.0, a) for a in (0.5, 0.2, 0.1, 0.05, 0.01)]
    assert all(a >= b for a, b in zip(rates, rates[1:])), rates


def test_the_uncertainty_metrics_are_nan_for_an_empty_set():
    import math

    from cpd.metrics import confident_feasibility_rate, expected_feasibility_rate

    assert math.isnan(expected_feasibility_rate([], 0.0))
    assert math.isnan(confident_feasibility_rate([], 0.0))


# --------------------------------------------------------------------------
# Diversity of the delivered set
# --------------------------------------------------------------------------


def test_a_set_of_identical_designs_has_no_diversity():
    from cpd.metrics import diversity
    from cpd.variants import Variant

    report = diversity([Variant.parse("E24K,L31P")] * 4)
    assert report.n_designs == 4
    assert report.n_unique == 1
    assert report.mean_pairwise_distance == pytest.approx(0.0)
    assert report.min_pairwise_distance == pytest.approx(0.0)
    assert report.n_positions_used == 2


def test_a_spread_out_set_scores_higher_than_a_concentrated_one():
    from cpd.metrics import diversity
    from cpd.variants import Variant

    concentrated = [Variant.parse(t) for t in ("E24K,L31P", "E24R,L31P", "E24K,L31A")]
    spread = [Variant.parse(t) for t in ("E24K,L31P", "T53G", "E24R")]
    assert (
        diversity(spread).mean_pairwise_distance
        > diversity(concentrated).mean_pairwise_distance
    )
    assert diversity(spread).n_positions_used >= diversity(concentrated).n_positions_used


def test_duplicates_are_counted_but_not_hidden():
    """A method returning the same design twice has returned one design."""
    from cpd.metrics import diversity
    from cpd.variants import Variant

    report = diversity([Variant.parse("E24K"), Variant.parse("E24K"), Variant.parse("T53G")])
    assert report.n_designs == 3
    assert report.n_unique == 2
    assert report.min_pairwise_distance == pytest.approx(0.0)


def test_diversity_of_a_single_design_is_undefined_not_zero():
    import math

    from cpd.metrics import diversity
    from cpd.variants import Variant

    report = diversity([Variant.parse("E24K")])
    assert math.isnan(report.mean_pairwise_distance)
    assert report.n_positions_used == 1


def test_diversity_of_an_empty_set():
    import math

    from cpd.metrics import diversity

    report = diversity([])
    assert report.n_designs == 0
    assert math.isnan(report.mean_pairwise_distance)
