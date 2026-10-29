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
