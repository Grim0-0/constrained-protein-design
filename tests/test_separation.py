"""Telling a real difference from seed noise.

These guard the claim-making machinery: a benchmark that ranks methods without
checking separation will happily invent a winner out of noise.
"""

import math

import pytest

from cpd.metrics import compare_groups, rank_with_ties


def test_clearly_different_groups_are_separated():
    better = [-3.0, -3.1, -2.9, -3.05, -2.95]
    worse = [-1.0, -1.1, -0.9, -1.05, -0.95]
    result = compare_groups(better, worse)

    assert result.separated
    assert result.p_value < 0.01
    assert result.gap == pytest.approx(2.0, abs=0.05)  # worse minus better
    assert result.mean_a < result.mean_b
    assert (result.n_a, result.n_b) == (5, 5)


def test_overlapping_groups_are_not_separated():
    """The real case from this benchmark: a visible gap that is still noise."""
    a = [-3.21, -2.60, -3.90, -2.95, -3.40]
    b = [-2.95, -2.40, -3.60, -2.70, -3.10]
    result = compare_groups(a, b)

    assert not result.separated
    assert result.p_value > 0.05
    assert result.gap > 0  # a looks better on the mean...
    # ...but the spread across seeds swallows it, so no claim may be made.


def test_not_separated_is_not_the_same_as_equal():
    identical = [1.0, 1.0, 1.0, 1.0]
    result = compare_groups(identical, [1.0, 1.0, 1.0, 1.0])
    assert not result.separated
    assert result.gap == pytest.approx(0.0)


def test_nans_are_dropped_and_counted():
    """A NaN means a seed returned nothing feasible; the n must reveal that."""
    partial = [-3.0, float("nan"), -3.1, float("nan"), -2.9]
    full = [-1.0, -1.1, -0.9, -1.05, -0.95]
    result = compare_groups(partial, full)

    assert result.n_a == 3  # two seeds failed outright
    assert result.n_b == 5
    assert result.separated


def test_too_few_usable_seeds_refuses_to_decide():
    result = compare_groups([-3.0], [-1.0, -1.1, -0.9])
    assert not result.separated
    assert math.isnan(result.p_value)
    assert result.n_a == 1

    empty = compare_groups([float("nan")] * 4, [-1.0, -1.1, -0.9])
    assert not empty.separated
    assert empty.n_a == 0
    assert math.isnan(empty.mean_a)


def test_alpha_is_respected():
    a = [-3.0, -2.8, -3.2, -2.9, -3.1]
    b = [-2.6, -2.4, -2.8, -2.5, -2.7]
    assert compare_groups(a, b, alpha=0.05).separated
    assert not compare_groups(a, b, alpha=1e-9).separated


def test_rank_with_ties_orders_best_first_and_flags_the_tie():
    scores = {
        "lagrangian": [-1.82, -1.90, -1.70, -1.85, -1.80],
        "fixed_penalty": [-1.77, -1.85, -1.65, -1.80, -1.75],
        "sweep": [-1.23, -1.30, -1.15, -1.25, -1.20],
    }
    ranked = rank_with_ties(scores)
    names = [name for name, _, _ in ranked]
    assert names[0] == "lagrangian"  # lowest mean wins

    flags = dict((name, sep) for name, _, sep in ranked)
    assert flags["lagrangian"] is False  # never separated from itself
    assert flags["fixed_penalty"] is False  # ties the leader; must not be called beaten
    assert flags["sweep"] is True  # genuinely worse


def test_rank_with_ties_handles_an_empty_input():
    assert rank_with_ties({}) == []


def test_zero_variance_groups_are_decided_on_their_means():
    """A rule that fails identically on every seed still has to be comparable."""
    constant_bad = [0.0, 0.0, 0.0, 0.0]
    constant_good = [-2.0, -2.0, -2.0, -2.0]
    result = compare_groups(constant_good, constant_bad)
    assert result.separated
    assert result.p_value == pytest.approx(0.0)
    assert result.gap == pytest.approx(2.0)
