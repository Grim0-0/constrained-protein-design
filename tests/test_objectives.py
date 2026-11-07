import pytest

from cpd.landscape import Measurement
from cpd.objectives import BindingOnly, FixedPenalty, Lagrangian, Scalarized


def test_scalarized_is_a_weighted_sum_and_never_sees_tau():
    m = Measurement(ddg_binding=-2.0, ddg_folding=4.0)
    assert Scalarized(0.5).score(m) == pytest.approx(0.5 * -2.0 + 0.5 * 4.0)
    assert Scalarized(1.0).score(m) == pytest.approx(-2.0)
    assert Scalarized(0.0).score(m) == pytest.approx(4.0)
    assert Scalarized(0.25).name == "scalarized_w0.25"
    with pytest.raises(ValueError):
        Scalarized(1.5)


def test_scalarized_is_stateless():
    objective = Scalarized(0.5)
    before = objective.score(Measurement(-2.0, 4.0))
    objective.update([Measurement(0.0, 100.0)] * 10)
    assert objective.score(Measurement(-2.0, 4.0)) == pytest.approx(before)


def test_fixed_penalty_charges_only_for_violations():
    objective = FixedPenalty(tau=1.0, rho=10.0)
    # Feasible: no charge, and no reward for extra stability either.
    assert objective.score(Measurement(-2.0, 1.0)) == pytest.approx(-2.0)
    assert objective.score(Measurement(-2.0, -5.0)) == pytest.approx(-2.0)
    # Violating by 0.5 costs 10 * 0.5.
    assert objective.score(Measurement(-2.0, 1.5)) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        FixedPenalty(tau=1.0, rho=-1.0)


def test_lagrangian_is_signed_so_slack_is_rewarded():
    objective = Lagrangian(tau=1.0, lam=2.0)
    # Unlike the hinged penalty, sitting below tau earns a bonus.
    assert objective.score(Measurement(-2.0, 1.0)) == pytest.approx(-2.0)
    assert objective.score(Measurement(-2.0, 0.0)) == pytest.approx(-2.0 + 2.0 * -1.0)
    assert objective.score(Measurement(-2.0, 2.0)) == pytest.approx(-2.0 + 2.0 * 1.0)


def test_lagrangian_multiplier_climbs_while_the_batch_violates():
    objective = Lagrangian(tau=1.0, lr=0.5, lam=1.0)
    # Mean folding 3.0 against tau 1.0 is a mean violation of 2.0.
    objective.update([Measurement(0.0, 2.0), Measurement(0.0, 4.0)])
    assert objective.lam == pytest.approx(1.0 + 0.5 * 2.0)


def test_lagrangian_multiplier_relaxes_once_the_batch_is_comfortably_feasible():
    objective = Lagrangian(tau=1.0, lr=0.5, lam=2.0)
    objective.update([Measurement(0.0, 0.0), Measurement(0.0, 0.0)])  # mean violation -1.0
    assert objective.lam == pytest.approx(1.5)


def test_lagrangian_multiplier_is_clipped_to_its_range():
    objective = Lagrangian(tau=1.0, lr=1.0, lam=0.5, lam_max=3.0)
    objective.update([Measurement(0.0, -100.0)])  # would drive lam far negative
    assert objective.lam == pytest.approx(0.0)

    objective.update([Measurement(0.0, 1000.0)])  # would overshoot lam_max
    assert objective.lam == pytest.approx(3.0)


def test_lagrangian_ignores_an_empty_batch():
    objective = Lagrangian(tau=1.0, lam=1.0)
    objective.update([])
    assert objective.lam == pytest.approx(1.0)
    assert objective.history == [1.0]


def test_lagrangian_validates_its_hyperparameters():
    with pytest.raises(ValueError):
        Lagrangian(tau=1.0, lr=0.0)
    with pytest.raises(ValueError):
        Lagrangian(tau=1.0, lam=-1.0)
    with pytest.raises(ValueError):
        Lagrangian(tau=1.0, lam=5.0, lam_max=1.0)


def test_binding_only_is_the_binding_energy():
    objective = BindingOnly()
    assert objective.score(Measurement(-2.0, 99.0)) == pytest.approx(-2.0)
    objective.update([Measurement(0.0, 99.0)])
    assert objective.score(Measurement(-2.0, 99.0)) == pytest.approx(-2.0)


# --------------------------------------------------------------------------
# Targets: unbounded versus a window
# --------------------------------------------------------------------------


def test_minimize_is_the_binding_value_and_is_always_achieved():
    from cpd.objectives import Minimize

    target = Minimize()
    assert target.cost(Measurement(-2.5, 0.1)) == pytest.approx(-2.5)
    assert target.achieved(Measurement(5.0, 0.1)) is True


def test_a_window_costs_nothing_inside_and_the_distance_outside():
    from cpd.objectives import Window

    window = Window(-3.0, -2.0)
    assert window.distance(-2.5) == pytest.approx(0.0)
    assert window.distance(-3.0) == pytest.approx(0.0)
    assert window.distance(-2.0) == pytest.approx(0.0)
    assert window.distance(-4.0) == pytest.approx(1.0)  # overshoot
    assert window.distance(-1.0) == pytest.approx(1.0)  # undershoot


def test_overshooting_a_window_costs_as_much_as_undershooting():
    """The property that makes this a different problem, not a harder one."""
    from cpd.objectives import Window

    window = Window(-3.0, -2.0)
    assert window.distance(-3.7) == pytest.approx(window.distance(-1.3))


def test_a_window_is_flat_inside_so_every_hit_ties():
    """Ranking within the window would quietly restore the unbounded objective."""
    from cpd.objectives import Window

    window = Window(-3.0, -2.0)
    costs = [window.cost(Measurement(b, 0.0)) for b in (-2.9, -2.5, -2.1)]
    assert costs == [0.0, 0.0, 0.0]


def test_a_window_knows_whether_it_was_hit():
    from cpd.objectives import Window

    window = Window(-3.0, -2.0)
    assert window.achieved(Measurement(-2.5, 0.1))
    assert not window.achieved(Measurement(-4.0, 0.1))
    assert not window.achieved(Measurement(-1.0, 0.1))


def test_an_inverted_window_is_rejected():
    from cpd.objectives import Window

    with pytest.raises(ValueError, match="lo must not exceed hi"):
        Window(-1.0, -2.0)


@pytest.mark.parametrize("rule_name", ["scalarized", "penalty", "lagrangian", "binding_only"])
def test_every_rule_scores_through_its_target(rule_name):
    """Swapping the target must change every rule, or the regime is not controlled."""
    from cpd.objectives import (
        BindingOnly,
        FixedPenalty,
        Lagrangian,
        Minimize,
        Scalarized,
        Window,
    )

    window = Window(-1.1, -0.9)
    builders = {
        "scalarized": (lambda t: Scalarized(weight=1.0, target=t)),
        "penalty": (lambda t: FixedPenalty(tau=1.0, rho=10.0, target=t)),
        "lagrangian": (lambda t: Lagrangian(tau=1.0, target=t)),
        "binding_only": (lambda t: BindingOnly(target=t)),
    }
    build = builders[rule_name]
    inside = Measurement(-1.0, 0.0)
    overshoot = Measurement(-3.0, 0.0)

    unbounded = build(Minimize())
    windowed = build(window)

    # Unbounded prefers the stronger binder; the window prefers the one inside.
    assert unbounded.score(overshoot) < unbounded.score(inside)
    assert windowed.score(inside) < windowed.score(overshoot)


def test_the_default_target_leaves_existing_behaviour_unchanged():
    from cpd.objectives import Lagrangian

    rule = Lagrangian(tau=0.5)
    m = Measurement(-2.0, 1.0)
    assert rule.score(m) == pytest.approx(-2.0 + rule.lam * (1.0 - 0.5))


# --------------------------------------------------------------------------
# NSGA-II
# --------------------------------------------------------------------------


def test_non_dominated_fronts_finds_the_pareto_front_first():
    from cpd.objectives import non_dominated_fronts

    # (0,0), (1,-1), (2,-2) trade off against each other; (3,3) is dominated by all.
    points = [(0.0, 0.0), (1.0, -1.0), (2.0, -2.0), (3.0, 3.0), (1.5, -0.5)]
    fronts = non_dominated_fronts(points)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[-1] == [3]


def test_a_dominated_point_lands_in_a_later_front():
    from cpd.objectives import non_dominated_fronts

    fronts = non_dominated_fronts([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    assert fronts == [[0], [1], [2]]


def test_identical_points_share_a_front():
    """Duplicates do not dominate each other, so they cannot be separated."""
    from cpd.objectives import non_dominated_fronts

    fronts = non_dominated_fronts([(1.0, 1.0)] * 3)
    assert sorted(fronts[0]) == [0, 1, 2]


def test_non_dominated_fronts_of_an_empty_population():
    from cpd.objectives import non_dominated_fronts

    assert non_dominated_fronts([]) == []


def test_crowding_puts_the_boundary_solutions_first():
    from cpd.objectives import crowding_distances

    points = [(0.0, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]
    distance = crowding_distances(points, [0, 1, 2, 3])
    assert distance[0] == float("inf")
    assert distance[3] == float("inf")
    assert distance[1] < float("inf")


def test_crowding_prefers_the_sparse_region():
    from cpd.objectives import crowding_distances

    # index 2 sits in a wide gap; index 1 is crowded against index 0.
    points = [(0.0, 10.0), (0.1, 9.9), (5.0, 5.0), (10.0, 0.0)]
    distance = crowding_distances(points, [0, 1, 2, 3])
    assert distance[2] > distance[1]


def test_nsga2_orders_the_front_before_anything_dominated():
    from cpd.landscape import Measurement
    from cpd.objectives import NSGA2

    designs = [
        Measurement(0.0, 0.0),
        Measurement(1.0, -1.0),
        Measurement(5.0, 5.0),   # dominated by both of the above
    ]
    order = NSGA2().population_order(designs)
    assert order[-1] == 2


def test_nsga2_never_sees_tau():
    """It treats folding as an objective, which is the framing being tested."""
    import inspect

    from cpd.objectives import NSGA2

    assert "tau" not in inspect.signature(NSGA2).parameters


def test_nsga2_respects_the_target_so_it_works_in_either_regime():
    from cpd.landscape import Measurement
    from cpd.objectives import NSGA2, Window

    inside = Measurement(-1.0, 0.0)
    overshoot = Measurement(-9.0, 0.0)
    windowed = NSGA2(target=Window(-1.1, -0.9))
    order = windowed.population_order([overshoot, inside])
    assert order[0] == 1, "the design inside the window should rank first"


def test_only_nsga2_opts_into_population_ranking():
    """Every other rule must keep the engine's original per-design path."""
    from cpd.landscape import Measurement
    from cpd.objectives import BindingOnly, FixedPenalty, Lagrangian, NSGA2, Scalarized

    batch = [Measurement(-1.0, 0.0), Measurement(-2.0, 1.0)]
    for rule in (Scalarized(), FixedPenalty(tau=0.5), Lagrangian(tau=0.5), BindingOnly()):
        assert rule.population_order(batch) is None, type(rule).__name__
    assert NSGA2().population_order(batch) is not None
