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
