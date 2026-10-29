import numpy as np
import pytest

from cpd.landscape import (
    BudgetedOracle,
    BudgetExhausted,
    ExactLandscape,
    Measurement,
    SyntheticLandscape,
)
from cpd.variants import Mutation, Variant


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def test_measurement_feasibility_is_inclusive():
    assert Measurement(0.0, 2.0).is_feasible(2.0)
    assert not Measurement(0.0, 2.01).is_feasible(2.0)
    assert Measurement(0.0, 3.5).violation(2.0) == pytest.approx(1.5)
    assert Measurement(0.0, 1.0).violation(2.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# SyntheticLandscape
# --------------------------------------------------------------------------


def test_synthetic_landscape_is_reproducible_from_its_seed():
    a = SyntheticLandscape(seed=7)
    b = SyntheticLandscape(seed=7)
    c = SyntheticLandscape(seed=8)
    variant = next(a.enumerate_singles())

    assert a.evaluate(variant) == b.evaluate(variant)
    assert a.wt_residues == b.wt_residues
    assert a.interface_positions == b.interface_positions
    assert (a.interface_positions, a.wt_residues) != (c.interface_positions, c.wt_residues)


def test_wild_type_sits_at_the_origin():
    landscape = SyntheticLandscape(seed=0)
    wt = landscape.evaluate(Variant())
    assert wt.ddg_binding == pytest.approx(0.0)
    assert wt.ddg_folding == pytest.approx(0.0)


def test_single_mutant_energies_add_up_at_uncoupled_positions():
    landscape = SyntheticLandscape(n_coupled_pairs=0, seed=3)
    singles = list(landscape.enumerate_singles())
    a, b = singles[0], singles[-1]
    assert a.positions != b.positions

    double = Variant.of(list(a) + list(b))
    observed = landscape.evaluate(double)
    expected_binding = landscape.evaluate(a).ddg_binding + landscape.evaluate(b).ddg_binding
    expected_folding = landscape.evaluate(a).ddg_folding + landscape.evaluate(b).ddg_folding
    assert observed.ddg_binding == pytest.approx(expected_binding)
    assert observed.ddg_folding == pytest.approx(expected_folding)


def test_couplings_make_the_landscape_non_additive():
    landscape = SyntheticLandscape(n_coupled_pairs=10, coupling_scale=0.5, seed=3)
    i, j = landscape.coupled_pairs[0]
    a = Variant((landscape.candidate_mutations(i)[0],))
    b = Variant((landscape.candidate_mutations(j)[0],))
    double = Variant.of(list(a) + list(b))

    additive = landscape.evaluate(a).ddg_binding + landscape.evaluate(b).ddg_binding
    assert landscape.evaluate(double).ddg_binding != pytest.approx(additive)


def test_interface_positions_trade_binding_against_folding():
    """The antagonism is the whole reason this benchmark exists; assert it exists."""
    landscape = SyntheticLandscape(n_positions=16, n_interface=6, n_core=4, seed=11)

    def correlation(positions):
        binding, folding = [], []
        for p in positions:
            for aa in landscape.substitutions_at(p):
                m = landscape.evaluate(Variant((Mutation(p, landscape.wt_residues[p], aa),)))
                binding.append(m.ddg_binding)
                folding.append(m.ddg_folding)
        return float(np.corrcoef(binding, folding)[0, 1])

    # Negative correlation at the interface: substitutions that improve binding
    # (more negative ddg_binding) tend to destabilize the fold.
    assert correlation(landscape.interface_positions) < -0.3
    # Core positions move stability without touching binding much.
    assert abs(correlation(landscape.core_positions)) < 0.3


def test_interface_positions_hold_the_strongest_binders():
    landscape = SyntheticLandscape(n_positions=16, n_interface=6, n_core=4, seed=11)
    best = min(landscape.enumerate_singles(), key=lambda v: landscape.evaluate(v).ddg_binding)
    assert set(best.positions) <= set(landscape.interface_positions)


def test_synthetic_landscape_validates_its_position_classes():
    with pytest.raises(ValueError, match="cannot exceed n_positions"):
        SyntheticLandscape(n_positions=5, n_interface=4, n_core=3)


def test_synthetic_landscape_rejects_out_of_space_variants():
    landscape = SyntheticLandscape(n_positions=5, n_interface=2, n_core=2, max_order=2, seed=0)
    assert not landscape.contains(Variant.parse("A99G"))  # position out of range
    too_deep = Variant.of(
        [landscape.candidate_mutations(p)[0] for p in landscape.positions[:3]]
    )
    assert not landscape.contains(too_deep)
    with pytest.raises(KeyError):
        landscape.evaluate(too_deep)

    wrong_wt_residue = "A" if landscape.wt_residues[1] != "A" else "C"
    assert not landscape.contains(Variant((Mutation(1, wrong_wt_residue, "W"),)))


# --------------------------------------------------------------------------
# ExactLandscape
# --------------------------------------------------------------------------


def _toy_exact() -> ExactLandscape:
    return ExactLandscape.from_records(
        [
            ("E24K", -1.0, 0.4),
            ("E24R", 0.2, -0.1),
            ("L31P", -0.5, 2.0),
            ("E24K,L31P", -1.8, 2.6),
        ],
        domain="toy",
    )


def test_exact_landscape_infers_its_design_space_from_the_data():
    landscape = _toy_exact()
    assert landscape.positions == (24, 31)
    assert landscape.wt_residues == {24: "E", 31: "L"}
    assert landscape.substitutions_at(24) == ("K", "R")
    assert landscape.max_order == 2


def test_exact_landscape_only_scores_measured_variants():
    landscape = _toy_exact()
    assert landscape.contains(Variant.parse("E24K,L31P"))
    assert not landscape.contains(Variant.parse("E24R,L31P"))  # never assayed
    with pytest.raises(KeyError, match="not measured"):
        landscape.evaluate(Variant.parse("E24R,L31P"))

    measured = landscape.evaluate(Variant.parse("E24K,L31P"))
    assert measured.ddg_binding == pytest.approx(-1.8)
    assert measured.ddg_folding == pytest.approx(2.6)


def test_exact_landscape_always_contains_the_wild_type_at_the_origin():
    landscape = _toy_exact()
    wt = landscape.evaluate(Variant())
    assert (wt.ddg_binding, wt.ddg_folding) == (0.0, 0.0)


def test_exact_landscape_catches_inconsistent_position_numbering():
    with pytest.raises(ValueError, match="two different wild-type"):
        ExactLandscape.from_records([("E24K", 0.0, 0.0), ("Q24R", 0.0, 0.0)])


def test_exact_landscape_carries_standard_errors_when_given_them():
    landscape = ExactLandscape.from_records([("E24K", -1.0, 0.4, 0.05, 0.08)])
    m = landscape.evaluate(Variant.parse("E24K"))
    assert m.ddg_binding_se == pytest.approx(0.05)
    assert m.ddg_folding_se == pytest.approx(0.08)


# --------------------------------------------------------------------------
# BudgetedOracle
# --------------------------------------------------------------------------


def test_budget_counts_distinct_variants_and_repeats_are_free():
    landscape = _toy_exact()
    oracle = BudgetedOracle(landscape, budget=3)
    v = Variant.parse("E24K")

    oracle(v)
    oracle(v)
    oracle(v)
    assert oracle.n_unique == 1  # the budget is spent on distinct designs
    assert oracle.n_calls == 3  # but every call is still counted
    assert oracle.remaining == 2
    assert oracle.seen(v)


def test_budget_raises_once_it_is_spent():
    landscape = _toy_exact()
    oracle = BudgetedOracle(landscape, budget=2)
    oracle(Variant.parse("E24K"))
    oracle(Variant.parse("E24R"))
    assert oracle.exhausted

    # A cached variant still answers after exhaustion; a new one does not.
    assert oracle(Variant.parse("E24K")).ddg_binding == pytest.approx(-1.0)
    with pytest.raises(BudgetExhausted):
        oracle(Variant.parse("L31P"))


def test_budget_history_is_the_evaluation_order():
    landscape = _toy_exact()
    oracle = BudgetedOracle(landscape, budget=None)
    order = ["L31P", "E24K", "L31P", "E24R"]
    for text in order:
        oracle(Variant.parse(text))

    assert [str(v) for v, _ in oracle.history] == ["L31P", "E24K", "E24R"]
    assert oracle.remaining == float("inf")

    oracle.reset()
    assert oracle.n_unique == 0 and oracle.history == []
