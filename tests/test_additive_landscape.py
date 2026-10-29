import math

import pytest

from cpd.landscape import MeasuredAdditiveLandscape, Measurement
from cpd.variants import Mutation, Variant

SINGLES = {
    Variant.parse("E24K"): Measurement(-1.0, 0.4, 0.10, 0.20),
    Variant.parse("E24R"): Measurement(0.2, -0.1, 0.05, 0.05),
    Variant.parse("L31P"): Measurement(-0.5, 2.0, 0.30, 0.40),
}


def _landscape(**kwargs) -> MeasuredAdditiveLandscape:
    return MeasuredAdditiveLandscape(SINGLES, domain="toy", **kwargs)


def test_it_reproduces_the_measured_singles_exactly():
    landscape = _landscape()
    m = landscape.evaluate(Variant.parse("E24K"))
    assert m.ddg_binding == pytest.approx(-1.0)
    assert m.ddg_folding == pytest.approx(0.4)


def test_energies_add_across_positions():
    landscape = _landscape()
    m = landscape.evaluate(Variant.parse("E24K,L31P"))
    assert m.ddg_binding == pytest.approx(-1.5)
    assert m.ddg_folding == pytest.approx(2.4)


def test_standard_errors_add_in_quadrature():
    landscape = _landscape()
    m = landscape.evaluate(Variant.parse("E24K,L31P"))
    assert m.ddg_binding_se == pytest.approx(math.hypot(0.10, 0.30))
    assert m.ddg_folding_se == pytest.approx(math.hypot(0.20, 0.40))


def test_the_wild_type_is_the_origin():
    m = _landscape().evaluate(Variant())
    assert (m.ddg_binding, m.ddg_folding) == (0.0, 0.0)


def test_it_composes_combinations_the_data_never_measured():
    """The point of Track 2, and the reason it is a model rather than a measurement."""
    landscape = _landscape()
    unmeasured = Variant.parse("E24K,L31P")
    assert unmeasured not in SINGLES
    assert landscape.contains(unmeasured)


def test_max_order_caps_the_design_space():
    landscape = _landscape(max_order=1)
    assert landscape.contains(Variant.parse("E24K"))
    assert not landscape.contains(Variant.parse("E24K,L31P"))
    with pytest.raises(KeyError):
        landscape.evaluate(Variant.parse("E24K,L31P"))
    with pytest.raises(ValueError):
        _landscape(max_order=0)


def test_it_refuses_substitutions_that_were_never_measured():
    landscape = _landscape()
    assert not landscape.contains(Variant.parse("E24W"))  # E24W is not in the table
    with pytest.raises(KeyError, match="cannot be composed"):
        landscape.evaluate(Variant.parse("E24W"))


def test_it_infers_its_design_space_from_the_measured_singles():
    landscape = _landscape()
    assert landscape.positions == (24, 31)
    assert landscape.wt_residues == {24: "E", 31: "L"}
    assert landscape.substitutions_at(24) == ("K", "R")
    assert len(landscape) == 3


def test_it_rejects_being_built_from_multi_mutants():
    with pytest.raises(ValueError, match="built from single mutants"):
        MeasuredAdditiveLandscape({Variant.parse("E24K,L31P"): Measurement(0.0, 0.0)})


def test_it_catches_inconsistent_position_numbering():
    bad = {
        Variant.parse("E24K"): Measurement(0.0, 0.0),
        Variant((Mutation(24, "Q", "R"),)): Measurement(0.0, 0.0),
    }
    with pytest.raises(ValueError, match="two different wild-type"):
        MeasuredAdditiveLandscape(bad)


def test_it_needs_some_data():
    with pytest.raises(ValueError):
        MeasuredAdditiveLandscape({})
