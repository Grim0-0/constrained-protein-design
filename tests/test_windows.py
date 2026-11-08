"""Does the window-selection rule do what the README says it does?

The windowed comparison is only as good as where the window is put, and the
first version of this experiment put it badly: it centred the target at twice
the strongest single-mutant effect and never consulted tau, so on a domain
where the constraint bites early the target sat past everything legal and the
only remedy left was to widen it until something overlapped. That produced one
window 147x wider than another and made three domains three different
experiments.

These tests pin the two properties that justify the corrected rule, because
they are what makes the comparison a comparison:

* **nothing reachable in one substitution lands in the window**, or the
  problem is not combinatorial and a sort solves it;
* **something legal does reach the window**, or every rule fails identically
  and the result says nothing.

Both are properties of the placement alone, checkable without running a
search, which is the point -- they are decided before any rule is scored.
"""

import pytest

from cpd.landscape import MeasuredAdditiveLandscape, Measurement, SyntheticLandscape
from cpd.variants import Variant

from experiments.windows import (
    WINDOW_RULES,
    attainable_window,
    choose_window,
    constrained_span,
)

#: A toy library where combining substitutions beats the best single one, and
#: where the strongest binder is also the one that breaks folding -- so the
#: constraint has to be consulted for the window to land anywhere legal.
SINGLES = {
    Variant.parse("E24K"): Measurement(-1.6, 2.4, 0.05, 0.05),
    Variant.parse("E24R"): Measurement(-0.5, -0.3, 0.05, 0.05),
    Variant.parse("L31P"): Measurement(-0.6, 2.0, 0.05, 0.05),
    Variant.parse("L31A"): Measurement(-0.4, -0.2, 0.05, 0.05),
    Variant.parse("T53G"): Measurement(-0.9, 1.1, 0.05, 0.05),
    Variant.parse("T53S"): Measurement(-0.3, -0.4, 0.05, 0.05),
}

TAU = 0.0


@pytest.fixture
def landscape() -> MeasuredAdditiveLandscape:
    return MeasuredAdditiveLandscape(SINGLES, domain="toy", max_order=3)


def _singles(landscape):
    return [
        landscape.evaluate(Variant((mutation,)))
        for position in landscape.positions
        for mutation in landscape.candidate_mutations(position)
    ]


# --------------------------------------------------------------------------
# The span the corrected rule places the window inside
# --------------------------------------------------------------------------


def test_the_span_runs_from_the_best_feasible_single_to_the_constrained_optimum(landscape):
    span = constrained_span(landscape, TAU)
    feasible = [m for m in _singles(landscape) if m.ddg_folding <= TAU]
    assert span["best_feasible_single"] == pytest.approx(
        min(m.ddg_binding for m in feasible)
    )
    assert span["constrained_optimum"] < span["best_feasible_single"]
    assert span["reach_margin"] == pytest.approx(
        span["best_feasible_single"] - span["constrained_optimum"]
    )


def test_the_span_ignores_singles_that_break_the_constraint(landscape):
    """E24K binds hardest and is infeasible, so it must not set the near end."""
    strongest = min(m.ddg_binding for m in _singles(landscape))
    span = constrained_span(landscape, TAU)
    assert strongest < span["best_feasible_single"], "the toy library lost its point"
    assert span["best_feasible_single"] > strongest


def test_a_tighter_constraint_shrinks_the_span(landscape):
    loose = constrained_span(landscape, 1.5)
    tight = constrained_span(landscape, TAU)
    assert tight["constrained_optimum"] > loose["constrained_optimum"]


# --------------------------------------------------------------------------
# The two properties a window has to have
# --------------------------------------------------------------------------


def test_no_single_substitution_can_land_in_the_window(landscape):
    """Otherwise the problem is not combinatorial and ranking singles solves it."""
    window, _ = choose_window(landscape, TAU)
    for measurement in _singles(landscape):
        if measurement.ddg_folding > TAU:
            continue
        assert not window.achieved(measurement), (
            "a feasible single substitution lands inside the window"
        )


def test_the_window_sits_inside_what_the_constraint_allows(landscape):
    window, meta = choose_window(landscape, TAU)
    assert window.lo > meta["constrained_optimum"], (
        "the window extends past the best legal design, so nothing can reach it"
    )
    assert window.hi < meta["best_feasible_single"]


def test_the_centre_is_the_midpoint_of_the_span(landscape):
    _, meta = choose_window(landscape, TAU)
    assert meta["centre"] == pytest.approx(
        0.5 * (meta["best_feasible_single"] + meta["constrained_optimum"])
    )


def test_the_half_width_is_twice_the_median_binding_error(landscape):
    _, meta = choose_window(landscape, TAU)
    assert meta["half_width"] == pytest.approx(2.0 * 0.05)


def test_widening_doubles_the_half_width_and_keeps_the_centre(landscape):
    narrow, first = choose_window(landscape, TAU, widenings=0)
    wide, second = choose_window(landscape, TAU, widenings=2)
    assert second["half_width"] == pytest.approx(4.0 * first["half_width"])
    assert second["centre"] == pytest.approx(first["centre"])
    assert wide.lo < narrow.lo and wide.hi > narrow.hi


# --------------------------------------------------------------------------
# The corrected rule against the one it replaced
# --------------------------------------------------------------------------


def test_the_corrected_rule_responds_to_the_constraint(landscape):
    """The whole defect in the superseded rule was that this did not happen."""
    loose, _ = choose_window(landscape, 1.5)
    tight, _ = choose_window(landscape, TAU)
    assert (loose.lo, loose.hi) != (tight.lo, tight.hi)


def test_the_superseded_rule_ignores_the_constraint(landscape):
    """Kept runnable so the correction can be checked, and documented as wrong."""
    loose, _ = choose_window(landscape, 1.5, rule="naive")
    tight, _ = choose_window(landscape, TAU, rule="naive")
    assert (loose.lo, loose.hi) == (tight.lo, tight.hi)


def test_the_superseded_rule_can_place_the_window_past_anything_legal(landscape):
    """This is how it ended up widening until the target was 147x wider."""
    window, _ = choose_window(landscape, TAU, rule="naive")
    span = constrained_span(landscape, TAU)
    assert window.hi < span["constrained_optimum"], (
        "the toy library no longer reproduces the defect the fix addresses"
    )


def test_an_unknown_rule_is_refused(landscape):
    with pytest.raises(ValueError, match="unknown window rule"):
        choose_window(landscape, TAU, rule="whatever")


def test_every_declared_rule_is_accepted(landscape):
    for rule in WINDOW_RULES:
        window, meta = choose_window(landscape, TAU, rule=rule)
        assert window.lo < window.hi
        assert meta["rule"] == rule


# --------------------------------------------------------------------------
# Reachability, which is checked before any search runs
# --------------------------------------------------------------------------


def test_the_corrected_rule_needs_no_widening_here(landscape):
    _, meta = attainable_window(landscape, TAU)
    assert meta["widenings"] == 0
    assert meta["exact_distance"] == pytest.approx(0.0, abs=1e-9)


def test_the_reachability_check_is_sharper_than_the_window_is_wide(landscape):
    """Binned DP error is bounded by max_order * bin_width / 2; keep it inside."""
    _, meta = attainable_window(landscape, TAU)
    error_bound = landscape.max_order * meta["bin_width"] / 2.0
    assert error_bound < meta["half_width"]


def test_a_landscape_with_epistasis_is_placed_from_a_first_order_model():
    synthetic = SyntheticLandscape(seed=0)
    tau = 0.5
    _, meta = attainable_window(synthetic, tau)
    assert meta["check_is_first_order"] is True
    assert meta["span_is_first_order"] is True


def test_a_constraint_nothing_survives_is_reported_not_worked_around(landscape):
    """No feasible single means no honest window, so say so rather than guess."""
    with pytest.raises(SystemExit, match="no single substitution is feasible"):
        constrained_span(landscape, -99.0)
