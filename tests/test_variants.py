import pytest

from cpd.variants import Mutation, Variant


def test_mutation_roundtrips_through_its_string_form():
    m = Mutation.parse("E24K")
    assert (m.position, m.wt, m.mut) == (24, "E", "K")
    assert str(m) == "E24K"


def test_mutation_rejects_nonsense():
    with pytest.raises(ValueError):
        Mutation(0, "E", "K")  # positions are 1-indexed
    with pytest.raises(ValueError):
        Mutation(24, "E", "E")  # not a substitution
    with pytest.raises(ValueError):
        Mutation(24, "E", "X")  # not a canonical residue
    with pytest.raises(ValueError):
        Mutation.parse("EK")


def test_variant_order_does_not_affect_identity():
    a = Variant.parse("E24K,L31P")
    b = Variant.parse("L31P,E24K")
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
    assert str(a) == "E24K,L31P"  # canonical, sorted by position


def test_wild_type_is_the_empty_variant():
    wt = Variant.parse("WT")
    assert wt == Variant()
    assert wt.order == 0
    assert str(wt) == "WT"


def test_two_mutations_at_one_position_is_an_error():
    with pytest.raises(ValueError):
        Variant.parse("E24K,E24R")


def test_with_mutation_replaces_at_the_same_position():
    v = Variant.parse("E24K,L31P")
    updated = v.with_mutation(Mutation(24, "E", "R"))
    assert str(updated) == "E24R,L31P"
    assert updated.order == 2


def test_without_position_reverts_to_wild_type():
    v = Variant.parse("E24K,L31P")
    assert str(v.without_position(24)) == "L31P"
    assert v.without_position(99) == v  # no-op at an unmutated position


def test_apply_to_sequence():
    #            123456
    sequence = "AEGLKP"
    v = Variant.of([Mutation(2, "E", "K"), Mutation(4, "L", "W")])
    assert v.apply_to(sequence) == "AKGWKP"


def test_apply_to_catches_a_numbering_offset():
    sequence = "AEGLKP"
    bad = Variant.of([Mutation(3, "E", "K")])  # position 3 is G, not E
    with pytest.raises(ValueError, match="disagrees with the wild-type sequence"):
        bad.apply_to(sequence)

    off_the_end = Variant.of([Mutation(99, "E", "K")])
    with pytest.raises(ValueError, match="outside a sequence"):
        off_the_end.apply_to(sequence)
