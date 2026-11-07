"""Mutation and Variant types.

A :class:`Variant` is an immutable, hashable set of point substitutions relative
to a wild-type sequence. Two variants listing the same substitutions in a
different order are equal and hash identically, so variants can be used as
dictionary keys in oracle caches and measured-data lookups.

Positions are 1-indexed, matching the convention used in the ddPCA
supplementary tables and in the PDB entries for these domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

#: The 20 canonical amino acids, in alphabetical order by one-letter code.
AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True, order=True)
class Mutation:
    """A single point substitution, e.g. ``E24K``.

    Attributes:
        position: 1-indexed residue position in the domain.
        wt: One-letter code of the wild-type residue at ``position``.
        mut: One-letter code of the substituted residue.
    """

    position: int
    wt: str
    mut: str

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError(f"position must be 1-indexed and positive, got {self.position}")
        for label, aa in (("wt", self.wt), ("mut", self.mut)):
            if aa not in AMINO_ACIDS:
                raise ValueError(f"{label} residue {aa!r} is not a canonical amino acid")
        if self.wt == self.mut:
            raise ValueError(f"synonymous 'mutation' at position {self.position}: {self.wt}->{self.mut}")

    def __str__(self) -> str:
        return f"{self.wt}{self.position}{self.mut}"

    @classmethod
    def parse(cls, token: str) -> "Mutation":
        """Parse a mutation written as ``E24K``."""
        token = token.strip()
        if len(token) < 3:
            raise ValueError(f"cannot parse mutation {token!r}")
        wt, mut, digits = token[0], token[-1], token[1:-1]
        if not digits.isdigit():
            raise ValueError(f"cannot parse mutation {token!r}: {digits!r} is not a position")
        return cls(position=int(digits), wt=wt, mut=mut)


@dataclass(frozen=True)
class Variant:
    """An immutable set of substitutions relative to the wild type.

    ``mutations`` is stored sorted by position so that equality and hashing are
    order-independent. The wild type itself is the variant with no mutations.
    """

    mutations: tuple[Mutation, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.mutations))
        positions = [m.position for m in ordered]
        if len(set(positions)) != len(positions):
            raise ValueError(f"two mutations at the same position in {positions}")
        object.__setattr__(self, "mutations", ordered)

    @property
    def order(self) -> int:
        """Number of substitutions: 0 for wild type, 1 for a single, 2 for a double."""
        return len(self.mutations)

    @property
    def positions(self) -> frozenset[int]:
        return frozenset(m.position for m in self.mutations)

    def __iter__(self) -> Iterator[Mutation]:
        return iter(self.mutations)

    def __len__(self) -> int:
        return len(self.mutations)

    def __str__(self) -> str:
        return ",".join(str(m) for m in self.mutations) if self.mutations else "WT"

    @classmethod
    def parse(cls, text: str) -> "Variant":
        """Parse a comma-separated variant string such as ``"E24K,L31P"`` or ``"WT"``."""
        text = text.strip()
        if not text or text.upper() == "WT":
            return cls()
        return cls(tuple(Mutation.parse(tok) for tok in text.split(",")))

    @classmethod
    def of(cls, mutations: Iterable[Mutation]) -> "Variant":
        return cls(tuple(mutations))

    def with_mutation(self, mutation: Mutation) -> "Variant":
        """Return a copy with ``mutation`` added, replacing any mutation at the same position."""
        kept = tuple(m for m in self.mutations if m.position != mutation.position)
        return Variant(kept + (mutation,))

    def without_position(self, position: int) -> "Variant":
        """Return a copy with the substitution at ``position`` reverted to wild type."""
        return Variant(tuple(m for m in self.mutations if m.position != position))

    def distance(self, other: "Variant") -> int:
        """Hamming distance: positions where the two differ in residue.

        A position counts when the two carry different substitutions there,
        and when one carries a substitution the other leaves wild type. Two
        designs hitting the same position with the same residue contribute
        nothing, which is what makes this a measure of how much of the design
        space a returned set actually covers.
        """
        mine = {m.position: m.mut for m in self.mutations}
        theirs = {m.position: m.mut for m in other.mutations}
        return sum(
            1 for p in mine.keys() | theirs.keys() if mine.get(p) != theirs.get(p)
        )

    def apply_to(self, wt_sequence: str) -> str:
        """Apply the substitutions to ``wt_sequence`` (a 1-indexed residue string).

        Raises:
            ValueError: if a mutation's wild-type residue disagrees with the sequence,
                which is the usual symptom of a position-numbering offset.
        """
        residues = list(wt_sequence)
        for m in self.mutations:
            index = m.position - 1
            if not 0 <= index < len(residues):
                raise ValueError(f"position {m.position} is outside a sequence of length {len(wt_sequence)}")
            if residues[index] != m.wt:
                raise ValueError(
                    f"{m} disagrees with the wild-type sequence, which has "
                    f"{residues[index]!r} at position {m.position}"
                )
            residues[index] = m.mut
        return "".join(residues)
