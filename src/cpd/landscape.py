"""Oracles over the folding/binding free-energy landscape.

Sign conventions (these match the ddPCA supplementary tables and are used
everywhere in this package, so read them once):

* ``ddg_binding`` -- change in binding free energy relative to wild type, in
  kcal/mol. **Positive means weaker binding.** It is the quantity we minimize.
* ``ddg_folding`` -- change in folding free energy relative to wild type, in
  kcal/mol. **Positive means destabilizing.** The hard constraint is
  ``ddg_folding <= tau``: the protein must stay folded.

Both objectives are therefore minimized, and the wild type sits at the origin
with both values exactly zero.

Three oracles live here:

* :class:`SyntheticLandscape` -- a seeded toy landscape with folding/binding
  antagonism deliberately built in at interface positions. It exists so the
  whole pipeline can be tested end to end, with a ground truth the tests
  control, before any data is downloaded.
* :class:`ExactLandscape` -- Track 1. A lookup table over the variants ddPCA
  actually measured. Oracle error is zero; the design space is restricted to
  the measured library.
* :class:`BudgetedOracle` -- wraps either of the above, caches repeats, and
  counts how many distinct variants a search has spent.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .variants import AMINO_ACIDS, Mutation, Variant

__all__ = [
    "Measurement",
    "Landscape",
    "SyntheticLandscape",
    "ExactLandscape",
    "BudgetedOracle",
    "BudgetExhausted",
]


@dataclass(frozen=True)
class Measurement:
    """What the oracle returns for one variant.

    Attributes:
        ddg_binding: Binding free-energy change, kcal/mol. Lower is better.
        ddg_folding: Folding free-energy change, kcal/mol. Lower is more stable.
        ddg_binding_se: Standard error on ``ddg_binding``, if the oracle reports one.
        ddg_folding_se: Standard error on ``ddg_folding``, if the oracle reports one.
    """

    ddg_binding: float
    ddg_folding: float
    ddg_binding_se: float | None = None
    ddg_folding_se: float | None = None

    def is_feasible(self, tau: float) -> bool:
        """True if this design satisfies the folding constraint at tightness ``tau``."""
        return self.ddg_folding <= tau

    def violation(self, tau: float) -> float:
        """How far past ``tau`` this design sits; zero when feasible."""
        return max(0.0, self.ddg_folding - tau)

    def feasibility_probability(self, tau: float) -> float:
        """``P(ddg_folding <= tau)``, treating the reading as normal about its SE.

        :meth:`is_feasible` answers a question the data cannot actually
        settle. A design sitting 0.05 kcal/mol inside the bar, measured to a
        standard error of 0.30, is not meaningfully feasible -- it is a coin
        flip reported as a pass. The published ddPCA tables ship a standard
        error for every variant, so the honest version of the test is
        available and simply was not being used.

        Normality is the assumption the ddPCA inference itself makes about its
        free-energy estimates, so this inherits it rather than adding one.
        With no reported error the answer collapses to :meth:`is_feasible`,
        which is the right behaviour for the synthetic track and for Track 1
        read as exact.
        """
        sigma = self.ddg_folding_se or 0.0
        if sigma <= 0.0:
            return 1.0 if self.ddg_folding <= tau else 0.0
        from math import erf, sqrt

        z = (tau - self.ddg_folding) / sigma
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def is_confidently_feasible(self, tau: float, alpha: float = 0.05) -> bool:
        """True when ``P(ddg_folding <= tau) >= 1 - alpha``.

        The chance-constrained test. ``alpha`` is the risk of handing over a
        design that turns out to break the constraint; ``alpha = 0.5`` is the
        nominal test, since the median of a normal is its mean.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
        return self.feasibility_probability(tau) >= 1.0 - alpha


class Landscape(ABC):
    """An oracle mapping variants to measured (or modelled) free energies."""

    #: Designable positions, 1-indexed.
    positions: tuple[int, ...]
    #: Highest-order variant the oracle can score.
    max_order: int
    #: Wild-type residue at each designable position.
    wt_residues: Mapping[int, str]

    @abstractmethod
    def evaluate(self, variant: Variant) -> Measurement:
        """Score one variant. Callers should check :meth:`contains` first."""

    @abstractmethod
    def contains(self, variant: Variant) -> bool:
        """True if ``variant`` lies inside this oracle's valid design space."""

    @abstractmethod
    def substitutions_at(self, position: int) -> tuple[str, ...]:
        """Non-wild-type residues this oracle can score at ``position``."""

    def candidate_mutations(self, position: int) -> tuple[Mutation, ...]:
        wt = self.wt_residues[position]
        return tuple(Mutation(position, wt, aa) for aa in self.substitutions_at(position))

    def wild_type(self) -> Variant:
        return Variant()


# --------------------------------------------------------------------------
# Track 0: synthetic landscape
# --------------------------------------------------------------------------

#: Per-class first-order effect distributions for :class:`SyntheticLandscape`.
#:
#: ``rho`` is the correlation between a mutation's binding and folding effects.
#: At interface positions it is negative: substitutions that *improve* binding
#: (negative ddg_binding) tend to *destabilize* the fold (positive ddg_folding).
#: That antagonism is the entire point of the benchmark, so the toy landscape
#: reproduces it rather than leaving the two objectives independent.
_CLASS_PARAMS: dict[str, dict[str, float]] = {
    "interface": {
        "mean_binding": 0.30,
        "sd_binding": 1.10,
        "mean_folding": 0.50,
        "sd_folding": 1.00,
        "rho": -0.60,
    },
    "core": {
        "mean_binding": 0.05,
        "sd_binding": 0.25,
        "mean_folding": 1.20,
        "sd_folding": 1.20,
        "rho": 0.0,
    },
    "surface": {
        "mean_binding": 0.05,
        "sd_binding": 0.30,
        "mean_folding": 0.10,
        "sd_folding": 0.40,
        "rho": 0.0,
    },
}


class SyntheticLandscape(Landscape):
    """A reproducible toy landscape with folding/binding antagonism built in.

    Positions are split into three classes. *Interface* positions drive binding
    and trade it against stability; *core* positions dominate folding stability
    and barely touch binding; *surface* positions are close to neutral in both.
    On top of those first-order effects a sparse set of position pairs carries
    pairwise couplings, so the landscape is not separable and the search engine
    has something to do.

    The landscape is fully determined by ``seed``: two instances built with the
    same arguments score every variant identically.
    """

    def __init__(
        self,
        n_positions: int = 12,
        n_interface: int = 4,
        n_core: int = 3,
        n_coupled_pairs: int = 10,
        coupling_scale: float = 0.35,
        max_order: int = 6,
        seed: int = 0,
    ) -> None:
        if n_interface + n_core > n_positions:
            raise ValueError("n_interface + n_core cannot exceed n_positions")
        rng = np.random.default_rng(seed)

        self.positions = tuple(range(1, n_positions + 1))
        self.max_order = max_order
        self.seed = seed

        shuffled = list(self.positions)
        rng.shuffle(shuffled)
        interface = sorted(shuffled[:n_interface])
        core = sorted(shuffled[n_interface : n_interface + n_core])
        self.position_class: dict[int, str] = {
            p: "interface" if p in interface else "core" if p in core else "surface"
            for p in self.positions
        }
        self.interface_positions: tuple[int, ...] = tuple(interface)
        self.core_positions: tuple[int, ...] = tuple(core)

        self.wt_residues = {p: str(rng.choice(AMINO_ACIDS)) for p in self.positions}

        # First-order effects: self._first[position][mut_aa] -> (ddg_binding, ddg_folding)
        self._first: dict[int, dict[str, tuple[float, float]]] = {}
        for p in self.positions:
            params = _CLASS_PARAMS[self.position_class[p]]
            table: dict[str, tuple[float, float]] = {}
            for aa in AMINO_ACIDS:
                if aa == self.wt_residues[p]:
                    continue
                z_bind, z_indep = rng.standard_normal(2)
                rho = params["rho"]
                # Correlate the folding draw with the binding draw at strength rho.
                z_fold = rho * z_bind + math.sqrt(1.0 - rho * rho) * z_indep
                table[aa] = (
                    params["mean_binding"] + params["sd_binding"] * float(z_bind),
                    params["mean_folding"] + params["sd_folding"] * float(z_fold),
                )
            self._first[p] = table

        # Sparse pairwise couplings over position pairs, as full 20x20x2 tables.
        all_pairs = [(i, j) for i in self.positions for j in self.positions if i < j]
        n_coupled_pairs = min(n_coupled_pairs, len(all_pairs))
        chosen = rng.choice(len(all_pairs), size=n_coupled_pairs, replace=False)
        self._aa_index = {aa: k for k, aa in enumerate(AMINO_ACIDS)}
        self._couplings: dict[tuple[int, int], np.ndarray] = {
            all_pairs[int(k)]: coupling_scale * rng.standard_normal((20, 20, 2)) for k in chosen
        }
        self.coupled_pairs: tuple[tuple[int, int], ...] = tuple(sorted(self._couplings))

    def substitutions_at(self, position: int) -> tuple[str, ...]:
        return tuple(self._first[position])

    def contains(self, variant: Variant) -> bool:
        if variant.order > self.max_order:
            return False
        for m in variant:
            if m.position not in self._first:
                return False
            if m.wt != self.wt_residues[m.position]:
                return False
            if m.mut not in self._first[m.position]:
                return False
        return True

    def evaluate(self, variant: Variant) -> Measurement:
        if not self.contains(variant):
            raise KeyError(f"{variant} is outside this landscape's design space")
        binding = 0.0
        folding = 0.0
        for m in variant:
            d_bind, d_fold = self._first[m.position][m.mut]
            binding += d_bind
            folding += d_fold
        by_position = {m.position: m.mut for m in variant}
        for (i, j), table in self._couplings.items():
            if i in by_position and j in by_position:
                cell = table[self._aa_index[by_position[i]], self._aa_index[by_position[j]]]
                binding += float(cell[0])
                folding += float(cell[1])
        return Measurement(ddg_binding=binding, ddg_folding=folding)

    def enumerate_singles(self) -> Iterator[Variant]:
        for p in self.positions:
            for aa in self.substitutions_at(p):
                yield Variant((Mutation(p, self.wt_residues[p], aa),))


# --------------------------------------------------------------------------
# Track 1: exact oracle over the measured library
# --------------------------------------------------------------------------


class ExactLandscape(Landscape):
    """Track 1. A lookup table over variants that were actually measured.

    Oracle error is zero by construction: every score returned is an inferred
    free energy from the ddPCA thermodynamic fit, for a variant that exists in
    the assayed library. The cost is that the design space is restricted to the
    measured set, so this is budget-limited combinatorial search over a known
    landscape rather than de-novo generation.
    """

    def __init__(self, measurements: Mapping[Variant, Measurement], domain: str = "") -> None:
        if not measurements:
            raise ValueError("ExactLandscape needs at least one measured variant")
        self._table: dict[Variant, Measurement] = dict(measurements)
        self.domain = domain

        wt_residues: dict[int, str] = {}
        subs: dict[int, set[str]] = {}
        for variant in self._table:
            for m in variant:
                existing = wt_residues.setdefault(m.position, m.wt)
                if existing != m.wt:
                    raise ValueError(
                        f"position {m.position} is listed with two different wild-type "
                        f"residues ({existing!r} and {m.wt!r}); check the position numbering"
                    )
                subs.setdefault(m.position, set()).add(m.mut)

        self.wt_residues = wt_residues
        self.positions = tuple(sorted(wt_residues))
        self._subs = {p: tuple(sorted(aas)) for p, aas in subs.items()}
        self.max_order = max((v.order for v in self._table), default=0)

        # The wild type is the origin of the ddG scale and is always feasible.
        self._table.setdefault(Variant(), Measurement(0.0, 0.0, 0.0, 0.0))

    def __len__(self) -> int:
        return len(self._table)

    def substitutions_at(self, position: int) -> tuple[str, ...]:
        return self._subs.get(position, ())

    def contains(self, variant: Variant) -> bool:
        return variant in self._table

    def evaluate(self, variant: Variant) -> Measurement:
        try:
            return self._table[variant]
        except KeyError:
            raise KeyError(f"{variant} was not measured in this library") from None

    def variants(self) -> tuple[Variant, ...]:
        return tuple(self._table)

    def measurements(self) -> tuple[Measurement, ...]:
        return tuple(self._table.values())

    def variants_of_order(self, order: int) -> tuple[Variant, ...]:
        return tuple(v for v in self._table if v.order == order)

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[str, float, float]] | Iterable[Sequence],
        domain: str = "",
    ) -> "ExactLandscape":
        """Build from ``(variant_string, ddg_binding, ddg_folding)`` rows.

        Rows may carry two further entries, the standard errors on each energy.
        """
        table: dict[Variant, Measurement] = {}
        for row in records:
            text, binding, folding, *rest = row
            se_bind = float(rest[0]) if len(rest) > 0 and rest[0] is not None else None
            se_fold = float(rest[1]) if len(rest) > 1 and rest[1] is not None else None
            table[Variant.parse(str(text))] = Measurement(
                float(binding), float(folding), se_bind, se_fold
            )
        return cls(table, domain=domain)


class MeasuredAdditiveLandscape(Landscape):
    """Track 2. Combinations of measured singles, scored by summing their ddGs.

    The published ddPCA tables infer free energies for **single** mutants only,
    so an exact oracle over multi-mutants does not exist in that data. This
    oracle composes measured singles additively, which is the first-order term
    of a MoCHI-style thermodynamic model and is how free energies behave in the
    absence of epistasis.

    It is a **model**, not a measurement, and the distinction matters: the
    assumption it makes is exactly the one that fails at interacting positions,
    which for binding domains are disproportionately the interface positions
    this benchmark cares about. Standard errors are propagated in quadrature as
    independent, which understates the true uncertainty for the same reason.
    Report results from this track as oracle-dependent.
    """

    def __init__(
        self,
        singles: Mapping[Variant, Measurement],
        domain: str = "",
        max_order: int = 4,
    ) -> None:
        if max_order < 1:
            raise ValueError(f"max_order must be at least 1, got {max_order}")
        self._singles: dict[tuple[int, str], Measurement] = {}
        wt_residues: dict[int, str] = {}
        subs: dict[int, set[str]] = {}
        for variant, measurement in singles.items():
            if variant.order != 1:
                raise ValueError(
                    f"MeasuredAdditiveLandscape is built from single mutants; got {variant}"
                )
            mutation = variant.mutations[0]
            existing = wt_residues.setdefault(mutation.position, mutation.wt)
            if existing != mutation.wt:
                raise ValueError(
                    f"position {mutation.position} is listed with two different wild-type "
                    f"residues ({existing!r} and {mutation.wt!r}); check the position numbering"
                )
            self._singles[(mutation.position, mutation.mut)] = measurement
            subs.setdefault(mutation.position, set()).add(mutation.mut)

        if not self._singles:
            raise ValueError("need at least one measured single mutant")

        self.domain = domain
        self.max_order = max_order
        self.wt_residues = wt_residues
        self.positions = tuple(sorted(wt_residues))
        self._subs = {p: tuple(sorted(aas)) for p, aas in subs.items()}

    def substitutions_at(self, position: int) -> tuple[str, ...]:
        return self._subs.get(position, ())

    def contains(self, variant: Variant) -> bool:
        if variant.order > self.max_order:
            return False
        return all(
            m.wt == self.wt_residues.get(m.position) and (m.position, m.mut) in self._singles
            for m in variant
        )

    def evaluate(self, variant: Variant) -> Measurement:
        if not self.contains(variant):
            raise KeyError(f"{variant} cannot be composed from the measured singles")
        binding = folding = 0.0
        var_binding = var_folding = 0.0
        for m in variant:
            single = self._singles[(m.position, m.mut)]
            binding += single.ddg_binding
            folding += single.ddg_folding
            var_binding += (single.ddg_binding_se or 0.0) ** 2
            var_folding += (single.ddg_folding_se or 0.0) ** 2
        return Measurement(
            ddg_binding=binding,
            ddg_folding=folding,
            ddg_binding_se=math.sqrt(var_binding),
            ddg_folding_se=math.sqrt(var_folding),
        )

    def __len__(self) -> int:
        return len(self._singles)


# --------------------------------------------------------------------------
# Budget accounting
# --------------------------------------------------------------------------


class BudgetExhausted(RuntimeError):
    """Raised when a search asks for a new variant past its oracle-call budget."""


@dataclass
class BudgetedOracle:
    """Wraps a landscape, caches repeats, and counts distinct variants spent.

    The budget counts **distinct** variants. Re-querying something already seen
    is free, which matches what a budget means in practice: the expensive step
    is assaying a design you have not assayed before, and any sane search keeps
    its own records.

    ``history`` records every distinct evaluation in order, which is what the
    budget curves are plotted from.
    """

    landscape: Landscape
    budget: int | None = None
    _cache: dict[Variant, Measurement] = field(default_factory=dict, repr=False)
    history: list[tuple[Variant, Measurement]] = field(default_factory=list, repr=False)
    n_calls: int = 0

    @property
    def n_unique(self) -> int:
        """Distinct variants evaluated so far; this is what the budget limits."""
        return len(self._cache)

    @property
    def remaining(self) -> float:
        return math.inf if self.budget is None else self.budget - self.n_unique

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def contains(self, variant: Variant) -> bool:
        return self.landscape.contains(variant)

    def __call__(self, variant: Variant) -> Measurement:
        self.n_calls += 1
        cached = self._cache.get(variant)
        if cached is not None:
            return cached
        if self.exhausted:
            raise BudgetExhausted(
                f"budget of {self.budget} distinct evaluations is spent; "
                f"{variant} has not been scored"
            )
        measurement = self.landscape.evaluate(variant)
        self._cache[variant] = measurement
        self.history.append((variant, measurement))
        return measurement

    def seen(self, variant: Variant) -> bool:
        """True if ``variant`` is cached, so scoring it costs nothing."""
        return variant in self._cache

    def reset(self) -> None:
        self._cache.clear()
        self.history.clear()
        self.n_calls = 0
