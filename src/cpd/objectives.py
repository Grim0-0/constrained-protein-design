"""The four constraint-handling rules.

Every rule is a scalar function of a :class:`~cpd.landscape.Measurement` that
the shared search engine minimizes. Swapping the rule is the *only* difference
between the methods compared in this benchmark -- the search engine, the
budget, the seeds and the initialization are all held fixed. That is what makes
constraint handling the single experimental variable.

A rule may be stateful. :meth:`Objective.update` is called once per generation
with that generation's measurements, which is how the Lagrangian multiplier
adapts. Scores are never cached by the engine, so a rule whose state changes
re-ranks the whole population on the next comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .landscape import Measurement

__all__ = [
    "Objective",
    "Scalarized",
    "FixedPenalty",
    "Lagrangian",
    "BindingOnly",
]


class Objective:
    """Base class: a scalar score to minimize, optionally with per-generation state."""

    #: Human-readable identifier used in result tables and figure legends.
    name: str = "objective"

    def score(self, measurement: Measurement) -> float:
        raise NotImplementedError

    def update(self, batch: Sequence[Measurement]) -> None:
        """Hook called once per generation. Stateless rules ignore it."""
        return None

    def state(self) -> dict[str, float]:
        """Current internal state, logged once per generation."""
        return {}

    def __call__(self, measurement: Measurement) -> float:
        return self.score(measurement)


@dataclass
class Scalarized(Objective):
    """Rule 1 -- weighted sum. The honest baseline: what most people actually do.

    ``score = w * ddg_binding + (1 - w) * ddg_folding``

    This rule never sees ``tau``. That is the point of including it: a
    practitioner picks a weight, runs the search, and finds out afterwards
    whether the designs clear the bar. Sweeping ``w`` traces an approximate
    Pareto front, but no single weight is the constraint.
    """

    weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must lie in [0, 1], got {self.weight}")
        self.name = f"scalarized_w{self.weight:g}"

    def score(self, measurement: Measurement) -> float:
        return self.weight * measurement.ddg_binding + (1.0 - self.weight) * measurement.ddg_folding

    def state(self) -> dict[str, float]:
        return {"weight": self.weight}


@dataclass
class FixedPenalty(Objective):
    """Rule 2 -- fixed penalty. The simplest thing that is aware of ``tau``.

    ``score = ddg_binding + rho * max(0, ddg_folding - tau)``

    Only violations are charged, so a design already past the bar gets no
    further reward for extra stability. The weakness is ``rho``: too small and
    the search buys binding with violations, too large and it never approaches
    the boundary where the good binders live.
    """

    tau: float
    rho: float = 10.0

    def __post_init__(self) -> None:
        if self.rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {self.rho}")
        self.name = f"penalty_rho{self.rho:g}"

    def score(self, measurement: Measurement) -> float:
        return measurement.ddg_binding + self.rho * measurement.violation(self.tau)

    def state(self) -> dict[str, float]:
        return {"rho": self.rho}


@dataclass
class Lagrangian(Objective):
    """Rule 3 -- the proposed method. Adaptive multiplier, batch-level update.

    ``score = ddg_binding + lam * (ddg_folding - tau)``

    with dual ascent on the batch mean constraint violation after each
    generation::

        lam <- clip(lam + lr * mean(ddg_folding - tau), 0, lam_max)

    The multiplier rises while the population sits on the wrong side of
    ``tau`` and relaxes once it is comfortably inside, so the search settles
    near the constraint boundary instead of needing a penalty weight chosen in
    advance. This is Ctrl-DNA's batch-level adaptive multiplier (arXiv
    2505.20578), which was introduced for regulatory DNA, applied here to
    protein sequence design.

    Note the term is signed, not hinged: a population deep inside the feasible
    region drives ``lam`` back down. That is what standard dual ascent on an
    inequality constraint does, and it is the difference from :class:`FixedPenalty`.
    """

    tau: float
    lr: float = 0.5
    lam: float = 1.0
    lam_max: float = 100.0
    name: str = "lagrangian"
    history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.lam < 0.0:
            raise ValueError(f"initial lam must be non-negative, got {self.lam}")
        if self.lam_max < self.lam:
            raise ValueError("lam_max must be at least the initial lam")
        self.history = [self.lam]

    def score(self, measurement: Measurement) -> float:
        return measurement.ddg_binding + self.lam * (measurement.ddg_folding - self.tau)

    def update(self, batch: Sequence[Measurement]) -> None:
        if not batch:
            return
        mean_violation = sum(m.ddg_folding - self.tau for m in batch) / len(batch)
        self.lam = min(self.lam_max, max(0.0, self.lam + self.lr * mean_violation))
        self.history.append(self.lam)

    def state(self) -> dict[str, float]:
        return {"lam": self.lam}


@dataclass
class BindingOnly(Objective):
    """Rule 4 -- the null hypothesis. Maximize binding and ignore the constraint.

    ``score = ddg_binding``

    On its own this is unconstrained optimization. It becomes *optimize-then-
    reject* through the selection policy applied afterwards, which discards
    infeasible designs from the returned set; see :mod:`cpd.methods`.
    """

    name: str = "binding_only"

    def score(self, measurement: Measurement) -> float:
        return measurement.ddg_binding
