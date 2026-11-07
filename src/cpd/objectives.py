"""The five constraint-handling rules.

Every rule is a scalar function of a :class:`~cpd.landscape.Measurement` that
the shared search engine minimizes. Swapping the rule is the *only* difference
between the methods compared in this benchmark -- the search engine, the
budget, the seeds and the initialization are all held fixed. That is what makes
constraint handling the single experimental variable.

A rule may be stateful. :meth:`Objective.update` is called once per generation
with that generation's measurements, which is how the Lagrangian multiplier
adapts. Scores are never cached by the engine, so a rule whose state changes
re-ranks the whole population on the next comparison.

Every rule scores two things: what you *want* (the target, on binding) and
what you must *not break* (the constraint, on folding). Only the first varies
between the two regimes this benchmark compares:

``Minimize``
    Drive binding as low as it will go. This is how the benchmark started and
    how most multi-objective design work frames affinity.
``Window``
    Land binding inside ``[lo, hi]``. Overshooting costs as much as
    undershooting, which is closer to what affinity engineering actually
    targets -- binding too tight carries its own liabilities.

The distinction matters for more than realism. Under ``Minimize`` on an
additive oracle the problem is a knapsack and a sort nearly solves it; see
:mod:`cpd.baselines`. Under ``Window`` the same oracle poses a subset-sum
problem, which greedy handles badly. Which of those a benchmark is really
measuring is the thing this repo is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .landscape import Measurement

__all__ = [
    "Target",
    "Minimize",
    "Window",
    "Objective",
    "Scalarized",
    "FixedPenalty",
    "Lagrangian",
    "BindingOnly",
    "NSGA2",
    "non_dominated_fronts",
    "crowding_distances",
]


class Target:
    """What the search is trying to achieve on ``ddg_binding``.

    Separated from the constraint handling so the two can be varied
    independently: every rule below works under either target, which is what
    makes "unbounded versus window" a clean experimental comparison rather
    than a different benchmark.
    """

    #: Identifier used in result tables and filenames.
    name: str = "target"

    def cost(self, measurement: Measurement) -> float:
        """Non-negative-or-signed scalar to minimize. Lower is better."""
        raise NotImplementedError

    def achieved(self, measurement: Measurement) -> bool:
        """True when this design has hit the target, where that is meaningful."""
        raise NotImplementedError


@dataclass
class Minimize(Target):
    """Push binding as low as possible. The original, unbounded objective.

    ``achieved`` is always True: there is no bar to clear, only a direction to
    go, which is exactly the property that makes the problem easy on a
    separable oracle.
    """

    name: str = "minimize"

    def cost(self, measurement: Measurement) -> float:
        return measurement.ddg_binding

    def achieved(self, measurement: Measurement) -> bool:
        return True


@dataclass
class Window(Target):
    """Land binding inside ``[lo, hi]``; overshooting costs as much as missing.

    ``cost`` is the two-sided hinge -- the distance from the interval, and zero
    inside it::

        cost = max(lo - b, 0) + max(b - hi, 0)

    Because the cost is flat inside the window, every design that lands in it
    ties. That is deliberate: the question a window asks is *did you hit it*,
    not *how far past the near edge did you go*. Ranking within the window
    would quietly restore the unbounded objective.
    """

    lo: float
    hi: float
    name: str = "window"

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"window lo must not exceed hi, got [{self.lo}, {self.hi}]")
        self.name = f"window[{self.lo:g},{self.hi:g}]"

    def cost(self, measurement: Measurement) -> float:
        return self.distance(measurement.ddg_binding)

    def distance(self, binding: float) -> float:
        """Distance from ``binding`` to the window; 0 inside it."""
        return max(self.lo - binding, 0.0) + max(binding - self.hi, 0.0)

    def achieved(self, measurement: Measurement) -> bool:
        return self.lo <= measurement.ddg_binding <= self.hi


class Objective:
    """Base class: a scalar score to minimize, optionally with per-generation state."""

    #: Human-readable identifier used in result tables and figure legends.
    name: str = "objective"

    def score(self, measurement: Measurement) -> float:
        raise NotImplementedError

    def update(self, batch: Sequence[Measurement]) -> None:
        """Hook called once per generation. Stateless rules ignore it."""
        return None

    def population_order(self, measurements: Sequence[Measurement]) -> list[int] | None:
        """Rank a whole population best-first, for rules that need to.

        Returning ``None`` -- the default -- means this rule scores designs
        independently and the engine should use :meth:`score`, exactly as it
        always has. Only a rule whose verdict on one design depends on the
        others, such as :class:`NSGA2`, overrides this.

        The opt-in matters: it keeps the engine's existing path byte-identical
        for every rule that does not need the hook, so adding a set-level rule
        cannot perturb results already committed.
        """
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
    target: Target = field(default_factory=Minimize)

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must lie in [0, 1], got {self.weight}")
        self.name = f"scalarized_w{self.weight:g}"

    def score(self, measurement: Measurement) -> float:
        return (
            self.weight * self.target.cost(measurement)
            + (1.0 - self.weight) * measurement.ddg_folding
        )

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
    target: Target = field(default_factory=Minimize)

    def __post_init__(self) -> None:
        if self.rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {self.rho}")
        self.name = f"penalty_rho{self.rho:g}"

    def score(self, measurement: Measurement) -> float:
        return self.target.cost(measurement) + self.rho * measurement.violation(self.tau)

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
    target: Target = field(default_factory=Minimize)
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
        return self.target.cost(measurement) + self.lam * (measurement.ddg_folding - self.tau)

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
    target: Target = field(default_factory=Minimize)

    def score(self, measurement: Measurement) -> float:
        return self.target.cost(measurement)


# --------------------------------------------------------------------------
# Rule 5 -- NSGA-II
# --------------------------------------------------------------------------


def non_dominated_fronts(points: Sequence[tuple[float, float]]) -> list[list[int]]:
    """Partition indices into Pareto fronts, best first. Both objectives minimized.

    The textbook fast non-dominated sort. Front 0 is non-dominated by anything,
    front 1 is non-dominated once front 0 is removed, and so on.
    """
    n = len(points)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            a, b = points[i], points[j]
            i_dominates = a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])
            j_dominates = b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1])
            if i_dominates:
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif j_dominates:
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        nxt: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
        current += 1
    return [front for front in fronts if front]


def crowding_distances(
    points: Sequence[tuple[float, float]], front: Sequence[int]
) -> dict[int, float]:
    """Crowding distance within one front; boundary solutions get infinity.

    This is NSGA-II's diversity term: it prefers solutions in sparse regions of
    the front, so the population spreads along the trade-off instead of
    bunching. Note what it optimizes for -- coverage of the front, including
    the stretch of it that violates the constraint.
    """
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for axis in (0, 1):
        ordered = sorted(front, key=lambda i: points[i][axis])
        low, high = points[ordered[0]][axis], points[ordered[-1]][axis]
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        span = high - low
        if span <= 0.0:
            continue
        for k in range(1, len(ordered) - 1):
            i = ordered[k]
            if distance[i] == float("inf"):
                continue
            gap = points[ordered[k + 1]][axis] - points[ordered[k - 1]][axis]
            distance[i] += gap / span
    return distance


@dataclass
class NSGA2(Objective):
    """Rule 5 -- non-dominated sorting. What the literature this repo argues with uses.

    Folding is treated as a second **objective**, not a constraint, so like
    :class:`Scalarized` this rule never sees ``tau``. That is the whole point
    of including it. The README's argument is about work that frames design as
    a trade-off and reports hypervolume over the resulting front; NSGA-II is
    how that framing is usually implemented, and it was missing from the
    comparison, which meant the claim and the evidence covered different
    methods.

    It cannot score a design on its own: whether a variant survives depends on
    what else is in the population. So it ranks the population instead, via
    :meth:`population_order` -- non-dominated front first, crowding distance
    within a front. :meth:`score` exists only as a fallback for callers that
    need some scalar, and returns the dominance rank from the most recent
    ordering.

    Under a target window the first objective becomes the window cost, so the
    rule transfers to either regime like the others.
    """

    target: Target = field(default_factory=Minimize)
    name: str = "nsga2"
    _rank: dict[int, int] = field(default_factory=dict, repr=False)

    def _points(self, measurements: Sequence[Measurement]) -> list[tuple[float, float]]:
        return [(self.target.cost(m), m.ddg_folding) for m in measurements]

    def population_order(self, measurements: Sequence[Measurement]) -> list[int]:
        """Indices best-first: by front, then by descending crowding distance."""
        if not measurements:
            return []
        points = self._points(measurements)
        order: list[int] = []
        for front in non_dominated_fronts(points):
            distance = crowding_distances(points, front)
            order.extend(sorted(front, key=lambda i: -distance[i]))
        self._rank = {index: position for position, index in enumerate(order)}
        return order

    def score(self, measurement: Measurement) -> float:
        """Fallback scalar. NSGA-II has no meaningful per-design score.

        Returns the first objective, which makes a lone design comparable to
        another lone design but is *not* what drives selection. Anything that
        needs the real ordering must call :meth:`population_order`.
        """
        return self.target.cost(measurement)

    def state(self) -> dict[str, float]:
        return {"front_size": float(sum(1 for r in self._rank.values() if r == 0))}
