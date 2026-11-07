"""Metrics for constrained design.

The argument this benchmark makes lives in these definitions, so they are
written to be unflattering where the truth is unflattering.

Two choices worth stating up front:

1. :func:`feasible_best_k` returns NaN when a method returns fewer than ``k``
   feasible designs. The alternative -- averaging over however many feasible
   designs happen to exist -- lets a method that returned one feasible design
   out of a hundred score the same as one that returned ten. That is precisely
   the failure this project is about, so it is not averaged away. ``n_feasible``
   is reported alongside every such number.

2. Hypervolume is computed twice, over all returned designs and over the
   feasible subset only. The gap between the two is the argument: hypervolume
   over all designs pays a method for points it bought in infeasible territory.

All objectives are minimized; see :mod:`cpd.landscape` for sign conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from .landscape import Measurement

__all__ = [
    "as_array",
    "feasibility_rate",
    "expected_feasibility_rate",
    "confident_feasibility_rate",
    "feasible_best",
    "feasible_best_k",
    "best_feasible_cost",
    "target_hit_rate",
    "ViolationStats",
    "violation_severity",
    "pareto_mask",
    "hypervolume",
    "DiversityReport",
    "diversity",
    "DesignSetReport",
    "summarize",
    "SeparationResult",
    "compare_groups",
    "compare_to_constant",
    "rank_with_ties",
]


def as_array(measurements: Iterable[Measurement]) -> np.ndarray:
    """Stack measurements into an ``(n, 2)`` array of ``[ddg_binding, ddg_folding]``."""
    rows = [(m.ddg_binding, m.ddg_folding) for m in measurements]
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.asarray(rows, dtype=float)


def _feasible_binding(measurements: Sequence[Measurement], tau: float) -> np.ndarray:
    points = as_array(measurements)
    if points.size == 0:
        return np.empty(0, dtype=float)
    return points[points[:, 1] <= tau, 0]


def feasibility_rate(measurements: Sequence[Measurement], tau: float) -> float:
    """Fraction of returned designs satisfying ``ddg_folding <= tau``.

    Returns NaN for an empty design set, since "no designs" is not the same
    claim as "zero percent feasible".
    """
    points = as_array(measurements)
    if points.size == 0:
        return float("nan")
    return float(np.mean(points[:, 1] <= tau))


def expected_feasibility_rate(measurements: Sequence[Measurement], tau: float) -> float:
    """Expected fraction of returned designs that really satisfy the constraint.

    The mean of ``P(ddg_folding <= tau)`` over the returned set, rather than
    the fraction whose point estimate clears the bar. Those differ whenever
    designs sit close to the boundary relative to their error -- which is
    exactly where a constraint-aware search puts them, so the gap is largest
    for the rules that work hardest.

    :func:`feasibility_rate` is what a benchmark reports. This is what the
    experiment would return. Reporting both, and the gap, is the point.
    """
    if not measurements:
        return float("nan")
    return float(np.mean([m.feasibility_probability(tau) for m in measurements]))


def confident_feasibility_rate(
    measurements: Sequence[Measurement], tau: float, alpha: float = 0.05
) -> float:
    """Fraction of returned designs passing ``P(ddg_folding <= tau) >= 1 - alpha``.

    The chance-constrained reading. ``alpha`` is the risk tolerated per design;
    at ``alpha = 0.5`` this equals :func:`feasibility_rate`, because the median
    of a normal is its mean, so sweeping ``alpha`` from 0.5 downward traces how
    fast a reported feasibility rate erodes once the error bars are honoured.
    """
    if not measurements:
        return float("nan")
    return float(np.mean([m.is_confidently_feasible(tau, alpha) for m in measurements]))


def feasible_best(measurements: Sequence[Measurement], tau: float) -> float:
    """Best (lowest) ``ddg_binding`` among feasible designs; NaN if none are feasible.

    This is the y-axis of the headline figure.
    """
    feasible = _feasible_binding(measurements, tau)
    if feasible.size == 0:
        return float("nan")
    return float(np.min(feasible))


def feasible_best_k(measurements: Sequence[Measurement], tau: float, k: int = 10) -> float:
    """Mean ``ddg_binding`` of the ``k`` best feasible designs.

    Returns NaN when fewer than ``k`` designs are feasible, rather than
    averaging over a smaller set -- see the module docstring.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    feasible = _feasible_binding(measurements, tau)
    if feasible.size < k:
        return float("nan")
    return float(np.mean(np.sort(feasible)[:k]))


def best_feasible_cost(
    measurements: Sequence[Measurement], tau: float, target
) -> float:
    """Lowest target cost among feasible designs; NaN if none are feasible.

    The generalization of :func:`feasible_best` to either regime. Under
    :class:`~cpd.objectives.Minimize` the target cost *is* ``ddg_binding``, so
    this returns exactly what ``feasible_best`` does. Under a window it returns
    the smallest distance from the window, and zero means something landed
    inside it.
    """
    feasible = [m for m in measurements if m.ddg_folding <= tau]
    if not feasible:
        return float("nan")
    return float(min(target.cost(m) for m in feasible))


def target_hit_rate(
    measurements: Sequence[Measurement], tau: float, target
) -> float:
    """Fraction of returned designs that are feasible *and* on target.

    Under an unbounded objective every feasible design counts, so this reduces
    to the feasibility rate. Under a window it is the metric that matters: a
    design that clears the stability bar but binds ten times tighter than
    asked for has not hit the target, and nothing in the unbounded metrics
    would say so.

    NaN for an empty design set, matching :func:`feasibility_rate` -- "no
    designs" is not the claim "zero percent on target".
    """
    if not measurements:
        return float("nan")
    hits = sum(
        1 for m in measurements if m.ddg_folding <= tau and target.achieved(m)
    )
    return hits / len(measurements)


@dataclass(frozen=True)
class ViolationStats:
    """How badly, and how often, a design set breaks the constraint."""

    n_violated: int
    mean_violation: float
    """Mean excess over ``tau``, averaged over violating designs only. NaN if none violate."""
    max_violation: float
    """Worst excess over ``tau``. 0.0 when nothing violates."""
    mean_violation_all: float
    """Mean excess over ``tau`` across every design, counting feasible ones as zero."""


def violation_severity(measurements: Sequence[Measurement], tau: float) -> ViolationStats:
    """Summarize how far past ``tau`` the infeasible designs sit."""
    points = as_array(measurements)
    if points.size == 0:
        return ViolationStats(0, float("nan"), 0.0, float("nan"))
    excess = np.maximum(0.0, points[:, 1] - tau)
    violated = excess[excess > 0.0]
    return ViolationStats(
        n_violated=int(violated.size),
        mean_violation=float(np.mean(violated)) if violated.size else float("nan"),
        max_violation=float(np.max(excess)),
        mean_violation_all=float(np.mean(excess)),
    )


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, for minimization in every column.

    A point is dominated if another point is no worse in every objective and
    strictly better in at least one. Duplicate points are all kept.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"expected a 2-D array of points, got shape {points.shape}")
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        no_worse = np.all(points <= points[i], axis=1)
        strictly_better = np.any(points < points[i], axis=1)
        if np.any(no_worse & strictly_better):
            mask[i] = False
    return mask


def hypervolume(points: np.ndarray, reference: Sequence[float]) -> float:
    """Two-objective hypervolume for minimization, measured against ``reference``.

    The reference point must be worse than the designs you care about in both
    objectives; points not strictly better than it in both contribute nothing.
    Returns 0.0 for an empty set.
    """
    points = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (2,):
        raise ValueError(f"reference must have 2 entries, got {ref.shape}")
    if points.size == 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) array of points, got shape {points.shape}")

    inside = points[np.all(points < ref, axis=1)]
    if inside.size == 0:
        return 0.0
    front = inside[pareto_mask(inside)]
    # Sort by the first objective ascending; on the front the second then descends.
    front = front[np.lexsort((front[:, 1], front[:, 0]))]

    total = 0.0
    prev_y = ref[1]
    for x, y in front:
        if y >= prev_y:  # a duplicate x with a worse y adds no area
            continue
        total += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(total)


@dataclass(frozen=True)
class DiversityReport:
    """How much of the design space a returned set actually covers.

    A method returns twenty designs, but twenty variations on one design is
    not twenty designs. The experimental budget a set is sized for assumes
    they probe different things; a set that collapses onto one region spends
    that budget on a single bet. Nothing else in this benchmark would notice.
    """

    n_designs: int
    n_unique: int
    """Distinct variants. Fewer than ``n_designs`` means duplicates were returned."""
    mean_pairwise_distance: float
    """Mean Hamming distance over all distinct pairs. NaN for fewer than two."""
    min_pairwise_distance: float
    """Closest pair. 0 means two returned designs are identical."""
    n_positions_used: int
    """Distinct positions touched anywhere in the set."""
    mean_order: float
    """Mean number of substitutions per design, for context on the distances."""

    def as_dict(self) -> dict:
        return asdict(self)


def diversity(variants: Sequence) -> DiversityReport:
    """Summarize how different a returned set of designs really is."""
    unique = list(dict.fromkeys(variants))
    n = len(variants)
    if n == 0:
        return DiversityReport(0, 0, float("nan"), float("nan"), 0, float("nan"))
    positions = {p for v in variants for p in v.positions}
    mean_order = float(np.mean([v.order for v in variants]))
    if n < 2:
        return DiversityReport(
            n, len(unique), float("nan"), float("nan"), len(positions), mean_order
        )
    pairs = [
        variants[i].distance(variants[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return DiversityReport(
        n_designs=n,
        n_unique=len(unique),
        mean_pairwise_distance=float(np.mean(pairs)),
        min_pairwise_distance=float(np.min(pairs)),
        n_positions_used=len(positions),
        mean_order=mean_order,
    )


@dataclass(frozen=True)
class DesignSetReport:
    """Everything worth reporting about one method's returned design set."""

    n_designs: int
    tau: float
    n_feasible: int
    feasibility_rate: float
    feasible_best: float
    feasible_best_k: float
    k: int
    n_violated: int
    mean_violation: float
    max_violation: float
    mean_violation_all: float
    hypervolume_all: float
    hypervolume_feasible: float
    reference: tuple[float, float]
    best_binding_any: float
    """Best binding ignoring the constraint -- the number an unconstrained method would quote."""

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(
    measurements: Sequence[Measurement],
    tau: float,
    reference: Sequence[float],
    k: int = 10,
) -> DesignSetReport:
    """Compute the full metric set for one method's returned designs.

    ``reference`` is the hypervolume reference point ``(ddg_binding, ddg_folding)``
    and must be held fixed across every method in a comparison for the
    hypervolume numbers to mean anything.
    """
    points = as_array(measurements)
    violations = violation_severity(measurements, tau)
    feasible_points = points[points[:, 1] <= tau] if points.size else points
    return DesignSetReport(
        n_designs=int(points.shape[0]),
        tau=float(tau),
        n_feasible=int(feasible_points.shape[0]),
        feasibility_rate=feasibility_rate(measurements, tau),
        feasible_best=feasible_best(measurements, tau),
        feasible_best_k=feasible_best_k(measurements, tau, k=k),
        k=k,
        n_violated=violations.n_violated,
        mean_violation=violations.mean_violation,
        max_violation=violations.max_violation,
        mean_violation_all=violations.mean_violation_all,
        hypervolume_all=hypervolume(points, reference),
        hypervolume_feasible=hypervolume(feasible_points, reference),
        reference=(float(reference[0]), float(reference[1])),
        best_binding_any=float(np.min(points[:, 0])) if points.size else float("nan"),
    )


# --------------------------------------------------------------------------
# Is a difference between two methods real, or seed noise?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeparationResult:
    """Whether two methods' scores are distinguishable across replicate seeds.

    Evolutionary search is stochastic, so two rules can differ by a visible
    margin on a plot and still be the same rule as far as the evidence goes.
    Ranking methods without checking this is how a benchmark invents findings.
    """

    mean_a: float
    mean_b: float
    gap: float
    """``mean_b - mean_a``. Positive means ``a`` scored lower, which for a
    minimized objective means ``a`` is better."""
    p_value: float
    n_a: int
    n_b: int
    separated: bool
    """True when the difference clears ``alpha``. False means *not shown to
    differ*, which is not the same as shown to be equal."""
    alpha: float


def compare_groups(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05
) -> SeparationResult:
    """Welch's t-test between two methods' per-seed scores.

    Welch rather than Student because the rules differ in variance as well as
    mean -- an unstable rule is a worse rule, and pooling the variance would
    hide that. NaNs are dropped, which matters here: a NaN means a seed
    returned no feasible design at all, so a method that failed outright on
    some seeds is compared on the seeds where it produced a number. Report
    ``n_a``/``n_b`` alongside, or that exclusion flatters the failing method.
    """
    from scipy import stats

    x = np.asarray([v for v in a if not np.isnan(v)], dtype=float)
    y = np.asarray([v for v in b if not np.isnan(v)], dtype=float)
    if x.size < 2 or y.size < 2:
        return SeparationResult(
            float(np.mean(x)) if x.size else float("nan"),
            float(np.mean(y)) if y.size else float("nan"),
            float("nan"), float("nan"), int(x.size), int(y.size), False, alpha,
        )
    # Zero variance in both groups breaks the t statistic (scipy warns about
    # catastrophic cancellation). It happens for real: a rule that fails the
    # same way on every seed returns an identical score each time. Decide it
    # on the means instead of letting a NaN through.
    if np.ptp(x) == 0.0 and np.ptp(y) == 0.0:
        identical = float(np.mean(x)) == float(np.mean(y))
        return SeparationResult(
            mean_a=float(np.mean(x)),
            mean_b=float(np.mean(y)),
            gap=float(np.mean(y) - np.mean(x)),
            p_value=1.0 if identical else 0.0,
            n_a=int(x.size),
            n_b=int(y.size),
            separated=not identical,
            alpha=alpha,
        )

    result = stats.ttest_ind(x, y, equal_var=False)
    p = float(result.pvalue)
    return SeparationResult(
        mean_a=float(np.mean(x)),
        mean_b=float(np.mean(y)),
        gap=float(np.mean(y) - np.mean(x)),
        p_value=p,
        n_a=int(x.size),
        n_b=int(y.size),
        separated=bool(p < alpha),
        alpha=alpha,
    )


def compare_to_constant(
    values: Sequence[float], constant: float, alpha: float = 0.05
) -> SeparationResult:
    """Does a stochastic method's score reach past a deterministic baseline?

    A search-free baseline returns the same number every time, so there is no
    spread to pool and the right test is one-sample against that constant.
    Welch against the constant repeated ``n`` times is algebraically the same
    test -- the second variance term vanishes and the Welch-Satterthwaite
    degrees of freedom collapse to ``n - 1`` -- but spelling it this way says
    what is being asked and keeps scipy from warning about a zero-variance
    group.

    ``gap`` keeps the same meaning as in :func:`compare_groups`: positive means
    ``values`` scored lower, which for a minimized objective means better.
    """
    from scipy import stats

    x = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if x.size < 2 or np.isnan(constant):
        return SeparationResult(
            float(np.mean(x)) if x.size else float("nan"),
            float(constant), float("nan"), float("nan"),
            int(x.size), 1, False, alpha,
        )
    if np.ptp(x) == 0.0:
        identical = float(x[0]) == float(constant)
        return SeparationResult(
            float(np.mean(x)), float(constant), float(constant - np.mean(x)),
            1.0 if identical else 0.0, int(x.size), 1, not identical, alpha,
        )
    p = float(stats.ttest_1samp(x, popmean=float(constant)).pvalue)
    return SeparationResult(
        mean_a=float(np.mean(x)),
        mean_b=float(constant),
        gap=float(constant - np.mean(x)),
        p_value=p,
        n_a=int(x.size),
        n_b=1,
        separated=bool(p < alpha),
        alpha=alpha,
    )


def rank_with_ties(
    scores: dict[str, Sequence[float]], alpha: float = 0.05
) -> list[tuple[str, float, bool]]:
    """Order methods best-first, flagging which are indistinguishable from the best.

    Returns ``(name, mean, separated_from_best)`` per method, lower mean first.
    The best method is always reported as not separated from itself. Anything
    else carrying ``False`` ties the leader on this evidence and must not be
    described as beaten by it.
    """
    means = {
        name: float(np.nanmean(values)) if len(values) else float("nan")
        for name, values in scores.items()
    }
    order = sorted(means, key=lambda n: (np.isnan(means[n]), means[n]))
    if not order:
        return []
    best = order[0]
    out = []
    for name in order:
        if name == best:
            out.append((name, means[name], False))
        else:
            out.append((name, means[name], compare_groups(scores[best], scores[name], alpha).separated))
    return out
