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
    "feasible_best",
    "feasible_best_k",
    "ViolationStats",
    "violation_severity",
    "pareto_mask",
    "hypervolume",
    "DesignSetReport",
    "summarize",
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
