"""Performance against oracle calls spent, at a fixed constraint tightness.

Run from the repository root::

    python experiments/budget_curves.py --tau -0.5

The question here is *where the budget goes*. A method that spends its calls in
infeasible territory can look fine on a final-design metric and still be the
wrong thing to run, because the same calls bought a competitor a better feasible
design much earlier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.landscape import Measurement
from cpd.methods import default_methods

from experiments import plotting
from experiments.common import (
    DEFAULT_SEEDS,
    FIGURES_DIR,
    add_common_args,
    build_landscape,
    save_results,
    search_config,
    tau_grid,
)


def running_best_feasible(
    trace: list[tuple[object, Measurement]], tau: float, length: int
) -> np.ndarray:
    """Best feasible ``ddg_binding`` seen after each oracle call.

    NaN until the first feasible design is found -- a method that has not yet
    produced anything usable does not get credited with a number.
    """
    out = np.full(length, np.nan, dtype=float)
    best = np.inf
    for i in range(length):
        if i < len(trace):
            measurement = trace[i][1]
            if measurement.is_feasible(tau):
                best = min(best, measurement.ddg_binding)
        out[i] = best if np.isfinite(best) else np.nan
    return out


def collect_curves(
    landscape,
    tau: float,
    seeds,
    budget: int,
    n_return: int,
    stride: int = 10,
) -> pd.DataFrame:
    """Run every method over every seed and tabulate mean/sd of the running best."""
    methods = default_methods()
    per_method: dict[str, list[np.ndarray]] = {name: [] for name in methods}

    for seed in seeds:
        config = search_config(seed, budget)
        for name, method in methods.items():
            run = method(landscape, tau, budget, n_return, config)
            per_method[name].append(running_best_feasible(run.trace, tau, budget))
        print(f"  seed {seed} done")

    grid = np.arange(0, budget, stride)
    rows = []
    for name, curves in per_method.items():
        stacked = np.vstack(curves)[:, grid]
        with np.errstate(invalid="ignore"):
            # A seed that has found nothing feasible yet contributes NaN, not a
            # value, so the mean is over seeds that actually have a design.
            mean = np.nanmean(np.where(np.isnan(stacked), np.nan, stacked), axis=0)
            std = np.nanstd(stacked, axis=0)
        n_valid = np.sum(~np.isnan(stacked), axis=0)
        for i, evaluations in enumerate(grid):
            rows.append(
                {
                    "method": name,
                    "evaluations": int(evaluations) + 1,
                    "mean": float(mean[i]) if n_valid[i] else float("nan"),
                    "std": float(std[i]) if n_valid[i] else float("nan"),
                    "n_valid": int(n_valid[i]),
                    "n_seeds": stacked.shape[0],
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="constraint tightness; defaults to the tight end of the sweep grid",
    )
    parser.add_argument("--stride", type=int, default=10, help="sampling stride along the budget")
    args = parser.parse_args(argv)

    landscape = build_landscape(args.landscape, args.domain)
    taus = tau_grid(landscape)
    tau = args.tau if args.tau is not None else float(taus[-2])
    seeds = tuple(DEFAULT_SEEDS[: args.seeds]) if args.seeds <= len(DEFAULT_SEEDS) else tuple(range(args.seeds))

    label = args.landscape if args.landscape == "synthetic" else f"{args.landscape}_{args.domain}"
    print(f"budget curves on the {label} landscape at tau = {tau:+.3f} kcal/mol")
    print(f"  budget {args.budget} oracle calls/method   seeds {list(seeds)}")

    curves = collect_curves(landscape, tau, seeds, args.budget, args.n_return, args.stride)
    path = save_results(
        curves,
        f"budget_curves_{label}",
        {"landscape": label, "tau": tau, "budget": args.budget, "seeds": list(seeds)},
    )
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    subtitle = f"{label} landscape, {len(seeds)} seeds, {args.n_return} designs returned"
    figure = plotting.budget_curves(
        curves, FIGURES_DIR / f"budget_curves_{label}.png", tau=tau, subtitle=subtitle
    )
    print(f"wrote {figure.relative_to(figure.parent.parent)}")

    final = curves[curves["evaluations"] == curves["evaluations"].max()]
    print("\nbest feasible binding at full budget:")
    print(
        final[["method", "mean", "std", "n_valid", "n_seeds"]].to_string(
            index=False, float_format=lambda v: f"{v:7.3f}"
        )
    )

    # How early did each method reach the eventual best-of-all-methods level?
    target = final["mean"].min()
    if np.isfinite(target):
        print(f"\noracle calls to reach {target:+.3f} kcal/mol (the best any method achieved):")
        for method, group in curves.groupby("method"):
            reached = group[group["mean"] <= target + 1e-9]
            if reached.empty:
                print(f"  {method:<24} never")
            else:
                print(f"  {method:<24} {int(reached['evaluations'].min())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
