"""Twenty designs, or one design twenty times?

    python experiments/diversity.py --domain GRB2-SH3

Every rule is asked for twenty designs and judged on how good the best one is
and how many clear the constraint. Neither question notices whether the twenty
are actually different. A set that collapses onto one region of sequence space
spends an experimental budget sized for twenty bets on a single bet, and every
metric in this benchmark would score it the same as a spread-out set.

Reported per rule, over the designs each one hands over on its own terms:

``filled``        how many of the twenty came back at all
``unique``        distinct variants among them -- duplicates are not designs
``mean distance`` mean pairwise Hamming distance, in substituted positions
``positions``     distinct positions touched anywhere in the set

The specific suspicion worth testing: optimize-then-reject already fills only
a fraction of the set it was asked for, because it discards most of what it
found. If what survives also sits in one place, its usable output is narrower
than the yield number alone suggests -- two separate costs, one of which the
benchmark has never reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.methods import default_methods  # noqa: E402
from cpd.metrics import diversity  # noqa: E402

from experiments.common import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_N_RETURN,
    DEFAULT_SEEDS,
    build_landscape,
    save_results,
    search_config,
    tau_grid,
)
from experiments.plotting import DISPLAY_NAMES  # noqa: E402

RULES = ["lagrangian", "fixed_penalty", "scalarization_w0.5", "scalarization_sweep",
         "optimize_then_reject", "nsga2"]


def run(landscape, tau: float, budget: int, seeds, n_return: int) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        config = search_config(seed, budget)
        for name, method in default_methods().items():
            run_result = method(landscape, tau, budget, n_return, config)
            delivered = run_result.designs
            feasible = [
                v for v, m in zip(run_result.designs, run_result.measurements)
                if m.is_feasible(tau)
            ]
            whole, usable = diversity(delivered), diversity(feasible)
            rows.append({
                "rule": name,
                "seed": seed,
                "filled": len(delivered) / n_return,
                "n_unique": whole.n_unique,
                "mean_distance": whole.mean_pairwise_distance,
                "min_distance": whole.min_pairwise_distance,
                "positions_used": whole.n_positions_used,
                "mean_order": whole.mean_order,
                "n_feasible": usable.n_designs,
                "feasible_mean_distance": usable.mean_pairwise_distance,
                "feasible_positions_used": usable.n_positions_used,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape", default="additive", choices=("additive", "synthetic"))
    parser.add_argument("--domain", default="GRB2-SH3")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--n-return", type=int, default=DEFAULT_N_RETURN)
    args = parser.parse_args(argv)

    landscape = build_landscape(args.landscape, domain=args.domain)
    label = args.domain if args.landscape == "additive" else "synthetic"
    tau = float(tau_grid(landscape)[-1])
    seeds = list(range(args.seeds))

    frame = run(landscape, tau, args.budget, seeds, args.n_return)
    frame["domain"] = label
    frame["tau"] = tau

    print(f"\n{'=' * 86}\n{label}, tau = {tau:+.3f}, {args.seeds} seeds, "
          f"{args.n_return} designs requested\n{'=' * 86}")
    print(f"{'rule':<24} {'filled':>7} {'unique':>7} {'mean dist':>10} {'positions':>10} "
          f"| {'feasible':>9} {'mean dist':>10} {'positions':>10}")

    summary = []
    for rule in RULES:
        sub = frame[frame["rule"] == rule]
        if sub.empty:
            continue
        row = {
            "domain": label, "tau": tau, "rule": rule,
            "filled": sub["filled"].mean(),
            "n_unique": sub["n_unique"].mean(),
            "mean_distance": sub["mean_distance"].mean(),
            "positions_used": sub["positions_used"].mean(),
            "n_feasible": sub["n_feasible"].mean(),
            "feasible_mean_distance": sub["feasible_mean_distance"].mean(),
            "feasible_positions_used": sub["feasible_positions_used"].mean(),
        }
        summary.append(row)
        feas_dist = row["feasible_mean_distance"]
        print(f"{DISPLAY_NAMES.get(rule, rule):<24} {row['filled']:>6.0%} "
              f"{row['n_unique']:>7.1f} {row['mean_distance']:>10.2f} "
              f"{row['positions_used']:>10.1f} | {row['n_feasible']:>9.1f} "
              f"{feas_dist:>10.2f} {row['feasible_positions_used']:>10.1f}")

    table = pd.DataFrame(summary)
    save_results(frame, f"diversity_{args.landscape}_{label}")
    path = save_results(table, f"diversity_summary_{args.landscape}_{label}")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    if not table.empty:
        otr = table[table["rule"] == "optimize_then_reject"]
        others = table[
            table["rule"].isin(["lagrangian", "fixed_penalty", "scalarization_w0.5"])
        ]
        if not otr.empty and not others.empty:
            mine = float(otr["feasible_mean_distance"].iloc[0])
            theirs = float(others["feasible_mean_distance"].mean())
            print(f"\n  optimize-then-reject fills {otr['filled'].iloc[0]:.0%} of the set, "
                  f"and what survives sits {mine:.2f} apart")
            print(f"  the constraint-aware rules average {theirs:.2f} apart")
            if np.isnan(mine) or np.isnan(theirs):
                print("  -> not comparable: too few feasible designs to measure a spread.")
            elif mine < theirs:
                print("  -> it concentrates as well as under-fills: two costs, not one.")
            else:
                print("  -> it under-fills but does NOT concentrate. Reported as found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
