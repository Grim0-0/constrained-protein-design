"""The headline experiment: how each constraint-handling rule degrades as tau tightens.

Run from the repository root::

    python experiments/tau_sweep.py                      # synthetic landscape (Track 0)
    python experiments/tau_sweep.py --landscape exact    # measured ddPCA library (Track 1)

Writes ``results/tau_sweep_<landscape>.csv`` and the figures in ``figures/``.

The hypothesis under test is that scalarization and optimize-then-reject
degrade sharply as tau tightens, because they keep spending budget in
infeasible regions, while the Lagrangian rule holds. This script does not
assume that. It prints what happened, including the case where the hypothesis
is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import plotting
from experiments.common import (
    DEFAULT_SEEDS,
    FIGURES_DIR,
    add_common_args,
    aggregate,
    build_landscape,
    reference_point,
    run_grid,
    save_results,
    tau_grid,
)


def summary_table(frame: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    """One row per method, at the tightest tau, under both readouts."""
    tight = float(min(taus))
    delivered = frame[frame["readout"] == "as_delivered"]
    filtered = frame[frame["readout"] == "feasible_filtered"]
    rows = []
    for method in sorted(frame["method"].unique()):
        d = delivered[(delivered["method"] == method) & np.isclose(delivered["tau"], tight)]
        f = filtered[(filtered["method"] == method) & np.isclose(filtered["tau"], tight)]
        all_d = delivered[delivered["method"] == method]
        rows.append(
            {
                "method": method,
                # Where the oracle budget went.
                "searched_feasible": all_d["search_feasibility_rate"].mean(),
                # What the rule hands over on its own terms.
                "delivered_feasible": d["feasibility_rate"].mean(),
                # Best usable design, selection held identical across rules.
                "best_feasible": f["feasible_best"].mean(),
                "design_yield": f["design_yield"].mean(),
                "seeds_with_nothing_feasible": int(f["feasible_best"].isna().sum()),
                "hv_overclaim": (all_d["hypervolume_all"] - all_d["hypervolume_feasible"]).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("best_feasible")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-tau", type=int, default=7, help="points on the tau grid")
    args = parser.parse_args(argv)

    landscape = build_landscape(args.landscape, args.domain)
    taus = tau_grid(landscape, n=args.n_tau)
    reference = reference_point(landscape)
    seeds = tuple(DEFAULT_SEEDS[: args.seeds]) if args.seeds <= len(DEFAULT_SEEDS) else tuple(range(args.seeds))

    label = args.landscape if args.landscape == "synthetic" else f"{args.landscape}_{args.domain}"
    print(f"tau sweep on the {label} landscape")
    print(f"  budget {args.budget} oracle calls/method   seeds {list(seeds)}")
    print(f"  tau from {taus[0]:+.3f} (loose) to {taus[-1]:+.3f} (tight) kcal/mol")
    print(f"  hypervolume reference {reference[0]:.3f}, {reference[1]:.3f}")

    frame = run_grid(
        landscape,
        taus,
        seeds=seeds,
        budget=args.budget,
        n_return=args.n_return,
        k=args.k,
        reference=reference,
    )

    meta = {
        "landscape": label,
        "budget": args.budget,
        "seeds": list(seeds),
        "n_return": args.n_return,
        "k": args.k,
        "taus": [float(t) for t in taus],
        "reference": list(reference),
    }
    path = save_results(frame, f"tau_sweep_{label}", meta)
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    # Panel A compares search quality with selection held identical, so it reads
    # the filtered readout. Panel B is the metric nobody reports -- how much of
    # what a rule hands over is actually usable -- so it reads as-delivered.
    best = aggregate(frame, "feasible_best", readout="feasible_filtered")
    rate = aggregate(frame, "feasibility_rate", readout="as_delivered")
    hv_all = aggregate(frame, "hypervolume_all", readout="as_delivered")
    hv_feasible = aggregate(frame, "hypervolume_feasible", readout="as_delivered")

    subtitle = (
        f"{label} landscape, {args.budget} oracle calls per method, "
        f"{len(seeds)} seeds, {args.n_return} designs returned"
    )
    headline = plotting.headline_tau_sweep(
        best, rate, FIGURES_DIR / f"headline_tau_sweep_{label}.png", subtitle=subtitle
    )
    gap = plotting.hypervolume_gap(
        hv_all, hv_feasible, FIGURES_DIR / f"hypervolume_gap_{label}.png", subtitle=subtitle
    )
    print(f"wrote {headline.relative_to(headline.parent.parent)}")
    print(f"wrote {gap.relative_to(gap.parent.parent)}")

    table = summary_table(frame, taus)
    print(f"\nat the tightest tau = {min(taus):+.3f} kcal/mol "
          f"(searched_feasible and hv_overclaim are averaged over all tau)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    # State the verdict rather than leaving it to the reader of a figure.
    tight = float(min(taus))
    at_tight = frame[np.isclose(frame["tau"], tight) & (frame["readout"] == "feasible_filtered")]
    scores = at_tight.groupby("method")["feasible_best"].mean()
    if scores.notna().any():
        winner = scores.idxmin()
        print(f"\nat the tightest tau the best mean feasible binding came from: {winner}")
        if winner != "lagrangian":
            print("  note: this is not the hypothesised result. Report it as it is.")
    else:
        print("\nat the tightest tau no method returned a feasible design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
