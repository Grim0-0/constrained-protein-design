"""What does a hyperparameter you have to choose in advance cost you?

Run from the repository root::

    python experiments/tuning_cost.py --landscape additive --domain GRB2-SH3

Every constraint-handling rule here has one knob: scalarization has its weight,
the fixed penalty has ``rho``, the Lagrangian has its multiplier learning rate.
Comparing the rules at their *best* knob setting flatters all of them equally
and answers a question nobody faces, because picking the best setting requires
already knowing the answer. A real run gets one guess.

So each rule is swept over its own knob and reported three ways: the best
setting, the worst, and the average over settings -- which is what choosing
blind gets you in expectation. The span between best and worst is the number
that matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.methods import run_fixed_penalty, run_lagrangian, run_scalarization

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

#: Each rule, its knob, and the settings a practitioner might plausibly pick.
SWEEPS: dict[str, tuple[str, tuple[float, ...], object]] = {
    "scalarization_sweep": ("weight", (0.1, 0.3, 0.5, 0.7, 0.9), run_scalarization),
    "fixed_penalty": ("rho", (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0), run_fixed_penalty),
    "lagrangian": ("lr", (0.05, 0.1, 0.25, 0.5, 1.0), run_lagrangian),
}


def best_feasible_binding(run, n_return: int) -> float:
    """Best binding among feasible designs, selection held identical across rules."""
    measurements = run.filtered(n_return)
    if not measurements:
        return float("nan")
    return float(np.nanmin([m.ddg_binding for m in measurements]))


def run_sweeps(landscape, taus, seeds, budget: int, n_return: int) -> pd.DataFrame:
    rows = []
    for rule, (parameter, values, runner) in SWEEPS.items():
        for value in values:
            for tau in taus:
                for seed in seeds:
                    run = runner(
                        landscape,
                        float(tau),
                        budget,
                        **{parameter: value},
                        n_return=n_return,
                        config=search_config(seed, budget),
                    )
                    rows.append(
                        {
                            "rule": rule,
                            "parameter": parameter,
                            "value": value,
                            "tau": float(tau),
                            "seed": seed,
                            "best_feasible": best_feasible_binding(run, n_return),
                        }
                    )
            print(f"  {rule} {parameter}={value} done")
    return pd.DataFrame(rows)


def summarize_sweeps(frame: pd.DataFrame) -> pd.DataFrame:
    """Per rule: its best setting, its worst, and the blind average over settings."""
    per_value = (
        frame.groupby(["rule", "parameter", "value"])["best_feasible"].mean().reset_index()
    )
    rows = []
    for (rule, parameter), group in per_value.groupby(["rule", "parameter"]):
        best_row = group.loc[group["best_feasible"].idxmin()]
        rows.append(
            {
                "rule": rule,
                "parameter": parameter,
                "n_settings": len(group),
                "best": float(group["best_feasible"].min()),
                "best_value": float(best_row["value"]),
                "worst": float(group["best_feasible"].max()),
                "blind_mean": float(group["best_feasible"].mean()),
                "spread": float(group["best_feasible"].max() - group["best_feasible"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("spread")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-tau", type=int, default=4, help="points on the tau grid")
    args = parser.parse_args(argv)

    landscape = build_landscape(args.landscape, args.domain)
    taus = tau_grid(landscape, n=args.n_tau)
    seeds = tuple(DEFAULT_SEEDS[: args.seeds]) if args.seeds <= len(DEFAULT_SEEDS) else tuple(range(args.seeds))
    label = args.landscape if args.landscape == "synthetic" else f"{args.landscape}_{args.domain}"

    print(f"hyperparameter sweeps on the {label} landscape")
    print(f"  budget {args.budget}   seeds {list(seeds)}   {args.n_tau} tau values")

    frame = run_sweeps(landscape, taus, seeds, args.budget, args.n_return)
    table = summarize_sweeps(frame)

    save_results(frame, f"tuning_cost_{label}_raw")
    path = save_results(
        table,
        f"tuning_cost_{label}",
        {"landscape": label, "budget": args.budget, "seeds": list(seeds),
         "taus": [float(t) for t in taus]},
    )
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    subtitle = (
        f"{label} landscape, {args.budget} oracle calls per run, "
        f"{len(seeds)} seeds x {args.n_tau} tau values per setting"
    )
    figure = plotting.tuning_cost(table, FIGURES_DIR / f"tuning_cost_{label}.png", subtitle)
    print(f"wrote {figure.relative_to(figure.parent.parent)}")

    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    tightest = table.iloc[0]
    widest = table.iloc[-1]
    print(
        f"\n{tightest['rule']} varies by {tightest['spread']:.2f} kcal/mol across its "
        f"{tightest['parameter']} settings; {widest['rule']} varies by {widest['spread']:.2f} "
        f"across its {widest['parameter']}."
    )
    if abs(tightest["best"] - widest["best"]) < 0.25:
        print(
            "At their best settings the rules are within 0.25 kcal/mol of each other, so the "
            "difference between them is robustness to the setting, not peak performance."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
