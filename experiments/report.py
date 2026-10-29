"""Print README-ready markdown tables from whatever is in ``results/``.

    python experiments/report.py --landscape additive_GRB2-SH3

The README quotes numbers from these tables. Generating them rather than
copying them by hand means the quoted numbers cannot drift from the runs that
produced them -- re-run the experiment, re-run this, paste.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS_DIR
from experiments.plotting import DISPLAY_NAMES

ORDER = ["lagrangian", "fixed_penalty", "scalarization_w0.5", "scalarization_sweep",
         "optimize_then_reject"]


def _fmt(value: float, places: int = 2, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    if percent:
        return f"{value * 100:.0f}%"
    return f"{value:+.{places}f}"


def tau_sweep_table(label: str) -> str:
    path = RESULTS_DIR / f"tau_sweep_{label}.csv"
    frame = pd.read_csv(path)
    tight = frame["tau"].min()
    delivered = frame[frame["readout"] == "as_delivered"]
    filtered = frame[frame["readout"] == "feasible_filtered"]

    lines = [
        "| rule | budget spent on feasible designs | feasible designs delivered | best usable design | designs filled |",
        "|---|---|---|---|---|",
    ]
    methods = [m for m in ORDER if m in set(frame["method"])]
    for method in methods:
        d = delivered[delivered["method"] == method]
        f = filtered[(filtered["method"] == method) & np.isclose(filtered["tau"], tight)]
        d_tight = d[np.isclose(d["tau"], tight)]
        lines.append(
            f"| {DISPLAY_NAMES.get(method, method)} "
            f"| {_fmt(d['search_feasibility_rate'].mean(), percent=True)} "
            f"| {_fmt(d_tight['feasibility_rate'].mean(), percent=True)} "
            f"| {_fmt(f['feasible_best'].mean())} kcal/mol "
            f"| {_fmt(f['design_yield'].mean(), percent=True)} |"
        )
    header = (
        f"Tightest constraint, tau = {tight:+.2f} kcal/mol. "
        f"Budget spent on feasible designs is averaged over every tau; the rest are at "
        f"the tightest. {frame['seed'].nunique()} seeds."
    )
    return header + "\n\n" + "\n".join(lines)


def tuning_table(label: str) -> str:
    path = RESULTS_DIR / f"tuning_cost_{label}.csv"
    frame = pd.read_csv(path).sort_values("spread")
    lines = [
        "| rule | knob | best setting | worst setting | choosing blind | spread |",
        "|---|---|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.rule, row.rule)} | `{row.parameter}` "
            f"| {_fmt(row.best)} (at {row.best_value:g}) | {_fmt(row.worst)} "
            f"| {_fmt(row.blind_mean)} | **{abs(row.spread):.2f}** |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape", default="additive_GRB2-SH3")
    parser.add_argument("--designs", action="store_true",
                        help="print the best usable designs instead of the metric tables")
    args = parser.parse_args(argv)

    if args.designs:
        for track in ("exact", "additive"):
            budget = 120 if track == "exact" else 1200
            print(f"\n### designs ({track})\n")
            print(designs_table(track, budget=budget))
        return 0

    for name, builder in (("tau sweep", tau_sweep_table), ("tuning cost", tuning_table)):
        try:
            print(f"\n### {name} ({args.landscape})\n")
            print(builder(args.landscape))
        except FileNotFoundError:
            print(f"(no {name} results for {args.landscape}; run the experiment first)")
    return 0



# --------------------------------------------------------------------------
# Actual designs
# --------------------------------------------------------------------------


def designs_table(track: str, domain: str = "GRB2-SH3", budget: int = 1200, n: int = 6) -> str:
    """The best usable designs the benchmark actually produced, and who found them.

    Pools the evaluation traces of every rule, keeps only designs satisfying the
    constraint, and ranks them by binding. Naming the rule that found each one
    keeps this honest: it is a record of what the benchmark produced, not a
    showcase for one method.
    """
    from experiments.common import build_landscape, search_config, tau_grid
    from cpd.methods import default_methods

    landscape = build_landscape(track, domain)
    tau = float(tau_grid(landscape, n=7)[3])  # mid-tightness

    pooled: dict = {}
    for name, method in default_methods().items():
        run = method(landscape, tau, budget, 20, search_config(0, budget))
        for variant, measurement in run.trace:
            if measurement.is_feasible(tau) and variant.order > 0:
                pooled.setdefault(variant, (measurement, name))

    ranked = sorted(pooled.items(), key=lambda kv: kv[1][0].ddg_binding)
    lines = [
        f"Constraint: `ddG_folding <= {tau:+.2f}` kcal/mol. "
        f"{len(ranked)} designs satisfy it; the best six:",
        "",
        "| design | ddG binding | ddG folding | found by |",
        "|---|---|---|---|",
    ]
    for variant, (m, name) in ranked[:n]:
        lines.append(
            f"| `{variant}` | **{m.ddg_binding:+.2f}** | {m.ddg_folding:+.2f} "
            f"| {DISPLAY_NAMES.get(name, name)} |"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    raise SystemExit(main())
