"""Does the finding hold on more than one protein?

    python experiments/replication.py

Reads the per-domain tau sweeps already written to ``results/`` and builds the
cross-domain figure plus a summary table. Run the sweeps first::

    for d in GB1 GRB2-SH3 PSD95-PDZ3; do
        python experiments/tau_sweep.py --landscape additive --domain $d
    done

One result on one protein is an anecdote. The question this script answers is
whether the ranking of the rules survives a change of protein, and whether the
size of the effect tracks how much the two objectives actually conflict --
which differs a lot between these three domains and is itself measured rather
than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import plotting
from cpd.metrics import rank_with_ties

from experiments.common import FIGURES_DIR, RESULTS_DIR, aggregate, save_results
from experiments.plotting import DOMAIN_ORDER

#: Rules compared across domains, in the order they appear in tables.
RULES = ["lagrangian", "fixed_penalty", "scalarization_w0.5", "scalarization_sweep",
         "optimize_then_reject"]


def conflict_strength(domain: str) -> dict:
    """How much binding the constraint actually costs, measured from the data.

    Computed on the measured singles, independent of any search: the best
    binding available overall, versus the best available once the folding
    constraint is applied at a tight quantile. The gap is what the constraint
    costs you at the optimum, which is the honest way to quantify the conflict
    -- the correlation between the two properties does not capture it, because
    the conflict lives in the tail.
    """
    frame = pd.read_csv(RESULTS_DIR.parent / "data" / "raw" / "ddpca_singles.csv")
    group = frame[frame["domain"] == domain]
    best_any = group["ddg_binding"].min()
    tau = group["ddg_folding"].quantile(0.10)
    feasible = group[group["ddg_folding"] <= tau]
    best_feasible = feasible["ddg_binding"].min() if len(feasible) else np.nan
    top = group.nsmallest(15, "ddg_binding")
    return {
        "domain": domain,
        "singles": len(group),
        "best_binding_any": float(best_any),
        "best_binding_feasible": float(best_feasible),
        "constraint_cost": float(best_feasible - best_any),
        "top15_median_ddg_folding": float(top["ddg_folding"].median()),
        "top15_pct_destabilizing": float((top["ddg_folding"] > 0).mean()),
    }


def load_sweep(track: str, domain: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"tau_sweep_{track}_{domain}.csv"
    return pd.read_csv(path) if path.exists() else None


def summary(track: str, domains) -> pd.DataFrame:
    """Per domain and rule, the numbers that decide whether the finding replicates."""
    rows = []
    for domain in domains:
        frame = load_sweep(track, domain)
        if frame is None:
            continue
        tight = frame["tau"].min()
        delivered = frame[frame["readout"] == "as_delivered"]
        filtered = frame[
            (frame["readout"] == "feasible_filtered") & np.isclose(frame["tau"], tight)
        ]
        for rule in RULES:
            d, f = delivered[delivered["method"] == rule], filtered[filtered["method"] == rule]
            if f.empty:
                continue
            rows.append({
                "domain": domain,
                "rule": rule,
                "searched_feasible": d["search_feasibility_rate"].mean(),
                "delivered_feasible": d[np.isclose(d["tau"], tight)]["feasibility_rate"].mean(),
                "best_feasible": f["feasible_best"].mean(),
                "design_yield": f["design_yield"].mean(),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="additive", choices=("additive", "exact"))
    args = parser.parse_args(argv)

    domains = [d for d in DOMAIN_ORDER if load_sweep(args.track, d) is not None]
    if len(domains) < 2:
        raise SystemExit(
            f"found sweeps for {domains or 'no domains'}; run tau_sweep.py for at least two "
            "domains before this script"
        )
    print(f"replication across {len(domains)} domains on the {args.track} track: {domains}")

    conflict = pd.DataFrame([conflict_strength(d) for d in domains])
    conflict = conflict.sort_values("constraint_cost")
    print("\nhow strongly do the objectives conflict, per domain?")
    print("  (constraint_cost = binding given up to satisfy the constraint; more negative"
          " best_binding is better)")
    print(conflict.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    table = summary(args.track, domains)
    save_results(conflict, f"conflict_strength_{args.track}")
    path = save_results(table, f"replication_{args.track}")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    per_domain = {}
    for domain in domains:
        frame = load_sweep(args.track, domain)
        per_domain[domain] = (
            aggregate(frame, "feasible_best", readout="feasible_filtered"),
            aggregate(frame, "feasibility_rate", readout="as_delivered"),
        )
    figure = plotting.replication_grid(
        per_domain,
        FIGURES_DIR / f"replication_{args.track}.png",
        subtitle=f"{args.track} track, 1200 oracle calls per rule, 5 seeds, 20 designs returned",
    )
    print(f"wrote {figure.relative_to(figure.parent.parent)}")

    print("\ndoes the ranking survive a change of protein, and is it real?")
    print("  (= means tied with the leader on this evidence; > means separated, p < 0.05)")
    for domain in domains:
        frame = load_sweep(args.track, domain)
        tight = frame["tau"].min()
        cell = frame[
            (frame["readout"] == "feasible_filtered") & np.isclose(frame["tau"], tight)
        ]
        scores = {
            rule: cell[cell["method"] == rule]["feasible_best"].tolist()
            for rule in RULES
            if not cell[cell["method"] == rule].empty
        }
        ranked = rank_with_ties(scores)
        parts = [ranked[0][0]]
        for name, _, separated in ranked[1:]:
            parts.append(f"{'>' if separated else '='} {name}")
        print(f"  {domain:<12} {' '.join(parts)}")

    print("\nwhat replicates across every domain?")
    always_separated = None
    for domain in domains:
        frame = load_sweep(args.track, domain)
        tight = frame["tau"].min()
        cell = frame[
            (frame["readout"] == "feasible_filtered") & np.isclose(frame["tau"], tight)
        ]
        scores = {
            rule: cell[cell["method"] == rule]["feasible_best"].tolist()
            for rule in RULES
            if not cell[cell["method"] == rule].empty
        }
        beaten = {name for name, _, sep in rank_with_ties(scores) if sep}
        always_separated = beaten if always_separated is None else (always_separated & beaten)
    print(f"  reliably worse than the leader in all {len(domains)} domains: "
          f"{sorted(always_separated) if always_separated else 'none'}")

    worst = table[table["rule"] == "optimize_then_reject"].sort_values("domain")
    print("\noptimize-then-reject, by domain (the null hypothesis):")
    print(worst[["domain", "searched_feasible", "design_yield", "best_feasible"]].to_string(
        index=False, float_format=lambda v: f"{v:7.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
