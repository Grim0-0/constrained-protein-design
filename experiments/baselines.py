"""Is this problem hard enough for the method comparison to mean anything?

    python experiments/baselines.py --track additive
    python experiments/baselines.py --track synthetic

Reads the tau sweeps already in ``results/`` and asks one question of them: can
a method with no search at all do as well? The baseline is a sort -- rank the
measured singles by binding, add substitutions while the constraint holds, stop
at the order cap. On the additive track the exact constrained optimum is also
computed, which turns every rule's score into a fraction of what was actually
attainable.

Run the sweeps first::

    for d in GB1 GRB2-SH3 PSD95-PDZ3; do
        python experiments/tau_sweep.py --landscape additive --domain $d
    done
    python experiments/tau_sweep.py --landscape synthetic

This script exists because the answer on the additive track is yes, and that
invalidates using it to rank constraint-handling rules. The two tracks together
localize the cause: additive objectives are separable, so the design problem is
an integer program with no structure for a search to exploit, and a 1200-call
evolutionary search does not reliably beat four oracle calls. Add epistasis and
the same baseline loses by about 2 kcal/mol. The lesson generalizes past this
repo: any multi-objective design benchmark built on an additive or per-residue
oracle should report a greedy baseline before it reports a method ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.baselines import exact_optimum, greedy_feasible  # noqa: E402
from cpd.landscape import MeasuredAdditiveLandscape  # noqa: E402
from cpd.metrics import compare_to_constant  # noqa: E402

from experiments import plotting  # noqa: E402
from experiments.common import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    build_landscape,
    save_results,
    tau_grid,
)
from experiments.plotting import DOMAIN_ORDER  # noqa: E402

#: Rules compared against the baseline, in table order.
RULES = [
    "lagrangian",
    "fixed_penalty",
    "scalarization_w0.5",
    "scalarization_sweep",
    "optimize_then_reject",
]


def sweep_path(track: str, domain: str | None) -> Path:
    stem = f"tau_sweep_{track}" + (f"_{domain}" if domain else "")
    return RESULTS_DIR / f"{stem}.csv"


def rule_scores(frame: pd.DataFrame, tau: float) -> dict[str, list[float]]:
    """Per-seed ``feasible_best`` for each rule at one tau."""
    cell = frame[
        (frame["readout"] == "feasible_filtered") & np.isclose(frame["tau"], tau)
    ]
    out = {}
    for rule in RULES:
        values = cell[cell["method"] == rule]["feasible_best"].tolist()
        if values:
            out[rule] = values
    return out


def evaluate(track: str, domain: str | None) -> pd.DataFrame:
    """One row per tau: the baseline, the optimum where available, every rule."""
    frame = pd.read_csv(sweep_path(track, domain))
    landscape = build_landscape(track, domain=domain or "GRB2-SH3")
    taus = sorted(frame["tau"].unique(), reverse=True)

    optima: dict[float, float] = {}
    if isinstance(landscape, MeasuredAdditiveLandscape):
        optima = exact_optimum(landscape, taus)

    rows = []
    for tau in taus:
        greedy = greedy_feasible(landscape, float(tau))
        scores = rule_scores(frame, float(tau))
        row = {
            "track": track,
            "domain": domain or track,
            "tau": float(tau),
            "exact_optimum": optima.get(float(tau), float("nan")),
            "greedy_binding": greedy.ddg_binding,
            "greedy_folding": greedy.ddg_folding,
            "greedy_order": greedy.order,
            "greedy_calls": greedy.n_evaluations,
        }
        for rule in RULES:
            row[rule] = float(np.nanmean(scores[rule])) if rule in scores else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(track: str, domain: str | None, table: pd.DataFrame) -> dict:
    """At the tightest tau, is the best rule actually better than the sort?"""
    frame = pd.read_csv(sweep_path(track, domain))
    tight = float(table["tau"].min())
    row = table[np.isclose(table["tau"], tight)].iloc[0]
    scores = rule_scores(frame, tight)
    if not scores:
        raise SystemExit(f"no rule produced a score at tau = {tight}")

    best_rule = min(scores, key=lambda r: float(np.nanmean(scores[r])))
    best = scores[best_rule]
    # The baseline is deterministic, so the question is whether the rule's seed
    # spread reaches past a fixed number -- a one-sample test, not Welch.
    separation = compare_to_constant(best, float(row.greedy_binding))
    return {
        "track": track,
        "domain": domain or track,
        "tau": tight,
        "exact_optimum": row.exact_optimum,
        "greedy": row.greedy_binding,
        "greedy_calls": int(row.greedy_calls),
        "best_rule": best_rule,
        "best_rule_mean": float(np.nanmean(best)),
        "gap_to_greedy": separation.gap,
        "p_value": separation.p_value,
        "beats_greedy": bool(separation.separated and separation.gap > 0),
        "loses_to_greedy": bool(separation.separated and separation.gap < 0),
    }


def print_table(table: pd.DataFrame) -> None:
    has_optimum = not table["exact_optimum"].isna().all()
    header = f"{'tau':>7} " + (f"{'optimum':>9} " if has_optimum else "") + f"{'greedy':>9} |"
    for rule in RULES:
        header += f" {plotting.DISPLAY_NAMES[rule][:11]:>11}"
    print(header)
    for _, row in table.iterrows():
        line = f"{row.tau:>7.3f} "
        if has_optimum:
            line += f"{row.exact_optimum:>9.3f} "
        line += f"{row.greedy_binding:>9.3f} |"
        for rule in RULES:
            line += f" {row[rule]:>11.3f}"
        print(line)


def print_attainment(v: dict, table: pd.DataFrame) -> None:
    """What fraction of the attainable objective each rule actually captured."""
    row = table[np.isclose(table["tau"], v["tau"])].iloc[0]
    optimum = row.exact_optimum
    if np.isnan(optimum):
        return
    print(f"\n  share of the attainable objective captured at tau = {v['tau']:.3f} "
          f"(optimum {optimum:.3f}):")
    entries = [("greedy (no search)", row.greedy_binding)]
    entries += [(plotting.DISPLAY_NAMES[r], row[r]) for r in RULES]
    for name, value in sorted(entries, key=lambda e: e[1]):
        if np.isnan(value):
            continue
        print(f"    {name:<24} {value:>8.3f}   gap {value - optimum:>6.3f}   "
              f"{100 * value / optimum:>5.0f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="additive", choices=("additive", "exact", "synthetic"))
    args = parser.parse_args(argv)

    if args.track == "synthetic":
        targets: list[str | None] = [None]
    else:
        targets = [d for d in DOMAIN_ORDER if sweep_path(args.track, d).exists()]
        if not targets:
            raise SystemExit(
                f"no {args.track} sweeps in results/; run tau_sweep.py first"
            )

    tables, verdicts = [], []
    for domain in targets:
        if not sweep_path(args.track, domain).exists():
            raise SystemExit(f"missing {sweep_path(args.track, domain)}; run tau_sweep.py first")
        table = evaluate(args.track, domain)
        v = verdict(args.track, domain, table)
        tables.append(table)
        verdicts.append(v)

        print(f"\n{'=' * 78}\n{v['domain']}  ({args.track} track)\n{'=' * 78}")
        print_table(table)
        print_attainment(v, table)
        print(f"\n  best rule ({plotting.DISPLAY_NAMES[v['best_rule']]}) "
              f"{v['best_rule_mean']:.3f}  vs  greedy {v['greedy']:.3f} "
              f"in {v['greedy_calls']} calls")
        print(f"  gap {v['gap_to_greedy']:+.3f} kcal/mol, p = {v['p_value']:.4f} -> ", end="")
        if v["beats_greedy"]:
            print("the search beats the sort")
        elif v["loses_to_greedy"]:
            print("THE SORT BEATS THE SEARCH")
        else:
            print("NOT SEPARATED: the search does not beat a sort")

    combined = pd.concat(tables, ignore_index=True)
    save_results(combined, f"baselines_{args.track}")
    summary = pd.DataFrame(verdicts)
    path = save_results(summary, f"baselines_verdict_{args.track}")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    print(f"\n{'=' * 78}\nverdict\n{'=' * 78}")
    beaten = summary[~summary["beats_greedy"]]["domain"].tolist()
    if beaten:
        print(f"  a search-free sort is not beaten on: {', '.join(beaten)}")
        print("  -> on those, differences between constraint-handling rules are not")
        print("     evidence about constraint handling; the search is the bottleneck.")
    else:
        print("  the search beats a search-free sort everywhere; the comparison stands.")

    figure = plotting.baseline_attainment(
        combined, summary, FIGURES_DIR / f"baselines_{args.track}.png",
        subtitle=f"{args.track} track, 1200 oracle calls per rule, 5 seeds, "
                 "greedy baseline is deterministic",
    )
    print(f"  wrote {figure.relative_to(figure.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
