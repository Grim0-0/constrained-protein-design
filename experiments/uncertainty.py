"""What does the reported feasibility rate promise, and what would it deliver?

    python experiments/uncertainty.py --domain GRB2-SH3

The benchmark treats the folding constraint as a noiseless test: a design
passes when its point estimate clears tau. The published ddPCA tables ship a
standard error for every variant and the parser already reads them, so that
test throws away information the data provides.

It matters most where the rules work hardest. A constraint-aware search is
supposed to settle near the boundary -- that is the whole argument for the
Lagrangian -- and a design sitting just inside the bar, measured to an error
several times its margin, is a coin flip reported as a pass. So the rules that
look best under the nominal test are the ones with the most to lose under an
honest one.

Three readings of the same returned designs:

``nominal``
    Fraction whose point estimate clears tau. What the benchmark reports.
``expected``
    Mean of ``P(ddg_folding <= tau)``. What the experiment would return.
``confident(alpha)``
    Fraction passing ``P(ddg_folding <= tau) >= 1 - alpha``. What you can
    promise. At ``alpha = 0.5`` this equals the nominal rate, so sweeping
    alpha downward from there traces how fast the promise erodes.

**Independence is assumed** when composing errors. The additive oracle adds
single-mutant variances in quadrature, which treats the published per-variant
errors as independent. They are not exactly: the same growth-rate assay and
the same fitted thermodynamic model produce all of them, so shared systematic
error is plausible and would make the true uncertainty larger than this. The
gap this script reports is therefore a lower bound on the real one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.methods import default_methods  # noqa: E402
from cpd.metrics import (  # noqa: E402
    confident_feasibility_rate,
    expected_feasibility_rate,
    feasibility_rate,
)

from experiments.common import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_N_RETURN,
    DEFAULT_SEEDS,
    FIGURES_DIR,
    build_landscape,
    save_results,
    search_config,
    tau_grid,
)
from experiments.plotting import DISPLAY_NAMES  # noqa: E402

RULES = ["lagrangian", "fixed_penalty", "scalarization_w0.5", "scalarization_sweep",
         "optimize_then_reject", "nsga2"]

#: Risk levels swept, from the nominal test down to a strict one.
ALPHAS: tuple[float, ...] = (0.5, 0.3, 0.2, 0.1, 0.05, 0.01)


def run(landscape, taus, budget, seeds, n_return) -> pd.DataFrame:
    rows = []
    for tau in taus:
        for seed in seeds:
            config = search_config(seed, budget)
            for name, method in default_methods().items():
                delivered = method(landscape, float(tau), budget, n_return, config).measurements
                row = {
                    "rule": name,
                    "tau": float(tau),
                    "seed": seed,
                    "nominal": feasibility_rate(delivered, float(tau)),
                    "expected": expected_feasibility_rate(delivered, float(tau)),
                }
                for alpha in ALPHAS:
                    row[f"confident_{alpha:g}"] = confident_feasibility_rate(
                        delivered, float(tau), alpha
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="GRB2-SH3")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--n-return", type=int, default=DEFAULT_N_RETURN)
    parser.add_argument("--all-tau", action="store_true",
                        help="sweep every tau rather than only the tightest")
    args = parser.parse_args(argv)

    landscape = build_landscape("additive", domain=args.domain)
    grid = tau_grid(landscape)
    taus = list(grid) if args.all_tau else [float(grid[-1])]
    seeds = list(range(args.seeds))

    frame = run(landscape, taus, args.budget, seeds, args.n_return)
    frame["domain"] = args.domain
    path = save_results(frame, f"uncertainty_additive_{args.domain}")

    tight = min(taus)
    cell = frame[np.isclose(frame["tau"], tight)]
    print(f"\n{'=' * 78}\n{args.domain}, tau = {tight:+.3f}, {args.seeds} seeds\n{'=' * 78}")
    header = f"{'rule':<24} {'nominal':>8} {'expected':>9} {'gap':>7} |"
    for alpha in ALPHAS:
        header += f" {f'a={alpha:g}':>7}"
    print(header)

    summary = []
    for rule in RULES:
        sub = cell[cell["rule"] == rule]
        if sub.empty:
            continue
        nominal, expected = sub["nominal"].mean(), sub["expected"].mean()
        line = (f"{DISPLAY_NAMES.get(rule, rule):<24} {nominal:>7.0%} {expected:>8.0%} "
                f"{nominal - expected:>6.0%} |")
        for alpha in ALPHAS:
            line += f" {sub[f'confident_{alpha:g}'].mean():>6.0%}"
        print(line)
        summary.append({
            "domain": args.domain, "tau": tight, "rule": rule,
            "nominal": nominal, "expected": expected, "gap": nominal - expected,
            **{f"confident_{a:g}": sub[f"confident_{a:g}"].mean() for a in ALPHAS},
        })

    table = pd.DataFrame(summary)
    save_results(table, f"uncertainty_summary_{args.domain}")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    if not table.empty:
        worst = table.loc[table["gap"].idxmax()]
        strict = table.loc[table["confident_0.05"].idxmax()]
        print(f"\n  largest nominal-to-expected gap: {DISPLAY_NAMES.get(worst['rule'])} "
              f"loses {worst['gap']:.0%} of its reported feasibility")
        print(f"  most dependable at alpha = 0.05: {DISPLAY_NAMES.get(strict['rule'])} "
              f"at {strict['confident_0.05']:.0%} (reports {strict['nominal']:.0%})")
        shuffled = table.sort_values("nominal", ascending=False)["rule"].tolist() != \
            table.sort_values("confident_0.05", ascending=False)["rule"].tolist()
        print("  honouring the error bars "
              + ("REORDERS the rules." if shuffled else "leaves the ordering unchanged."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
