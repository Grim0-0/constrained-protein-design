"""Does the search advantage come back when the objective has a target range?

    python experiments/windows.py --landscape additive --domain GRB2-SH3
    python experiments/windows.py --landscape synthetic

The unbounded benchmark has a problem: on an additive oracle a search-free
sort ties or beats every rule, because minimizing a sum of constants is a
knapsack and ranking the terms nearly solves it. See
:mod:`experiments.baselines`.

Affinity engineering does not generally ask for the tightest possible binder,
though. It asks for one inside a range, because binding too tight carries its
own liabilities. That changes the problem rather than the method: hitting a
cumulative sum inside a window is subset-sum, where the locally best step
routinely rules out the combination that would have landed inside.

So this runs the identical rule comparison twice -- once minimizing binding
without bound, once against a target window -- on the same landscape, the same
engine, the same budget and the same seeds. The only thing that changes is the
target. Each regime is scored against its own fair greedy baseline.

The hypothesis was that the search advantage returns under the window. It
returns on two of the three measured domains. On the third, and on the
synthetic track, greedy forward selection reaches the window as well and
everything ties at zero distance. That is reported as found.

**On the window-selection rule.** The first version of this experiment centred
the target at twice the strongest single-mutant effect and never consulted
tau, which in a benchmark about a constraint is wrong on inspection: on a
domain where the constraint bites early it puts the target past everything
legal, leaving nothing to do but widen the window until something overlaps.
That produced one window 147x wider than another, so the domains were not
running the same experiment. :func:`choose_window` now centres on the midpoint
of the constrained span, and the superseded rule is kept behind
``--rule naive`` so the correction can be checked rather than taken on trust.
Correcting it cost the finding one domain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpd.baselines import (  # noqa: E402
    exact_optimum,
    exact_optimum_window,
    greedy_feasible,
    greedy_window,
)
from cpd.landscape import MeasuredAdditiveLandscape  # noqa: E402
from cpd.methods import default_methods  # noqa: E402
from cpd.metrics import best_feasible_cost, target_hit_rate  # noqa: E402
from cpd.objectives import Minimize, Window  # noqa: E402
from cpd.variants import Variant  # noqa: E402

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

#: Half-width of the window when the landscape carries no measurement error,
#: in kcal/mol. Only the synthetic track uses it.
FALLBACK_HALF_WIDTH = 0.10

#: How the window centre is placed. ``reachable`` consults the constraint;
#: ``naive`` is the original rule, kept so the correction is auditable.
WINDOW_RULES = ("reachable", "naive")


def _single_measurements(landscape):
    """Every single substitution the library offers, measured."""
    out = []
    for position in landscape.positions:
        for mutation in landscape.candidate_mutations(position):
            variant = Variant((mutation,))
            if landscape.contains(variant):
                out.append(landscape.evaluate(variant))
    return out


def _additive_surrogate(landscape) -> MeasuredAdditiveLandscape:
    """A first-order model of any landscape, built from its own single mutants.

    Used to place the window and to check in advance whether it is reachable.
    On a landscape with epistasis both are approximate, but both are computed
    before any search runs and from the singles alone, so neither can be
    influenced by how the rules happen to perform.
    """
    singles = {}
    for position in landscape.positions:
        for mutation in landscape.candidate_mutations(position):
            variant = Variant((mutation,))
            if landscape.contains(variant):
                singles[variant] = landscape.evaluate(variant)
    return MeasuredAdditiveLandscape(singles, max_order=landscape.max_order)


def _exact_model(landscape):
    """The landscape itself if it is additive, else a first-order stand-in."""
    if isinstance(landscape, MeasuredAdditiveLandscape):
        return landscape, False
    return _additive_surrogate(landscape), True


def constrained_span(landscape, tau: float) -> dict:
    """The stretch of the binding axis that needs a combination and obeys tau.

    ``best_feasible_single`` is the strongest binding reached by any one
    substitution that is itself feasible. ``constrained_optimum`` is the
    strongest reached by any legal combination subject to
    ``ddg_folding <= tau``, computed exactly by
    :func:`cpd.baselines.exact_optimum`. Between them lies the part of the
    axis that one substitution cannot get to and the constraint still allows.

    If the optimum is no better than the best single, the domain cannot
    support a windowed test at this tau: combining substitutions buys nothing
    a single does not already give, so there is nowhere to put a target that
    is both combinatorial and legal. That is reported, not worked around.
    """
    model, approximate = _exact_model(landscape)
    feasible = [m for m in _single_measurements(landscape) if m.ddg_folding <= tau]
    if not feasible:
        raise SystemExit(
            f"no single substitution is feasible at tau = {tau:+.3f}; "
            "nothing can be said about a window here"
        )
    best_single = float(min(m.ddg_binding for m in feasible))
    optimum = float(exact_optimum(model, [tau])[tau])
    return {
        "best_feasible_single": best_single,
        "constrained_optimum": optimum,
        "reach_margin": best_single - optimum,
        "span_is_first_order": approximate,
    }


def choose_window(
    landscape, tau: float, rule: str = "reachable", widenings: int = 0
) -> tuple[Window, dict]:
    """Pick the target window from the data, by a rule fixed before any run.

    Two decisions, both made from the landscape and tau alone, neither tuned
    to any outcome.

    **Centre.** Under ``rule="reachable"``, the midpoint between the strongest
    feasible single substitution and the exact constrained optimum. Every
    point in that interval needs more than one substitution to reach -- so the
    problem is genuinely combinatorial -- and every point in it is allowed by
    the constraint, because the optimum is its far end. Those are the only two
    properties the window has to have, and the midpoint is the furthest a
    target can sit from both failure modes at once.

    Under ``rule="naive"``, twice the strongest single-mutant binding effect.
    That was the rule the first run of this experiment used, and it is kept
    only so the correction can be checked. It never consults tau, so on a
    domain where the constraint bites early it places the target past anything
    legal, and the only remedy left to it is to widen the window until the
    target is broad enough to overlap something reachable. That is how
    PSD95-PDZ3 ended up with a window many times wider than GB1's, which makes
    the three domains three different experiments rather than one repeated.

    **Half-width: twice the median standard error on binding.** The data's own
    resolution sets how precisely a target can be meant at all. A window
    narrower than the error bars would be asking for precision the oracle
    cannot express. The same number on every domain, so the widths compare.

    If no combination can reach the window, ``widenings`` doubles the
    half-width that many times. The count is reported, because a window that
    needed widening is a weaker test than one that did not.
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"unknown window rule {rule!r}; expected one of {WINDOW_RULES}")

    singles = _single_measurements(landscape)
    strongest = float(min(m.ddg_binding for m in singles))
    median_se = float(np.median([m.ddg_binding_se or 0.0 for m in singles]))
    half = 2.0 * median_se if median_se > 0 else FALLBACK_HALF_WIDTH
    half *= 2.0 ** widenings

    meta = {
        "rule": rule,
        "strongest_single": strongest,
        "median_binding_se": median_se,
        "half_width": half,
        "widenings": widenings,
    }

    if rule == "naive":
        centre = 2.0 * strongest
    else:
        span = constrained_span(landscape, tau)
        meta.update(span)
        if span["reach_margin"] <= 0.0:
            raise SystemExit(
                f"at tau = {tau:+.3f} the constrained optimum "
                f"({span['constrained_optimum']:+.3f}) is no better than the best "
                f"feasible single ({span['best_feasible_single']:+.3f}); this domain "
                "cannot support a windowed test at this tau"
            )
        centre = 0.5 * (span["best_feasible_single"] + span["constrained_optimum"])

    meta["centre"] = centre
    return Window(centre - half, centre + half), meta


def attainable_window(landscape, tau: float, rule: str = "reachable",
                      max_widenings: int = 4):
    """Widen the window by the declared rule until something can reach it.

    A window nothing can reach is not a hard test, it is a broken one: every
    rule fails identically and the comparison says nothing. So reachability is
    checked before any search runs, exactly on the additive track and against
    a first-order surrogate elsewhere, and the window is widened by doubling
    its half-width until the check passes. The number of doublings is
    reported, because a window that needed several is a weaker test than one
    that needed none.

    Under the ``reachable`` rule the centre is by construction inside the
    constrained-attainable range, so widening should not be needed, and a
    nonzero count means the attainable set is sparse near the target rather
    than that the target was misplaced.

    The reachability check bins the binding axis, and its error is bounded by
    ``max_order * bin_width / 2``. The bin width is therefore tied to the
    half-width rather than fixed, so the check stays sharp for the narrow
    windows the ``reachable`` rule produces.
    """
    model, approximate = _exact_model(landscape)
    window, meta = choose_window(landscape, tau, rule, 0)
    for widenings in range(max_widenings + 1):
        window, meta = choose_window(landscape, tau, rule, widenings)
        bin_width = max(min(0.01, meta["half_width"] / 4.0), 1e-3)
        best = exact_optimum_window(model, window, [tau], bin_width=bin_width)[tau]
        meta["exact_distance"] = best
        meta["check_is_first_order"] = approximate
        meta["bin_width"] = bin_width
        if not np.isnan(best) and best <= 1e-9:
            return window, meta
    return window, meta


def run_regime(landscape, target, tau, budget, seeds, n_return):
    """Every rule under one target, at one tau, over the seeds."""
    rows = []
    for seed in seeds:
        config = search_config(seed, budget)
        for name, method in default_methods(target=target).items():
            run = method(landscape, float(tau), budget, n_return, config)
            delivered = run.measurements
            rows.append({
                "rule": name,
                "seed": seed,
                "best_cost": best_feasible_cost(delivered, float(tau), target),
                "hit_rate": target_hit_rate(delivered, float(tau), target),
                "n_evaluations": run.n_evaluations,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape", default="additive", choices=("additive", "synthetic"))
    parser.add_argument("--domain", default="GRB2-SH3")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--n-return", type=int, default=DEFAULT_N_RETURN)
    parser.add_argument(
        "--rule", default="reachable", choices=WINDOW_RULES,
        help="how the window centre is placed; naive reproduces the first run",
    )
    args = parser.parse_args(argv)

    from cpd.metrics import compare_to_constant

    landscape = build_landscape(args.landscape, domain=args.domain)
    label = args.domain if args.landscape == "additive" else "synthetic"
    seeds = list(range(args.seeds))
    # The tightest tau the sweep uses, where the constraint actually bites.
    tau = float(tau_grid(landscape)[-1])

    window, meta = attainable_window(landscape, tau, args.rule)
    print(f"\n{'=' * 78}\n{label}  ({args.landscape} track), tau = {tau:+.3f}, "
          f"{args.rule} window\n{'=' * 78}")
    print(f"  strongest single binder   {meta['strongest_single']:+.3f} kcal/mol")
    print(f"  median binding std error   {meta['median_binding_se']:.3f}")
    if args.rule == "reachable":
        print(f"  strongest feasible single {meta['best_feasible_single']:+.3f}")
        print(f"  constrained optimum       {meta['constrained_optimum']:+.3f}  "
              f"(a combination buys {meta['reach_margin']:.3f} over the best single)")
    placement = ("midpoint of the constrained span" if args.rule == "reachable"
                 else "2x strongest single, tau ignored")
    print(f"  window                    [{window.lo:+.3f}, {window.hi:+.3f}]  "
          f"width {window.hi - window.lo:.3f}  ({placement}, "
          f"half-width 2x median SE"
          + (f", widened {meta['widenings']}x" if meta["widenings"] else "") + ")")
    if "exact_distance" in meta:
        reach = meta["exact_distance"]
        kind = "first-order check" if meta.get("check_is_first_order") else "exact"
        ok = (not np.isnan(reach)) and reach <= 1e-9
        print(f"  reachable ({kind})?      "
              f"{'yes' if ok else f'NO, best distance {reach:.3f}'}")

    results, summary = [], []
    for target in (Minimize(), Window(window.lo, window.hi)):
        regime = "minimize" if isinstance(target, Minimize) else "window"
        frame = run_regime(landscape, target, tau, args.budget, seeds, args.n_return)
        frame["regime"] = regime
        frame["domain"] = label
        frame["tau"] = tau
        results.append(frame)

        if regime == "minimize":
            baseline = greedy_feasible(landscape, tau)
            baseline_cost = baseline.ddg_binding
        else:
            baseline = greedy_window(landscape, target, tau, budget=args.budget)
            baseline_cost = (
                float("nan") if baseline.measurement is None
                else target.distance(baseline.ddg_binding)
            )

        means = frame.groupby("rule")["best_cost"].mean()
        usable = means.dropna()
        if usable.empty:
            print(f"\n  {regime}: no rule produced a feasible design")
            continue
        best_rule = usable.idxmin()
        per_seed = frame[frame["rule"] == best_rule]["best_cost"].tolist()
        sep = compare_to_constant(per_seed, baseline_cost)

        print(f"\n  --- {regime} ---")
        print(f"  {'rule':<24} {'best cost':>10} {'on-target rate':>15}")
        for rule in RULES:
            sub = frame[frame["rule"] == rule]
            if sub.empty:
                continue
            print(f"  {DISPLAY_NAMES.get(rule, rule):<24} {sub['best_cost'].mean():>10.3f} "
                  f"{sub['hit_rate'].mean():>14.0%}")
        print(f"  {'greedy (fair baseline)':<24} {baseline_cost:>10.3f} "
              f"{'-' if np.isnan(baseline_cost) else f'{int(baseline_cost <= 1e-9):>13.0%}'}")
        verdict = (
            "search wins" if sep.separated and sep.gap > 0
            else "SORT WINS" if sep.separated and sep.gap < 0
            else "not separated"
        )
        print(f"  best rule {DISPLAY_NAMES.get(best_rule, best_rule)}: "
              f"gap {sep.gap:+.3f}, p = {sep.p_value:.4f} -> {verdict}")

        summary.append({
            "domain": label, "track": args.landscape, "regime": regime, "tau": tau,
            "window_rule": args.rule,
            "window_lo": window.lo, "window_hi": window.hi,
            "window_width": window.hi - window.lo,
            "widenings": meta["widenings"],
            "best_feasible_single": meta.get("best_feasible_single", float("nan")),
            "constrained_optimum": meta.get("constrained_optimum", float("nan")),
            "best_rule": best_rule, "best_rule_cost": float(usable.min()),
            "greedy_cost": baseline_cost, "greedy_calls": baseline.n_evaluations,
            "gap": sep.gap, "p_value": sep.p_value,
            "beats_greedy": bool(sep.separated and sep.gap > 0),
            "loses_to_greedy": bool(sep.separated and sep.gap < 0),
        })

    stem = f"windows_{args.landscape}"
    if args.landscape == "additive":
        stem += f"_{args.domain}"
    if args.rule != "reachable":
        stem += f"_{args.rule}"
    save_results(pd.concat(results, ignore_index=True), stem)
    path = save_results(pd.DataFrame(summary), f"{stem}_summary")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    if len(summary) == 2:
        unbounded, windowed = summary[0], summary[1]
        print(f"\n{'=' * 78}\nverdict\n{'=' * 78}")
        print(f"  unbounded: {'search wins' if unbounded['beats_greedy'] else 'search does not beat the sort'}"
              f"  (p = {unbounded['p_value']:.3f})")
        print(f"  window:    {'search wins' if windowed['beats_greedy'] else 'search does not beat the sort'}"
              f"  (p = {windowed['p_value']:.3f})")
        if windowed["beats_greedy"] and not unbounded["beats_greedy"]:
            print("  -> the search advantage returns once the objective has a target range.")
        elif not windowed["beats_greedy"]:
            print("  -> the window does not rescue the search here. Reported as found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
