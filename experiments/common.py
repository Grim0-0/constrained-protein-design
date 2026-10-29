"""Shared setup for the experiments: landscapes, grids, replicates, result IO.

Two things here are deliberate and worth reading before changing them.

**The budget must bind.** Every method is compared at equal oracle cost, so
``n_generations`` is set high enough that the run ends because the budget ran
out, not because the generations did. :func:`check_budget_binds` asserts it,
because if generations run out first the comparison is silently unfair.

**Replicates, not single runs.** Evolutionary search is stochastic. Every
(method, tau) cell is run over several seeds and the figures show the spread.
A single-seed curve would be a story, not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Repository root, resolved relative to this file so nothing depends on the
#: working directory the script is launched from.
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Put the package on the path before importing it, so a fresh clone can run the
# experiments with or without `pip install -e .`.
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cpd.landscape import Landscape, SyntheticLandscape  # noqa: E402
from cpd.methods import DesignRun, default_methods  # noqa: E402
from cpd.metrics import feasibility_rate, summarize  # noqa: E402
from cpd.search import SearchConfig  # noqa: E402
from cpd.variants import Variant  # noqa: E402

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_BUDGET = 1500
DEFAULT_N_RETURN = 20
DEFAULT_K = 10


# --------------------------------------------------------------------------
# Landscapes
# --------------------------------------------------------------------------


def build_synthetic(seed: int = 5) -> SyntheticLandscape:
    """The reference synthetic landscape used by every experiment on Track 0."""
    return SyntheticLandscape(
        n_positions=20,
        n_interface=6,
        n_core=5,
        n_coupled_pairs=24,
        coupling_scale=0.35,
        max_order=5,
        seed=seed,
    )


def build_landscape(
    kind: str, domain: str = "GRB2-SH3", seed: int = 5, max_order: int = 4
) -> Landscape:
    """Build one of the three oracles.

    ``synthetic`` is the controlled toy used by the tests; ``exact`` is Track 1
    over the measured ddPCA singles; ``additive`` is Track 2, combinations of
    those singles under a first-order model.
    """
    if kind == "synthetic":
        return build_synthetic(seed=seed)
    if kind in ("exact", "additive"):
        from data.download import load_domain  # imported late; needs the downloaded data

        return load_domain(domain, track=kind, max_order=max_order)
    raise ValueError(
        f"unknown landscape kind {kind!r}; expected 'synthetic', 'exact' or 'additive'"
    )


# --------------------------------------------------------------------------
# Grids derived from the landscape itself
# --------------------------------------------------------------------------


def _single_mutant_measurements(landscape: Landscape) -> np.ndarray:
    """``(n, 2)`` array of single-mutant energies, used to calibrate grids."""
    rows = []
    for position in landscape.positions:
        for mutation in landscape.candidate_mutations(position):
            variant = Variant((mutation,))
            if landscape.contains(variant):
                m = landscape.evaluate(variant)
                rows.append((m.ddg_binding, m.ddg_folding))
    if not rows:
        raise ValueError("landscape exposes no single mutants to calibrate against")
    return np.asarray(rows, dtype=float)


def tau_grid(landscape: Landscape, n: int = 7, loose_q: float = 0.90, tight_q: float = 0.05) -> np.ndarray:
    """Constraint tightness from loose to tight, in the landscape's own units.

    Quantiles of the single-mutant folding distribution, so the grid spans a
    range where the constraint actually bites rather than arbitrary kcal/mol
    values that might all be trivial or all be impossible.
    """
    folding = _single_mutant_measurements(landscape)[:, 1]
    loose, tight = np.quantile(folding, [loose_q, tight_q])
    return np.linspace(float(loose), float(tight), n)


def reference_point(landscape: Landscape, quantile: float = 0.95) -> tuple[float, float]:
    """Hypervolume reference, held fixed across every method and every tau.

    Hypervolume numbers are only comparable against a shared reference, so this
    is computed once from the landscape and reused everywhere.
    """
    singles = _single_mutant_measurements(landscape)
    ref = np.quantile(singles, quantile, axis=0)
    return (float(ref[0]), float(ref[1]))


def search_config(seed: int, budget: int = DEFAULT_BUDGET) -> SearchConfig:
    """Shared engine settings. Generations are set so the budget is what binds."""
    population_size = 32
    n_offspring = 32
    # Generous enough that the oracle budget, not the generation count, ends the
    # run. The headroom is large because a converged population re-proposes
    # cached variants, which are free and so make no progress against the budget
    # -- most of all on Track 1, where the whole library is a few hundred designs.
    n_generations = max(40, int(12 * budget / n_offspring))
    return SearchConfig(
        population_size=population_size,
        n_generations=n_generations,
        n_offspring=n_offspring,
        tournament_size=3,
        init_order=1,
        seed=seed,
    )


def check_budget_binds(runs: Iterable[DesignRun], budget: int) -> None:
    """Fail loudly if any method stopped before spending its budget."""
    underspent = {
        run.method: run.n_evaluations for run in runs if run.n_evaluations < budget
    }
    if underspent:
        raise RuntimeError(
            "these methods did not spend the full oracle budget, so the comparison "
            f"is not at equal cost: {underspent} (budget {budget}). "
            "Raise n_generations in search_config()."
        )


# --------------------------------------------------------------------------
# Running a grid of (method, tau, seed)
# --------------------------------------------------------------------------


def run_grid(
    landscape: Landscape,
    taus: Sequence[float],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    budget: int = DEFAULT_BUDGET,
    n_return: int = DEFAULT_N_RETURN,
    k: int = DEFAULT_K,
    reference: tuple[float, float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every method at every tau over every seed and tabulate the metrics."""
    reference = reference or reference_point(landscape)
    methods = default_methods()
    rows: list[dict] = []

    for tau in taus:
        for seed in seeds:
            config = search_config(seed, budget)
            runs = []
            for name, method in methods.items():
                run = method(landscape, float(tau), budget, n_return, config)
                runs.append(run)
                # Both readouts come from the same paid-for trace; see
                # DesignRun.filtered for why both are reported.
                readouts = {
                    "as_delivered": run.measurements,
                    "feasible_filtered": run.filtered(n_return),
                }
                searched = feasibility_rate(run.trace_measurements, float(tau))
                for readout, measurements in readouts.items():
                    report = summarize(measurements, float(tau), reference, k=k)
                    rows.append(
                        {
                            "method": name,
                            "readout": readout,
                            "seed": seed,
                            **report.as_dict(),
                            "search_feasibility_rate": searched,
                            "design_yield": len(measurements) / n_return,
                            "n_evaluations": run.n_evaluations,
                            "final_lambda": (
                                run.objective_log[-1].get("lam") if run.objective_log else None
                            ),
                        }
                    )
            check_budget_binds(runs, budget)
        if verbose:
            print(f"  tau = {tau:+.3f}  done ({len(seeds)} seeds x {len(methods)} methods)")

    frame = pd.DataFrame(rows)
    # reference is a tuple per row; split it so the CSV stays flat and readable.
    frame[["reference_binding", "reference_folding"]] = pd.DataFrame(
        frame["reference"].tolist(), index=frame.index
    )
    return frame.drop(columns=["reference"])


def aggregate(frame: pd.DataFrame, value: str, readout: str = "feasible_filtered") -> pd.DataFrame:
    """Mean, standard deviation and seed count per (method, tau) for one metric.

    NaNs are counted, not dropped silently: ``n_valid`` says how many seeds
    actually produced a number, which for the feasible-only metrics is the
    difference between "this method scored well" and "this method mostly
    returned nothing".
    """
    if "readout" in frame.columns:
        frame = frame[frame["readout"] == readout]
    grouped = frame.groupby(["method", "tau"])[value]
    out = grouped.agg(mean="mean", std="std", n_valid="count").reset_index()
    out["n_seeds"] = grouped.size().values
    return out


# --------------------------------------------------------------------------
# Result IO
# --------------------------------------------------------------------------


def save_results(frame: pd.DataFrame, name: str, meta: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    if meta is not None:
        (RESULTS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--landscape",
        choices=("synthetic", "exact", "additive"),
        default="synthetic",
        help="toy landscape, Track 1 measured singles, or Track 2 additive model",
    )
    parser.add_argument("--domain", default="GRB2-SH3", help="domain to use when --landscape exact")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="oracle calls per method")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS), help="number of replicates")
    parser.add_argument("--n-return", type=int, default=DEFAULT_N_RETURN, help="designs returned per method")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k for feasible best-k")
    return parser
