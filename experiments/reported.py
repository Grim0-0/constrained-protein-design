"""The numbers the README quotes, in machine-readable form.

    python experiments/reported.py          # regenerate reported_numbers.json

This file exists because the README drifted. Three figures in it were produced
by a parameter setting the committed code no longer used, and one was copied
from a script that was never committed at all. Nothing caught it, because the
README was the only place those numbers lived.

So they now live here too. :func:`collect` reads ``results/`` and writes
``reported_numbers.json`` at the repository root -- committed, unlike
``results/`` itself, because it is the record of what the prose claims.
``tests/test_reported_numbers.py`` parses the README and fails if any quoted
figure disagrees with this file.

That makes drift a test failure in both directions:

* change the code, re-run the experiments, re-run this -- the JSON moves, and
  the test fails until the README is updated to match;
* edit a number in the README by hand -- the test fails immediately.

The JSON is small and diffable on purpose. A reviewer can see in one ``git
diff`` which reported figures a change moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_N_RETURN,
    DEFAULT_SEEDS,
    ROOT,
    RESULTS_DIR,
)

#: Where the committed record lives. Not under results/, which is gitignored.
REPORTED_PATH = ROOT / "reported_numbers.json"

TRACKS = ("additive", "synthetic")


def _verdict_rows(track: str) -> dict:
    path = RESULTS_DIR / f"baselines_verdict_{track}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for _, row in frame.iterrows():
        out[str(row["domain"])] = {
            "exact_optimum": None if np.isnan(row["exact_optimum"]) else round(float(row["exact_optimum"]), 3),
            "greedy": round(float(row["greedy"]), 3),
            "greedy_calls": int(row["greedy_calls"]),
            "forward": round(float(row["forward"]), 3),
            "forward_calls": int(row["forward_calls"]),
            "best_rule": str(row["best_rule"]),
            "best_rule_mean": round(float(row["best_rule_mean"]), 3),
            "toughest_baseline": str(row["toughest_baseline"]),
            "gap": round(float(row["gap_to_greedy"]), 3),
            "p_value": round(float(row["p_value"]), 3),
            "beats_greedy": bool(row["beats_greedy"]),
            "loses_to_greedy": bool(row["loses_to_greedy"]),
        }
    return out


def _budgets_used() -> list[int]:
    """Every oracle budget any committed result was produced at.

    The scripts deliberately differ -- the tuning sweep runs many more
    configurations so it gets less each, and Track 1's measured library is too
    small to spend the default on. The README may quote any of these and no
    others, which is what the test checks.
    """
    budgets = set()
    for path in sorted(RESULTS_DIR.glob("*_meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "budget" in meta:
            budgets.add(int(meta["budget"]))
    budgets.add(DEFAULT_BUDGET)
    return sorted(budgets)


def _window_rows() -> dict:
    """The two-regime comparison, per domain, from experiments/windows.py."""
    out: dict = {}
    for path in sorted(RESULTS_DIR.glob("windows_*_summary.csv")):
        frame = pd.read_csv(path)
        for domain, group in frame.groupby("domain"):
            entry = out.setdefault(str(domain), {})
            first = group.iloc[0]
            entry["window_lo"] = round(float(first["window_lo"]), 3)
            entry["window_hi"] = round(float(first["window_hi"]), 3)
            entry["widenings"] = int(first["widenings"])
            for _, row in group.iterrows():
                entry[str(row["regime"])] = {
                    "best_rule": str(row["best_rule"]),
                    "best_rule_cost": round(float(row["best_rule_cost"]), 3),
                    "greedy_cost": round(float(row["greedy_cost"]), 3),
                    "gap": round(float(row["gap"]), 3),
                    "p_value": round(float(row["p_value"]), 3),
                    "beats_greedy": bool(row["beats_greedy"]),
                }
    return out


def _uncertainty_rows() -> dict:
    """Nominal vs expected vs chance-constrained feasibility, per domain and rule."""
    out: dict = {}
    for path in sorted(RESULTS_DIR.glob("uncertainty_summary_*.csv")):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            entry = out.setdefault(str(row["domain"]), {})
            entry[str(row["rule"])] = {
                "nominal": round(float(row["nominal"]), 3),
                "expected": round(float(row["expected"]), 3),
                "confident_0.05": round(float(row["confident_0.05"]), 3),
                "confident_0.01": round(float(row["confident_0.01"]), 3),
            }
    return out


def _diversity_rows() -> dict:
    """How spread out each rule's delivered set is, per domain."""
    out: dict = {}
    for path in sorted(RESULTS_DIR.glob("diversity_summary_*.csv")):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            out.setdefault(str(row["domain"]), {})[str(row["rule"])] = {
                "filled": round(float(row["filled"]), 3),
                "mean_distance": round(float(row["mean_distance"]), 2),
                "positions_used": round(float(row["positions_used"]), 1),
                "n_feasible": round(float(row["n_feasible"]), 1),
                "feasible_mean_distance": (
                    None if np.isnan(row["feasible_mean_distance"])
                    else round(float(row["feasible_mean_distance"]), 2)
                ),
                "feasible_positions_used": round(float(row["feasible_positions_used"]), 1),
            }
    return out


def collect() -> dict:
    """Gather every figure the README quotes, from the committed result files."""
    record: dict = {
        "settings": {
            "budget": DEFAULT_BUDGET,
            "budgets_used": _budgets_used(),
            "n_seeds": len(DEFAULT_SEEDS),
            "n_designs_returned": DEFAULT_N_RETURN,
        },
        "baselines": {track: _verdict_rows(track) for track in TRACKS},
        "windows": _window_rows(),
        "uncertainty": _uncertainty_rows(),
        "diversity": _diversity_rows(),
    }
    missing = [t for t, rows in record["baselines"].items() if not rows]
    if missing:
        raise SystemExit(
            f"no baseline verdicts for {missing}; run experiments/baselines.py "
            "for those tracks before regenerating the record"
        )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the file on disk is stale, without rewriting it",
    )
    args = parser.parse_args(argv)

    record = collect()
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"

    if args.check:
        current = REPORTED_PATH.read_text(encoding="utf-8") if REPORTED_PATH.exists() else ""
        if current != text:
            print(f"{REPORTED_PATH.name} is stale; re-run experiments/reported.py")
            return 1
        print(f"{REPORTED_PATH.name} is up to date")
        return 0

    REPORTED_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {REPORTED_PATH.relative_to(ROOT)}")
    for track, rows in record["baselines"].items():
        for domain, values in rows.items():
            print(f"  {track:<10} {domain:<12} best rule {values['best_rule_mean']:+.3f}  "
                  f"greedy {values['greedy']:+.3f}  p {values['p_value']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
