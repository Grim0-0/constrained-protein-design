"""Every figure the README quotes must match the record the code produced.

This file exists because the README drifted and nothing caught it. Three
reported figures came from a parameter setting the committed code no longer
used, and one was copied from a script that was never committed. The prose was
the only place those numbers lived, so there was nothing to disagree with.

Now ``experiments/reported.py`` writes ``reported_numbers.json`` from
``results/``, and these tests parse the README and compare. Drift fails in
both directions: regenerate the record after a code change and these fail
until the prose is updated; hand-edit a number in the prose and they fail at
once.

They need no data and no network -- both the README and the record are
committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RECORD = ROOT / "reported_numbers.json"

#: README prose uses a typographic minus; the record uses ASCII.
MINUS = "−"

#: A README figure is printed to 2 or 3 decimals, so it may differ from the
#: stored value by up to half of the last place shown.
TOLERANCE = 0.0051


@pytest.fixture(scope="module")
def record() -> dict:
    assert RECORD.exists(), (
        f"{RECORD.name} is missing; run `python experiments/reported.py` to write it"
    )
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def _number(cell: str) -> float | None:
    """Pull a signed float out of one markdown table cell, or None if absent."""
    text = cell.replace(MINUS, "-").replace("**", "").replace("`", "").strip()
    match = re.search(r"[-+]?\d+\.\d+", text)
    return float(match.group()) if match else None


def _table_rows(text: str, first_header: str) -> dict[str, list[str]]:
    """Rows of the markdown table whose header line starts with ``first_header``.

    Returns a mapping from the first cell to the remaining cells.
    """
    rows: dict[str, list[str]] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(f"| {first_header} |"):
            continue
        for body in lines[i + 2:]:
            if not body.startswith("|"):
                break
            cells = [c.strip() for c in body.strip().strip("|").split("|")]
            if cells and not set(cells[0]) <= {"-", ":"}:
                rows[cells[0]] = cells[1:]
        break
    return rows


# --------------------------------------------------------------------------
# Settings quoted in prose
# --------------------------------------------------------------------------


def test_the_readme_quotes_the_budget_the_results_were_made_with(readme, record):
    budget = record["settings"]["budget"]
    assert f"{budget} oracle calls" in readme, (
        f"the README should say '{budget} oracle calls'; the results were produced "
        "at that budget. This is the exact mismatch that made the documented "
        "commands irreproducible once before."
    )
    # Scripts deliberately use different budgets, but the README must not
    # advertise one that nothing was actually run at.
    allowed = {str(b) for b in record["settings"]["budgets_used"]}
    quoted = set(re.findall(r"(\d{2,5}) oracle calls", readme))
    assert quoted <= allowed, (
        f"README quotes budgets nothing was run at: {sorted(quoted - allowed)}; "
        f"results exist for {sorted(allowed)}"
    )


def test_the_readme_quotes_the_seed_count_that_was_run(readme, record):
    assert f"{record['settings']['n_seeds']} seeds" in readme


def test_the_readme_quotes_the_number_of_designs_returned(readme, record):
    # The prose wraps, so compare on whitespace-normalised text.
    flat = " ".join(readme.split())
    assert f"{record['settings']['n_designs_returned']} designs returned" in flat


# --------------------------------------------------------------------------
# The baselines table
# --------------------------------------------------------------------------


def test_every_domain_in_the_record_has_a_row_in_the_baselines_table(readme, record):
    rows = _table_rows(readme, "domain")
    assert rows, "could not find the baselines table in the README"
    for domain in record["baselines"]["additive"]:
        assert domain in rows, f"{domain} is in the record but has no README row"


@pytest.mark.parametrize("domain", ["GB1", "GRB2-SH3", "PSD95-PDZ3"])
def test_the_baselines_table_matches_the_record(readme, record, domain):
    """Column by column, the prose must equal what the code emitted."""
    stored = record["baselines"]["additive"][domain]
    cells = _table_rows(readme, "domain")[domain]
    # exact optimum | rank once | forward | best rule | gap | p | verdict
    expected = [
        ("exact optimum", stored["exact_optimum"]),
        ("greedy rank-once", stored["greedy"]),
        ("greedy forward", stored["forward"]),
        ("best rule", stored["best_rule_mean"]),
        ("gap", stored["gap"]),
        ("p value", stored["p_value"]),
    ]
    for index, (label, want) in enumerate(expected):
        got = _number(cells[index])
        assert got is not None, f"{domain}: no number in the {label} cell"
        assert abs(got - want) <= TOLERANCE, (
            f"{domain} {label}: README says {got}, the record says {want}. "
            "Re-run experiments/reported.py and update the README."
        )


def test_the_verdict_wording_matches_the_significance_result(readme, record):
    """A domain the sort wins must not be described as tied, or vice versa."""
    rows = _table_rows(readme, "domain")
    for domain, stored in record["baselines"]["additive"].items():
        verdict = rows[domain][-1].lower()
        if stored["loses_to_greedy"]:
            assert "sort wins" in verdict, (
                f"{domain}: the sort beats the search (p = {stored['p_value']}) "
                f"but the README says {verdict!r}"
            )
        elif stored["beats_greedy"]:
            assert "search" in verdict
        else:
            assert "tied" in verdict or "not separated" in verdict, (
                f"{domain}: not separated, but the README says {verdict!r}"
            )


# --------------------------------------------------------------------------
# The synthetic block
# --------------------------------------------------------------------------


def _labelled(readme: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s+(-?\d+\.\d+)", readme)
    assert match, f"could not find {label!r} in the README"
    return float(match.group(1))


def test_the_synthetic_block_matches_the_record(readme, record):
    stored = record["baselines"]["synthetic"]["synthetic"]
    assert _labelled(readme, "best rule (Lagrangian)") == pytest.approx(
        stored["best_rule_mean"], abs=TOLERANCE
    )
    assert _labelled(readme, "greedy, rank once") == pytest.approx(
        stored["greedy"], abs=TOLERANCE
    )
    assert _labelled(readme, "greedy, forward") == pytest.approx(
        stored["forward"], abs=TOLERANCE
    )

    gap = re.search(r"gap \+(\d+\.\d+) kcal/mol, p = (\d+\.\d+)", readme)
    assert gap, "could not find the synthetic gap/p line in the README"
    assert float(gap.group(1)) == pytest.approx(stored["gap"], abs=TOLERANCE)
    assert float(gap.group(2)) == pytest.approx(stored["p_value"], abs=TOLERANCE)


#: Words that would oversell a p = 0.05 result, and words that honestly hedge it.
OVERSTATED = ("decisive", "decisively", "comfortably", "convincingly", "robustly")
HEDGED = ("marginal", "only just", "slim", "by a hair", "barely", "narrow")


def _section(readme: str, heading: str) -> str:
    """The text under one heading, up to the next heading of any level."""
    lines = readme.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == heading:
            out = []
            for body in lines[i + 1:]:
                if body.startswith("#"):
                    break
                out.append(body)
            return chr(10).join(out)
    raise AssertionError(f"heading {heading!r} not found in the README")


def test_the_synthetic_claim_is_not_overstated(readme, record):
    """The one place the search wins is marginal; that section has to say so.

    Scoped to the section that makes the claim, not the whole document. A
    document-wide keyword search passes as soon as any hedge word appears
    anywhere, which is not the same as hedging the claim in question -- it let
    an edit from "only just" to "decisively" through unnoticed.
    """
    stored = record["baselines"]["synthetic"]["synthetic"]
    assert stored["beats_greedy"], "the record no longer shows a synthetic win"
    if stored["p_value"] <= 0.01:
        return
    body = _section(readme, "### The cause is the oracle, not the rules").lower()
    assert any(word in body for word in HEDGED), (
        f"the synthetic win is p = {stored['p_value']}, which is marginal, but the "
        "section claiming it uses none of: " + ", ".join(HEDGED)
    )
    overstated = [word for word in OVERSTATED if word in body]
    assert not overstated, (
        f"the synthetic win is p = {stored['p_value']}, but the section describes "
        f"it as {overstated}"
    )


# --------------------------------------------------------------------------
# The record itself
# --------------------------------------------------------------------------


def test_the_record_is_sorted_and_diffable():
    """A reviewer should see which figures a change moved, in one git diff."""
    raw = RECORD.read_text(encoding="utf-8")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n", (
        "reported_numbers.json is not in canonical form; "
        "re-run experiments/reported.py"
    )
