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


#: Rule keys as the prose spells them.
RULE_DISPLAY = {
    "lagrangian": "Lagrangian",
    "fixed_penalty": "Fixed penalty",
    "scalarization_w0.5": "Scalarization (w=0.5)",
    "scalarization_sweep": "Scalarization",
    "optimize_then_reject": "Optimize-then-reject",
    "nsga2": "NSGA-II",
}


def test_the_synthetic_block_matches_the_record(readme, record):
    stored = record["baselines"]["synthetic"]["synthetic"]
    # The winning rule is read from the record, not hardcoded, so a change of
    # winner shows up as a prose mismatch rather than silently passing.
    label = f"best rule ({RULE_DISPLAY[stored['best_rule']]})"
    assert _labelled(readme, label) == pytest.approx(
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


# --------------------------------------------------------------------------
# The two-regime window table
# --------------------------------------------------------------------------

WINDOW_HEADER = (
    "| domain | window | width | widened | regime | best rule | greedy | gap "
    "| *p* | verdict |"
)

#: The table comparing the superseded window rule against the corrected one.
WINDOW_RULE_HEADER = (
    "| domain | old window | width | widened | new window | width | widened |"
)


def _window_table(readme: str) -> dict[tuple[str, str], list[str]]:
    """Rows of the window table, keyed by (domain, regime).

    The table repeats the domain only on its first row, so the domain carries
    down until the next one names itself.
    """
    lines = readme.splitlines()
    start = lines.index(WINDOW_HEADER)
    rows: dict[tuple[str, str], list[str]] = {}
    domain = ""
    for body in lines[start + 2:]:
        if not body.startswith("|"):
            break
        cells = [c.strip() for c in body.strip().strip("|").split("|")]
        if cells[0]:
            domain = cells[0]
        regime = cells[4].replace("**", "").strip()
        rows[(domain, regime)] = cells
    return rows


def test_the_window_table_lists_every_domain_in_the_record(readme, record):
    rows = _window_table(readme)
    for domain in record["windows"]:
        for regime in ("unbounded", "window"):
            assert (domain, regime) in rows, f"{domain}/{regime} missing from the window table"


@pytest.mark.parametrize("domain", ["GB1", "GRB2-SH3", "PSD95-PDZ3", "synthetic"])
def test_the_window_table_matches_the_record(readme, record, domain):
    stored = record["windows"][domain]
    rows = _window_table(readme)

    # The window, its width and the widening count, stated once per domain.
    first = rows[(domain, "unbounded")]
    bounds = re.findall(r"[-+]?\d+\.\d+", first[1].replace(MINUS, "-"))
    assert len(bounds) == 2, f"{domain}: could not read the window bounds"
    assert float(bounds[0]) == pytest.approx(stored["window_lo"], abs=TOLERANCE)
    assert float(bounds[1]) == pytest.approx(stored["window_hi"], abs=TOLERANCE)
    assert _number(first[2]) == pytest.approx(stored["window_width"], abs=TOLERANCE)
    assert first[3].rstrip("x×") == str(stored["widenings"])

    for regime, key in (("unbounded", "minimize"), ("window", "window")):
        cells = rows[(domain, regime)]
        want = stored[key]
        for index, name in ((5, "best_rule_cost"), (6, "greedy_cost"), (7, "gap"), (8, "p_value")):
            got = _number(cells[index])
            assert got is not None, f"{domain}/{regime}: no number in the {name} cell"
            assert abs(got - want[name]) <= TOLERANCE, (
                f"{domain}/{regime} {name}: README says {got}, record says {want[name]}"
            )


def test_the_window_verdicts_match_the_significance_results(readme, record):
    rows = _window_table(readme)
    for domain, stored in record["windows"].items():
        for regime, key in (("unbounded", "minimize"), ("window", "window")):
            verdict = rows[(domain, regime)][-1].replace("**", "").lower()
            if stored[key]["beats_greedy"]:
                assert "search wins" in verdict, (
                    f"{domain}/{regime}: the search separates "
                    f"(p = {stored[key]['p_value']}) but the README says {verdict!r}"
                )
            else:
                assert "search wins" not in verdict, (
                    f"{domain}/{regime}: not separated "
                    f"(p = {stored[key]['p_value']}) but the README claims a win"
                )


MEASURED = ("GB1", "GRB2-SH3", "PSD95-PDZ3")


def _flipped(record) -> list[str]:
    """"Flips" means a specific thing: not separated unbounded, separated windowed."""
    return [
        d for d in MEASURED
        if not record["windows"][d]["minimize"]["beats_greedy"]
        and record["windows"][d]["window"]["beats_greedy"]
    ]


def _claimed(readme: str, label: str) -> list[str]:
    """The domains the README names after ``label``, in its summary line."""
    match = re.search(rf"\*\*{label}:\*\* ([^.]*)\.", readme)
    assert match, f"the window section states no {label!r} line"
    return [d.strip() for d in match.group(1).split(",") if d.strip()]


def test_the_window_rescues_exactly_the_domains_the_prose_claims(readme, record):
    """The prose names which domains flip; the record decides which ones do.

    An earlier version of this test asserted that all three flip. That made
    it a test of the result rather than of agreement between the prose and
    the record, so correcting the window-selection rule broke the test
    instead of the claim.
    """
    flipped = _flipped(record)
    held = [d for d in MEASURED if d not in flipped]
    assert sorted(_claimed(readme, "Flips on")) == sorted(flipped), (
        f"the record says {flipped} flip"
    )
    assert sorted(_claimed(readme, "Does not flip on")) == sorted(held), (
        f"the record says {held} do not flip"
    )
    assert "On all three measured domains the finding flips" not in " ".join(
        readme.split()
    ) or len(flipped) == 3


def test_the_synthetic_row_is_described_the_way_it_behaves(readme, record):
    """It wins unbounded, so only its windowed row can claim anything new."""
    synthetic = record["windows"]["synthetic"]
    flat = " ".join(readme.split())
    if not synthetic["window"]["beats_greedy"]:
        assert "wins under the window too" not in flat, (
            "the synthetic track does not separate under the window; the README "
            "must not say it does"
        )
    else:
        assert "wins under the window too" in flat


def _rule_table(readme: str) -> dict[str, list[str]]:
    """Rows of the old-rule/new-rule comparison, keyed by domain, cells intact."""
    lines = readme.splitlines()
    start = lines.index(WINDOW_RULE_HEADER)
    rows: dict[str, list[str]] = {}
    for body in lines[start + 2:]:
        if not body.startswith("|"):
            break
        cells = [c.strip() for c in body.strip().strip("|").split("|")]
        rows[cells[0]] = cells
    return rows


def test_the_window_rule_comparison_matches_the_record(readme, record):
    """Both rules are reported, so both have to agree with what was run."""
    rows = _rule_table(readme)
    for domain, stored in record["windows"].items():
        assert domain in rows, f"{domain} missing from the window-rule comparison"
        cells = rows[domain]
        for offset, entry in ((1, stored["naive"]), (4, stored)):
            bounds = re.findall(r"[-+]?\d+\.\d+", cells[offset].replace(MINUS, "-"))
            assert len(bounds) == 2, f"{domain}: unreadable window in column {offset}"
            assert float(bounds[0]) == pytest.approx(entry["window_lo"], abs=TOLERANCE)
            assert float(bounds[1]) == pytest.approx(entry["window_hi"], abs=TOLERANCE)
            assert _number(cells[offset + 1]) == pytest.approx(
                entry["window_width"], abs=TOLERANCE
            )
            assert cells[offset + 2].replace("**", "").rstrip("x×") == str(
                entry["widenings"]
            )


def test_the_width_disparity_the_correction_removed_is_stated_correctly(readme, record):
    """The headline number for why the old rule was broken, from the old runs."""
    widths = [e["naive"]["window_width"] for e in record["windows"].values()]
    ratio = max(widths) / min(widths)
    flat = " ".join(readme.split())
    assert f"{ratio:.0f}× wider" in flat, (
        f"the superseded rule's widest window was {ratio:.0f}x its narrowest"
    )


def test_the_corrected_rule_never_needs_widening(readme, record):
    """The README claims this as the reason to consult tau; check it holds."""
    widened = {d: e["widenings"] for d, e in record["windows"].items()
               if e["widenings"]}
    flat = " ".join(readme.split())
    if "0× on all four tracks" in flat:
        assert not widened, f"the README claims no widening, but {widened} widened"


def test_a_withdrawn_claim_is_withdrawn_in_the_open(readme):
    """Results that moved under a change get a note, not a silent edit."""
    flat = " ".join(readme.split())
    assert "**An earlier version of this section**" in flat
    assert "--rule naive" in readme, (
        "the superseded window rule must stay runnable for the correction to "
        "be checkable"
    )


# --------------------------------------------------------------------------
# The uncertainty table
# --------------------------------------------------------------------------

UNCERTAINTY_HEADER = "| rule | GB1 | GRB2-SH3 | PSD95-PDZ3 |"
DISPLAY_TO_KEY = {
    "Optimize-then-reject": "optimize_then_reject",
    "Fixed penalty": "fixed_penalty",
    "Lagrangian": "lagrangian",
    "Scalarization (`w`=0.5)": "scalarization_w0.5",
    "Scalarization (sweep)": "scalarization_sweep",
    "NSGA-II": "nsga2",
}


def _uncertainty_table(readme: str) -> dict[str, list[str]]:
    lines = readme.splitlines()
    start = lines.index(UNCERTAINTY_HEADER)
    rows = {}
    for body in lines[start + 2:]:
        if not body.startswith("|"):
            break
        cells = [c.strip() for c in body.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def _percents(cell: str) -> list[float]:
    """The nominal -> expected -> confident triple from one cell, as fractions."""
    text = cell.replace("**", "")
    return [int(v) / 100.0 for v in re.findall(r"(\d+)%", text)]


@pytest.mark.parametrize("display,key", list(DISPLAY_TO_KEY.items()))
def test_the_uncertainty_table_matches_the_record(readme, record, display, key):
    rows = _uncertainty_table(readme)
    assert display in rows, f"{display} has no row in the uncertainty table"
    for column, domain in enumerate(("GB1", "GRB2-SH3", "PSD95-PDZ3")):
        stored = record["uncertainty"][domain][key]
        got = _percents(rows[display][column])
        assert len(got) == 3, f"{display}/{domain}: expected three percentages, got {got}"
        for value, name in zip(got, ("nominal", "expected", "confident_0.05")):
            # The README rounds to whole percent, so half a point of slack,
            # plus a hair so an exact 0.005 does not fail on binary rounding.
            assert abs(value - stored[name]) <= TOLERANCE, (
                f"{display}/{domain} {name}: README says {value:.1%}, "
                f"record says {stored[name]:.1%}"
            )


def test_the_prose_names_the_rule_that_actually_degrades_most(readme, record):
    """The headline claim must follow the data, not the other way round."""
    worst = {}
    for domain, rules in record["uncertainty"].items():
        # Among rules that report a high nominal rate, which keeps least of it?
        candidates = {r: v for r, v in rules.items() if v["nominal"] >= 0.9}
        worst[domain] = min(candidates, key=lambda r: candidates[r]["confident_0.05"])
    assert worst["GB1"] == "optimize_then_reject"
    assert worst["PSD95-PDZ3"] == "optimize_then_reject"
    flat = " ".join(readme.split())
    assert "Optimize-then-reject reports a perfect 100% on all three domains" in flat

    otr = record["uncertainty"]
    low = sorted(
        otr[d]["optimize_then_reject"]["confident_0.05"] for d in ("GB1", "PSD95-PDZ3")
    )
    assert f"{low[0] * 100:.0f}–{low[1]:.0%} of it on two of them" in flat, (
        "the range quoted for optimize-then-reject no longer matches the record"
    )


def test_the_independence_assumption_is_stated(readme):
    """A propagated error that assumes independence has to say so."""
    flat = " ".join(readme.split())
    assert "quadrature, which treats them as independent" in flat
    assert "lower bound on the real ones" in flat


# --------------------------------------------------------------------------
# The diversity table
# --------------------------------------------------------------------------

DIVERSITY_RULES = {
    "Lagrangian": "lagrangian",
    "Fixed penalty": "fixed_penalty",
    "Scalarization (`w`=0.5)": "scalarization_w0.5",
    "Scalarization (sweep)": "scalarization_sweep",
    "Optimize-then-reject": "optimize_then_reject",
    "NSGA-II": "nsga2",
}


def _diversity_table(readme: str) -> dict[str, list[str]]:
    """The diversity table, distinguished from the uncertainty one by its cells.

    Both tables share a header, so the diversity table is the one whose cells
    hold a bare number rather than a percentage triple.
    """
    lines = readme.splitlines()
    for start, line in enumerate(lines):
        if line != UNCERTAINTY_HEADER:
            continue
        rows = {}
        for body in lines[start + 2:]:
            if not body.startswith("|"):
                break
            cells = [c.strip() for c in body.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
        if not any("%" in c for cells in rows.values() for c in cells):
            return rows
    raise AssertionError("could not find the diversity table in the README")


@pytest.mark.parametrize("display,key", list(DIVERSITY_RULES.items()))
def test_the_diversity_table_matches_the_record(readme, record, display, key):
    rows = _diversity_table(readme)
    assert display in rows, f"{display} has no row in the diversity table"
    for column, domain in enumerate(("GB1", "GRB2-SH3", "PSD95-PDZ3")):
        stored = record["diversity"][domain][key]["feasible_mean_distance"]
        cell = rows[display][column].replace("**", "").strip()
        if stored is None:
            assert cell in ("\u2014", "-", "none"), (
                f"{display}/{domain}: no feasible designs, so the cell must be a dash"
            )
            continue
        got = _number(cell)
        assert got is not None, f"{display}/{domain}: no number in the cell"
        assert abs(got - stored) <= 0.0051, (
            f"{display}/{domain}: README says {got}, record says {stored}"
        )


def test_the_prose_names_the_domains_where_the_suspicion_held(readme, record):
    """Whether optimize-then-reject concentrates is a finding, not an assumption."""
    peers = ("lagrangian", "fixed_penalty", "scalarization_w0.5")
    concentrates = []
    for domain, rules in record["diversity"].items():
        mine = rules["optimize_then_reject"]["feasible_mean_distance"]
        theirs = [rules[r]["feasible_mean_distance"] for r in peers]
        if mine is not None and all(t is not None for t in theirs):
            if mine < sum(theirs) / len(theirs):
                concentrates.append(domain)
    assert concentrates == ["GRB2-SH3"], (
        f"which domains optimize-then-reject concentrates on has changed: "
        f"{concentrates}; the README says GRB2-SH3 only"
    )
    flat = " ".join(readme.split())
    assert "On GRB2-SH3, yes." in flat
    assert "On GB1 and PSD95-PDZ3, no." in flat


# --------------------------------------------------------------------------
# The additivity bound
# --------------------------------------------------------------------------

ADDITIVITY_HEADER = "| domain | assay | doubles | ρ |"


def test_the_additivity_table_matches_the_record(readme, record):
    lines = readme.splitlines()
    start = lines.index(ADDITIVITY_HEADER)
    seen = set()
    for body in lines[start + 2:]:
        if not body.startswith("|"):
            break
        cells = [c.strip() for c in body.strip().strip("|").split("|")]
        domain, assay = cells[0], cells[1]
        stored = record["additivity"][domain][assay]
        seen.add((domain, assay))
        n = int(cells[2].replace(",", ""))
        assert n == stored["n_doubles"], f"{domain}/{assay}: double count differs"
        rho = _number(cells[3])
        assert rho is not None and abs(rho - stored["rho_agreement"]) <= TOLERANCE, (
            f"{domain}/{assay}: README says rho={rho}, record says "
            f"{stored['rho_agreement']}"
        )
    expected = {(d, a) for d, assays in record["additivity"].items() for a in assays}
    assert seen == expected, f"table and record disagree on rows: {seen ^ expected}"


def test_the_quoted_rho_range_matches_the_record(readme, record):
    values = [
        v["rho_agreement"]
        for assays in record["additivity"].values()
        for v in assays.values()
    ]
    flat = " ".join(readme.split())
    assert f"ρ = {min(values):.2f}–{max(values):.2f} at order 2" in flat, (
        f"the quoted rho range no longer matches the record "
        f"({min(values):.2f}-{max(values):.2f})"
    )


def test_the_readme_admits_the_headline_domain_is_the_weakest(readme, record):
    """If that stops being true the prose must change, not stay flattering."""
    worst = min(
        ((d, a, v["rho_agreement"]) for d, assays in record["additivity"].items()
         for a, v in assays.items()),
        key=lambda row: row[2],
    )
    assert worst[:2] == ("GRB2-SH3", "Binding"), (
        f"the weakest additivity is now {worst[0]}/{worst[1]}; the README says "
        "GRB2-SH3 binding"
    )
    flat = " ".join(readme.split())
    assert "The headline domain is the weakest one" in flat


def test_the_readme_says_the_bound_stops_at_doubles(readme):
    """Overclaiming the scope of this bound would be the easy mistake."""
    flat = " ".join(readme.split())
    assert "Table S6 stops at doubles" in flat
    assert "nothing here measures it at three or four" in flat


# --------------------------------------------------------------------------
# The scope of the hyperparameter-robustness claim
# --------------------------------------------------------------------------
#
# This is the one claim in the README that survived the baseline result, so
# it is the one most worth overstating. It compares three rules that take a
# knob. NSGA-II takes none, and runs in the same comparison, so an unscoped
# version of the claim -- "treat the constraint as a constraint and you get
# indifference to a hyperparameter" -- is answered by "or use a rule that has
# no hyperparameter". The scope has to be stated, and these pin it to the
# code rather than to prose.


def _feasible_fraction(record, rule: str) -> list[float]:
    """Share of the returned set that clears tau, per domain, for one rule."""
    returned = record["settings"]["n_designs_returned"]
    return [
        round(entry[rule]["n_feasible"] / returned, 2)
        for entry in record["diversity"].values()
        if rule in entry
    ]


def test_the_tuning_claim_is_scoped_to_rules_that_have_a_knob(readme):
    """Any benchmarked rule the sweep does not cover forces the qualifier."""
    from cpd.methods import default_methods

    from experiments.tuning_cost import SWEEPS

    uncovered = sorted(set(default_methods()) - set(SWEEPS) - {"scalarization_w0.5"})
    if not uncovered:
        return
    flat = " ".join(readme.split())
    assert "Among the rules that take a knob" in flat, (
        f"{uncovered} are benchmarked but have no knob to sweep, so the "
        "hyperparameter-robustness claim cannot be stated over all the rules"
    )
    assert "has no weight, no penalty coefficient and no learning rate" in flat, (
        "the README must say which rule sidesteps the hyperparameter problem "
        "entirely, not merely hedge the claim"
    )


def test_the_price_nsga2_pays_matches_the_record(readme, record):
    """The trade-off is the whole argument, so its numbers are load-bearing."""
    nsga2 = _feasible_fraction(record, "nsga2")
    penalty = _feasible_fraction(record, "fixed_penalty")
    flat = " ".join(readme.split())
    claim = f"{100 * min(nsga2):.0f}\N{EN DASH}{100 * max(nsga2):.0f}% feasible"
    assert claim in flat, (
        f"the record puts NSGA-II at {claim}; the README says something else"
    )
    penalty_claim = (
        f"{100 * min(penalty):.0f}\N{EN DASH}{100 * max(penalty):.0f}% "
        "for the fixed penalty"
    )
    assert flat.count(penalty_claim) == 2, (
        f"the record puts the fixed penalty at {penalty_claim}, and the README "
        "states that range in both the NSGA-II section and the tuning section"
    )
    worst = f"{100 * (1 - min(nsga2)):.0f}% unusable at worst"
    assert flat.count(worst) == 2, (
        f"the worst NSGA-II set in the record is {worst}; both blockquotes that "
        "price the trade-off have to say so, and neither may round it down"
    )
