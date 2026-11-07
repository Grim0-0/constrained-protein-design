"""Parsing the published ddPCA table, exercised on a fixture rather than the network.

The fixture mirrors the real column names and dtypes of Supplementary Table S7,
including the wild-type row and the authors' confidence flags.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download import DOMAINS, load_domain, parse_table, summarize

COLUMNS = [
    "Pos_ref", "id", "id_ref", "Pos", "mut_order",
    "f_dg_pred", "f_ddg_pred", "f_ddg_pred_sd",
    "b_dg_pred", "b_ddg_pred", "b_ddg_pred_sd",
    "f_ddg_pred_conf", "b_ddg_pred_conf",
    "Pos_class", "protein", "WT_AA", "Mut",
]

ROWS = [
    # wild type: mut_order 0, dropped by the single-mutant filter
    (None, "-0-", "-0-", None, 0, -4.19, 0.0, 0.0, -0.57, 0.0, 0.0, True, True, None, "GRB2-SH3", None, None),
    # confident singles
    (24, "Q23A", "Q24A", 23, 1, -3.08, 1.10, 0.03, -0.79, -0.23, 0.01, True, True, "binding_interface", "GRB2-SH3", "Q", "A"),
    (31, "L30P", "L31P", 30, 1, -1.20, 2.00, 0.05, -0.20, 0.40, 0.02, True, True, "core", "GRB2-SH3", "L", "P"),
    # unconfident in folding only -- must be dropped by default
    (40, "W39A", "W40A", 39, 1, -2.00, 0.50, 1.90, -0.40, -0.90, 0.02, False, True, "surface", "GRB2-SH3", "W", "A"),
    # a different domain
    (12, "K11E", "K12E", 11, 1, -3.50, 0.30, 0.02, -0.60, -0.10, 0.01, True, True, "surface", "GB1", "K", "E"),
]


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    path = tmp_path / "TableS7_fixture.xlsx"
    frame.to_excel(path, sheet_name="TableS7", index=False)
    return path


def test_parse_keeps_confident_single_mutants_only(workbook):
    tidy = parse_table(workbook)
    assert set(tidy["variant"]) == {"Q24A", "L31P", "K12E"}  # WT and W40A dropped
    assert set(tidy["domain"]) == {"GRB2-SH3", "GB1"}


def test_parse_can_keep_unconfident_rows_when_asked(workbook):
    tidy = parse_table(workbook, confident_only=False)
    assert "W40A" in set(tidy["variant"])


def test_parse_uses_pdb_reference_numbering(workbook):
    """id_ref/Pos_ref are PDB numbering; id/Pos are construct numbering."""
    tidy = parse_table(workbook)
    row = tidy[tidy["variant"] == "Q24A"].iloc[0]
    assert row["position"] == 24  # not 23, the construct number
    assert (row["wt"], row["mut"]) == ("Q", "A")


def test_parse_preserves_the_published_sign_conventions(workbook):
    tidy = parse_table(workbook)
    row = tidy[tidy["variant"] == "Q24A"].iloc[0]
    assert row["ddg_folding"] == pytest.approx(1.10)  # positive = destabilizing
    assert row["ddg_binding"] == pytest.approx(-0.23)  # negative = stronger binding
    assert row["ddg_binding_se"] == pytest.approx(0.01)
    assert row["ddg_folding_se"] == pytest.approx(0.03)


def test_load_domain_builds_an_exact_oracle(workbook):
    landscape = load_domain("GRB2-SH3", track="exact", path=workbook)
    assert landscape.positions == (24, 31)
    assert landscape.max_order == 1  # the published energies are singles only
    from cpd.variants import Variant

    assert landscape.evaluate(Variant.parse("Q24A")).ddg_binding == pytest.approx(-0.23)


def test_load_domain_builds_an_additive_oracle_that_combines_singles(workbook):
    from cpd.variants import Variant

    landscape = load_domain("GRB2-SH3", track="additive", max_order=2, path=workbook)
    combined = landscape.evaluate(Variant.parse("Q24A,L31P"))
    assert combined.ddg_binding == pytest.approx(-0.23 + 0.40)
    assert combined.ddg_folding == pytest.approx(1.10 + 2.00)


def test_load_domain_validates_its_arguments(workbook):
    with pytest.raises(ValueError, match="unknown domain"):
        load_domain("NOT-A-DOMAIN", path=workbook)
    with pytest.raises(ValueError, match="unknown track"):
        load_domain("GRB2-SH3", track="telepathy", path=workbook)
    with pytest.raises(ValueError, match="no confident single mutants"):
        load_domain("PSD95-PDZ3", path=workbook)  # absent from the fixture


def test_missing_data_file_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download.py"):
        parse_table(tmp_path / "absent.xlsx")


def test_summarize_reports_coverage_and_the_interface_correlation(workbook):
    table = summarize(parse_table(workbook))
    assert set(table["domain"]) == {"GRB2-SH3", "GB1"}
    grb2 = table[table["domain"] == "GRB2-SH3"].iloc[0]
    assert grb2["singles"] == 2
    assert grb2["pdb"] == "2VWF"


def test_the_three_domains_are_the_ones_the_paper_measured():
    assert DOMAINS == ("GB1", "GRB2-SH3", "PSD95-PDZ3")


# --------------------------------------------------------------------------
# The summary correlations
# --------------------------------------------------------------------------


def test_a_thin_domain_reports_nan_rather_than_a_spurious_correlation():
    """Two points cannot support a correlation; both columns must say so.

    corr_all used to lack the length guard that corr_interface had two lines
    below it. Two points always lie on a line, so it reported a perfect +/-1
    correlation for any domain with two rows, while corr_interface correctly
    reported NaN for the same data. A fabricated 1.0 is worse than a missing
    value in a table whose whole subject is folding/binding antagonism.
    """
    import math

    import pandas as pd

    from data.download import summarize

    thin = pd.DataFrame(
        {
            "domain": ["TINY", "TINY"],
            "position": [1, 2],
            "position_class": ["binding_interface", "other"],
            "ddg_binding": [-1.0, 0.5],
            "ddg_folding": [0.3, 1.2],
            "ddg_binding_se": [0.1, 0.1],
            "ddg_folding_se": [0.1, 0.1],
        }
    )
    row = summarize(thin).iloc[0]
    assert math.isnan(row["corr_all"])
    assert math.isnan(row["corr_interface"])


def test_a_domain_with_no_spread_reports_nan():
    """A constant column has no correlation with anything."""
    import math

    import pandas as pd

    from data.download import summarize

    flat = pd.DataFrame(
        {
            "domain": ["FLAT"] * 4,
            "position": [1, 2, 3, 4],
            "position_class": ["other"] * 4,
            "ddg_binding": [-1.0, -1.0, -1.0, -1.0],
            "ddg_folding": [0.1, 0.4, 0.9, 1.3],
            "ddg_binding_se": [0.1] * 4,
            "ddg_folding_se": [0.1] * 4,
        }
    )
    assert math.isnan(summarize(flat).iloc[0]["corr_all"])


def test_a_real_sized_domain_still_reports_a_number():
    import pandas as pd

    from data.download import summarize

    n = 12
    frame = pd.DataFrame(
        {
            "domain": ["REAL"] * n,
            "position": list(range(n)),
            "position_class": ["binding_interface"] * n,
            "ddg_binding": [float(i) * 0.3 - 1.0 for i in range(n)],
            "ddg_folding": [float(i) * 0.2 + 0.1 for i in range(n)],
            "ddg_binding_se": [0.1] * n,
            "ddg_folding_se": [0.1] * n,
        }
    )
    row = summarize(frame).iloc[0]
    assert row["corr_all"] == pytest.approx(1.0)
    assert row["corr_interface"] == pytest.approx(1.0)
