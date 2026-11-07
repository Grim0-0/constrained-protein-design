"""Fetch and parse the ddPCA free energies from Faure, Domingo & Schmiedel et al.

    Mapping the energetic and allosteric landscapes of protein binding domains
    Nature 606, 175-183 (2022).  doi:10.1038/s41586-022-04586-4

Run from the repository root::

    python data/download.py              # fetch and summarize all three domains
    python data/download.py --domain GRB2-SH3

The raw workbook and the parsed CSVs land in ``data/raw/``, which is gitignored.
**The data is never committed**; this script is how you get it.

What the published tables actually contain
------------------------------------------

Supplementary Table S7 holds the inferred free energies -- ``f_ddg_pred``
(folding) and ``b_ddg_pred`` (binding) -- for the **same variants**, which is
the property this whole benchmark is built on. But it holds them for **single
mutants only**. Supplementary Table S6 covers higher-order variants and is raw
BindingPCA/AbundancePCA *fitness*, not free energies.

That shapes the two oracle tracks, and the distinction is not cosmetic:

* **Track 1, exact.** Design space = the measured singles. Ground truth is the
  published inferred energy, oracle error is zero, and the search problem is
  budget-limited *selection* over a few hundred measured variants.
* **Track 2, additive model.** Design space = combinations of those singles,
  scored by summing their ddGs. Free energies are additive to first order and
  this is the base term of a MoCHI-style model, but it is a **model**: it
  assumes no epistasis, so oracle error is not zero and results under it are
  oracle-dependent by construction.

Sign conventions in the published data match the ones this package uses
throughout: ``f_ddg_pred`` positive means destabilizing, ``b_ddg_pred``
negative means stronger binding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpd.landscape import ExactLandscape, MeasuredAdditiveLandscape, Measurement
from cpd.variants import Mutation, Variant

RAW_DIR = Path(__file__).resolve().parent / "raw"

DOI = "10.1038/s41586-022-04586-4"
#: Supplementary Table S7 -- the inferred folding and binding free energies.
TABLE_S7_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-022-04586-4"
    "/MediaObjects/41586_2022_4586_MOESM10_ESM.xlsx"
)
TABLE_S7_FILE = "41586_2022_4586_MOESM10_ESM.xlsx"

DOMAINS: tuple[str, ...] = ("GB1", "GRB2-SH3", "PSD95-PDZ3")

#: PDB entries the reference ("_ref") position numbering matches.
STRUCTURES = {"GB1": "1FCC", "GRB2-SH3": "2VWF", "PSD95-PDZ3": "1BE9"}


def fetch(force: bool = False) -> Path:
    """Download Supplementary Table S7 into ``data/raw/`` if it is not there."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / TABLE_S7_FILE
    if path.exists() and not force:
        print(f"already downloaded: {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        return path

    try:
        import requests
    except ImportError:  # pragma: no cover - depends on the optional extra
        raise SystemExit(
            "requests is needed to download the data: pip install -e '.[data]'"
        ) from None

    print(f"downloading Supplementary Table S7 from doi:{DOI}")
    response = requests.get(TABLE_S7_URL, timeout=300)
    if not response.ok:
        raise SystemExit(
            f"download failed with HTTP {response.status_code}.\n"
            f"Fetch it by hand instead: open https://doi.org/{DOI}, download\n"
            f"Supplementary Table 7, and save it as {path}"
        )
    path.write_bytes(response.content)
    print(f"wrote {path} ({len(response.content) / 1024:.0f} KB)")
    return path


def parse_table(path: Path | None = None, confident_only: bool = True) -> pd.DataFrame:
    """Parse Table S7 into tidy single-mutant rows.

    Args:
        path: Workbook location; defaults to the downloaded copy.
        confident_only: Keep only variants the authors flagged as confidently
            inferred in **both** energies. Defaulting this to True is a real
            choice: the unfiltered table contains estimates with standard
            errors larger than the effects themselves, and an oracle built on
            those is measuring its own noise.
    """
    path = path or (RAW_DIR / TABLE_S7_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python data/download.py` first")

    frame = pd.read_excel(path, sheet_name="TableS7")
    frame = frame[frame["mut_order"] == 1].copy()
    if confident_only:
        frame = frame[frame["f_ddg_pred_conf"].astype(bool) & frame["b_ddg_pred_conf"].astype(bool)]

    # "_ref" columns are PDB numbering, which is what the structures use.
    tidy = pd.DataFrame(
        {
            "domain": frame["protein"].astype(str),
            "variant": frame["id_ref"].astype(str),
            "position": frame["Pos_ref"].astype(int),
            "wt": frame["WT_AA"].astype(str),
            "mut": frame["Mut"].astype(str),
            "ddg_binding": frame["b_ddg_pred"].astype(float),
            "ddg_folding": frame["f_ddg_pred"].astype(float),
            "ddg_binding_se": frame["b_ddg_pred_sd"].astype(float),
            "ddg_folding_se": frame["f_ddg_pred_sd"].astype(float),
            "position_class": frame["Pos_class"].astype(str),
        }
    )
    return tidy.dropna(subset=["ddg_binding", "ddg_folding"]).reset_index(drop=True)


def _measurements(frame: pd.DataFrame) -> dict[Variant, Measurement]:
    table: dict[Variant, Measurement] = {}
    for row in frame.itertuples(index=False):
        variant = Variant((Mutation(int(row.position), row.wt, row.mut),))
        table[variant] = Measurement(
            ddg_binding=float(row.ddg_binding),
            ddg_folding=float(row.ddg_folding),
            ddg_binding_se=float(row.ddg_binding_se),
            ddg_folding_se=float(row.ddg_folding_se),
        )
    return table


def load_domain(
    domain: str = "GRB2-SH3",
    track: str = "exact",
    max_order: int = 4,
    confident_only: bool = True,
    path: Path | None = None,
):
    """Build an oracle for one domain.

    Args:
        domain: One of :data:`DOMAINS`.
        track: ``"exact"`` for the measured singles (zero oracle error), or
            ``"additive"`` for combinations of them under a first-order model.
        max_order: Highest variant order the additive track may compose.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    frame = parse_table(path, confident_only=confident_only)
    frame = frame[frame["domain"] == domain]
    if frame.empty:
        raise ValueError(f"no confident single mutants found for {domain}")

    table = _measurements(frame)
    if track == "exact":
        return ExactLandscape(table, domain=domain)
    if track == "additive":
        return MeasuredAdditiveLandscape(table, domain=domain, max_order=max_order)
    raise ValueError(f"unknown track {track!r}; expected 'exact' or 'additive'")


def _safe_corr(group: pd.DataFrame, min_n: int = 3) -> float:
    """Pearson r between the two energies, or NaN when there is too little to say.

    Fewer than three points, or no spread in either column, makes the
    correlation undefined; pandas returns NaN but warns while dividing by a
    zero standard deviation, and this project turns warnings into errors. A
    domain this thin should report NaN rather than a number.
    """
    if len(group) < min_n:
        return float("nan")
    binding, folding = group["ddg_binding"], group["ddg_folding"]
    if binding.nunique() < 2 or folding.nunique() < 2:
        return float("nan")
    return float(binding.corr(folding))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-domain coverage, plus the antagonism this benchmark depends on."""
    rows = []
    for domain, group in frame.groupby("domain"):
        interface = group[group["position_class"] == "binding_interface"]
        rows.append(
            {
                "domain": domain,
                "pdb": STRUCTURES.get(str(domain), ""),
                "singles": len(group),
                "positions": group["position"].nunique(),
                "interface_singles": len(interface),
                "corr_all": _safe_corr(group),
                "corr_interface": _safe_corr(interface),
                "median_binding_se": group["ddg_binding_se"].median(),
                "median_folding_se": group["ddg_folding_se"].median(),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and parse the ddPCA free energies.")
    parser.add_argument("--domain", choices=DOMAINS, default=None, help="summarize one domain")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--keep-unconfident",
        action="store_true",
        help="keep variants the authors did not flag as confidently inferred",
    )
    args = parser.parse_args(argv)

    path = fetch(force=args.force)
    frame = parse_table(path, confident_only=not args.keep_unconfident)
    if args.domain:
        frame = frame[frame["domain"] == args.domain]

    out = RAW_DIR / "ddpca_singles.csv"
    frame.to_csv(out, index=False)
    print(f"parsed {len(frame)} confident single mutants -> {out}")
    print()
    print(summarize(frame).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(
        "\ncorr_interface is the folding/binding antagonism at interface positions."
        "\nA negative value means substitutions that improve binding tend to destabilize"
        "\nthe fold -- the experimentally measured conflict this benchmark is built on."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
