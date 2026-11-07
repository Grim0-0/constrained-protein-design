"""How far can the additive oracle be trusted?

    python experiments/additivity.py

Track 2 composes multi-mutant free energies by adding single-mutant ones. The
README calls that the largest caveat on every result produced from it, and so
far the caveat has been unquantified -- which makes it an apology rather than
a measurement.

It can be measured, partly. Supplementary Table S6 reports **raw fitness** for
variants of every order, including the doubles Table S7 has no free energies
for. Absolute validation is therefore impossible: there is nothing to compare
a predicted kcal/mol against. Ordering can be compared, and ordering is what
a search actually consumes -- an oracle that ranks designs correctly leads the
search to the same places even if its numbers are off.

So: for every measured double, add the two single-mutant free energies from
Table S7, and ask how well that ordering reproduces the measured fitness
ordering from Table S6.

**Why rank correlation and not R-squared.** Fitness is a sigmoidal function of
free energy -- it saturates once a protein is fully folded or fully unfolded --
so a perfect free-energy model would still show a curved fitness plot. Spearman
is invariant to that monotone transform. Pearson would mostly measure the
sigmoid.

**Sign.** Lower ddG means tighter binding and more stability, which means
*higher* fitness, so perfect agreement is rho = -1. The reported
``rho_agreement`` flips the sign, so +1 is perfect and 0 is useless.

**What this does and does not bound.** Table S6 stops at doubles -- its
maximum Hamming distance is 2 on every domain -- so order 2 is not a choice
here, it is the only order the published data can test at all. The oracle
composes up to four substitutions, and nothing measures the gap at three or
four. Couplings accumulate, so those are likely worse, but that is an
inference, not a result.

Two further limits. GB1 has no AbundancePCA data in Table S6, so its folding
model is unbounded by this. And saturation compresses ranks at both ends of
the fitness scale, which depresses rho for reasons that are not the model's
fault -- so a strong value is trustworthy, while a weak one is partly an
artefact of the assay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download import DOMAINS, RAW_DIR, TABLE_S6_FILE, parse_table  # noqa: E402

from experiments.common import save_results  # noqa: E402

#: Which assay each energy drives.
ASSAYS = {"ddg_binding": "Binding", "ddg_folding": "Abundance"}

#: Cached decomposition, so the 43 MB workbook is read once.
CACHE = RAW_DIR / "ddpca_doubles.csv"


def decompose(path: Path | None = None, max_per_domain: int | None = None) -> pd.DataFrame:
    """Every measured double, split into the two singles that compose it."""
    path = path or (RAW_DIR / TABLE_S6_FILE)
    if not path.exists():
        raise SystemExit(
            f"{path.name} is missing; fetch it with\n"
            "  python -c \"import sys; sys.path.insert(0,'.'); "
            'from data.download import fetch_s6; fetch_s6()"'
        )
    raw = pd.read_excel(path, sheet_name="TableS6")

    rows = []
    for domain in DOMAINS:
        for assay in ("Binding", "Abundance"):
            group = raw[(raw["protein"] == domain) & (raw["pca_type"] == assay)]
            wt_rows = group[group["Nham_aa"] == 0]
            if wt_rows.empty:
                continue
            wt = str(wt_rows.iloc[0]["aa_seq"])
            doubles = group[group["Nham_aa"] == 2]
            if max_per_domain:
                doubles = doubles.head(max_per_domain)
            for _, row in doubles.iterrows():
                seq = str(row["aa_seq"])
                if len(seq) != len(wt):
                    continue
                diffs = [
                    (i, w, m)
                    for i, (w, m) in enumerate(zip(wt, seq), start=1)
                    if w != m
                ]
                if len(diffs) != 2:
                    continue
                (p1, w1, m1), (p2, w2, m2) = diffs
                rows.append({
                    "domain": domain, "assay": assay,
                    "pos1": p1, "wt1": w1, "mut1": m1,
                    "pos2": p2, "wt2": w2, "mut2": m2,
                    "fitness": float(row["fitness"]),
                    "sigma": float(row["sigma"]),
                })
    return pd.DataFrame(rows)


def singles_lookup(confident_only: bool = True) -> pd.DataFrame:
    """Single-mutant free energies keyed by construct position, as S6 numbers them."""
    frame = parse_table(confident_only=confident_only)
    # parse_table reports PDB numbering; the raw table carries both, so re-read
    # the construct numbering that S6's sequences index into.
    raw = pd.read_excel(RAW_DIR / "41586_2022_4586_MOESM10_ESM.xlsx", sheet_name="TableS7")
    raw = raw[raw["mut_order"] == 1]
    if confident_only:
        raw = raw[raw["f_ddg_pred_conf"].astype(bool) & raw["b_ddg_pred_conf"].astype(bool)]
    return pd.DataFrame({
        "domain": raw["protein"].astype(str),
        "pos": raw["Pos"].astype(int),
        "wt": raw["WT_AA"].astype(str),
        "mut": raw["Mut"].astype(str),
        "ddg_binding": raw["b_ddg_pred"].astype(float),
        "ddg_folding": raw["f_ddg_pred"].astype(float),
    })


def build(max_per_domain: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Doubles joined to their two singles' predicted energies, cached on disk."""
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)

    doubles = decompose(max_per_domain=max_per_domain)
    singles = singles_lookup()
    merged = doubles
    for which in ("1", "2"):
        merged = merged.merge(
            singles.rename(columns={
                "pos": f"pos{which}", "wt": f"wt{which}", "mut": f"mut{which}",
                "ddg_binding": f"b{which}", "ddg_folding": f"f{which}",
            }),
            on=["domain", f"pos{which}", f"wt{which}", f"mut{which}"],
            how="inner",
        )
    merged["pred_ddg_binding"] = merged["b1"] + merged["b2"]
    merged["pred_ddg_folding"] = merged["f1"] + merged["f2"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CACHE, index=False)
    return merged


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman between the additive prediction and the measured fitness."""
    from scipy import stats

    rows = []
    for (domain, assay), group in frame.groupby(["domain", "assay"]):
        energy = "pred_ddg_binding" if assay == "Binding" else "pred_ddg_folding"
        usable = group[[energy, "fitness"]].dropna()
        if len(usable) < 30:
            continue
        rho, p = stats.spearmanr(usable[energy], usable["fitness"])
        # Lower ddG means higher fitness, so flip the sign: +1 is perfect.
        rows.append({
            "domain": domain,
            "assay": assay,
            "energy": energy.replace("pred_", ""),
            "n_doubles": len(usable),
            "rho_agreement": -float(rho),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-read the workbook")
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="cap doubles per domain and assay, for a quick run")
    args = parser.parse_args(argv)

    frame = build(max_per_domain=args.max_per_domain, refresh=args.refresh)
    table = correlations(frame)
    if table.empty:
        raise SystemExit("no domain had enough decomposable doubles to correlate")

    print(f"\n{'=' * 74}\nAdditive prediction vs measured fitness, on doubles\n{'=' * 74}")
    print(f"{'domain':<12} {'assay':<10} {'energy':<14} {'doubles':>8} {'rho':>7}")
    for _, row in table.sort_values(["assay", "domain"]).iterrows():
        print(f"{row['domain']:<12} {row['assay']:<10} {row['energy']:<14} "
              f"{row['n_doubles']:>8,} {row['rho_agreement']:>7.3f}")

    path = save_results(table, "additivity_spearman")
    print(f"\nwrote {path.relative_to(path.parent.parent)}")

    worst = table.loc[table["rho_agreement"].idxmin()]
    best = table.loc[table["rho_agreement"].idxmax()]
    print(f"\n  best  {best['domain']} {best['assay']}: rho = {best['rho_agreement']:.3f}")
    print(f"  worst {worst['domain']} {worst['assay']}: rho = {worst['rho_agreement']:.3f}")
    print()
    print("  Read as: Track 2 rankings hold to about this resolution AT ORDER 2,")
    print("  which is as far as Table S6 goes. It stops at doubles, so nothing here")
    print("  measures the oracle at the three and four substitutions it actually")
    print("  composes. GB1 has no AbundancePCA data, so its folding model is")
    print("  unbounded by this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
