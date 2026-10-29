# constrained-protein-design

**When one design objective is a hard constraint rather than something to trade off, do standard multi-objective methods produce usable designs?**

```
maximize    binding affinity      (minimize ddG_binding)
subject to  ddG_folding <= tau    (the protein must stay folded)
```

A binder that does not fold is not a different point on the Pareto front. It is
worthless. But the metric the field reports — hypervolume — pays for it anyway.

This repo benchmarks four constraint-handling rules through a single shared
search engine, on a landscape built from **experimentally measured** folding and
binding free energies, and reports **feasibility rate**: the fraction of
returned designs that actually satisfy the constraint. That metric is almost
never reported.

---

## The headline

![Best usable design and feasibility rate against constraint tightness](figures/headline_tau_sweep_additive_GRB2-SH3.png)

GRB2-SH3, binding its GAB2 ligand. 1200 oracle calls per rule, 5 seeds, 20
designs returned.

**This is Track 2.** Single-mutant free energies are the measured values from
Faure, Domingo & Schmiedel et al., *Nature* 2022; multi-mutant values are
**composed additively from them, not measured** — the published tables report
energies for singles only. See [the oracle problem](#the-oracle-problem-and-what-the-published-data-actually-supports)
below, which is the single biggest caveat on everything here. The same
comparison on the zero-error Track 1 oracle ranks the rules identically.

**Left:** the best design you can actually use, with selection held identical
across rules. **Right:** how much of what each rule hands over clears the bar.

The two panels disagree, and that is the point. Optimize-then-reject reports
100% feasibility because it discards failures — while filling only a fifth of
the designs it was asked for. The scalarization sweep returns a full set of
twenty designs of which **none** satisfy the constraint.

---

## What actually happened

Tightest constraint, tau = -0.27 kcal/mol. Budget spent on feasible designs is
averaged over every tau; the rest are at the tightest. 5 seeds.

| rule | budget spent on feasible designs | feasible designs delivered | best usable design | designs filled |
|---|---|---|---|---|
| Lagrangian (`lr`=0.5) | 50% | 80% | −2.95 kcal/mol | 100% |
| Fixed penalty (`rho`=10) | 57% | 97% | −3.19 kcal/mol | 100% |
| Scalarization (`w`=0.5) | 77% | 94% | −3.21 kcal/mol | 100% |
| Scalarization (weight sweep) | 52% | **0%** | −1.69 kcal/mol | 100% |
| Optimize-then-reject | **5%** | 100% | −0.59 kcal/mol | **21%** |

Three findings, including one that contradicts the hypothesis this project
started with.

### 1. Optimize-then-reject fails, as predicted

It spends **95% of its oracle budget on designs that break the constraint**, and
at the tightest tau it can only fill 21% of the designs it was asked for. Its
best usable design is 2.6 kcal/mol worse than the best rule's. Maximizing first
and filtering later is not a cheap approximation to constrained design — at a
fixed budget it is a different and much worse algorithm.

### 2. Hypervolume rewards designs nobody can use

![Hypervolume over all designs versus feasible designs only](figures/hypervolume_gap_additive_GRB2-SH3.png)

For the scalarization sweep, hypervolume computed over all returned designs is
nearly **flat** as the constraint tightens, while hypervolume over the feasible
subset collapses. Reported the usual way, the metric says nothing changed. The
shaded gap is hypervolume bought entirely with designs that would fail.

### 3. The Lagrangian does *not* beat a well-chosen weight — it beats having to choose

The table above is at the **tightest** tau, and that is the one place the
Lagrangian loses: scalarization at w=0.5 (−3.21) and the fixed penalty at ρ=10
(−3.19) both beat it (−2.95) there. Across the four tau values swept at default
settings it leads at three of them and trails only at the tightest:

| tau (kcal/mol) | −0.27 | +0.51 | +1.28 | +2.05 |
|---|---|---|---|---|
| Lagrangian | −2.67 | **−3.37** | **−3.53** | **−3.84** |
| Fixed penalty | **−2.94** | −3.16 | −3.43 | −3.80 |
| Scalarization (w=0.5) | −2.75 | −2.98 | −3.09 | −3.22 |

So **the original hypothesis — that the Lagrangian holds where the others
degrade as tau tightens — is not supported as stated.** If anything the
direction is reversed: its advantage is largest when the constraint is loose
and vanishes exactly where the problem is hardest. Signed dual ascent drives the
multiplier toward zero whenever the population sits comfortably inside the
feasible region, which is good exploration when tau is loose and leaves the
search thrashing across the boundary when tau is tight; the multiplier
trajectories in `results/` show the oscillation.

There is a second reason to distrust the comparison above: w=0.5 and ρ=10 are
*hindsight*. A real run gets one guess at the knob. Sweep each rule over its own:

![What a hyperparameter you must choose in advance costs you](figures/tuning_cost_additive_GRB2-SH3.png)

| rule | knob | best setting | worst setting | choosing blind | spread |
|---|---|---|---|---|---|
| Lagrangian | `lr` | −3.38 (at 0.25) | −3.29 | −3.34 | **0.09** |
| Fixed penalty | `rho` | −3.33 (at 10) | −3.08 | −3.25 | **0.26** |
| Scalarization | `weight` | −3.19 (at 0.7) | −1.71 | −2.55 | **1.48** |

Each cell averages 5 seeds over the four tau values above, at 800 oracle calls —
a different budget and a wider tau range than the headline table, which is why
the absolute numbers differ from it.

At their best settings the three rules sit within 0.19 kcal/mol of each other.
The difference between them is not peak performance — it is **how much it costs
to guess the knob wrong**. A badly chosen scalarization weight costs 1.48
kcal/mol of binding; a badly chosen Lagrangian learning rate costs 0.09, a
**17× difference in sensitivity**. Choosing blind, the Lagrangian is the best of
the three, and it is also the best at its own best setting — but only by 0.19,
which is inside the seed-to-seed spread and is not a result worth leaning on.

So the honest claim is narrower and more useful than the one this started with:

> Treating a hard constraint as a constraint does not buy a better design than a
> perfectly tuned weighted sum. It buys near-indifference to a hyperparameter
> that cannot be tuned before the run, and it never ships a design set that is
> 0% feasible.

### Where the budget goes

![Best feasible design against oracle calls spent](figures/budget_curves_additive_GRB2-SH3.png)

At a middling constraint (tau = +0.12), the Lagrangian and the fixed penalty are
still improving at 1200 calls. Optimize-then-reject is not: it reaches its final
value after about 800 and then flatlines, having found its feasible designs
early by accident and spent the rest of the budget somewhere it cannot use. The
scalarization sweep settles a full kcal/mol short, having split its budget five
ways to hedge a weight choice.

This is the practical reading of the feasibility-rate column. A rule that spends
95% of its assays on designs that break the constraint is not merely scoring
badly on a metric — it is buying nothing with most of its budget.

---

## The designs themselves

Metrics are the contribution, but a design benchmark should show what it
designed. Both tables pool every rule's evaluations at a mid-tightness
constraint and keep only designs that satisfy it, ranked by binding. Negative
ddG binding means tighter binding than wild type; ddG folding at or below the
limit means the protein still folds.

**Track 1 — measured variants, zero oracle error.** These are real assayed
single mutants; both numbers are the published inferred free energies, not
predictions of this work.

Constraint: `ddG_folding <= +0.89` kcal/mol. 171 designs satisfy it; the best six:

| design | ddG binding | ddG folding | found by |
|---|---|---|---|
| `T53G` | **−1.37** | −0.05 | Scalarization (w=0.5) |
| `K37T` | **−1.04** | +0.61 | Lagrangian |
| `H26E` | **−0.88** | −0.66 | Fixed penalty |
| `K37S` | **−0.88** | +0.86 | Scalarization (w=0.5) |
| `Q43D` | **−0.74** | +0.49 | Scalarization (w=0.5) |
| `C40L` | **−0.72** | +0.33 | Lagrangian |

**Track 2 — composed from measured singles under the additive model.** These
combinations were never assayed; their energies are model predictions and carry
the oracle caveat below.

Constraint: `ddG_folding <= +0.89` kcal/mol. 2677 designs satisfy it; the best six:

| design | ddG binding | ddG folding | found by |
|---|---|---|---|
| `H26E,K37D,R49A,T53G` | **−4.12** | +0.71 | Lagrangian |
| `H26E,K37T,R49T,T53G` | **−3.91** | +0.07 | Lagrangian |
| `H26E,K37T,R49N,T53G` | **−3.90** | +0.72 | Lagrangian |
| `G15D,K37T,R49A,T53G` | **−3.90** | +0.41 | Fixed penalty |
| `H26E,K37D,R49S,T53G` | **−3.85** | +0.82 | Lagrangian |
| `H26E,K37T,R49A,T53G` | **−3.79** | −0.74 | Lagrangian |

`H26E,K37T,R49T,T53G` is the one to look at: roughly 4 kcal/mol tighter binding
at a folding cost of +0.07 kcal/mol, which is within the measurement error on
the underlying singles. Under the additive model it is close to free.

Position 37 recurs across the whole table — K37T, K37S, K37D all improve
binding — so the search converges on a binding hotspot that is present in the
measured data rather than scattering over the sequence.

Regenerate with `python experiments/report.py --designs`.

**None of these have been made or tested in a lab.** Track 1 rows are measured
free energies for variants that exist; Track 2 rows are model output. Neither
is a claim about behaviour in a cell.

---

## The oracle problem, and what the published data actually supports

This is where projects like this usually die: a search proposes sequences nobody
measured, so the "ground truth" becomes a model's opinion. Two tracks, both
reported.

**Track 1 — exact oracle.** Design space restricted to the measured variants.
Oracle error is exactly zero.

**Track 2 — additive model.** Combinations of measured singles, scored by summing
their ddGs. This is the first-order term of a MoCHI-style thermodynamic model.
It is a model, and results under it are oracle-dependent by construction.

**A finding from the data itself:** the spec for this project assumed Track 1
would be a *combinatorial* search over measured multi-mutants. It cannot be.
Supplementary Table S7 reports inferred free energies for **single mutants
only** (588 confident singles for GRB2-SH3); Table S6 covers higher-order
variants but holds raw BindingPCA/AbundancePCA *fitness*, not free energies. So
Track 1 is budget-limited **selection** over a few hundred measured variants,
and anything combinatorial is Track 2 and model-dependent. The two tracks rank
the rules identically, but Track 1 compresses all the differences — with only
singles available, every rule finds a decent variant.

**A second finding, and a check that nearly killed the project.** The folding
and binding effects are *not* globally anticorrelated. Correlation between
ddG_folding and ddG_binding at interface positions:

| domain | all positions | interface positions |
|---|---|---|
| GB1 | +0.39 | +0.48 |
| GRB2-SH3 | −0.15 | −0.07 |
| PSD95-PDZ3 | +0.22 | +0.15 |

Mostly *positive* — unsurprising, since a mutation that unfolds a domain also
destroys its binding site. A correlation test alone would have said "no
antagonism here" and ended the project.

The conflict is real but lives in the **tail**, not the correlation. For
GRB2-SH3, the 15 strongest binding improvers have a median ddG_folding of
**+1.94 kcal/mol** and 93% of them are destabilizing. Tightening tau from the
90th to the 5th percentile of the folding distribution costs 0.76 kcal/mol of
achievable binding. The right question was never "are the objectives
anticorrelated" but "what does the constraint cost you at the optimum".

---

## Why the comparison is clean

Every rule runs through the **same** evolutionary search engine, with the same
budget, the same seeds, the same initialization and the same population size.
Only the scalar objective changes. Comparing a policy-gradient agent against
NSGA-II would confound the search algorithm with the constraint-handling
mechanism; this design cannot.

Three choices worth stating, because each one could have quietly rigged the result:

- **The budget binds.** Methods are compared at equal oracle cost, and the run
  asserts that every rule actually spent its full budget rather than running out
  of generations first ([`common.py`](experiments/common.py)). A comparison where
  one method stopped early is not a comparison.
- **Optimize-then-reject filters its whole trace,** not just its final
  population. A weak null hypothesis would make the headline result worthless.
- **The weight sweep is charged for the sweep.** Five weights share one budget,
  split evenly. Handing it five full budgets would make every other rule look
  better than it is.

Results are reported under two readouts, both computed from the same paid-for
trace: **as delivered** (what the rule hands over on its own terms — this is
where the feasibility rate lives) and **feasible-filtered** (every rule made to
discard infeasible designs, so selection is identical and only the search
differs). Reporting only the first would flatter rules that quietly return
unusable designs; only the second would hide that they do.

---

## Metrics

- **Feasibility rate** — fraction of returned designs satisfying the constraint.
- **Design yield** — how many of the N requested designs a rule could fill with
  feasible ones. The honest version of "it returned 100% feasible designs".
- **Feasible best-k** — mean binding of the k best feasible designs. Returns
  `NaN` when fewer than k are feasible, rather than averaging over however many
  happen to exist; a rule that returned one feasible design out of a hundred
  should not score like one that returned ten.
- **Violation severity** — how far past tau, when violated.
- **Oracle-call budget curves** — performance against evaluations spent.
- **Hypervolume, twice** — over all designs and over the feasible subset. The gap
  is the argument.

---

## Running it

```bash
pip install -e ".[data,dev]"
pytest                                    # 107 tests, no data or network needed

python data/download.py                   # fetches the ddPCA tables into data/raw/
python experiments/tau_sweep.py --landscape additive --domain GRB2-SH3
python experiments/tuning_cost.py --landscape additive --domain GRB2-SH3
python experiments/budget_curves.py --landscape additive --domain GRB2-SH3
python experiments/report.py              # regenerates the tables above
```

Everything runs on CPU in minutes. All seeds are fixed; `--landscape synthetic`
runs the whole pipeline on a built-in toy landscape with no download at all.

```
src/cpd/
  variants.py    Mutation, Variant            landscape.py   the three oracles + budget accounting
  objectives.py  the four rules               search.py      the one shared engine
  methods.py     engine x rule, two readouts  metrics.py     feasibility, yield, hypervolume
experiments/     tau_sweep, tuning_cost, budget_curves, report, plotting
data/download.py fetches and parses the ddPCA supplementary tables
```

The data is **never committed**; `data/raw/` is gitignored and `download.py` is
how you get it.

---

## Limitations

- **One domain carries the headline.** GRB2-SH3 only. GB1 and PSD95-PDZ3 are
  parsed and available but not swept; GB1's constraint barely binds, so it would
  show less.
- **Two small binding domains.** May not transfer to large proteins or to
  designed binders.
- **ddPCA free energies are inferred** from a thermodynamic model fitted to
  growth-rate assays, not measured by calorimetry. Median standard error on
  GRB2-SH3 binding ddG is 0.06 kcal/mol.
- **Track 2 is oracle-dependent by construction.** Additivity is exactly the
  assumption that fails at interacting positions, which for binding domains are
  disproportionately the interface positions this benchmark cares about.
  Propagating the published standard errors in quadrature as independent
  understates the true uncertainty for the same reason.
- **No wet-lab validation.** None of these designs have been made. This work is
  entirely computational.
- **One search family.** Conclusions about constraint handling may interact with
  the evolutionary engine. A second engine running all four rules would test
  that; it is not done here.
- **Hyperparameter ranges are a judgement call.** The spread numbers depend on
  which settings are considered plausible. The ranges used are in
  [`tuning_cost.py`](experiments/tuning_cost.py); they are deliberately wide
  rather than chosen to flatter any rule.

---

## Data and citation

Faure, A.J., Domingo, J., Schmiedel, J.M. et al. **Mapping the energetic and
allosteric landscapes of protein binding domains.** *Nature* **604**, 175–183
(2022). [doi:10.1038/s41586-022-04586-4](https://doi.org/10.1038/s41586-022-04586-4)

Supplementary Table S7 (inferred free energies) is fetched by
`data/download.py`. Raw sequencing is at GEO `GSE184042`; the authors' analysis
code is at [lehner-lab/doubledeepms](https://github.com/lehner-lab/doubledeepms)
and the thermodynamic model fitting at
[lehner-lab/MoCHI](https://github.com/lehner-lab/MoCHI).

The adaptive-multiplier rule is adapted from **Ctrl-DNA**
([arXiv:2505.20578](https://arxiv.org/abs/2505.20578)), which introduced
batch-level Lagrangian relaxation for constrained regulatory *DNA* design. The
contribution here is porting it to protein sequence design and, more
importantly, the measurement: a benchmark where the hard constraint is a
biophysical property with experimentally measured ground truth, and where
feasibility rate is a reported metric.

Structures referenced by the position numbering: `2VWF` (GRB2-SH3), `1BE9`
(PSD95-PDZ3), `1FCC` (GB1).

## License

MIT. See [LICENSE](LICENSE).
