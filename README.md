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
returned designs that actually satisfy the constraint — a metric standard in
constrained optimization and essentially absent from multi-objective protein
design. See [relation to prior work](#relation-to-prior-work) for what that
does and does not claim.

> **Read ["is the problem hard?"](#before-any-of-that-is-the-problem-hard) first.**
> A search-free greedy baseline is not beaten by any rule on the measured
> track, and beats them all on one domain. That track therefore cannot rank
> the rules on design quality, and the section says what survives.

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

## Before any of that: is the problem hard?

A comparison between search methods is only evidence about search if a method
that does not search would do worse. So: rank the measured singles by binding,
add substitutions one at a time while the constraint still holds, stop at the
order cap. No population, no stochasticity, no hyperparameter. Four oracle
calls after the singles are measured.

Because both objectives on Track 2 are **sums of per-substitution constants**,
the constrained problem is also solvable exactly — by dynamic programming over
the Pareto front of (folding, binding), which is lossless here precisely
because the objectives are separable. So every score below can be stated as a
fraction of what was actually attainable, not just relative to the other rules.

![Every rule against a search-free sort and against the exact optimum](figures/baselines_additive.png)

At the tightest constraint, best of 5 seeds per rule, one-sample *t* against the
deterministic baseline:

| domain | exact optimum | greedy, no search | best rule | gap | *p* | verdict |
|---|---|---|---|---|---|---|
| GB1 | −2.47 | **−1.97** | −2.01 (fixed penalty) | +0.04 | 0.81 | tied |
| GRB2-SH3 | −3.79 | **−2.58** | −3.21 (scalarization `w`=0.5) | +0.63 | 0.096 | not separated |
| PSD95-PDZ3 | −2.44 | **−2.20** | −1.82 (Lagrangian) | −0.38 | **0.019** | **the sort wins** |

**No rule reliably beats the sort on any of the three domains, and on PSD95-PDZ3
the sort wins outright.** The best rules capture 68–85% of the attainable
objective; the sort captures 68–90%. On GB1 the sort hits the exact optimum at
five of seven constraint levels while every evolutionary rule falls short.

The gaps between rules that this benchmark was built to measure are 0.03–0.26
kcal/mol. The gap from the best rule to optimal is 0.47–1.21 kcal/mol. **The
shortfall is an order of magnitude larger than the signal.**

### The cause is the oracle, not the rules

Same engine, same rules, same budget, run instead on the synthetic landscape —
which carries 24 coupled position pairs, so the objectives are *not* separable:

```
at the tightest tau
  best rule (Lagrangian)          -3.948
  greedy, additive plan           -2.127   gap +1.82 kcal/mol, p = 0.0042
  greedy, forward on true oracle  -1.909   gap +2.04 kcal/mol, p = 0.0028
```

With epistasis present the search wins decisively. Remove it and the
constrained design problem degenerates into a knapsack that a sort solves
nearly optimally in four calls.

### What this invalidates, and what survives

**Invalidated.** Track 2 cannot rank constraint-handling rules on design
quality. Any such ranking — including ones earlier versions of this README
reported — is measuring a search that is not the binding factor. The Track 2
design-quality numbers are retained below as a record, not as a result.

**Survives.** The failure modes are about *feasibility and yield*, not about
reaching the optimum, and the baseline does not touch them: optimize-then-reject
still spends 5% of its budget on feasible designs and still fills a fifth of the
set, still in proportion to measured conflict strength; the weight sweep still
returns twenty designs of which none are feasible; hypervolume over all designs
still reports improvement where no usable design exists. The
[hyperparameter-robustness result](#3-the-lagrangian-does-not-beat-a-well-chosen-weight--it-beats-having-to-choose)
is a spread across knob settings rather than a ranking against a baseline, and
it replicates on the synthetic track where the search does bind.

**The generalizable part.** Additive ΔΔG models and per-residue predictors are
common oracles in multi-objective protein design, and a separable oracle makes
the design problem an integer program with no structure for a search to
exploit. A greedy baseline costs four oracle calls and detects this. Run one
before reporting a method comparison.

```
python experiments/baselines.py --track additive
python experiments/baselines.py --track synthetic
```

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

At the tightest tau the Lagrangian posts −2.95 against −3.21 for scalarization
at w=0.5 and −3.19 for the fixed penalty. Ranked on the means, it loses. But
those gaps do not survive a significance test across the five replicate seeds
(Welch, p = 0.50 and p = 0.94), and the same holds on all three domains:

| domain | ranking at the tightest tau |
|---|---|
| GB1 | fixed penalty **=** optimize-then-reject **=** Lagrangian **=** scalarization **>** sweep |
| GRB2-SH3 | scalarization **=** fixed penalty **=** Lagrangian **>** sweep **>** optimize-then-reject |
| PSD95-PDZ3 | Lagrangian **=** fixed penalty **=** scalarization **>** sweep **=** optimize-then-reject |

`=` means tied on this evidence; `>` means separated at p < 0.05. The leader
changes between domains, and in every case the top three are indistinguishable.
**The apparent winner is seed noise.** Reporting that ordering as a result would
have been inventing a finding, which is why
[`compare_groups`](src/cpd/metrics.py) now gates every ranking the repo prints.

So the honest verdict on the original hypothesis: **not supported.** The
Lagrangian does not pull ahead as tau tightens — it ties. What it does not do
is lose, and the earlier draft of this README said it did.

The second reason to distrust a ranking on means: w=0.5 and ρ=10 are
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
to guess the knob wrong**.

**And unlike the rankings, this one replicates.** The same sweep on all three
domains, spread between the best and worst setting of each rule:

| domain | Lagrangian `lr` | Fixed penalty `rho` | Scalarization `weight` |
|---|---|---|---|
| GB1 | **0.088** | 0.136 | 1.168 |
| GRB2-SH3 | **0.085** | 0.255 | 1.483 |
| PSD95-PDZ3 | **0.129** | 0.205 | 0.928 |

Same ordering every time, on three proteins with different conflict strengths:
the Lagrangian is least sensitive to its knob, the fixed penalty next, the
weighted sum between 7× and 17× worse than the Lagrangian. This is the only
claim in the project that both replicates across domains *and* distinguishes
the rules — the design-quality rankings do neither.

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

## Does it replicate? Three domains, and a dose-response

One result on one protein is an anecdote. All three ddPCA domains were run, and
they differ a lot in how much the two objectives actually fight — measured, not
assumed, as the binding you give up to satisfy the constraint:

| domain | binding given up to satisfy the constraint | top-15 binders destabilizing |
|---|---|---|
| GB1 | 0.41 kcal/mol | 73% |
| GRB2-SH3 | 0.76 kcal/mol | 93% |
| PSD95-PDZ3 | 1.12 kcal/mol | 80% |

![Best feasible design and feasibility rate across three domains](figures/replication_additive.png)

Two things replicate, and one thing turns out to be conditional.

**Replicates: the weight sweep is reliably worst.** Separated from the leader at
p < 0.05 in all three domains, and it delivers near-zero feasible designs
everywhere. This is the most robust negative result in the project.

**Replicates: the constraint-aware rules tie.** In all three domains the top
three are statistically indistinguishable. No rule wins on design quality.

**Conditional, and this is the useful finding: optimize-then-reject only fails
when the objectives genuinely conflict.**

| domain | conflict | budget spent on feasible designs | designs filled | verdict |
|---|---|---|---|---|
| GB1 | 0.41 | 45% | 100% | **ties the leader** |
| GRB2-SH3 | 0.76 | 5% | 21% | collapses |
| PSD95-PDZ3 | 1.12 | 8% | 61% | collapses |

On the domain where the constraint barely bites, ignoring it costs nothing
measurable. On the two where it bites, the same rule wastes most of its budget
and cannot fill the design set. The failure tracks the underlying biophysics
rather than being a quirk of one protein.

That makes the practical advice conditional and testable rather than universal:
**measure what the constraint costs you at the optimum before choosing a
method.** It is a cheap calculation on the data you already have — no search
required — and it tells you whether you can get away with filtering at the end.

Regenerate with `python experiments/replication.py`.

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

## Relation to prior work

Worth being exact about, because the loose version of this claim is wrong.

**Hypervolume is not broken.** It measures the volume of objective space a
solution set dominates, and it was defined for *unconstrained* multi-objective
problems where every point is a legitimate trade-off. It does that correctly.
The failure shown above is what happens when it is applied **unchanged** to a
problem with a hard constraint, so that violating designs still contribute
volume.

**Constrained optimization already solved the reporting problem.** The
constrained multi-objective evolutionary computation literature has handled
this for decades: hypervolume computed over feasible solutions only, set to
zero when nothing is feasible, with feasibility ratio reported as a matter of
course, against benchmark suites built around constraints (CF, MW, DAS-CMOP,
LIR-CMOP). None of that is new here, and this project claims none of it.

**The gap is in protein design specifically.** Multi-objective protein design
work -- LaMBO, EvoMOBO, MP2D, NSGA-II with ProteinMPNN -- frames these problems
as trade-offs and reports hypervolume and front coverage. Because no
feasibility threshold is defined, feasibility rate never appears. There are
reasonable causes for that, not an oversight:

- for most design objectives there is no principled threshold; what ddG_folding
  counts as "unfolded" is continuous and context-dependent
- until datasets measured *both* properties for the *same* variants, there was
  no measured ground truth to define a constraint against
- Pareto framing is genuinely more useful when the threshold is unknown
- the two literatures barely overlap: constrained-MOEA machinery sits in
  evolutionary computation, protein design ML grew out of generative modelling

**So what is contributed here** is a benchmark, not a method or a metric: a
protein sequence-design problem where the hard constraint is a biophysical
property with experimentally measured ground truth; one search engine with the
constraint rule as the only variable; and a measurement of what the current
protein-design reporting convention hides. The Lagrangian rule itself is
adapted from Ctrl-DNA, and the constraint-handling rules are all standard.

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
pytest                                    # 159 tests, no data or network needed

python data/download.py                   # fetches the ddPCA tables into data/raw/
for d in GB1 GRB2-SH3 PSD95-PDZ3; do
  python experiments/tau_sweep.py --landscape additive --domain $d
done
python experiments/replication.py         # cross-domain figure and significance tests
python experiments/baselines.py --track additive    # greedy baseline + exact optimum
python experiments/baselines.py --track synthetic   # the same check, with epistasis
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
  methods.py     engine x rule, two readouts  metrics.py     feasibility, yield, hypervolume, significance
  baselines.py   greedy + exact optimum
experiments/     tau_sweep, replication, baselines, tuning_cost, budget_curves, report, plotting
data/download.py fetches and parses the ddPCA supplementary tables
```

The data is **never committed**; `data/raw/` is gitignored and `download.py` is
how you get it.

---

## Limitations

- **The measured track cannot rank the rules on design quality.** A
  search-free sort ties or beats every rule on all three domains, so the
  search is not the binding factor there. Established by
  [`baselines.py`](experiments/baselines.py) and stated up front rather
  than buried here. The rankings that survive are on the synthetic track,
  where the same baseline loses by about 2 kcal/mol.
- **No method gets close to optimal.** Where the optimum is computable the
  best rules capture 68-85% of it. Nothing here is evidence that any of
  these rules searches well in absolute terms.
- **Three domains is still three.** All three ddPCA domains are swept, which is
  enough to show the effect tracks conflict strength but not enough to fit that
  relationship. Three points define a direction, not a curve.
- **Five seeds is a thin basis for a significance test.** Welch at n=5 detects
  only large effects. "Tied" here means *not shown to differ*, which is weaker
  than *shown to be equivalent*; a 20-seed run could separate rules this one
  cannot.
- **No correction for multiple comparisons.** Each domain runs four pairwise
  tests against the leader. The separated results (p < 0.05, and p < 0.005 for
  the clearest) mostly survive a Bonferroni-style adjustment, but the marginal
  ones do not, and none of the ties would change.
- **Small binding domains.** May not transfer to large proteins or to designed
  binders.
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
