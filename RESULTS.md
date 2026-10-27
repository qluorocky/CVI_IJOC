# Constrained-game numerical results

The two studies use ten paired random seeds per setting, a three-second cap
per algorithm and seed, and the common target
`max(weak VI gap, norm of positive constraint violations) <= 0.02`.
Gap evaluation runs after the measured solve. Times below are medians at the
first saved checkpoint meeting the target; a dash means no seed met it.

## Settings

| Study | Dimensions | Coupling constraints | Proposed method | Baseline |
| --- | --- | --- | --- | --- |
| Single constraint | n = 25, 50, 100, 200, 500, 1000 | m = 1 | PFAOS | Projected OE |
| Constraint count | n = 50 | m = 2, 5, 10, 20, 40, 100, 250, 500 | MC-PFAOS | Projected OE |
| Additional dimension check | n = 25, 50, 100 | m = 5 | MC-PFAOS | Projected OE |

Instances are continuous-strategy regularized bilinear zero-sum games on the
unit ball, with a monotone but non-strongly-monotone affine VI operator and
convex singular quadratic resource constraints. Each instance has a common
strict feasible point. For larger `m`, five resource families are repeated
with small distinct perturbations; each constraint is rescaled by
`1/sqrt(m)` without altering feasibility. The algorithms start at the same
feasible point and do not receive global Lipschitz constants. Every nontrivial
OE trial, including one with a feasible center, builds a fresh CVXPY problem and solves it with the generic
Clarabel conic solver, without a parameterized cache or warm start. Its
accepted projections receive a separate NumPy
feasibility check. The audit kept the existing PFAOS/MC-PFAOS traces, whose
updates were unchanged, and reran OE on the same seeds and instances with the
fresh-solve policy. The code also includes full drivers to rerun both methods.
All runs use one numerical linear algebra thread on an
AMD EPYC 9V74 host. Detailed formulas and tolerances are in the LaTeX section.

## Single-constraint results

| n | PFAOS success | PFAOS median seconds | OE success | OE median seconds |
| ---: | :---: | ---: | :---: | ---: |
| 25 | 10/10 | 0.088 | 10/10 | 0.163 |
| 50 | 10/10 | 0.073 | 10/10 | 0.185 |
| 100 | 10/10 | 0.085 | 10/10 | 0.208 |
| 200 | 10/10 | 0.081 | 10/10 | 0.285 |
| 500 | 10/10 | 0.090 | 10/10 | 0.564 |
| 1000 | 10/10 | 0.102 | 10/10 | 1.026 |

PFAOS reaches the target sooner at every tested dimension with fresh generic
OE projections. At `n=1000`, PFAOS uses a median 2,039 operator evaluations and about
124 constraint-gradient calls by the target checkpoint. OE uses 28 operator
evaluations and 27 generic nonlinear projections. Thus PFAOS's observed time
gain comes with much more operator work; for OE, nearly all measured solver
time is spent on model construction and projection solves at `n=1000`.

## Many-constraint results at n = 50

| m | MC-PFAOS success | MC-PFAOS median seconds | OE success | OE median seconds |
| ---: | :---: | ---: | :---: | ---: |
| 2 | 10/10 | 0.625 | 10/10 | 0.175 |
| 5 | 10/10 | 1.146 | 10/10 | 0.129 |
| 10 | 10/10 | 1.143 | 10/10 | 0.128 |
| 20 | 10/10 | 1.258 | 10/10 | 0.144 |
| 40 | 10/10 | 1.199 | 10/10 | 0.181 |
| 100 | 10/10 | 1.198 | 10/10 | 1.890 |
| 250 | 10/10 | 1.326 | 9/10 | 2.490* |
| 500 | 10/10 | 1.577 | 0/10 | — |

*The OE median at `m=250` is over its nine successful runs. At `m=500`, its
median final residual across all ten runs is 0.140.*

OE remains faster through `m=40`: even a full generic projection can be
cheaper overall when OE needs only 18 to 27 projections and MC-PFAOS needs
tens of thousands of operator evaluations. MC-PFAOS becomes faster at `m=100`
under this generic projection implementation. At `m=500`, MC-PFAOS succeeds in all
ten seeds, using a median 38,545 operator evaluations, 419 full Jacobian
evaluations, and 209,500 individual component-gradient equivalents. OE reaches
the target in none of ten before the cap. In the additional `m=5` dimension
check, OE is faster at every tested `n=25,50,100` (median times
`0.107,0.129,0.149` seconds versus `0.971,1.146,0.889` for MC-PFAOS).
Increasing dimension at this low fixed constraint count does not create a
wall-time advantage for MC-PFAOS in this synthetic game.

## Residual rebound and separate figures

Four standalone plots show the weak VI gap and positive constraint-violation
norm separately for each study. Both methods start from the same initial point
at time zero in the figures; subsequent wall-clock time includes setup work.
The curves compare the most recently saved diagnostic checkpoint at each
common time and are not smoothed to make them monotone. All horizontal axes
show actual elapsed seconds linearly. The first comparison presents all six
dimensions on the same 0--1.2 second range, encompassing all observed target
crossings. Both MC-PFAOS figures show the early transient alongside the full
three-second run, with linear axes in both views.

At `n=50,m=100`, MC-PFAOS's median weak gap falls from `0.256` initially
to `0.066` at checkpoint 8 as its median constraint violation rises from
zero to `0.058`. At checkpoint 27, its median weak gap is zero, but the
median violation is `0.524`; the *combined target metric* therefore rises
before dual updates reduce the violation. Zero weak gap alone does not imply
feasibility or attainment of the target. The raw weak gap is nonincreasing
at all saved MC-PFAOS checkpoints across this experiment. This empirical
behavior is not a pointwise monotonicity theorem; the convergence upper bound
does not require the combined target metric to decrease. PFAOS and projected OE have
zero coupling-constraint violation at every saved single-constraint
checkpoint, which the separate PFAOS violation figure makes explicit.

## Audit and interpretation

Nine regression, numerical, and release-integrity checks pass. The independent SciPy/SLSQP
crosscheck agrees with the Clarabel weak-gap result to `8.49e-9`; all saved
gap checks report optimal status. The released `AUDIT.md` documents the
algorithm correspondence, the projection feasibility check, counted
initialization and rejected trials, and the memory-layout correction that
prompted a complete rerun of the PFAOS grid.

The times are checkpoint times, so they conservatively bound unobserved first
crossings. A time-limited run is censored, not assigned a successful three
second result. The generic projected OE baseline is not a claim about every
specialized projection method. The maximum dimension and maximum constraint
count belong to different experimental grids; no run combines `n=1000` and
`m=500`. The larger constraint sets contain near-duplicate resource families.
The expanded LaTeX section in `latex/` uses these released CSVs and figures;
`latex/preview.tex` provides a standalone wrapper without the journal class.
