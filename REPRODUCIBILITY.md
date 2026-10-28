# Reproducing the numerical experiments

## Environment

The reference runs used Python 3.12 on Linux, an AMD EPYC 9V74 processor,
and one numerical linear algebra thread. Direct dependencies are pinned in
`requirements.txt`. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` before starting Python. The algorithms run on a CPU and
use double precision. Matplotlib uses its noninteractive `Agg` backend.

Exact wall times depend on hardware, numerical libraries, and operating-system
scheduling. A wall-clock budget also changes which final iteration completes.
Seeds reproduce the generated problems; they do not fix elapsed times or final
iteration counts. Published counts and times belong to the included CSVs.

## Instance generation

`game_problem.make_game(seed, n, m)` constructs a unit-ball VI with
`F(x) = M @ x - d`, where the skew part of `M` represents a sparse bilinear
game and its symmetric part is diagonal, positive semidefinite, and singular.
The construction bounds the operator norm by 1.4 and normalizes `||d||_2=1`.
The strictly feasible initializer is `x_f = -0.35*d`.

Every coupling constraint is a convex diagonal quadratic. Five recurring
resource prototypes have independent row perturbations and offsets giving
margin `-0.04` at the initializer before scaling. Dividing all constraints by
`sqrt(m)` preserves the feasible set. Consequently, the reported violation
norm is the root-mean-square violation of the unscaled resource limits.
Increasing `m` adds related limits, rather than independent resource types.

NumPy `SeedSequence([seed, 0])` initializes the operator stream;
`SeedSequence([seed, i+1])` initializes the perturbation stream for zero-based
constraint `i`; and `SeedSequence([seed, 100+i%5])` initializes its prototype.
Curvature samples use seed `seed+10000`. The full construction and game
interpretation are given in `latex/sec_numerical_experiments.tex`.

| Study | Settings | Seeds per setting | Paired instances |
| --- | --- | ---: | ---: |
| PFAOS | `m=1`, `n=25,50,100,200,500,1000` | 0 through 9 | 60 |
| MC-PFAOS, constraint count | `n=50`, `m=2,5,10,20,40,100,250,500` | 0 through 9 | 80 |
| MC-PFAOS, dimension | `m=5`, `n=25,50,100` | 0 through 9 | 30 |

The shared setting `(n,m)=(50,5)` is run once, leaving 100 distinct paired
MC-PFAOS instances. Both methods receive the same realized instance and point.

## Algorithm and measurement conventions

PFAOS projects onto a ball intersected with an affine halfspace. MC-PFAOS
uses a ball projection for its primal update and nonnegative-orthant projection
for its dual update. Both use `Lambda=100` and four local curvature sample
pairs; initialization costs and first-order calls are charged to the methods.
No global Lipschitz or constraint-smoothness constants are supplied.

The reported projected OE is `cvx_benchmark.projected_oe`. It is run three
times per instance, starting at `M0=M`, `0.1*M`, and `10*M`. Here `M=||B||_2`
is computed to numerical precision using a deterministic sparse singular-value
solve outside algorithm timing. The operator line search is unchanged. No
constraint-smoothness constant is an OE input: the original constraints are
handled by projection. Each trial,
including a feasible center, builds a fresh CVXPY model for the complete
nonlinear projection. Clarabel uses feasibility, absolute-gap, and relative-gap
tolerances of `1e-7`, with warm starts disabled. An accepted result must be
optimal and independently feasible to `2e-6`. Model construction,
canonicalization, conic solves, checks, and rejected trials are timed. This
policy is part of the benchmark definition; it is not a cached or specialized
projection implementation.

Each method runs for three seconds, with a safeguard of 10,000 outer
iterations. Only completed outer averages are reported. Checkpoints include
zero, the final completed iteration, and the sequence starting at one with
`k_next = max(k+1, ceil(1.5*k))`.

The weak VI gap uses `F(y)` at the comparison point:

```text
gap(x_bar) = max(0, sup_{y in feasible set} <F(y), x_bar-y>)
violation(x_bar) = ||[G(x_bar)]_+||_2
residual(x_bar) = max(gap(x_bar), violation(x_bar))
```

For the affine operator, the supremum is evaluated through an equivalent
convex quadratic minimization over the original feasible set. A separate
CVXPY/Clarabel model evaluates gaps after all timed runs. Diagnostic time
is saved separately. Only an optimal diagnostic result qualifies for a
numerical target crossing. A zero weak gap alone need not imply feasibility.

The saved `gap` uses the numerical **primal** objective of that minimization;
it is an estimate, not an independently certified upper bound. Write the true
minimum as `p*`, and let `c = -d.T @ x_bar`. A feasible primal value `p`
satisfies `p >= p*`, so `max(0, c-p)` is a **lower** bound on the true gap.
A verified dual-feasible lower bound `ell <= p*` would instead give the upper
bound `max(0, c-ell)` required to certify a gap threshold. Solver status and
the requested `1e-7` tolerances alone do not establish that certificate.
The released CSVs contain neither dual bounds nor checkpoint decision vectors;
their target counts describe numerical estimates and cannot be relabeled as
certified counts. New certification would require recovering the iterates and
verifying valid bounds, including numerical feasibility and rounding error.

`--target` specifies the retrospective reporting threshold. It does not stop
the timed algorithms; they continue to the budget or iteration safeguard.
Target times are first observed checkpoint times, which can overestimate a
first crossing between checkpoints. Tables report success fractions, median
time and work over successful runs, and median final residual over all runs.
Timeouts are never counted as successful three-second target times. All times
are wall-clock seconds without normalization; only the generated instances
are normalized. All three OE configurations are reported separately, so no
best-trajectory selection or fallback is applied.

Plots use each seed's latest saved checkpoint at a common time, without
interpolation or running minima. They show medians and 25th--75th percentile
bands. Time axes are linear. The weak-gap axis includes zero using a linear
region below `1e-4` and a logarithmic region above it. The initial point is
displayed at time zero; subsequent times retain initialization costs.

The additional two-panel summaries use the same checkpoint convention at
`(n,m)=(1000,1)` and `(50,100)`, with logarithmic time from `1e-4` to three
seconds. Their shaded bands are pointwise 95% percentile-bootstrap confidence
intervals for the median, using 2000 resamples of whole seed trajectories
with fixed random seed `20261028`. They are not interquartile ranges or
simultaneous confidence bands. The gap summary retains a linear region below
`1e-4` to include actual zero estimates. The combined-error summary uses
logarithmic axes and aggregates each seed's saved `residual`, so infeasible
points with zero gap are not shown as jointly accurate. Neither summary
smooths rebounds, interpolates between checkpoints, or changes the raw results.

## Checkpoint CSV columns

Each row describes one averaged iterate. Published CSVs contain no decision
vectors; the instances and iterates can be regenerated by the drivers.

| Column | Meaning |
| --- | --- |
| `seed`, `n`, `m`, `method` | Paired instance and method identifiers |
| `k` | Completed outer iteration, with zero for the initializer |
| `elapsed` | Cumulative algorithm seconds at this checkpoint |
| `F_calls` | Cumulative operator calls |
| `grad_calls` | Scalar constraint-gradient calls or full Jacobian calls, including the four calibration calls; zero for OE |
| `projection_calls` | Cumulative successful generic projection solves on the recorded trajectory |
| `projection_time` | Cumulative time in those projection requests, including construction and independent checks |
| `gap`, `violation`, `residual` | Numerical primal-objective weak-gap estimate, measured violation, and their maximum; no certified upper gap bound is saved |
| `gap_status` | Diagnostic solver status |
| `diagnostic_seconds` | Gap/violation postprocessing time for this row, outside algorithm timing |
| `solver_status` | Termination status of the complete run, repeated on its checkpoint rows |
| `target` | Retrospective residual target |
| `L0` | Sampled initial constraint-curvature estimate for the proposed method; blank for OE |
| `M_true` | Spectral norm of the realized operator matrix, retained as experiment metadata |
| `M0`, `M0_factor` | OE starting operator estimate and its factor relative to `M_true`; blank for the proposed method |
| `multiplier_max` | Maximum recorded outer dual norm for the proposed run; zero for PFAOS, blank for OE |
| `gradient_label` | Proposed method's first-order counter label |

`solver_status="Time limit"` can coexist with a successful target crossing:
the method continues after that crossing. `gradient_label` is carried as
instance metadata on OE rows; OE does not make the corresponding calls.
Failed or unfinished final projections do not add a new checkpoint. Thus the
last checkpoint's time and work need not include unfinished terminal work.

## Source and result provenance

The repository's original experiment package supplies the unchanged generator,
PFAOS/MC-PFAOS recursions, and CVXPY/Clarabel projection policy. The current
release reruns all methods on the same 160 instances, with three OE starting
operator estimates per instance. Raw CSVs, summaries, figures, and the LaTeX
results discussion belong to this rerun.

The only algorithm input changed in OE is its initial operator estimate.
Constraint handling, solver tolerances, the operator line search, initialization,
seed grids, target, and time budgets remain as before. Each main driver now
runs its proposed method and all three OE configurations before evaluating
weak gaps outside their timed runs.

The files `problem.py`, `verification.py`, and `projected_oe.py` retain the
earlier dense generator and specialized Dykstra baseline for regression tests.
They do not generate the released experiment tables. `rerun_generic_oe.py` is
a maintenance utility that preserves proposed-method rows and remeasures the
three OE initializations; the released rerun uses the two main drivers to
remeasure all methods together.

## LaTeX

From the repository root, a standalone preview can be built with a standard
LaTeX installation containing `amsmath`, `amssymb`, `graphicx`, `geometry`,
and `hyperref`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=latex latex/preview.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=latex latex/preview.tex
```

The wrapper supplies the two algorithm references that would normally resolve
in the full manuscript. It needs no journal font or class package. The
numerical section reads figures and tables from `results/figures/`; the wrapper
and section are intended to be compiled from the repository root. To use a
new experiment run in the section, regenerate into that directory in a working
copy or change the section's figure/table paths together.

## Automated checks

GitHub Actions installs the reference Python version and pinned dependencies,
runs the test suite, regenerates the figures and tables from released
checkpoints, and compares the three regenerated LaTeX tables with the tracked
copies. It does not rerun the full time-limited study in CI.
