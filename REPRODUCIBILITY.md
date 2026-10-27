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

The reported projected OE is `cvx_benchmark.projected_oe`. Each trial,
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
CVXPY/Clarabel model evaluates gaps after both timed runs. Diagnostic time
is saved separately. Only an optimal diagnostic result qualifies for a
target crossing. A zero weak gap alone need not imply feasibility.

`--target` specifies the retrospective reporting threshold. It does not stop
the timed algorithms; they continue to the budget or iteration safeguard.
Target times are first observed checkpoint times, which can overestimate a
first crossing between checkpoints. Tables report success fractions, median
time and work over successful runs, and median final residual over all runs.
Timeouts are never counted as successful three-second target times.

Plots use each seed's latest saved checkpoint at a common time, without
interpolation or running minima. They show medians and 25th--75th percentile
bands. Time axes are linear. The weak-gap axis includes zero using a linear
region below `1e-4` and a logarithmic region above it. The initial point is
displayed at time zero; subsequent times retain initialization costs.

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
| `gap`, `violation`, `residual` | Accuracy components and their maximum |
| `gap_status` | Diagnostic solver status |
| `diagnostic_seconds` | Gap/violation postprocessing time for this row, outside algorithm timing |
| `solver_status` | Termination status of the complete run, repeated on its checkpoint rows |
| `target` | Retrospective residual target |
| `L0` | Sampled initial curvature estimate; blank for OE |
| `multiplier_max` | Maximum recorded outer dual norm for the proposed run; zero for PFAOS, blank for OE |
| `gradient_label` | Proposed method's first-order counter label |

`solver_status="Time limit"` can coexist with a successful target crossing:
the method continues after that crossing. `gradient_label` is carried as
instance metadata on OE rows; OE does not make the corresponding calls.
Failed or unfinished final projections do not add a new checkpoint. Thus the
last checkpoint's time and work need not include unfinished terminal work.

## Source and result provenance

The reference code and raw data were taken from
`pfaos_mc_pfaos_projected_oe_python_code.zip`. Its checkpoint CSVs are preserved
byte for byte. The newer manuscript section had slightly different timing
summaries; the repository uses the released raw CSVs as its source of truth.
The expanded section's method description and numerical discussion have been
reconciled with those CSVs and the implemented all-trial projection policy.
Figures and tables can be regenerated directly from them.

Repository preparation changes only driver defaults, output-collision
protection, documentation, and checks. It does not change either proposed
algorithm's recursions, the game generator, or the reported OE implementation.

The files `problem.py`, `verification.py`, and `projected_oe.py` retain the
earlier dense generator and specialized Dykstra baseline for regression tests.
They do not generate the released experiment tables. `rerun_generic_oe.py` is
a historical maintenance utility that preserves proposed-method rows and
remeasures OE; use the two main drivers for independent full reruns.

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
