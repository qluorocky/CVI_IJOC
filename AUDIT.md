# Numerical implementation audit

## Source-to-paper correspondence

- PFAOS and MC-PFAOS start from the attached implementations. Their inner
  proximal weights are respectively `2*M + eta*M*sum(1/M_i)` and
  `3*M + eta*M*sum(1/M_i)`. Their middle tests and operator/coupling line
  searches match Algorithms 1 and 2 in the manuscript. The changes add
  elapsed-time histories and stop only at safe, completed outer iterates.
- PFAOS's large checkpoint histories now use column-contiguous uninitialized
  arrays. Only completed columns are read. Profiling found that the original
  row-major arrays spent about `0.3--0.4` seconds touching scattered memory
  pages before the first update at `n=1000`; this is a history-layout issue,
  not an algorithm step. Every PFAOS result in the release is rerun with
  column-contiguous storage.
- The **reported** OE implementation is `cvx_benchmark.py`, not the retained
  original Dykstra implementation in `projected_oe.py`. OE applies the same
  extrapolation, adaptive trial `M`, and ergodic averaging as the manuscript,
  while every projection trial, including a feasible center, uses a fresh CVXPY problem
  and the generic Clarabel conic solver. Model construction, canonicalization,
  rejected projections, and the solver are all on OE's clock. The fresh solves
  use no warm start or parameterized compilation cache.
- To isolate this correction, `rerun_generic_oe.py` retains the prior
  PFAOS/MC-PFAOS rows and remeasures only OE on exactly the same seeds and
  generated instances. The unchanged proposed implementations can also be
  rerun end to end with the main drivers.
- The four local curvature pairs are generated exactly once per proposed
  method and seed. Their four gradient/Jacobian queries and elapsed time are
  included in the proposed method's figures and tables. Rejected search
  trials are included in operator calls, OE projection counts, and time.

## Mathematical and numerical checks

- The interaction block is skew symmetric. The symmetric part of `M` is the
  singular diagonal `H >= 0` and `||M||_2 <= 1.4` by construction.
  `||d||_2 = 1`, `||x_f||_2 = 0.35`, and each unscaled constraint has margin
  `-0.04` at `x_f`. Quadratic constraint matrices are positive diagonal with
  zeros, so the realized test problems are convex without requiring strong
  convexity. Scaling every row by `1/sqrt(m)` preserves feasibility and keeps
  the vector curvature scale bounded.
- The diagnostic solver minimizes
  `y.T @ H @ y - (M.T @ x_bar + d).T @ y` over the *original nonlinear feasible
  set* and adds the constant `-d.T @ x_bar` afterward. This is exactly the
  sign-reversed affine weak VI gap. The convex gap is measured outside timed
  solver runs and only `optimal` results count as target crossings. On an
  independently solved 12-dimensional check, CVXPY/Clarabel and SciPy/SLSQP
  gap values differed by `8.49e-9`, and SLSQP feasibility was `1.43e-11`.
- The CVXPY projection is accepted only when the solver reports `optimal`
  and an independent NumPy evaluation confirms ball and constraint
  feasibility within `2e-6`. A projection that consumes the remaining time
  without an accepted OE iterate is recorded as a time limit. The earlier
  benchmark reused a DPP-compiled, warm-started CVXPY projection and could
  substantially understate the wall time of the requested fresh generic solve.
  Both policies solve the same mathematical projection; no slow specialized
  projector was substituted to alter the ordering.
- The legacy source regression suite and the added monotonicity,
  linearization, weak-gap, generic-projection, and chart-alignment checks
  pass; the GitHub-ready release adds checks for raw-data consistency and
  protection of existing output directories (nine tests total).
- A fresh replay of seed 0 at `n=50,m=100` reproduced both saved weak gaps
  and constraint-violation norms at checkpoints 0, 8, 12, and 27 to numerical
  precision. The observed rebound is in the *combined* target diagnostic
  `max(gap, violation)`, caused by temporary coupling violations: the primal
  update projects onto the ball and dualizes the coupling limits. In the
  existing MC-PFAOS dataset, the weak VI gap is nonincreasing at every saved
  checkpoint for all 100 seeds across ten settings; this is an observation,
  not a theorem asserting per-iterate monotonicity on arbitrary instances.
  Its plot now displays the weak gap; a separate plot displays violation.
  The paper's `O(1/N^2)` bound does not assert that the combined diagnostic
  decreases at every checkpoint. The alignment check preserves the starting
  point at time zero without smoothing either raw component.

## Interpretation limits

- The performance claim concerns fresh, independently modeled generic
  CVXPY/Clarabel projections. A specialized analytic or Dykstra projector,
  or a cached and warm-started generic projection, is a different baseline
  and can perform differently. Rebuilding each generic model is the requested
  measurement convention and its cost is charged on every projection.
- The larger `m` instances reuse five resource families with small distinct
  perturbations to control the underlying feasible geometry. They illustrate
  projection cost under additional, partly redundant nonlinear limits, not
  five hundred independent resource types.
- All tabulated target times are observed at geometric checkpoints and are
  upper bounds on the unobserved first hitting times. A three-second timeout
  is censored; it is not an observation equal to three seconds. The weak gap
  and violation can be nonmonotone, especially for MC-PFAOS.
- Figures align seeds at common wall-clock times using their latest *saved*
  diagnostic checkpoint, and display the supplied initial point at time
  zero. Raw CSV elapsed times are unchanged; the proposed methods' setup
  time remains charged to subsequent iterates. A median curve's target
  crossing need not equal the median per-seed first target time in a table.
- Dimensions up to 1000 and constraint counts up to 500 are tested on
  separate axes. No experiment combines `n=1000` with `m=500`. The study is
  synthetic and reports medians and interquartile bands over ten seeds.

## GitHub-ready release

- The scientific recursions, normalized generator, reported OE implementation,
  and raw checkpoint CSVs are unchanged from the reference code archive.
- Driver defaults now use the full reported grids and three-second budget,
  and write to separate reproduction directories. Existing experiment output
  is rejected before its configuration or checkpoint CSV can be modified.
- Raw CSVs cover all 160 paired cases without duplicate checkpoints. Their
  per-seed summary CSVs agree with recomputed summaries. The expanded LaTeX
  numerical discussion uses this same dataset; slightly different timings
  in the later manuscript archive are not mixed into the release.
- GitHub Actions runs tests and regenerates figures and tables from the raw
  checkpoints; it compares the regenerated LaTeX tables with tracked copies.
