# Constrained-game numerical results

All methods were rerun on the same 160 paired instances, with ten seeds per
setting and a three-second budget per method. There are 640 timed trajectories
and 8,890 saved checkpoints. The target is
`max(weak VI gap, ||[G(x_bar)]_+||_2) <= 0.02`.

The three OE configurations start at `M0=M`, `0.1*M`, and `10*M`, with
`M=||B||_2` computed from the realized operator outside algorithm timing.
OE retains its original operator line search and fresh CVXPY/Clarabel
projection at every trial. No constraint-smoothness constant is supplied to
OE: the original nonlinear constraints are handled by projection. Instance
generation, proposed algorithms, local curvature calibration, projection
settings, target, and budgets remain as before.

Times below are medians at the first saved target checkpoint among successful
runs. A dash indicates no target crossing; final residuals use all ten seeds.
Offline weak-gap evaluation is excluded from algorithm timing.

## Single-constraint study

The dimension grid is `n=25,50,100,200,500,1000`, with `m=1`.

| n | Method | Reached | Median seconds | Median final residual |
| ---: | --- | :---: | ---: | ---: |
| 25 | PFAOS | 10/10 | 0.102 | 0.000193 |
| 25 | Projected OE (M) | 10/10 | 0.181 | 0.000711 |
| 25 | Projected OE (0.1M) | 10/10 | 0.305 | 0.00115 |
| 25 | Projected OE (10M) | 10/10 | 2.169 | 0.0101 |
| 50 | PFAOS | 10/10 | 0.079 | 0.000223 |
| 50 | Projected OE (M) | 10/10 | 0.181 | 0.000727 |
| 50 | Projected OE (0.1M) | 10/10 | 0.290 | 0.00106 |
| 50 | Projected OE (10M) | 10/10 | 2.349 | 0.011 |
| 100 | PFAOS | 10/10 | 0.089 | 0.000157 |
| 100 | Projected OE (M) | 10/10 | 0.234 | 0.000943 |
| 100 | Projected OE (0.1M) | 10/10 | 0.254 | 0.00112 |
| 100 | Projected OE (10M) | 10/10 | 2.858 | 0.0145 |
| 200 | PFAOS | 10/10 | 0.088 | 0.000118 |
| 200 | Projected OE (M) | 10/10 | 0.322 | 0.00116 |
| 200 | Projected OE (0.1M) | 10/10 | 0.487 | 0.00181 |
| 200 | Projected OE (10M) | 3/10 | 2.991 | 0.0208 |
| 500 | PFAOS | 10/10 | 0.107 | 0.000186 |
| 500 | Projected OE (M) | 10/10 | 0.616 | 0.00245 |
| 500 | Projected OE (0.1M) | 10/10 | 0.493 | 0.00194 |
| 500 | Projected OE (10M) | 0/10 | -- | 0.0515 |
| 1000 | PFAOS | 10/10 | 0.108 | 0.000189 |
| 1000 | Projected OE (M) | 10/10 | 1.131 | 0.00456 |
| 1000 | Projected OE (0.1M) | 10/10 | 0.884 | 0.00348 |
| 1000 | Projected OE (10M) | 0/10 | -- | 0.113 |

PFAOS reaches the target in all seeds, as do the OE configurations initialized
at `M` and `0.1*M`. At `n=1000`, their median target times are 0.108, 1.131,
and 0.884 seconds, respectively. The lower OE initialization can give faster
locally accepted steps despite its initial backtracking. The true global
operator norm is a valid bound, rather than an optimal local step estimate.

Starting at `10*M` is much slower: at `n=25`, it uses a median 315 projections
and 2.169 seconds to target, versus 27 projections and 0.181 seconds from `M`.
It succeeds in 3/10 seeds at `n=200` and none at `n=500` or `1000`. This line
search only increases its estimate, so it does not reduce a conservative
initial value. At `n=1000`, PFAOS uses a median 2,039 operator calls and about
124 constraint-gradient calls; OE uses 28 operator calls and 27 projections
from `M`, or 22 operator calls and 21 projections from `0.1*M`.

## Constraint-count study

Here `n=50` and `m=2,5,10,20,40,100,250,500`.

| m | Method | Reached | Median seconds | Median final residual |
| ---: | --- | :---: | ---: | ---: |
| 2 | MC-PFAOS | 10/10 | 0.633 | 0.00323 |
| 2 | Projected OE (M) | 10/10 | 0.202 | 0.000761 |
| 2 | Projected OE (0.1M) | 10/10 | 0.296 | 0.00101 |
| 2 | Projected OE (10M) | 10/10 | 2.418 | 0.0104 |
| 5 | MC-PFAOS | 10/10 | 1.241 | 0.00405 |
| 5 | Projected OE (M) | 10/10 | 0.115 | 0.000293 |
| 5 | Projected OE (0.1M) | 10/10 | 0.182 | 0.000393 |
| 5 | Projected OE (10M) | 10/10 | 1.518 | 0.00516 |
| 10 | MC-PFAOS | 10/10 | 1.194 | 0.00404 |
| 10 | Projected OE (M) | 10/10 | 0.136 | 0.000289 |
| 10 | Projected OE (0.1M) | 10/10 | 0.170 | 0.000358 |
| 10 | Projected OE (10M) | 10/10 | 1.520 | 0.00594 |
| 20 | MC-PFAOS | 10/10 | 1.182 | 0.00449 |
| 20 | Projected OE (M) | 10/10 | 0.149 | 0.000317 |
| 20 | Projected OE (0.1M) | 10/10 | 0.220 | 0.000464 |
| 20 | Projected OE (10M) | 10/10 | 1.624 | 0.00633 |
| 40 | MC-PFAOS | 10/10 | 1.209 | 0.0042 |
| 40 | Projected OE (M) | 10/10 | 0.166 | 0.000383 |
| 40 | Projected OE (0.1M) | 10/10 | 0.262 | 0.000536 |
| 40 | Projected OE (10M) | 10/10 | 1.931 | 0.00801 |
| 100 | MC-PFAOS | 10/10 | 1.156 | 0.00437 |
| 100 | Projected OE (M) | 10/10 | 1.742 | 0.00811 |
| 100 | Projected OE (0.1M) | 8/10 | 2.364 | 0.0126 |
| 100 | Projected OE (10M) | 0/10 | -- | 0.125 |
| 250 | MC-PFAOS | 10/10 | 1.314 | 0.00514 |
| 250 | Projected OE (M) | 9/10 | 2.140 | 0.0118 |
| 250 | Projected OE (0.1M) | 5/10 | 2.881 | 0.02 |
| 250 | Projected OE (10M) | 0/10 | -- | 0.135 |
| 500 | MC-PFAOS | 10/10 | 1.577 | 0.00576 |
| 500 | Projected OE (M) | 0/10 | -- | 0.138 |
| 500 | Projected OE (0.1M) | 0/10 | -- | 0.256 |
| 500 | Projected OE (10M) | 0/10 | -- | 0.182 |

OE initialized at `M` or `0.1*M` remains faster than MC-PFAOS through `m=40`.
At `m=100`, MC-PFAOS succeeds in 10/10 seeds at a median 1.156 seconds. OE
succeeds in 10/10 from `M` (1.742 seconds), 8/10 from `0.1*M` (2.364 seconds
among successful runs), and 0/10 from `10*M`.

At `m=250`, MC-PFAOS succeeds in 10/10 at 1.314 seconds; the three OE success
fractions are 9/10, 5/10, and 0/10. Their two defined conditional median times
are 2.140 and 2.881 seconds. At `m=500`, MC-PFAOS succeeds in 10/10 at
1.577 seconds, while no OE configuration reaches the target. The corresponding
median final residuals are 0.00576 for MC-PFAOS and 0.138, 0.256, and 0.182
for OE starting at `M`, `0.1*M`, and `10*M`. MC-PFAOS uses a median 38,545
operator calls and 419 full Jacobian calls, or 209,500 component-gradient
equivalents, by its target checkpoint.

The projection policy is identical in all three OE configurations. A low
operator estimate can require rejected full projections before a step is
accepted; these are costly at large `m`. A large estimate avoids backtracking
but gives smaller steps. The larger constraint sets reuse five resource
families with small perturbations and are scaled by `1/sqrt(m)`, preserving
feasibility and making the reported violation an unscaled root-mean-square
measure. These results concern that synthetic family and the specified generic
projection implementation.

## Dimension check with five constraints

| n | Method | Reached | Median seconds | Median final residual |
| ---: | --- | :---: | ---: | ---: |
| 25 | MC-PFAOS | 10/10 | 1.085 | 0.00425 |
| 25 | Projected OE (M) | 10/10 | 0.115 | 0.00025 |
| 25 | Projected OE (0.1M) | 10/10 | 0.208 | 0.000461 |
| 25 | Projected OE (10M) | 10/10 | 1.028 | 0.00389 |
| 50 | MC-PFAOS | 10/10 | 1.241 | 0.00405 |
| 50 | Projected OE (M) | 10/10 | 0.115 | 0.000293 |
| 50 | Projected OE (0.1M) | 10/10 | 0.182 | 0.000393 |
| 50 | Projected OE (10M) | 10/10 | 1.518 | 0.00516 |
| 100 | MC-PFAOS | 10/10 | 0.945 | 0.00373 |
| 100 | Projected OE (M) | 10/10 | 0.189 | 0.000548 |
| 100 | Projected OE (0.1M) | 10/10 | 0.242 | 0.000655 |
| 100 | Projected OE (10M) | 10/10 | 2.046 | 0.00934 |

With `M0=M`, OE remains faster at all three tested dimensions. The additional
initializations show its sensitivity to the operator estimate without changing
constraint handling.

## Figures and validation

Four PDF/PNG figures show weak VI gap and constraint violation separately.
Each has the same proposed method and the three OE starting estimates. Time
axes are linear; the single-constraint figures now show the full three-second
budget so late crossings and unsuccessful `10*M` runs remain visible. Lines
and interquartile bands hold the latest saved checkpoint, without interpolation
or running-minimum smoothing.

For `n=50,m=100`, MC-PFAOS's median weak gap falls from 0.256 initially to
0.066 at checkpoint 8, while median violation rises to 0.058. At checkpoint
27 the median weak gap is zero, but median violation is 0.524. The joint target
therefore still requires subsequent feasibility recovery. This transient is
retained in the figures. All saved single-constraint outputs and all OE
outputs have zero measured coupling-constraint violation.

All 8,890 diagnostic gap solves returned optimal status. Ten tests pass,
including a spectral-norm cross-check, OE initialization and backtracking
checks, released-data consistency, and output protection. See `AUDIT.md` and
`REPRODUCIBILITY.md` for implementation and reporting details.
