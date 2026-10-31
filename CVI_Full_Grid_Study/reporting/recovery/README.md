# Separate offline numerical gap-recovery probe

This directory is **not** part of the frozen timed method source. Its program
reads saved checkpoint points and writes new sidecars here. It does not modify
case manifests, run records, points, diagnostic records, or running workers.
Existing timed method failures remain observed outcomes.

The independent reconstruction reproduces the original instance SHA-256. For
large cases it opens `quadratic_factors.dat` in read-only mode, consumes the same
random QR-input draws without performing QR, and reconstructs only the small
operator/vector arrays. The full 4 GB factor mapping uses the same underlying
file and shared OS pages. It never builds a dense stack of A_i or a full conic
model. Environment variables and `threadpool_limits(1)` request one numerical
thread. The main owned arrays are the 8 MB B, the 8 MB b, selected Jacobians, and
SLSQP's quadratic workspace, well below 512 MiB; reported peak RSS includes
resident pages of the shared read-only mapping and is not private memory.

For saved z, the convex subproblem minimizes

    f(y) = h @ (y*y) - (B.T @ z + d) @ y

subject to the radius-2 ball and the original quadratic constraints. The weak
gap is max(0, -min(f) - d @ z). The solver starts with the constraints most
violated by the ball-only minimizer. It solves selected constraints with fresh
SLSQP, scans **every** original constraint, and monotonically adds the worst
violations. A global minimizer of a restricted problem which is feasible for
every original constraint is also a global minimizer of the complete problem.
SLSQP success and the original 1e-7 feasibility tolerance establish numerical
optimization evidence, not formal certification.

Version 1's initial largest-case probe solved 10 selected constraints but failed
with SLSQP status 8 after expanding to 20. That unresolved result and source are
preserved. Version 2 restores a strictly feasible selected-set start on the
segment from the original strict-feasible x0 to the previous solution before
each restricted solve. It does not convert failed SLSQP statuses into success.

Version 2 also fits nonnegative multipliers for nearly active original
constraints and the ball. Convexity gives the numerical estimate

    min_full(f) >= L(y) - (radius + norm(y)) * norm(gradient(L)(y)).

The sidecar records that estimate, original feasibility, stationarity,
complementarity, active IDs, all solver statuses, deadlines, saved candidate
points, input/source hashes, and the original diagnostic record. Floating-point
arithmetic is not an interval proof, so this must not be described as a formal
upper gap certificate. Failed recovery reports retain gap=null.

The optimization deadline uses a wall-clock alarm and checks in callbacks. A
separately recorded final original-feasibility scan and small KKT check follow
the timed solve. Instance reconstruction and hash validation are setup work and
are recorded separately. These times are offline postprocessing, never method
runtime measurements.

## Future retry admission and reconstruction

The current infrastructure revision refuses working/incomplete cases. It accepts
only the controller's `verified` snapshot, which is copied after worker exit,
with all requested methods complete and no recorded active worker for the case.
It validates plan/source/budget identity, run metadata, timed-pair checksums,
finite point arrays, checkpoint counts/times/iteration indices, and original
diagnostic identities. It snapshots plan/manifest/run/point/diagnostic hashes and
checks them again before writing a result. The CLI also waits for a quiescent
controller progress state, with no recorded active workers. These are admission
barriers; the script still has no integration/adoption operation, and explicit
controller coordination remains necessary to prevent a concurrent resume.
The requested method must belong to the validated plan; checkpoint indices
must be nonnegative and unique, and existing candidate files are never
overwritten. Output directories must be new and outside the original results.

Serialize recovery, factor reconstruction, and the factorwise structural audit:
run only one of these heavy jobs at a time. The guard checks recorded numerical
workers; it is not a global process lock and does not detect a separately
launched audit. Root/controller coordination must enforce this serialization.

`--reconstruct-factors` is optional and required when the original cache was
deleted. It generates one QR factor at a time into a uniquely named temporary
memmap outside the original results tree, checks the complete original instance
hash, and reopens the mapping read-only. Temporary storage is removed at exit.
The largest case needs 4 GB scratch disk plus the retained 12 GiB disk reserve.
Large reconstruction refuses concurrent workers; owned QR buffers remain
factorwise. Its setup budget defaults to 300 seconds, checked between factors,
and its actual time is recorded separately from optimization and hash validation.
Reconstruction may be expensive. No large reconstruction was run by this
revision; only a 10 KB n=50,m=1 fixture confirmed exact hash identity, read-only
reopening, temporary cleanup, and guard rejection behavior.

A **potential manual** 300-second diagnostic retry after worker completion is
shown below. This command has not been run or scheduled. It uses a new output
directory, preserves prior attempts, and has no resampling or automatic retry.

    ../.venv/bin/python reporting/recovery/constraint_generation_gap.py \
      --case results/verified/multi_n1000_m1000_seed0 \
      --checkpoints 0 --seconds 300 --reconstruct-factors \
      --reconstruction-seconds 300 --attempt-label manual_postcompletion_300s \
      --output reporting/recovery/future_manual_300s_attempt

The reconstruction/setup budget and the 300-second optimization budget are
separate offline costs. The final original-feasibility scan is timed separately;
the numerical multiplier/KKT check has its own 5-second deadline. Its failure is
recorded without losing the solver result, candidate, or original status, and
the recovered gap remains unresolved. Any threshold decision near 1e-5 still
needs numerical-error assessment and an independent audit; SLSQP success alone
does not establish that level of gap accuracy.

Do not insert sidecar values into a case while its worker is evaluating that
case. Integration, if warranted, requires a documented new diagnostic policy,
preservation of original failure attempts, and separate audit after the worker
has finished.

## Completed bounded probes

No further large probes are currently scheduled by this directory.

| Probe | Outcome | Offline optimization time | Maximum all-original violation |
|---|---|---:|---:|
| n=50,m=10,seed=0 checkpoints 0,1,2,34 (v1 and v2) | All numerical solves valid; exact instance hash verified | 0.003–0.008 s each | 5.1e-11 |
| n=1000,m=1000,seed=0 checkpoint 0 (v1) | Unresolved: selected-set SLSQP status 8 after expansion to 20 constraints | 1.538 s | 1.093 |
| Same largest checkpoint (v2) | Unresolved: 55 s optimization deadline in 40-constraint round | 55.010 s | 0.1942 |

The small-case first three gaps agree with saved independent conic evaluations
within 1.8e-8. Both largest-case diagnostic attempts retain gap=null. The
v2 feasibility restoration allowed the 10/20/30-constraint subproblems to finish
with SLSQP success, but each took 10.9–12.0 seconds under the concurrent worker
load. Largest-case hash validation/setup took 7.12 s for v1 and 21.42 s for v2;
v2's final full feasibility scan and quality check took about 2.20 s. The two
large attempts used approximately 56.55 s total optimization time. The wall
alarm behaved as intended. Peak RSS, including the shared factor mapping,
was 4.11 GB (v1) and 3.86 GB (v2).

These probes do **not** establish cheap recovery for the largest case. Do not
integrate their invalid values. Any future attempt should be a documented new
offline diagnostic policy after controller/worker coordination. Possible future
experiments include objective normalization, larger constraint batches, or a
selected-set dual/trust-region method; none was run here. The original source,
timed outcomes, and diagnostic failure records are preserved.

The v1 and v2 program snapshots match the source hashes stored in their
respective benchmark JSON files. Each benchmark directory also contains the
actual final candidate point, including failed candidates, for later inspection.

`attempt_history.json` inventories all four original benchmark directories and
their artifact checksums. Those attempt files and immutable source snapshots
were not changed by the infrastructure revision.

## Correction to preserved v2 quality labels

The preserved v2 failure has full-set violation 0.194. Its field named
`numerical_objective_bracket_width` (about 5.96e-5) is therefore **not** a full
primal optimum bracket: an infeasible candidate objective is not a full primal
upper bound. Its multiplier-derived objective lower estimate remains useful,
and its reported gap was correctly null. The current source exposes full
objective upper/bracket fields only for full-feasible candidates; otherwise it
labels candidate-minus-lower only as a residual estimate. It also rejects
nonfinite candidate/objective/all-G/raw-gap values. Preserved v1/v2 JSON and
source snapshots retain their original fields as actual attempt history.

In v2, `final_verification_seconds` included the numerical KKT-check duration.
The current source records feasibility verification and KKT time separately to
avoid double counting. No revised largest-case numerical result is claimed.

## Single-only release recovery

`--stable-single-snapshot` is an explicit read-only exception to whole-study
quiescence. It requires `release_scope=single_only`, all six single dimensions,
ten seeds, five methods, and a completed verified target absent from pending or
active records. It allows other pending single workers to finish; nonsingle
work is refused. All target identity/checksum/stability guards remain active.
Root explicitly authorized this mode while holding the new MC controller.

Root authorized one 300-second SLSQP retry for each of eight original invalid
single diagnostics. `single_delivery_v1` preserves all eight status-8 outcomes
with gaps unresolved; no original diagnostic was overwritten. Root then
authorized one independent two-multiplier QCQP dual retry per point. The
separate `recover_single_dual.py` creates only one A=W.T@W (at most 8 MB), uses
positive log multipliers and an analytic two-dimensional dual Jacobian, and
checks the original factor constraints, ball, stationarity, complementarity and
numerical primal-dual bracket. A tiny segment toward x0 supplies a strictly
feasible primal endpoint when necessary.

All eight `single_delivery_v2` posterior checks passed: maximum original
violation zero, stationarity norm below 2.04e-14, numerical bracket widths
3.83e-14–1.92e-12. These are numerical floating-point bounds, not formal interval
certificates. Five root solves returned status 1. Three Golden Ratio root
solves returned stagnation status 5 despite machine-precision residuals; those
statuses remain recorded. Numerical acceptance uses the independent posterior
primal/dual/KKT checks, rather than rewriting the optimizer's exit status.

Each v2 sidecar binds input/source/instance identity, complete original and v1
attempts, feasible candidate NPZ and hash, exact measured solver/module source
snapshots, and the numerical quality checks. Candidate binding added metadata
only; original solver-produced JSON is preserved separately. Root controls any
subsequent adoption into copied release data after independent review. The
recovery scripts never edit original diagnostics or timed method pairs.
