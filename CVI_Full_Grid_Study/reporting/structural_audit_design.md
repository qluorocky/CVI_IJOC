# Factorwise saved-record audit

`audit_saved_records.py` reads saved measurements without importing experiment
code or rerunning a timed algorithm. It writes only a new structural audit
report. The completed experiment source, data, and reports remain untouched.

The default contract is the requested grid: single-constraint cases with
`n = 10, 50, 100, 200, 500, 1000`; multiple-constraint cases with `n = 100` and
`m = 10, 50, 100, 200, 500`; seeds `0..9`; methods `proposed`, `oe_true`,
`oe_low`, `oe_high`, and `golden_ratio`; and a 1000-second cap per method. This gives 110 instances
and 550 method runs. A different declared plan cannot silently pass that
coverage contract. The amended plan retains original frozen worker hashes;
new single-constraint records inherited from the same study keep their original
run metadata and saved-point checksums. Plan snapshots preserve the amendment,
separate orchestrator hashes, and inheritance provenance.

The raw dataset and figures use all five methods. Main comparison tables use
only `proposed`, true-`M` OE (`oe_true`), and `golden_ratio`; that presentation
choice does not reduce the audit's full-grid coverage requirement.

`--single-only` is a separate real release contract: the six single-constraint
settings, seeds 0..9, five methods, a 1000-second cap, and
`release_scope="single_only"` in the plan. It requires all 60 instances and
300 method runs. Combining `--plan-scope --single-only` is permitted only for
explicitly marked single-constraint layout fixtures; it never establishes a real
release. The real release report scope is
`single_constraint_release`, while `full_requested_grid_records_complete` stays
false. Present lineage binds origin/current-plan identities and inherited timed
run/point hashes; original diagnostic/manifest copy hashes are historical and
current diagnostic versions receive new audit input hashes.

## Running it

From the workspace, after the runner's source and plan have been frozen:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/audit_saved_records.py \
  --results CVI_Full_Grid_Study/code/FULL_RESULTS_DIRECTORY \
  --code CVI_Full_Grid_Study/code \
  --out CVI_Full_Grid_Study/reporting/full_grid_structural_audit
```

The output consists of `structural_audit.json` and `structural_audit.md`.
An explicitly selected `--out` keeps audit output separate from measurements.
The default output is `RESULTS/report/structural`.

`--plan-scope` audits exactly the declared plan, for legacy or smoke data. It
changes the report's scope to `declared_plan`; it does not claim full-grid
coverage. A plan-scope method set must be exactly the three legacy methods
(`proposed`, `oe_true`, `golden_ratio`) or all five current methods. New worker run metadata and point-file checksums are required in the
default full-grid scope. In plan scope they may be absent for legacy data, but
any fields that are present are checked. `--preflight` checks files and record structure without generating
factors and can never return `structural_pass`. `--verify-operator-norm` adds a
numerical SVD check of the recorded spectral norm of `B`; without it the report
explicitly marks that value as not independently checked. `--temp-dir` selects
the parent for temporary factor files. `--point-block` bounds the number of
simultaneous point products and defaults to eight.

## Outcomes

| Status | Meaning | Exit code |
|---|---|---:|
| `structural_pass` | Complete selected scope and all implemented structural checks passed | 0 |
| `preflight_only` | Complete selected scope and preflight checks passed; factors were not checked | 0 |
| `incomplete` | Missing plan, coverage, records, or completion markers prevent a structural pass | 2 |
| `failed` | At least one identity, format, arithmetic, budget, or consistency check failed | 1 |

Failures take precedence over incomplete coverage. An incomplete report still
lists every available case that could be inspected. A missing plan produces an
incomplete report without inventing expected identities or source hashes.

Every outcome has `gap_validation: not_independently_performed` and
`independent_gap_solves: 0`. Neither exit code zero nor `structural_pass` means
that numerical gaps were independently validated. Publication or benchmark
completion must not be inferred from this structural audit alone.

## Independent reconstruction and memory

The auditor independently reproduces the documented NumPy `SeedSequence`
draws, QR factors, unscaled linear terms, operator, initial point, and offsets.
It does not call `DenseProblem`, `FullSolver`, or any source generator.

Each factor is stored in a C-contiguous memmap slice **before** multiplying
by the initial point to compute its offset. This matches the source's
preallocated storage and avoids a layout-dependent last-bit change in `c`.
Exact instance SHA256 equality remains required. The digest order is
`B, d, b, c, h`, followed by factor slices `W[0], ..., W[m-1]`, in native
little-endian float64 C order. The auditor refuses other byte orders.

The auditor never forms `A_i = W_i.T @ W_i` or an `(m,n,n)` stack. PSD follows
the sum-of-squares representation. The known generated QR eigenvalues provide
individual spectral maxima and the prescribed rank; there is no dense
constraint-matrix eigenvalue computation. The report states that these are
construction-based numerical checks, not formal interval proofs or an
independent re-solve of an optimization problem.

The largest factor file in the revised grid is
`500 * 50 * 100 * 8 = 20,000,000` bytes, about 19.1 MiB; the single
`n=1000,m=1` factor file uses 4,000,000 bytes. It is temporary disk storage, closed and removed after each case.
Persistent in-memory arrays include `B`, `b`, small scalar vectors, and saved
points. QR generation uses `O(n^2)` transient storage. At default block size,
constraint evaluation uses a factor product of at most `(ceil(n/2), 8)` and a
result block of at most `(8, m)`. The operating system may cache mapped pages;
the process's resident-memory reading can therefore include file-backed pages.
Only one instance is reconstructed at a time, with one numerical thread.

For row points `Z`, the auditor evaluates

```text
G_i(Z) = 0.5 * column_sum((W_i @ Z.T)^2) + Z @ b_i + c_i.
```

Its `J(point)` helper evaluates `W_i.T @ (W_i @ point) + b_i` in bounded factor
blocks. Saved-point auditing needs only `G`, so it does not pay for unnecessary
Jacobians. Arrays and files are hashed in 8 MiB chunks; no whole factor stack
is copied into a Python `bytes` object.

## Checks and interpretation

Checks cover the exact declared setting/seed/method identities, duplicate and
unexpected cases, canonical directory names, manifest completion, root Python
source file sets and hashes, instance digests, method completion markers,
run/point/diagnostic counts, float64 dimensions, checkpoint iteration/time
identity, finite points, initial points, and budget timestamps. Timeout runtime
must reach the declared cap; saved outputs must lie at or before the cap and
within the recorded runtime. Runtime can exceed the cap because the runner
charges work that fails to complete before its deadline.

For the new worker schema, each run's `metadata` must match the manifest's
`n`, `m`, `seed`, `cap_seconds`, `instance_hash`, and `source_hashes`, plus the
selected method. Its `points_sha256` must equal the streamed digest of the
actual compressed point archive. These checks bind each saved run to its
point file; the audit records the recorded and recomputed archive digests.
An exact initial-point comparison applies when this metadata is required or
present. Checkpoint times must strictly increase after checkpoint zero.
When recorded, fixed OE `fixed_M` and checkpoint `M_estimate` fields must equal
`true_M`, `0.1 * true_M`, or `10 * true_M` for `oe_true`, `oe_low`, or `oe_high`,
respectively.

The auditor recomputes ball membership and every saved constraint violation
factorwise. It checks finite nonnegative violations, residual arithmetic,
diagnostic durations, and baseline feasibility within the source's declared
projection tolerance. Valid gaps require finite nonnegative gap and residual
values and a recognized successful solver status. A failed diagnostic is
permitted when its validity flag and status/reason are honest; it is counted
as unresolved. A finite estimate with `gap_valid: false` remains unresolved.
It cannot support a numerical threshold hit or a claim of nonattainment.
Invalid diagnostics may have an unavailable residual. An unavailable violation
is accepted only for an invalid diagnostic with an explicit recorded
`diagnostic_error` in the `violation_check` phase. Its violation is still
independently recomputed from the saved point for structural and stopping
checks. A finite residual, when present, must satisfy the same arithmetic
identity even for an invalid gap estimate.

For proposed checkpoints after initialization, paper `C` must be finite,
nonnegative, consistent between timed and diagnostic records, and below its
geometric bound. A saved gap **marked** valid is compared with `C` using
`5e-8 + 2e-8 * max(1, abs(gap))`. This is a consistency comparison, not an
independent numerical validation of either quantity. The auditor does not
have the accumulated affine functional or all inner states needed to
reconstruct the certificate. Certificate-stop endpoints must satisfy the
saved certificate, violation, and middle-test conditions.

Other record comparisons use absolute and relative tolerances of `2e-12`;
ball tolerance is `1e-10`. The report records all tolerances. Per-point SHA256
values identify the newly inspected data. Since the source format has no
earlier authenticated point digests, these hashes cannot prove historical
authenticity or the absence of edits before auditing.

For multiple constraints, the minimal vector remainder constant is bracketed
by `max_i lambda_max(A_i)` and `norm((lambda_max(A_i))_i, 2)`. These bounds are
not claimed exact for `m > 1`. A valid theorem smoothness bound is
`L = max(L0, upper_bound)`. The initialization does not require `L0` to exceed
the smallest smoothness constant. No optimal multiplier bound is established;
the worst-case complexity assumption on `Lambda` remains unverified.

## Small-data validation

The frozen completed 20-instance dataset can be checked without writing to
any completed source, data, or report directory:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/audit_saved_records.py \
  --results CVI_Completed_Experiments/code_completed/measured_small/verified \
  --code CVI_Completed_Experiments/code_completed \
  --out /private/tmp/cvi_factor_audit_smoke \
  --plan-scope --verify-operator-norm
```

This validates small `n=10,m=1` and `n=50,m=10` cases only. It does not benchmark
or reconstruct any `n=1000,m=1000` case, and it makes no full-grid numerical
gap-validation claim.

The smoke run inspected 20 instances, 60 method runs, and 2,027 points with
`structural_pass`, zero failures, and zero incomplete reasons. All exact
instance digests and source hashes matched. The maximum factorwise constraint
violation difference was `2.562728675103987e-15`.

Temporary copied small fixtures also verified the following outcomes: missing
full-grid coverage, missing plan, and missing diagnostic files produced
`incomplete`; an honest invalid/timed-out gap remained structurally acceptable
and unresolved; nonfinite gaps marked valid, corrupted residuals, checkpoint
identity, instance hashes, and source hashes produced `failed`. A malformed
diagnostic object produced a report rather than crashing. The factorwise
Jacobian agreed with a small central directional finite difference. These
checks wrote only temporary copies and new audit reports.

After the worker added pair provenance, copied small fixtures verified that
matching metadata/checksums pass; mismatched present metadata/checksums fail
even in plan scope; and legacy fields are optional only in plan scope. Default
full-grid scope rejects absent pair provenance. Honest invalid diagnostics with
a finite unvalidated gap and unavailable residual remain unresolved, as do
explicitly explained violation-check exceptions; unexplained nonfinite
violations fail. The frozen 20-instance smoke still passes with the new checks.
