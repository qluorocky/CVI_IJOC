# Full-grid measured reporting and article assembly

These scripts are outside the frozen timed experiment source directory. They
never read old table values or estimate runtimes. They operate on the new saved
records from the following exact release plan:

- Single constraint: m=1; n=10, 50, 100, 200, 500, 1000.
- Multiple constraints: n=100; m=10, 50, 100, 200, 500.
- Seeds 0–9; 1,000 seconds for each independently timed trajectory.
- Five methods: proposed, oe_true, oe_low, oe_high, golden_ratio.
- Total: 110 instances and 550 timed method runs.

Both figures compare all five methods: PFAOS/MC-PFAOS, Golden Ratio, and fixed
OE at M, 0.1M, and 10M. The main tables preserve three rows per configuration
(proposed, true-M OE, Golden Ratio). The lower/higher norm choices are separate
measured sensitivity trajectories; no selected best-OE row is constructed.

The amended plan changes only the MC dimension from n=1000 to n=100 and
removes the MC m=1000 setting. Original frozen worker source hashes stay unchanged.
Valid new single-constraint records from the same corrected study may be carried
forward; the earlier completed small study is not used. The plan preserves
`plan_amendment`, `inherited_single_provenance`, and separate `orchestrator_hashes`;
per-case `lineage.json` records inherited-file origins. Full plan snapshots and
input hashes are retained in reporting, independent audits, and article assembly.

## Prioritized single-constraint LaTeX release

All four release tools accept `--single-only`. This is a real complete release
scope, distinct from validation fixtures: the plan must declare
`release_scope="single_only"`, all six single-constraint dimensions, seeds 0–9,
all five ordered methods, and the 1,000-second budget (60 instances / 300 runs).
Do not use `--plan-scope` or `--validation-subset-plan` for this release.

```sh
python reporting/report_full_grid.py --results results_single/verified --single-only
python reporting/audit_saved_records.py --results results_single/verified --code code --single-only
python reporting/audit_table_statistics.py --results results_single/verified --single-only
python reporting/assemble_manuscript.py --template ../CVI_Completed_Experiments/manuscript_revised --results results_single/verified --destination manuscript_single --single-only
sh manuscript_single/build.sh
```

Only the PFAOS main table, companion, six-panel figure, and data-derived discussion
are generated. The editable full article retains all theory sources; the numerical
section, abstract, and conclusion identify the single-constraint empirical scope.
`sec_multi_constraint_results_pending.tex` is the MC insertion point. Traditional
MC table/figure fragments contain explicit empty placeholders, and no unmeasured
MC result or missing table/figure reference is introduced.

When `lineage.json` is present, all auditors check its origin, amended-plan hash,
and every listed copied timed run/point archive hash. The original manifest and
diagnostic copy hashes remain historical because completion markers and recorded
diagnostic retries may advance; current versions are independently input-hashed.
No extra inheritance boolean flag is required beyond the agreed provenance fields.

## Commands after the full record inventory is complete

Run from the full study directory, using the project's Python environment:

```sh
python reporting/report_full_grid.py --results results/verified
python reporting/audit_saved_records.py --results results/verified --code code
python reporting/audit_table_statistics.py --results results/verified
python reporting/assemble_manuscript.py --template ../CVI_Completed_Experiments/manuscript_revised --results results/verified --destination manuscript_revised
sh manuscript_revised/build.sh
cd manuscript_revised
pdflatex -interaction=nonstopmode -halt-on-error numerical_review.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_review.tex
```

Replace `results/verified` with the runner's actual verified output directory.
Use fail-fast command execution in orchestration. Assembly refuses an existing
destination instead of overwriting an author's source tree. The assembler reruns
the independent table-statistics audit automatically when it is absent or stale.

`report_full_grid.py` preflights the complete case/method/file inventory before
writing paper outputs. It requires the eleven settings, ten seeds, five methods,
1,000-second cap, consistent source metadata, and saved-point checksums. It writes:

- `pfaos_table.tex`, `mc_table.tex`: primary joint numerical-error tables.
- `pfaos_table_gap_only.tex`, `mc_table_gap_only.tex`: companion gap-only tables.
- `checkpoints.csv`, `per_run_summary.csv`: all five trajectories.
- `pfaos_full_grid.pdf/.png` and `mc_full_grid.pdf/.png`: a six-panel single-constraint figure and a five-panel MC figure,
  with five measured method curves per panel when diagnostic coverage permits.
- Matching `_figure.tex` captions, numerical protocol, factual discussion,
  coverage/status audit, SHA256 evidence, and `validation.json`.

The coefficient limit and gap-call budgets are read from actual manifests and
diagnostic rows; they are not inferred from a hardware rule or historical input.
The current worker records the memory-bounded conic policy and full-set SLSQP
fallback. Such a backend change does not change the original feasible set.

## Explicit adoption of the eight selected independent m=1 refinements

`adopt_single_dual_gaps.py` performs no optimization. Its default is a read-only
review of the exact eight original invalid rows listed in the frozen delivery
inventory and their hash-bound v2 sidecars. The target is restricted to this
study's copied `results_single/verified`; the original `results` tree, frozen
`code`, run files, point files, plan, and every other diagnostic row stay unchanged.
The conservative preflight requires the complete 60-instance/300-run copied
single release, consistent source/instance/point identities, and original frozen
worker source hashes before either dry-run or apply.

```sh
python reporting/adopt_single_dual_gaps.py --report /private/tmp/single_dual_preview.json
```

Optionally add `--review-file notes/single_dual_review_approved.json` to bind the
independent/root review document to the exact summary and eight sidecar hashes.
This does not replace the explicit `--quality-reviewed` flag.

Review the eight candidate identities, posterior quality evidence, and retained
optimizer statuses first. Computed quality checks and review authorization are
separate: `--quality-reviewed` is an explicit root review flag and is never set
from solver output. Only after that review, an explicit root invocation may apply:

```sh
python reporting/adopt_single_dual_gaps.py --apply --quality-reviewed --report /private/tmp/single_dual_applied.json
```

Every accepted sidecar must have finite nonnegative gap, full original feasibility
at most 1e-8, stationarity L2 at most 1e-8, and primal-dual width at most 1e-8.
The additional recorded KKT checks must pass, multipliers must be nonnegative,
source/candidate files must match hashes, and no numerical bracket may straddle a
paper threshold. Three actual v2 root calls retain non-success status 5; posterior
quality may pass despite that raw termination flag. The adopted status is the
separate `independent_dual_bracket_valid`, preserving raw optimizer success/status.
Ordinary-float primal/dual checks are not formal interval certificates.

Apply creates immutable backups of the affected primary diagnostic files and
manifests under `results_single/diagnostic_adoption_backups`, plus a helper source
snapshot. It changes exactly the eight authorized rows and recomputes manifest
diagnostic counts/status counts. Full original rows, all failed v1 attempts, and
complete v2 sidecars remain in per-row attempt histories. Updated diagnostic
seconds sum the three solve attempts; setup is separately recorded in sidecars.
All timed pairs and every other valid row are checked unchanged. A failed write
rolls back primary changes. A lock prevents concurrent helper applications.

`verified/diagnostic_adoption_manifest.json` records review authorization, before/
after hashes, selected candidates, sidecar/source hashes, backups, and unchanged
protected inputs. Report generation and both independent auditors bind this
manifest and its referenced evidence into their input checksums. Re-run reporting,
structural auditing, table-statistics auditing, and assembly after adoption;
earlier reports are stale. The measured protocol and article explicitly state
that only these eight selected gaps received independent dual evaluation, preserve
the three raw non-success flags, and make no claim that every gap was independently
re-solved. The structural auditor itself still performs zero gap solves.

`test_dual_adoption.py` checks quality rejection, the review gate, temporary-file
backup integrity, protected timed files, and transaction rollback. Its apply tests
use synthetic temporary files only; they never apply to measured data.

## Missing, failed, or unresolved records

Missing case/run/point/diagnostic files are incomplete data and are refused.
A solver/algorithm error must preserve its last valid reported output, result
status, and diagnostic records; it is enumerated separately from a time limit.

Invalid numerical gaps remain recorded as unresolved. For a threshold cell,
`s/N` is the number of observed valid crossings. If additional no-hit runs have
unresolved diagnostics, the cell also gives `u=...`. Such runs are not counted
as demonstrated nonattainment. Success times use valid saved checkpoints only;
a prior unresolved checkpoint is flagged per run. Final V is over every planned
terminal output, or marked unresolved when any terminal violation is unavailable.
The time cap is never imputed as a successful hitting time.

`validation.json` is `complete` only if all diagnostics are valid and all runs
have normal cap/certificate terminations. Otherwise a complete record inventory
is labeled `complete_with_exceptions`. The default assembler refuses that state;
`--allow-unresolved` is an explicit option for an honestly qualified draft, with
unknown counts and failure counts retained. It does not turn a draft into an
accepted numerical release.

The aggregate median never drops problematic seeds. It requires at least two
saved checkpoints and complete valid joint diagnostics for every seed of that
method; unavailable groups are labeled. Each curve ends at the earliest terminal
checkpoint among its ten seeds. Display interpolation, a 1e-12 floor, and bootstrap
bands do not modify table values or threshold decisions.

## Independent checks and limits

`audit_saved_records.py` independently reconstructs deterministic C-contiguous
factors into temporary memory maps, streams hashes, and recomputes constraints in
bounded point blocks. It never stacks all A_i matrices. In this revised grid the
largest factor file is 20 MB for n=100, m=500 (the single n=1000, m=1
factor file is 4 MB), released per case. Its `structural_pass`
verifies record/source/instance identity, point budgets, ball membership,
constraints, diagnostic honesty, and saved certificate/residual consistency.
It explicitly performs **zero independent gap solves**. A structural pass must
not be described as independent numerical gap verification.

`audit_table_statistics.py` imports no experiment or reporter code. It independently
derives first valid gap/joint hits, unresolved flags, conditional timing triples,
and all-run terminal-V triples. It checks every primary table cell and all five
methods' CSV records, including unknown counts and pair checksums. Table rounding
uses half the displayed unit; target comparisons use no added tolerance.

Selected independent numerical gap solves should be run separately with a bounded
factorwise implementation if required. The old small-study auditor constructs
an 8 GB A stack at m=n=1000 and an all-constraint conic model; do not use it unchanged.
Neither primary numerical solver status nor a primal objective is a formal
finite-precision upper gap certificate.

### Separate recovery preparation

The [recovery notes](recovery/README.md) describe a separate factorwise
constraint-generation probe and its preserved attempt history. Small validation
cases agree with existing numerical gap solves; both initial largest-case probes
remain unresolved. No probe value has been adopted into measured records.

The current recovery CLI admits only complete, stable verified snapshots after
worker quiescence. It validates plan/source/run/point identities, preserves original
diagnostics, and can reconstruct deleted factors in temporary disk-backed storage
under an explicit reserve and setup budget. Run it separately from the structural
audit. A later retry needs a new output directory, separately recorded optimization
budget, full original-feasibility checks and numerical-quality review, particularly
near the requested thresholds. It provides no operation for replacing primary
diagnostics; any adoption requires a documented provenance layer and independent
audit. Exact-point reuse candidates recorded in monitor notes are also unadopted.

## Source assembly and traditional paths

The assembler derives the new grid, solver-memory cap, call budgets, table and
figure references from the complete records. It updates abstract, introduction,
numerical section, conclusion, and numerical supplement, while requiring the five
theory/stopping/proof sources to remain byte-identical to the template. It records
actual input and copy hashes in `assembly_manifest.json` and leaves `compiled=false`
until the build and visual review actually occur.

The active article can import `ResultsDir`, and the author's traditional paths
are also updated: `JoC_submission/results/pfaos_table.tex`, `mc_table.tex`, their
gap-only companions, and `pfaos_figures.tex`/`mc_figures.tex`. Figure fragments use
`ResultsDir` for the new PDF assets. No old numerical value remains in those
traditional table/figure fragments.

A final PDF still needs reference/citation/log checks and visual inspection of
the eighteen-row single-constraint and fifteen-row MC main tables, both companions, and the six-panel single-constraint and five-panel MC figures. The
assembly manifest records this distinction rather than claiming a completed build.

## Safe progress and validation fixtures

`progress_report.py --results results` writes only a clearly labeled inventory
of started/complete instances and timed method stages by row. It never emits
partial paper tables or performance claims.

Reporter `--validation-subset-plan` accepts a copied plan only when it is explicitly
marked `validation_fixture=true`; all captions/discussion are then labeled as
fixtures. Auditor `--plan-scope` supports declared three-method legacy or five-method
fixture plans. Default release paths still require the full 110/550 contract.
Validation fixtures and temporary previews are kept under `/private/tmp`, outside
new measured data and deliverable manuscripts.

Run `python reporting/test_revised_contract.py` for lightweight contract and
synthetic layout checks. It rejects the superseded MC n=1000 grid and removed
m=1000 row, checks the real 60/300 single-release contract, and tests copied
pair/lineage integrity on an earlier tiny worker fixture when available. It runs
no timed algorithm or gap solve and writes temporary previews outside measured data.
