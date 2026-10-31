# Priority two-panel sensitivity phase

`run_priority_sensitivity.py` is a separate untimed orchestrator. It runs only
MC n=100,m=100, seeds 0–9, all five original methods, at the existing
`results_n100/working/multi_n100_m100_seed*` paths. The frozen worker remains
`code/study.py`, with its original 1000-second method and separate 300-second
diagnostic budgets. The priority-only pool defaults to ten one-thread workers
so all ten cases can run in one wave. Its modeled footprint is
10 x 0.502736902 = 5.027369 GiB, below the retained 8-GiB RAM allowance. The pool
must fit the available CPU count and the ten cases. The 12-GiB free-disk reserve
is retained. This is an explicit override of the original seven-worker queue;
the original plan/source stay unchanged and original resumption restores seven.
The two figure panels therefore do not claim matched seven-worker workloads.

Root must first interrupt the original wrapper/controller/owned workers and
record the interruption separately from method failures. This runner never
stops an original actor. It refuses alive or unknown original PIDs, checks the
original launch ledger for late admissions, and validates source, plan,
controller, wrapper and frozen grid-transition hashes. It also reconstructs the
original controller's plan in memory to confirm `--resume` compatibility.
**Do not edit `notes/grid_transition.json` or the original plan/controller.**

The separate transition note defaults to
`notes/priority_sensitivity_transition.json`. Its required fields are:

```json
{
  "original_plan_sha256": "SHA256 of results_n100/plan.json",
  "original_wrapper_sha256": "SHA256 of run_n100_study.py",
  "original_queue_quiescent": true,
  "priority_workers": 10,
  "original_controller": {"pid": 43774, "process_start": "recorded ps lstart"},
  "original_wrapper": {"pid": 43771, "process_start": "recorded ps lstart"},
  "interrupted_workers": ["full PID/start/path record for each stopped worker"],
  "render_command": ["python executable", "explicit audit/plot wrapper", "arguments"],
  "figure_output": "/absolute/new/figure/output"
}
```

The controller/wrapper records may instead contain a full `process_identity`
row. Exact birth identities distinguish a reused PID from the original actor.
The render command is an argv list, not a shell string. It must perform the
selected-right-case structural audit and invoke the plotting helper with the
refined left `results_single/verified` n=1000,m=1 data and the right n=100,m=100
data. Root supplies this command; the runner does not choose figure statistics
or style. The plot helper requires all ten seeds, five methods and a passed
selected-right structural audit.

Launch after review/controlled interruption:

    ../.venv/bin/python orchestration/run_priority_sensitivity.py \
      --transition notes/priority_sensitivity_transition.json

`--validate-only` performs the hash/plan/data/quiescence checks without creating
state or launching actors. `--resume` requires exactly the same priority plan,
script and authorization note. State defaults to `priority_sensitivity/`.
`--priority-workers` defaults to 10 and must agree with the explicit phase note.
The phase plan records original/priority/restored pool limits, memory model,
CPU count, one numerical thread per worker and an immutable runner snapshot.

Each exported target has a `priority_execution.json` sidecar binding that phase
plan, unchanged original plan, runner snapshot and timed-pair checksums. It
distinguishes already committed methods from methods scheduled in the priority
phase, preserves any prior execution sidecar, and records configured pool limits
separately from observed live-process counts. The observed peak is the largest
priority process pool seen by that case's export, not a guarantee of simultaneous
CPU occupancy throughout every method. Reused complete cases are identified
explicitly instead of being relabeled as new ten-worker measurements. Existing
completed single/PFAOS data is only read and is never rerun.

The runner holds both its own controller lock and the original
`results_n100/controller.lock`. Before each spawn it durably writes the same
launch intent into its own ledger and the compatible **original** launch
ledger, preserving every unrelated old entry. After spawning it records the
PID in the original ledger first, then its own, before any further admission.
A PID-less unresolved intent blocks retries; a partially committed PID can be
reconciled by its unique intent ID. Workers and the render process inherit the
original controller-lock descriptor, so even a killed priority parent cannot
release the original queue while those children remain alive. A resumed
priority controller waits read-only for its orphan actors to exit.

Complete verified targets are skipped. Incomplete cases are strictly checked
and resumed by the unchanged worker with `--resume`. Existing committed
run/point checksums and all diagnostic prefixes must survive unchanged. Actual
method errors remain outcomes, and invalid diagnostics are not automatically
retried. Only an uncommitted interrupted method can restart at its original
full budget. Finished targets are verified and copied atomically to the
original `results_n100/verified`, where the unchanged original controller skips
them during its next resume.

After ten complete targets and fifty method pairs, the runner checks that the
refined left inputs remain unchanged and executes the supplied audit/plot
command while retaining the original queue lock. It requires and hashes
`sensitivity_residual.pdf` and `sensitivity_residual.png`. On success it records
the handoff, releases locks after all numerical children have exited, and
replaces itself with the unchanged `run_n100_study.py`. That wrapper
automatically adds `--resume` to the original controller, which skips the ten
verified targets and resumes partial interrupted cases without rerunning
committed methods. A worker, export, audit, or plot error is recorded as needing
attention; it does not trigger resampling or automatic queue resumption.

Focused tests (`python -m unittest test_priority_sensitivity -v` from this
directory) cover target scope, exact original-plan reconstruction, dual-ledger
spawn-gap reconciliation, unresolved-intent refusal, unknown-liveness refusal,
and inheritance of the original lock by workers/rendering. They spawn no
processes and write only temporary fixtures.
