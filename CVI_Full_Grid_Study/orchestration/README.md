# Amended MC grid controller

`batch_n100.py` changes orchestration only. It executes the unchanged frozen
`code/study.py` worker. The final amended grid is six original single-constraint
dimensions and MC n=100 with m=10,50,100,200,500: 110 instances, ten seeds per
setting, five methods, 550 method runs, 1000-second method budgets and separate
300-second diagnostics. There is no MC m=1000 job in the amended grid.

The transition note and original plan checksum identify the only permitted
reuse source. Complete original **new full-study single cases** are copied only
after strict metadata, source, initial-point, point-file checksum, checkpoint,
diagnostic-prefix, terminal-status and budget checks. All original method-error
and invalid-diagnostic records remain unchanged. Original MC and small-study
records cannot enter the amended study. Each adopted single gets `lineage.json`
with original/new plan hashes, source path, transition hash and all copied file
checksums. Copies are verified in a temporary directory and atomically published.

The three single workers listed in `notes/grid_transition.json` remain external
and read-only while their original PIDs exist. A start-time check detects PID
reuse; permission errors or unavailable process identity are treated as still
alive. Zombies are treated as exited before command-identity comparison. The
controller never signals an external worker. It waits for exit
before exporting a complete case. If an inherited worker exits incomplete, it
validates saved identities/pairs and schedules the frozen worker with `--resume`
at its original case path. Committed methods and observed failures are retained;
only an uncommitted interrupted method can restart at its original full budget.

External singles count toward both the seven-worker limit and the 8 GiB modeled
RAM allowance. Admission retains the original factor-plus-conic-reserve formula
but defines a heavy case by a factor footprint of at least 1 GiB, rather than by
m alone. The largest amended MC factor file is 20 MB. Seven largest amended MC
weights total about 7.80 GiB, so ordinary RAM/slot limits govern their concurrency.
At least 12 GiB free disk is reserved in addition to the next factor allocation.

A controller file lock prevents duplicate controllers for one output. The old
controller must be absent before launch. A restart refuses any leftover owned
worker still alive/unknown instead of launching duplicate methods; root must
coordinate that separate handoff before resuming. An atomic `launch_ledger.json`
persists a launch intent before spawning, then records the PID before any further
admission. A crash with an unresolved PID-less intent blocks resume until root
reconciles process identity, closing the spawn-to-progress duplicate window.
`worker_identity_records` in progress also retain PID/start/path information.
Frozen source hashes
are checked again before every worker launch. The orchestrator has a separate
hash in the new plan and is not added to the timed worker's source-hash set.

Read-only preflight (launches no processes and writes no results):

    ../.venv/bin/python orchestration/batch_n100.py --validate-only

Root launches the controller after review/coordination, using:

    ../.venv/bin/python orchestration/batch_n100.py \
      --seconds 1000 --diagnostic-seconds 300 --seeds 10 --workers 7 \
      --memory-budget-gib 8 --min-free-disk-gib 12 \
      --out results_n100 --transition notes/grid_transition.json

Use `--resume` only with the exact same new plan/orchestrator hash. This module
does not enable diagnostic retries or resample observed method failures.
