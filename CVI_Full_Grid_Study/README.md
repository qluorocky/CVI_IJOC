# Published corrected CVI experiment code and results

This update publishes the completed corrected experiments and the code used to
run, diagnose, audit, and report them. Publishing the update does not rerun an
experiment or alter the saved measurements. The supplied manuscript and proofs
are outside this code update.

## Completed experimental scope

| Study | Fixed parameter | Varying parameter | Instances | Method runs |
| --- | --- | --- | ---: | ---: |
| PFAOS, single constraint | m = 1 | n = 10, 50, 100, 200, 500, 1000 | 60 | 300 |
| MC-PFAOS, multiple constraints | n = 100 | m = 10, 50, 100, 200, 500 | 50 | 250 |
| Total | | Ten paired seeds per setting | 110 | 550 |

Each instance has five independently timed methods: PFAOS or MC-PFAOS, Golden
Ratio, and fixed OE with the estimate equal to M, 0.1M, or 10M. The primary
tables retain the proposed method, true-M OE, and Golden Ratio; the figures show
all five methods. No selected best-OE trajectory is constructed.

The method budget is 1,000 seconds per trajectory. Primary offline gap
calculations have a separate 300-second budget; their time is not included in
method hitting times. The MC dimension was revised from 1,000 to 100, and the
MC m = 1,000 setting was removed. Original single-constraint measurements remain
part of the corrected study with their recorded lineage.

The regular controller admitted up to seven workers, each using one numerical
thread. The ten n = 100, m = 100 instances were prioritized for the requested
two-panel sensitivity figure with an admission ceiling of ten workers. Per-case
sidecars record this temporary scheduling override and bind the phase, source,
and timed-file hashes. The regular queue subsequently returned to seven workers.
These are recorded workload conditions, not a guarantee of simultaneous CPU use.

## Published outputs

The separately published result folders are:

- `published/single/`: measured PFAOS tables and six-panel comparison figure.
- `published/multi/`: measured MC tables, five-panel comparison figure, and the
  brief discussion for manual LaTeX insertion.
- `published/sensitivity/`: the requested two-panel sensitivity comparison,
  with PFAOS n = 1,000, m = 1 and MC-PFAOS n = 100, m = 100, preserving the
  supplied 11 by 4.8 inch grayscale layout.

Primary table cells report the observed success count and best/mean/worst
hitting times conditional on observed valid crossings. The time cap is never
inserted as a successful hitting time. The joint numerical error requires the
numerical VI gap and constraint violation to meet the threshold at the same
saved averaged point. Gap-only companion tables do not imply simultaneous
feasibility. Saved-checkpoint times, rather than interpolated figure crossings,
determine table values.

The completed MC subset contains 50 instances, 250 methods, and 6,327 valid
diagnostic checkpoints, with no unresolved MC diagnostic or method failure.
MC-PFAOS attains joint error 1e-4 in all 50 runs and 1e-5 in ten m = 10 runs
plus two m = 50 runs. True-M OE and Golden Ratio record no joint 1e-3 crossing.
The single-only release separately preserves eight accepted independent
numerical gap refinements and their original attempts; historical primary
single diagnostic exceptions in the combined snapshot do not enter MC metrics.

Factorwise structural checks and independent table-statistics checks passed.
Structural auditing reconstructs instances, checks saved points and constraints,
and verifies provenance; it performs zero independent numerical gap solves.
Numerical solver outcomes and the selected numerical dual brackets are not
formal finite-precision interval certificates. All actual failure histories
remain preserved in the original local records.

## Code map and historical entry points

- `code/`: the frozen numerical implementation, projections, worker, and tests.
- `orchestration/batch_n100.py`: the amended grid controller used by the local
  study; it executes the unchanged numerical worker.
- `orchestration/run_priority_sensitivity.py`: the recorded priority-phase
  orchestration, with locking, worker identities, and scheduling sidecars.
- `reporting/`: full-grid reporting, structural and table-statistics auditors,
  selected diagnostic refinement/adoption helpers, figure rendering, and
  separate single/MC delivery helpers.

The original `code/batch.py` and `run_full_study.py` retain the historical
120-instance/600-method plan with MC n = 1,000. They are not the corrected grid's
launch defaults. The amended and priority controllers preserve local plan hashes,
absolute provenance paths, and process-transition records. Their retained
commands document the original run; they are not portable launch defaults for a
fresh clone. Do not restart the completed local queues merely to publish or view
the results. Some retained historical READMEs describe that earlier scope; the
scope table above identifies the completed corrected study.

## Optional new single-instance run

Python 3.12 and the pinned packages in `code/requirements.txt` reproduce the
recorded software environment. From the repository root:

```sh
python -m pip install -r CVI_Full_Grid_Study/code/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
python CVI_Full_Grid_Study/code/study.py \
  --n 100 --m 10 --seed 0 \
  --seconds 1000 --diagnostic-seconds 300 \
  --methods proposed,oe_true,oe_low,oe_high,golden_ratio \
  --out /tmp/cvi-n100-m10-seed0-new
```

This command intentionally creates a new measurement set in a fresh output
folder; it does not recreate the published timings or modify their source files.
Use a new output path for a new experiment. The worker's `--resume` option is
for the same saved instance, methods, budgets, source hashes, and checked file
pairs; it is not a way to relabel a previous dimension or discard failures.

## Lightweight checks without a full grid

After installing the requirements, run these from the repository root:

```sh
cd CVI_Full_Grid_Study/code
python tests/test_robustness.py
python tests/test_large_worker.py
cd ../reporting
python -m unittest test_priority_execution_context test_sensitivity_two_instances
```

The numerical checks use small temporary instances, mocked failures, and short
synthetic deadlines. The reporting checks use explicitly marked temporary
fixtures. They do not run the 550-method grid or modify published data. Some
other retained orchestration or release tests deliberately require original
plan/data snapshots or a TeX installation; they are not all standalone tests for
this reduced publication checkout.

## Limits of this checkout

The publication includes selected measured outputs, source, plans and provenance;
it does not include every original raw saved point, diagnostic attempt, factor
file, working directory, or local recovery snapshot. A published summary CSV, if
present, is an output summary and does not replace those inputs.

The full reporting and auditor entry points require the original complete saved
record inventory and its exact hash-bound provenance. They must not regenerate
primary tables from partial CSVs, copied figures, historical pasted values, or
runtime estimates. Reconstructing a portable full-grid runner or distributing a
complete raw-data archive would be a separate change; no such runner is added
by this publication.
