> Historical worker documentation: this frozen code directory includes the original
> 120-instance batch default. The corrected completed release uses 110 instances
> and 550 methods, with MC n=100 and m=10,50,100,200,500. Read
> [the published guide](../PUBLISHED_GUIDE.md) before choosing a launch command.
> The measured Python source files are unchanged.

# Full-grid experiment code

The active study has 120 cases and 600 method runs. Single-constraint rows use
m=1 and n=10,50,100,200,500,1000. Multiple-constraint rows fix n=1000 and use
m=10,50,100,200,500,1000. Each row has ten seeds and five measured trajectories:
PFAOS/MC-PFAOS, fixed OE at M, 0.1M and 10M, and Golden Ratio.

Source hashes are frozen in `../results/plan.json`. No historical values,
previous small-study timings, or runtime estimates enter the new records.

Activate Python 3.12 and install `requirements.txt`. From this directory:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
python test_checks.py
python tests/test_robustness.py
python tests/test_large_worker.py
python batch.py --seconds 1000 --diagnostic-seconds 300 --seeds 10 --workers 7 --memory-budget-gib 8 --min-free-disk-gib 12 --heavy-companion-m 0 --out ../results
```

Add `--resume` for the unchanged plan and source. The supervised entry point is
`../run_full_study.py`; do not launch a second copy while its recorded PID is
active. `../pipeline_status.json` and `../results/progress.json` show current state.

The worker saves actual runtime, pre-deadline averaged outputs, source and
instance hashes, checksummed JSON/NPZ pairs, work counters, solver statuses,
initialization progress, and peak RSS. Interrupted methods restart; completed
pairs are reused only after validation. Diagnostic prefixes resume, and invalid
rows retain their reasons and attempt histories. Unknown values are not imputed
as failures to attain a threshold.

Conic models are limited to five million dense factor coefficients. A full-set
SLSQP fallback follows a conic failure or resource guard, with all original
constraints checked. Numerical libraries use one thread per worker. Admission
reserves memory for factor storage and profiled solver costs, rather than
assuming seven large models fit in 16 GiB. The 1,000-second method budget and
300-second diagnostic budget are separate. Modeling and cleanup overruns are
recorded; outputs completed after the method deadline are excluded.

Active reporting tools are outside the timed source tree:

```sh
python ../reporting/report_full_grid.py --results ../results/verified
python ../reporting/audit_saved_records.py --results ../results/verified --temp-dir ../validation
```

The local `report.py`, `pack.py`, and `preview.tex` are retained legacy snapshot
helpers; they are not full-grid reporting entry points. Use the commands above
and the project pipeline. Reporting hashes are captured separately from timed
source hashes.

Algorithm formulas and instance draws are unchanged. The original twelve
implementation checks, nine large-worker regressions, and three fallback,
admission and deadline checks pass. Validation fixtures are stored outside
`../results` and are not paper measurements.
