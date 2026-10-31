"""Revised MC pipeline; delivered single manuscript remains a separate artifact."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
NOTES = ROOT / 'notes'
RESULTS = ROOT / 'results_n100'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
           MPLCONFIGDIR=str(ROOT / 'validation/matplotlib_cache'))
ENV['PATH'] = '/Library/TeX/texbin:' + ENV.get('PATH', '')


def update(state, **extra):
    data = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                pipeline_pid=os.getpid(), active_results=str(RESULTS),
                single_delivery=str(ROOT / 'manuscript_single_ready'),
                scope='single6dimensions;multi n100,m10,50,100,200,500;10seeds;5methods',
                planned_instances=110, planned_method_runs=550, **extra)
    p = ROOT / 'pipeline_status.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(p)
    print(json.dumps(data), flush=True)


def run(args, log):
    with (NOTES / log).open('a', buffering=1) as stream:
        subprocess.run(args, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    transition = ROOT / 'notes/grid_transition.json'
    assert json.loads(transition.read_text())['old_controller_stopped']
    update('running_revised_n100_grid', note='Original timed source unchanged; inherited new singles retained with lineage.')
    cmd = [sys.executable, str(ROOT / 'orchestration/batch_n100.py'), '--seconds', '1000',
           '--diagnostic-seconds', '300', '--seeds', '10', '--workers', '7',
           '--memory-budget-gib', '8', '--min-free-disk-gib', '12', '--out', str(RESULTS),
           '--transition', str(transition)]
    if (RESULTS / 'plan.json').exists():
        cmd.append('--resume')
    run(cmd, 'n100_batch.log')
    verified = RESULTS / 'verified'
    validation = json.loads((verified / 'report/validation.json').read_text())
    update('auditing_revised_n100_grid', measured_report_status=validation['status'])
    run([sys.executable, str(ROOT / 'reporting/audit_saved_records.py'), '--results', str(verified),
         '--code', str(ROOT / 'code'), '--temp-dir', str(ROOT / 'validation')], 'n100_structural_audit.log')
    run([sys.executable, str(ROOT / 'reporting/audit_table_statistics.py'), '--results', str(verified)],
        'n100_statistics_audit.log')
    report = verified / 'report'
    checksums = {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in report.rglob('*') if p.is_file()}
    (NOTES / 'n100_report_sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
    update('ready_for_multi_artifact_review', instances=validation['instances'],
           method_runs=validation['method_runs'], report_status=validation['status'],
           multi_table=str(report / 'mc_table.tex'), multi_figure=str(report / 'mc_full_grid.pdf'),
           note='Review unresolved diagnostics, actual tables/figure and numerical limits before final delivery. '
                'User will insert MC table; do not overwrite delivered single manuscript.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update('needs_attention', error_type=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise
