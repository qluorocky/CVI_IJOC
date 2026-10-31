"""Snapshot only the new measured single-constraint grid after each worker exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def live(pid):
    out = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    if out.returncode == 1 and not out.stdout.strip():
        return False
    if out.returncode != 0:
        raise RuntimeError('Cannot establish inherited worker quiescence: ' + out.stderr)
    return not out.stdout.strip().startswith('Z')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--transition', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=1800)
    a = parser.parse_args()
    source, output = a.source.resolve(), a.out.resolve()
    study = Path(__file__).resolve().parents[1]
    sourceplan = source / 'plan.json'
    old = json.loads(sourceplan.read_text())
    transition = json.loads(a.transition.read_text())
    assert transition['old_controller_stopped']
    assert transition['source_plan_sha256'] == sha(sourceplan)
    assert old['seeds'] == list(range(10)) and old['seconds'] == 1000
    for name, value in old['source_hashes'].items():
        assert sha(study / 'code' / name) == value
    sys.path.insert(0, str(study / 'code'))
    import numpy as np
    from study import validate_method_pair, validate_diagnostic_prefix
    settings = [dict(study='single', n=n, m=1) for n in (10, 50, 100, 200, 500, 1000)]
    plan = dict(old, settings=settings, release_scope='single_only',
                study_version='single_constraint_release_v1', prior_runs_reused=True,
                purpose='new_measured_single_constraint_delivery',
                inherited_single_provenance=dict(source_results=str(source),
                    source_plan_sha256=sha(sourceplan), earlier_small_study_reused=False,
                    historical_table_values_reused=False),
                plan_amendment=dict(reason='User requested immediate single-constraint LaTeX; multi-constraint table will be inserted later.',
                    source_plan_sha256=sha(sourceplan)),
                orchestrator_hashes={Path(__file__).name: sha(Path(__file__))})
    output.mkdir(exist_ok=True, parents=True)
    verified = output / 'verified'
    verified.mkdir(exist_ok=True)
    for path in (output / 'plan.json', verified / 'plan.json'):
        if path.exists():
            assert json.loads(path.read_text()) == plan, 'Existing single release plan differs'
        else:
            atomic(path, plan)
    planhash = sha(output / 'plan.json')
    inherited = {item['case']: item['pid'] for item in transition['inherited_single_workers']}
    todo = [(s['n'], seed) for s in settings for seed in old['seeds']]
    deadline = time.monotonic() + a.wait_seconds
    last = 0
    while todo:
        for n, seed in list(todo):
            name = f'single_n{n}_m1_seed{seed}'
            case = source / 'working' / name
            if name in inherited and live(inherited[name]):
                continue
            meta = json.loads((case / 'manifest.json').read_text())
            if meta['state'] != 'complete':
                if name in inherited:
                    raise RuntimeError(f'{name}: exited before a complete case was saved')
                continue
            assert meta['source_hashes'] == old['source_hashes'] and meta['cap_seconds'] == old['seconds']
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, 101]))
            rng.uniform(0, 1, (n, n)); rng.uniform(0, 1, n - n // 2)
            d = rng.uniform(0, 1, n); x0 = -d / np.linalg.norm(d)
            for method in old['methods']:
                saved, points = validate_method_pair(case, method, meta, x0)
                rows = json.loads((case / f'{method}_diagnostics.json').read_text())
                validate_diagnostic_prefix(rows, saved, method, meta)
                assert len(rows) == len(points) and meta['methods'][method]['stage'] == 'complete'
            files = [f for f in case.iterdir() if f.is_file() and f.suffix in {'.json', '.npz', '.log'}]
            hashes = {f.name: sha(f) for f in files}
            target = verified / name
            if target.exists():
                lineage = json.loads((target / 'lineage.json').read_text())
                assert lineage['copied_artifact_sha256'] == hashes
                for fname, value in hashes.items():
                    assert sha(target / fname) == value
            else:
                staging = verified / ('.' + name + '.copying')
                staging.mkdir(exist_ok=False)
                for f in files:
                    shutil.copy2(f, staging / f.name)
                assert hashes == {f.name: sha(f) for f in files}, 'Original changed during copy'
                assert all(sha(staging / k) == v for k, v in hashes.items())
                atomic(staging / 'lineage.json', dict(source_case=str(case), source_results=str(source),
                    source_plan_sha256=sha(sourceplan), new_plan_sha256=planhash,
                    copied_artifact_sha256=hashes, copied_utc=datetime.now(timezone.utc).isoformat(),
                    validation='Original frozen worker pair/prefix identity and checksums; exited worker; copy digests verified.'))
                staging.replace(target)
            todo.remove((n, seed))
        completed = 60 - len(todo)
        atomic(output / 'progress.json', dict(state='waiting_for_single_workers' if todo else 'complete',
            updated_utc=datetime.now(timezone.utc).isoformat(), planned_instances=60, planned_methods=300,
            instances_finished=completed, methods_complete=5 * completed,
            pending=[f'single_n{n}_m1_seed{s}' for n, s in todo]))
        if time.monotonic() - last > 30 or not todo:
            print(f'Preserved single cases {completed}/60; pending {len(todo)}', flush=True)
            last = time.monotonic()
        if todo:
            if time.monotonic() > deadline:
                raise TimeoutError('Single snapshot waits on finishing workers; existing copies preserved')
            time.sleep(5)


if __name__ == '__main__':
    main()
