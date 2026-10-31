"""Coordinate an intentional method-boundary handoff, never pausing worker clocks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / 'notes/priority_sensitivity_transition.json'


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def process(pid):
    out = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,command='], capture_output=True, text=True)
    if out.returncode == 1 and not out.stdout.strip():
        return None
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    stat, command = out.stdout.strip().split(None, 1)
    return None if stat.startswith('Z') else command


def main():
    note = json.loads(NOTE.read_text())
    source = Path(note['source_results'])
    assert hashlib.sha256((source / 'plan.json').read_bytes()).hexdigest() == note['source_plan_sha256']
    assert note['original_controller_frozen']
    sys.path.insert(0, str(ROOT / 'code'))
    import numpy as np
    from study import validate_method_pair
    pending = []
    for item in note['stopped_owned_workers']:
        case = Path(item['path'])
        live = json.loads((case / 'live.json').read_text())
        item['boundary_method'] = live.get('method')
        item['boundary_watch_started_utc'] = datetime.now(timezone.utc).isoformat()
        pending.append(item)
    atomic(NOTE, note)
    deadline = time.monotonic() + 1200
    last_probe = {}
    while pending:
        for item in list(pending):
            case, pid = Path(item['path']), item['pid']
            meta = json.loads((case / 'manifest.json').read_text())
            all_saved = all((case / f'{m}_run.json').exists() for m in meta['chosen_methods']) if 'chosen_methods' in meta else all((case / f'{m}_run.json').exists() for m in ('proposed', 'oe_true', 'oe_low', 'oe_high', 'golden_ratio'))
            boundary = item['boundary_method']
            ready = meta['state'] == 'complete' or all_saved or (case / f'{boundary}_run.json').exists()
            if not ready and time.monotonic() - last_probe.get(pid, 0) < 2:
                continue
            command = process(pid)
            last_probe[pid] = time.monotonic()
            if command is not None:
                assert str(ROOT / 'code/study.py') in command and str(case) in command
                if not ready:
                    continue
                if meta['state'] != 'complete' and not all_saved:
                    rng = np.random.default_rng(np.random.SeedSequence([meta['seed'], meta['n'], 101]))
                    rng.uniform(0, 1, (meta['n'], meta['n'])); rng.uniform(0, 1, meta['n'] - meta['n'] // 2)
                    d = rng.uniform(0, 1, meta['n']); x0 = -d / np.linalg.norm(d)
                    assert validate_method_pair(case, boundary, meta, x0) is not None
                os.kill(pid, signal.SIGTERM)
                item['signal'] = 'SIGTERM after saved current-method pair (or all timed pairs saved)'
                item['signal_utc'] = datetime.now(timezone.utc).isoformat()
                until = time.monotonic() + 10
                while process(pid) is not None:
                    if time.monotonic() > until:
                        raise TimeoutError(f'{pid} did not stop')
                    time.sleep(.05)
            item['quiescent'] = True
            item['stop_reason'] = 'user_requested_figure_priority_not_algorithm_failure'
            item['manifest_after_stop'] = json.loads((case / 'manifest.json').read_text())
            item['live_after_stop'] = json.loads((case / 'live.json').read_text())
            item['committed_artifact_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in case.iterdir() if f.name.endswith(('_run.json', '_points.npz', '_diagnostics.json'))}
            pending.remove(item)
            atomic(NOTE, note)
            print(f'Preserved boundary and stopped {item["case"]}; remaining {len(pending)}', flush=True)
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError('Method-boundary wait exceeded; scheduler remains held; records preserved')
            time.sleep(.05)
    controller = note['original_controller_pid']
    command = process(controller)
    if command is not None:
        assert str(ROOT / 'orchestration/batch_n100.py') in command
        os.kill(controller, signal.SIGKILL)
    wrapper = note['original_wrapper_pid']
    until = time.monotonic() + 10
    while process(wrapper) is not None and time.monotonic() < until:
        time.sleep(.1)
    command = process(wrapper)
    if command is not None:
        assert str(ROOT / 'run_n100_study.py') in command
        os.kill(wrapper, signal.SIGTERM)
        time.sleep(.2)
    assert process(controller) is None and process(wrapper) is None
    note['old_controller_stopped'] = True
    note['quiescence_confirmed'] = True
    note['quiescent_utc'] = datetime.now(timezone.utc).isoformat()
    note['phase'] = 'ready_for_priority_runner'
    atomic(NOTE, note)
    atomic(ROOT / 'priority_sensitivity/status.json', dict(state=note['phase'], transition=str(NOTE),
           quiescent_utc=note['quiescent_utc']))
    print('Priority handoff is quiescent; original plans, committed records and source remain unchanged.', flush=True)


if __name__ == '__main__':
    main()
