"""Audit the ten completed priority MC cases, then render the requested figure."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = ROOT / 'results_n100'
    phase = ROOT / 'priority_sensitivity'
    plan = json.loads((results / 'plan.json').read_text())
    assert all(sha(ROOT / 'code' / name) == value for name, value in plan['source_hashes'].items())
    sys.path.insert(0, str(ROOT / 'orchestration'))
    import batch_n100 as original
    setting = dict(study='multi', n=100, m=100)
    snapshot = phase / 'selected_mc_snapshot'
    assert not snapshot.exists(), 'Existing selected snapshot needs review before reuse'
    snapshot.mkdir()
    selected = dict(plan, settings=[setting], release_scope='requested_sensitivity_MC_panel',
                    parent_plan_sha256=sha(results / 'plan.json'),
                    purpose='Ten measured MC instances for requested two-panel figure; declared-plan audit scope')
    (snapshot / 'plan.json').write_text(json.dumps(selected, indent=2) + '\n')
    copies = []
    for seed in range(10):
        name = f'multi_n100_m100_seed{seed}'
        source = results / 'verified' / name
        original.verify_case(source, plan, setting, seed)
        files = [f for f in source.iterdir() if f.is_file() and f.suffix in {'.json', '.npz', '.log'}]
        before = {f.name: sha(f) for f in files}
        target = snapshot / name
        target.mkdir()
        for f in files:
            shutil.copy2(f, target / f.name)
        assert before == {f.name: sha(f) for f in files}
        assert all(sha(target / name) == value for name, value in before.items())
        copies.append(dict(case=name, source=str(source), copied_sha256=before))
    (snapshot / 'selection_provenance.json').write_text(json.dumps(dict(
        selected_utc=datetime.now(timezone.utc).isoformat(), parent_plan_sha256=sha(results / 'plan.json'),
        scope='n100,m100,seeds0..9,fivemethods; actual corrected measurements', copies=copies), indent=2) + '\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
               MPLCONFIGDIR=str(ROOT / 'validation/matplotlib_cache'))
    audit = snapshot / 'report/structural'
    subprocess.run([sys.executable, str(ROOT / 'reporting/audit_saved_records.py'),
                    '--results', str(snapshot), '--code', str(ROOT / 'code'), '--plan-scope',
                    '--temp-dir', str(ROOT / 'validation'), '--out', str(audit)], cwd=ROOT, env=env, check=True)
    result = json.loads((audit / 'structural_audit.json').read_text())
    assert result['status'] == 'structural_pass'
    subprocess.run([sys.executable, str(ROOT / 'reporting/plot_sensitivity_two_instances.py'), 'plot',
                    '--left-results', str(ROOT / 'results_single/verified'), '--left-n', '1000', '--left-m', '1',
                    '--right-results', str(results / 'verified'), '--right-n', '100', '--right-m', '100',
                    '--right-audit', str(audit / 'structural_audit.json'),
                    '--out', str(phase / 'figure')], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
