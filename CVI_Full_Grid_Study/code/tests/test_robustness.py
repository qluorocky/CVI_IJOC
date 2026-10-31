import sys
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import numpy as np
import cvxpy as cp
sys.path.insert(0, str(Path.cwd()))
import batch, study
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as td:
    p = study.DenseProblem(4, 1, 7, td)
    solver = study.FullSolver(p)
    with patch.object(cp.Problem, 'solve', side_effect=cp.error.SolverError('injected failure')):
        x, info = solver.project(p.x0 + 0.1, p.x0, float('inf'))
    assert info['status'] == 'slsqp_optimal'
    assert info['fallback_from'] == 'clarabel_solver_error'
    assert max(p.G(x)) <= solver.feasibility_tolerance
    print('PASS raised conic SolverError invokes checked SLSQP fallback')
    policy=SimpleNamespace(workers=7,memory_budget_gib=8.,heavy_companion_m=0)
    heavy={'n':1000,'m':1000,'study':'multi'}
    active={0:{'setting':heavy}}
    assert batch.admissible({'n':10,'m':1,'study':'single'},active,policy)
    assert not batch.admissible({'n':1000,'m':500,'study':'multi'},active,policy)
    assert not batch.admissible({'n':1000,'m':10,'study':'multi'},active,policy)
    print('PASS bounded-memory admission permits light companions and excludes simultaneous large models')
    def late_run(p, seconds, observe):
        state = SimpleNamespace(average=p.x0)
        observe(0, 0., state, {'certificate': float('nan'), 'violation': 0.})
        observe(1, seconds + .01, state, {'certificate': 0., 'violation': 0., 'certificate_stop': True})
        return dict(status='certificate_stop', runtime=seconds+.01)
    out=Path(td)/'boundary'
    with patch.object(study, 'run_proposed', late_run):
        study.worker(4,1,7,.01,out,methods=['proposed'])
    saved=json.loads((out/'proposed_run.json').read_text())
    assert len(saved['checkpoints'])==1
    assert saved['result']['status']=='time_limit'
    print('PASS post-deadline output excluded and late certificate stop classified as timeout')
