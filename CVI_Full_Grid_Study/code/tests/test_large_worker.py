from pathlib import Path
from unittest.mock import patch
import tempfile,json,time,sys
import numpy as np
sys.path.insert(0,str(Path.cwd()))
import study
from algorithms import Deadline
from threadpoolctl import threadpool_limits
checks=[]
def done(name):checks.append(name);print('PASS',name,flush=True)
with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as td, patch.object(study,'source_hashes',lambda:{'fixture':'fixed'}):
 root=Path(td);p=study.DenseProblem(4,1,2,root)
 solver=study.FullSolver(p);solver.max_conic_coefficients=0
 x,info=solver.project(p.x0+.2,p.x0,float('inf'))
 assert info['status']=='slsqp_optimal' and info['fallback_from']=='memory_bounded_conic_guard'
 assert max(p.G(x))<=solver.feasibility_tolerance
 done('memory_coefficient_guard_falls_back_to_checked_full_problem')
 def expired(*args):
  time.sleep(.025);return None,None,dict(valid=False,status='injected_nonoptimal',rounds=1)
 with patch.object(solver,'_solve',expired):
  try:solver.gap(p.x0,seconds=.01)
  except Deadline:pass
  else:raise AssertionError('Fallback received a fresh diagnostic budget')
 done('gap_conic_and_fallback_share_one_absolute_deadline')
 out=root/'resumption';study.worker(4,1,2,.03,out,methods=['oe_true'],diagnostic_seconds=2)
 manifest=out/'manifest.json';diagfile=out/'oe_true_diagnostics.json';rows=json.loads(diagfile.read_text());assert len(rows)>1
 original_count=len(rows);diagfile.write_text(json.dumps(rows[:1]));meta=json.loads(manifest.read_text());meta['state']='diagnostics';manifest.write_text(json.dumps(meta))
 realgap=study.FullSolver.gap;calls=[]
 def counted(self,z,seconds=120):calls.append(1);return realgap(self,z,seconds)
 with patch.object(study.FullSolver,'gap',counted),patch.object(study,'run_baseline',side_effect=AssertionError('timed method rerun')):
  study.worker(4,1,2,.03,out,methods=['oe_true'],resume=True,diagnostic_seconds=2)
 assert len(calls)==original_count-1
 done('partial_diagnostics_resume_without_repeating_timed_methods_or_valid_prefix')
 rows=json.loads(diagfile.read_text());rows[0].update(gap_valid=False,gap_status='injected_invalid',gap=float('nan'),residual=float('nan'));diagfile.write_text(json.dumps(rows));meta=json.loads(manifest.read_text());meta['state']='diagnostics';manifest.write_text(json.dumps(meta));calls.clear()
 with patch.object(study.FullSolver,'gap',counted),patch.object(study,'run_baseline',side_effect=AssertionError('timed method rerun')):
  study.worker(4,1,2,.03,out,methods=['oe_true'],resume=True,diagnostic_seconds=2,retry_invalid_diagnostics=True)
 repaired=json.loads(diagfile.read_text());assert len(calls)==1 and repaired[0]['gap_valid'] and repaired[0]['attempt_history'][0]['gap_status']=='injected_invalid'
 done('explicit_invalid_diagnostic_retry_preserves_attempt_history')
 try:study.worker(4,1,2,.04,out,methods=['oe_true'],resume=True,diagnostic_seconds=2)
 except RuntimeError as error:assert 'cap_seconds' in str(error)
 else:raise AssertionError('Changed method cap accepted on resume')
 done('resume_refuses_changed_method_budget')
 pointfile=out/'oe_true_points.npz';pointfile.write_bytes(pointfile.read_bytes()+b'tamper')
 try:study.worker(4,1,2,.03,out,methods=['oe_true'],resume=True,diagnostic_seconds=2)
 except RuntimeError as error:assert 'checksum' in str(error)
 else:raise AssertionError('Tampered point pair accepted')
 done('resume_refuses_tampered_point_pair')
 out=root/'invalid'
 with patch.object(study.FullSolver,'gap',side_effect=Deadline()):
  study.worker(4,1,3,.01,out,methods=['oe_true'],diagnostic_seconds=.01)
 rows=json.loads((out/'oe_true_diagnostics.json').read_text());meta=json.loads((out/'manifest.json').read_text())
 assert meta['state']=='complete' and meta['invalid_diagnostic_count']==len(rows)>0 and all(not r['gap_valid'] and r['gap_status']=='diagnostic_time_limit' for r in rows)
 done('invalid_diagnostics_are_complete_explicit_records_not_missing_values')
 out=root/'method_error'
 with patch.object(study,'run_proposed',side_effect=ArithmeticError('injected algorithm failure')):
  study.worker(4,1,4,.01,out,methods=['proposed'],diagnostic_seconds=2)
 saved=json.loads((out/'proposed_run.json').read_text());zs=np.load(out/'proposed_points.npz')['x']
 assert saved['result']['status']=='method_error' and saved['result']['error']['exception_type']=='ArithmeticError'
 assert len(zs)==1 and np.allclose(zs[0],study.DenseProblem(4,1,4,root).x0)
 done('method_exception_preserves_only_known_initial_output_with_failure_status')
 out=root/'late_output'
 from types import SimpleNamespace
 def late_run(p,seconds,observe):
  state=SimpleNamespace(average=p.x0)
  observe(0,0.,state,{'certificate':float('nan'),'violation':0.})
  observe(1,seconds+.01,state,{'certificate':0.,'violation':0.,'certificate_stop':True})
  return dict(status='certificate_stop',runtime=seconds+.01)
 with patch.object(study,'run_proposed',late_run):
  study.worker(4,1,5,.01,out,methods=['proposed'],diagnostic_seconds=2)
 saved=json.loads((out/'proposed_run.json').read_text())
 assert len(saved['checkpoints'])==1 and saved['result']['status']=='time_limit'
 done('post_deadline_output_excluded_and_late_certificate_is_timeout')
print(json.dumps({'checks':checks,'count':len(checks),'all_passed':True}))
