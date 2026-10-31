"""Fresh full-grid measurements with bounded-memory worker admission."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
from time import monotonic,sleep
from study import atomic_json,source_hashes

GIB=1024**3
COEF_CAP=5_000_000
THREAD_ENV={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
SETTINGS=([dict(study='single',n=n,m=1) for n in (10,50,100,200,500,1000)]+
          [dict(study='multi',n=1000,m=m) for m in (10,50,100,200,500,1000)])
METHODS=['proposed','oe_true','oe_low','oe_high','golden_ratio']

def weight(s):
    count=s['m']*(s['n']-s['n']//2)*s['n']
    return dict(factor_gib=8*count/GIB,conic_reserve_gib=320*min(count,COEF_CAP)/GIB,
                resident_gib=.35+8*count/GIB+320*min(count,COEF_CAP)/GIB)

def admissible(s,running,a):
    ss=[e['setting'] for e in running.values()]
    if len(ss)>=a.workers or weight(s)['resident_gib']+sum(weight(x)['resident_gib'] for x in ss)>a.memory_budget_gib:return False
    if s['m']>=500 and (any(x['m']>=500 for x in ss) or any(x['m']>max(1,a.heavy_companion_m) for x in ss)):return False
    if any(x['m']>=500 for x in ss) and s['m']>max(1,a.heavy_companion_m):return False
    if s['m']==200 and sum(x['m']==200 for x in ss)>=2:return False
    return True

def name(s,seed):return f"{s['study']}_n{s['n']}_m{s['m']}_seed{seed}"

def verify(case,plan):
    import numpy as np
    meta=json.loads((case/'manifest.json').read_text())
    assert meta['state']=='complete' and meta['source_hashes']==plan['source_hashes'] and meta['cap_seconds']==plan['seconds'],case
    for method in METHODS:
        rr=json.loads((case/f'{method}_run.json').read_text());dd=json.loads((case/f'{method}_diagnostics.json').read_text())
        with np.load(case/f'{method}_points.npz',allow_pickle=False) as saved:xs=saved['x']
        assert xs.shape==(len(rr['checkpoints']),meta['n']) and np.all(np.isfinite(xs)),(case,method)
        assert len(dd)==len(xs)>0 and all(0<=v['elapsed']<=plan['seconds'] for v in dd),(case,method)
        assert meta['methods'][method]['stage']=='complete',(case,method)
    return meta

def main():
    ap=argparse.ArgumentParser()
    for flag,typ,default in [('seconds',float,1000.),('diagnostic-seconds',float,300.),('seeds',int,10),('workers',int,7),('memory-budget-gib',float,8.),('min-free-disk-gib',float,12.),('heavy-companion-m',int,0)]:ap.add_argument('--'+flag,type=typ,default=default)
    ap.add_argument('--out',default='../results');ap.add_argument('--resume',action='store_true');ap.add_argument('--retry-invalid-diagnostics',action='store_true');ap.add_argument('--fixture',action='store_true');ap.add_argument('--skip-report',action='store_true')
    a=ap.parse_args()
    if not 1<=a.workers<=7 or min(a.seconds,a.diagnostic_seconds,a.seeds,a.memory_budget_gib)<=0 or not 0<=a.heavy_companion_m<=100:raise ValueError('Invalid budget/worker parameters')
    root=Path(__file__).resolve().parent;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    settings=[dict(study='single',n=10,m=1),dict(study='multi',n=50,m=10)] if a.fixture else SETTINGS
    plan=dict(study_version='full_grid_corrected_v1',validation_fixture=a.fixture,purpose='validation_fixture' if a.fixture else 'full_grid_measured',
      settings=settings,seeds=list(range(a.seeds)),methods=METHODS,seconds=a.seconds,diagnostic_seconds=a.diagnostic_seconds,workers=a.workers,source_hashes=source_hashes(),prior_runs_reused=False,
      execution_policy=dict(workers=a.workers,affinity='os_managed',logical_cpus=os.cpu_count(),hardware='Apple M4,10cores,16GiB',platform=platform.platform(),machine=platform.machine(),numerical_threads=1,thread_environment=THREAD_ENV,
        memory_budget_gib=a.memory_budget_gib,min_free_disk_gib=a.min_free_disk_gib,model_reserve_bytes_per_coefficient=320,max_simultaneous_m_ge500=1,heavy_companion_max_m=a.heavy_companion_m,max_simultaneous_m200=2,
        admission='First pending admissible case; factor storage plus profiled conic/model/module allowance; heavy cases before light companions'),
      projection_policy=dict(dense_coefficient_cap=COEF_CAP,conic_tolerance=1e-8,feasibility_tolerance=1e-7,slsqp_ftol=1e-10,fallback='full-constraint SLSQP after nonoptimal/error/resource-guard conic result',gap_seconds=a.diagnostic_seconds,shared_absolute_gap_deadline=True))
    pf=out/'plan.json'
    if pf.exists():
        if not a.resume:raise FileExistsError('Use --resume for existing results')
        if json.loads(pf.read_text())!=plan:raise RuntimeError('Plan/source mismatch: use a new output directory')
    atomic_json(pf,plan);working=out/'working';working.mkdir(exist_ok=True);verified=out/'verified';verified.mkdir(exist_ok=True);atomic_json(verified/'plan.json',plan)
    jobs=[(s,seed) for seed in plan['seeds'] for s in settings];jobs.sort(key=lambda j:(0 if j[0]['m']>=500 else 1 if j[0]['m']==1 else 2,j[1],-j[0]['m'],j[0]['n']))
    running={};failed=[];done=[];start=monotonic();last=-60.;env=dict(os.environ,**THREAD_ENV);interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while jobs or running:
        if interrupted:
            for e in running.values():e['process'].terminate()
            for e in running.values():
                try:e['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:e['process'].kill();e['process'].wait()
                e['log'].close()
            atomic_json(out/'controller_status.json',dict(state='interrupted',controller_pid=os.getpid(),note='Completed method pairs retained; interrupted methods restart at full budget'))
            raise SystemExit(130)
        for slot in range(a.workers):
            if slot in running:continue
            index=next((i for i,(s,seed) in enumerate(jobs) if admissible(s,running,a) and shutil.disk_usage(out).free/GIB>=a.min_free_disk_gib+weight(s)['factor_gib']),None)
            if index is None:continue
            s,seed=jobs.pop(index);case_name=name(s,seed);case=working/case_name;case.mkdir(exist_ok=True);log=(case/'worker.log').open('a',buffering=1)
            cmd=[sys.executable,str(root/'study.py'),'--n',str(s['n']),'--m',str(s['m']),'--seed',str(seed),'--seconds',str(a.seconds),'--diagnostic-seconds',str(a.diagnostic_seconds),'--out',str(case),'--methods',','.join(METHODS)]
            if a.resume:cmd.append('--resume')
            if a.retry_invalid_diagnostics:cmd.append('--retry-invalid-diagnostics')
            process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[slot]=dict(process=process,name=case_name,setting=s,seed=seed,log=log)
        for slot,e in list(running.items()):
            if e['process'].poll() is None:continue
            e['log'].close();case=working/e['name']
            if e['process'].returncode==0:
                try:
                    verify(case,plan);target=verified/e['name'];target.mkdir(exist_ok=True)
                    for f in case.iterdir():
                        if f.is_file() and f.suffix in {'.json','.npz','.log'}:shutil.copy2(f,target/f.name)
                    done.append(e['name'])
                except Exception as error:failed.append(dict(case=e['name'],reason=repr(error)))
            else:failed.append(dict(case=e['name'],exit_code=e['process'].returncode))
            del running[slot]
        timed=complete=invalid=0;stages={};method_errors=[]
        for mf in working.glob('*/manifest.json'):
            meta=json.loads(mf.read_text());stages[mf.parent.name]=meta['state'];invalid+=meta.get('invalid_diagnostic_count',0)
            for method,v in meta.get('methods',{}).items():
                timed+=v['stage'] in {'timed_complete','complete'};complete+=v['stage']=='complete'
                if v.get('status') not in {'time_limit','certificate_stop'}:method_errors.append(dict(case=mf.parent.name,method=method,status=v.get('status')))
        live={}
        for e in running.values():
            p=working/e['name']/'live.json'
            if p.exists():live[e['name']]=json.loads(p.read_text())
        state='running' if jobs or running else ('failed' if failed else 'complete')
        progress=dict(state=state,controller_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=monotonic()-start,
          planned_instances=len(settings)*a.seeds,planned_methods=len(settings)*a.seeds*len(METHODS),instances_finished=len(done),methods_timed=int(timed),methods_complete=int(complete),pending_instances=len(jobs),
          active=[e['name'] for e in running.values()],worker_pids={e['name']:e['process'].pid for e in running.values()},active_modeled_resident_gib=sum(weight(e['setting'])['resident_gib'] for e in running.values()),invalid_diagnostics=invalid,failed=failed,method_errors=method_errors,stages=stages,live=live)
        atomic_json(out/'progress.json',progress);atomic_json(out/'controller_status.json',{k:progress[k] for k in ('state','controller_pid','updated_utc','elapsed_seconds','instances_finished','pending_instances','active')})
        if monotonic()-start-last>=30 or state!='running':print(json.dumps({k:v for k,v in progress.items() if k not in {'stages','live'}}),flush=True);last=monotonic()-start
        if jobs and not running and not any(weight(s)['resident_gib']<=a.memory_budget_gib and shutil.disk_usage(out).free/GIB>=a.min_free_disk_gib+weight(s)['factor_gib'] for s,_ in jobs):raise RuntimeError('Pending jobs cannot fit configured RAM/disk admission limits')
        if jobs or running:sleep(2)
    if failed:raise SystemExit('Worker/export failures recorded in progress.json')
    if not a.skip_report and not a.fixture:subprocess.run([sys.executable,str(root.parent/'reporting/report_full_grid.py'),'--results',str(verified)],cwd=root.parent,env=env,check=True)
    atomic_json(out/'controller_status.json',dict(state='complete',controller_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),instances_finished=len(done)))

if __name__=='__main__':main()
