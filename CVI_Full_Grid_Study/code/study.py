"""Dense-paper instances, full projections, diagnostics, and auditable workers.

Fresh study: independent dense rank-deficient quadratic constraints, radius 2,
unscaled coefficients. Large PSD factors are memory mapped to respect 8 GB RAM.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from collections import Counter
from time import perf_counter

import numpy as np
import scipy
import cvxpy as cp
import clarabel
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from algorithms import Deadline, run_proposed

ROOT = Path(__file__).resolve().parent
METHODS = ['proposed', 'oe_true', 'oe_low', 'oe_high', 'golden_ratio']
THRESHOLDS = [1e-2, 1e-3, 1e-4, 1e-5]
MAX_CONIC_COEFFICIENTS = 5_000_000


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == 'Darwin' else value*1024)


class DenseProblem:
    def __init__(self, n, m, seed, cache_dir, initialize_progress=None):
        self.n, self.m, self.seed, self.r = n, m, seed, 2.
        if n < 2 or m < 1:
            raise ValueError('n>=2 and m>=1 required')
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, 101]))
        raw = rng.uniform(0.,1.,(n,n))
        self.h = np.r_[rng.uniform(0.,1., n-n//2), np.zeros(n//2)]
        self.B = raw-raw.T+np.diag(self.h)
        self.d = rng.uniform(0.,1.,n)
        self.x0 = -self.d/np.linalg.norm(self.d)
        if initialize_progress:
            initialize_progress('operator_norm', 0, m)
        self.M = float(np.linalg.norm(self.B,2))
        rank = n-n//2
        shape = (m,rank,n)
        self.factor_file = None
        self.factor_bytes = int(np.prod(shape)*8)
        if np.prod(shape)*8 > 128*1024**2:
            self.factor_file = Path(cache_dir)/'quadratic_factors.dat'
            self.W = np.memmap(self.factor_file, dtype='float64', mode='w+', shape=shape)
        else:
            self.W = np.empty(shape)
        self.b = np.empty((m,n))
        self.c = np.empty(m)
        self.eigen_max = np.empty(m)
        for i in range(m):
            if initialize_progress:
                initialize_progress('constraint_factors', i, m)
            local = np.random.default_rng(np.random.SeedSequence([seed,n,i,202]))
            # Each independent uniform matrix has its own orthogonal basis.
            Q, _ = np.linalg.qr(local.uniform(0.,1.,(n,n)))
            values = local.uniform(0.,1.,rank)
            self.W[i] = np.sqrt(values)[:,None]*Q[:,:rank].T
            self.b[i] = local.uniform(0.,1.,n)
            wx = self.W[i]@self.x0
            self.c[i] = -.5*(wx@wx)-self.b[i]@self.x0-.02
            self.eigen_max[i] = max(values)
        if isinstance(self.W,np.memmap):
            self.W.flush()
        if initialize_progress:
            initialize_progress('instance_digest', m, m)
        self._x = self._wx = self._J = None
        self.instance_hash = self.digest()
        assert np.max(self.G(self.x0)) < -.0199999
        assert np.linalg.norm(self.x0) < self.r

    def digest(self):
        h = hashlib.sha256()
        for a in [self.B,self.d,self.b,self.c,self.h]:
            h.update(a.tobytes())
        for i in range(self.m):
            h.update(np.asarray(self.W[i]).tobytes())
        return h.hexdigest()

    def _products(self,x):
        x = np.asarray(x)
        if self._x is None or not np.array_equal(self._x,x):
            self._x = x.copy()
            self._wx = np.einsum('mkn,n->mk',self.W,x,optimize=False)
            self._J = None
        return self._wx

    def F(self,x):
        return self.B@x-self.d

    def G(self,x):
        wx = self._products(x)
        return .5*np.einsum('mk,mk->m',wx,wx)+self.b@x+self.c

    def J(self,x):
        wx = self._products(x)
        if self._J is None:
            self._J = np.einsum('mkn,mk->mn',self.W,wx,optimize=False)+self.b
        return self._J.copy()

    def remainder(self,delta):
        wd = np.einsum('mkn,n->mk',self.W,delta,optimize=False)
        return .5*np.einsum('mk,mk->m',wd,wd)


class ReferenceSLSQP:
    """Fresh SLSQP minimizations over the full original feasible set.

    Constraint callables are reused, optimizer state and factorizations are not.
    Projection initial guesses use the previous feasible projected point.
    """
    ftol = 1e-10
    feasibility_tolerance = 1e-7

    def __init__(self,p):
        self.p = p
        self.deadline = float('inf')
        self.last_gap_point = p.x0.copy()

    def check(self):
        if perf_counter() >= self.deadline:
            raise Deadline()

    def constraints(self,x):
        self.check()
        out = np.r_[self.p.r**2-x@x, -self.p.G(x)]
        self.check()
        return out

    def jacobian(self,x):
        self.check()
        out = np.vstack([-2*x,-self.p.J(x)])
        self.check()
        return out

    def solve(self,fun,jac,x0,deadline):
        self.deadline = deadline
        def f(x):
            self.check()
            return fun(x)
        def j(x):
            self.check()
            return jac(x)
        result = minimize(f,np.asarray(x0),jac=j,method='SLSQP',
            constraints=[dict(type='ineq',fun=self.constraints,jac=self.jacobian)],
            options=dict(ftol=self.ftol,maxiter=2000,disp=False))
        self.check()
        feasible = float(min(self.constraints(result.x))) >= -self.feasibility_tolerance
        return result, bool(result.success and feasible)

    def project(self,center,x0,deadline):
        out,valid = self.solve(lambda y:.5*float((y-center)@(y-center)),
                              lambda y:y-center,x0,deadline)
        if not valid:
            raise RuntimeError(f'projection_status={out.status}: {out.message}')
        return out.x.copy(), dict(status=int(out.status),nit=int(out.nit),
                                  feasibility=max(0.,-float(min(self.constraints(out.x)))))

    def gap(self,z,seconds=120,deadline=None):
        p = self.p
        linear = p.B.T@z+p.d
        out,valid = self.solve(lambda y:float(p.h@(y*y)-linear@y),
                              lambda y:2*p.h*y-linear,self.last_gap_point,
                              perf_counter()+seconds if deadline is None else deadline)
        if valid:
            self.last_gap_point = out.x.copy()
        value = max(0.,-float(out.fun)-float(p.d@z)) if valid else np.nan
        return value, dict(status=int(out.status),valid=valid,nit=int(out.nit),
                          feasibility=max(0.,-float(min(self.constraints(out.x)))),
                          raw_objective=float(out.fun), message=str(out.message))


class FullSolver:
    """Generic conic solver with exact constraint generation in principle.

    A restricted convex problem is solved, then ALL original constraints are
    checked. If its minimizer is feasible for all constraints, it minimizes the
    full problem: its restricted optimum is a lower bound, and its full-feasible
    objective is an upper bound. Only numerical optimality is claimed in code.
    Every construction, restricted solve, and full check is charged to runtime.
    No CVXPY model/factorization/warm start is reused across projections.
    """
    tolerance=1e-8
    feasibility_tolerance=1e-7
    max_conic_coefficients=MAX_CONIC_COEFFICIENTS

    def __init__(self,p):
        self.p=p
        self.projection_active=set(range(p.m)) if p.m<=20 else set()
        self.gap_active=set(range(p.m)) if p.m<=20 else set()
        self.fallback=ReferenceSLSQP(p)

    def _solve(self,objective,active,deadline):
        p=self.p
        rounds=0
        for _ in range(p.m+1):
            if perf_counter()>=deadline:
                raise Deadline()
            coefficients=sum(int(p.W[i].size) for i in active)
            if coefficients>self.max_conic_coefficients:
                return None,None,dict(valid=False,status='memory_bounded_conic_guard',
                    rounds=rounds,conic_coefficients=coefficients,
                    max_conic_coefficients=self.max_conic_coefficients)
            y=cp.Variable(p.n)
            constraints=[cp.sum_squares(y)<=p.r**2]
            for i in sorted(active):
                if perf_counter()>=deadline:
                    raise Deadline()
                constraints.append(.5*cp.sum_squares(np.asarray(p.W[i])@y)
                                   +p.b[i]@y+p.c[i]<=0)
            model=cp.Problem(cp.Minimize(objective(y)),constraints)
            remaining=deadline-perf_counter()
            if remaining<=0:
                raise Deadline()
            try:
                model.solve(solver=cp.CLARABEL,warm_start=False,verbose=False,
                            tol_feas=self.tolerance,tol_gap_abs=self.tolerance,
                            tol_gap_rel=self.tolerance,max_threads=1,
                            time_limit=max(1e-4,remaining))
            except cp.error.SolverError as error:
                if perf_counter()>=deadline:
                    raise Deadline() from error
                return None,model,dict(valid=False,status='clarabel_solver_error',
                    rounds=rounds+1,message=str(error))
            rounds+=1
            if perf_counter()>=deadline:
                raise Deadline()
            if model.status!=cp.OPTIMAL or y.value is None:
                return None,model,dict(valid=False,status=str(model.status),rounds=rounds)
            point=np.asarray(y.value).reshape(-1)
            violation=p.G(point)
            ball=max(0.,np.linalg.norm(point)-p.r)
            maximum=max(0.,float(max(violation)))
            if perf_counter()>=deadline:
                raise Deadline()
            if maximum<=self.feasibility_tolerance and ball<=self.feasibility_tolerance:
                return point,model,dict(valid=True,status=str(model.status),rounds=rounds,
                    active_constraints=len(active),feasibility=max(maximum,ball),
                    active_ids=sorted(active))
            missed=[int(i) for i in np.argsort(violation)[::-1]
                    if violation[i]>self.feasibility_tolerance and int(i) not in active]
            if not missed:
                return None,model,dict(valid=False,status='feasibility_check_failed',rounds=rounds)
            active.update(missed[:10])
        raise ArithmeticError('Full-constraint generation failed')

    def project(self,center,x0,deadline):
        point,model,info=self._solve(lambda y:.5*cp.sum_squares(y-center),
                                    self.projection_active,deadline)
        if not info['valid']:
            failed=info['status']
            point,reference=self.fallback.project(center,x0,deadline)
            info=dict(valid=True,status='slsqp_optimal',feasibility=reference['feasibility'],
                      rounds=info['rounds']+1,active_constraints=self.p.m,
                      fallback_from=failed)
        return point,info

    def gap(self,z,seconds=120):
        deadline=perf_counter()+seconds
        linear=self.p.B.T@z+self.p.d
        point,model,info=self._solve(
            lambda y:cp.sum(cp.multiply(self.p.h,cp.square(y)))-linear@y,
            self.gap_active,deadline)
        if not info['valid']:
            value,reference=self.fallback.gap(z,seconds,deadline=deadline)
            reference['status']='slsqp_optimal' if reference['valid'] else reference['status']
            reference['fallback_from']=info['status']
            return value,reference
        value=max(0.,-float(model.value)-float(self.p.d@z)) if info['valid'] else np.nan
        info['raw_objective']=float(model.value) if model.value is not None else np.nan
        return value,info


def run_baseline(p,method,seconds,observe,epsilon=1e-5,max_steps=10_000_000):
    start = perf_counter()
    end = start+seconds
    solver = FullSolver(p)
    # Source constants and setup are supplied before starting the method clock.
    estimate = {'oe_true':1.,'oe_low':.1,'oe_high':10.}.get(method)
    x,old_x = p.x0.copy(),p.x0.copy()
    f = p.F(x)
    old_f = f.copy()
    average = p.x0.copy()
    counts = dict(F_calls=1,projection_calls=0,completed_projections=0,projection_seconds=0.,
                  projection_rounds=0,fallback_projections=0,max_projection_feasibility=0.,
                  memory_guard_fallback_projections=0)
    phi = (1+np.sqrt(5))/2
    rho = 1/phi+1/phi**2
    # The original paper's local secant initialization uses no global M.
    alpha,alpha_max,theta = 1.,1e6,1.
    if estimate is None:
        old_x=x.copy()
        old_x[0]+=1e-3
        old_f=p.F(old_x)
        counts['F_calls']+=1
        denominator=np.linalg.norm(f-old_f)
        alpha=np.linalg.norm(x-old_x)/denominator if denominator else 1.
    initial_alpha=float(alpha)
    observe(0,0.,average,dict(counts))
    auxiliary = x.copy()
    weight,total = 0.,np.zeros(p.n)
    status = 'time_limit'
    failure = None
    k=0
    try:
        while True:
            if perf_counter() >= end:
                break
            k += 1
            if estimate is not None:
                center = x-(2*f-old_f)/(2*estimate*p.M)
                used_alpha=1/(2*estimate*p.M)
            else:
                # Algorithm 1 theta_0=1, including the first step.
                difference = f-old_f
                denominator=float(difference@difference)
                ratio=float((x-old_x)@(x-old_x))/denominator if denominator else np.inf
                new_alpha=min(rho*alpha,phi*theta/(4*alpha)*ratio,alpha_max)
                auxiliary=((phi-1)*x+auxiliary)/phi
                center=auxiliary-new_alpha*f
                used_alpha=new_alpha
            clock=perf_counter()
            counts['projection_calls'] += 1
            try:
                xn,info=solver.project(center,x,end)
            finally:
                counts['projection_seconds'] += perf_counter()-clock
            counts['completed_projections']+=1
            counts['projection_rounds']+=info['rounds']
            counts['fallback_projections']+=int(info['status']=='slsqp_optimal')
            counts['memory_guard_fallback_projections']+=int(
                info.get('fallback_from')=='memory_bounded_conic_guard')
            counts['max_projection_feasibility']=max(counts['max_projection_feasibility'],info['feasibility'])
            fn=p.F(xn)
            counts['F_calls'] += 1
            if perf_counter() >= end:
                break
            if estimate is not None:
                average += (xn-average)/k
            else:
                # Section 2.2: alpha_k*z_k, not the auxiliary or z_{k+1}.
                weight += new_alpha
                total += new_alpha*x
                average=total/weight
                theta=phi*new_alpha/alpha
                alpha=new_alpha
            observe(k,perf_counter()-start,average,
                dict(counts,M_estimate=estimate*p.M if estimate else None,
                     alpha=used_alpha,weight=weight,projection_status=info['status'],
                     projection_feasibility=info['feasibility'],
                     projection_rounds=info['rounds'],
                     active_constraints=info['active_constraints']))
            old_x,x=x,xn
            old_f,f=f,fn
            if k>=max_steps:
                status='iteration_limit'
                break
    except Deadline:
        pass
    except RuntimeError as e:
        status='method_error'
        failure=dict(exception_type=type(e).__name__,message=str(e))
    except Exception as e:
        status='method_error'
        failure=dict(exception_type=type(e).__name__,message=str(e))
    return dict(status=status,runtime=perf_counter()-start,attempted_steps=k,
                counts=counts,phi=float(phi),alpha0=initial_alpha,alpha_max=alpha_max,
                theta0=1.,fixed_M=(estimate*p.M if estimate else None),error=failure)


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=True,default=float)+'\n')
    temporary.replace(path)


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.glob('*.py'))}


def atomic_points(path, points):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,x=np.stack(points))
    temporary.replace(path)


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def validate_method_pair(out,method,config,x0):
    """A pair is reusable only with matching identity, digest and checkpoints."""
    runfile=out/f'{method}_run.json';pointfile=out/f'{method}_points.npz'
    if not (runfile.exists() and pointfile.exists()):
        return None
    saved=json.loads(runfile.read_text())
    meta=saved.get('metadata',{})
    for key in ['n','m','seed','cap_seconds','instance_hash','source_hashes']:
        if meta.get(key)!=config[key]:
            raise RuntimeError(f'{method}: saved pair identity mismatch for {key}')
    if meta.get('method')!=method:
        raise RuntimeError(f'{method}: saved method identity mismatch')
    if saved.get('points_sha256')!=file_sha256(pointfile):
        raise RuntimeError(f'{method}: saved point checksum mismatch')
    with np.load(pointfile,allow_pickle=False) as archive:
        zs=archive['x'].copy()
    samples=saved['checkpoints']
    if zs.shape!=(len(samples),config['n']) or not len(samples) or not np.all(np.isfinite(zs)):
        raise RuntimeError(f'{method}: saved point/checkpoint shape or values invalid')
    if not np.array_equal(zs[0],x0):
        raise RuntimeError(f'{method}: saved initial point mismatch')
    times=np.array([s['elapsed'] for s in samples],dtype=float)
    if not np.all(np.isfinite(times)) or min(times)<0 or max(times)>config['cap_seconds'] or np.any(np.diff(times)<=0):
        raise RuntimeError(f'{method}: saved checkpoint times invalid')
    runtime=saved['result'].get('runtime')
    if runtime is None or not np.isfinite(runtime) or runtime<times[-1]:
        raise RuntimeError(f'{method}: saved runtime invalid')
    return saved,zs


def validate_diagnostic_prefix(rows,saved,method,config):
    if len(rows)>len(saved['checkpoints']):
        raise RuntimeError(f'{method}: too many saved diagnostic rows')
    for i,(row,sample) in enumerate(zip(rows,saved['checkpoints'])):
        identity=(row.get('n'),row.get('m'),row.get('seed'),row.get('method'),row.get('checkpoint'))
        if identity!=(config['n'],config['m'],config['seed'],method,i):
            raise RuntimeError(f'{method}: saved diagnostic identity mismatch at {i}')
        if row.get('elapsed')!=sample['elapsed'] or row.get('k')!=sample['k']:
            raise RuntimeError(f'{method}: saved diagnostic checkpoint mismatch at {i}')


def worker(n,m,seed,seconds,out,core=None,methods=None,resume=False,
           diagnostic_seconds=300.,retry_invalid_diagnostics=False):
    gap_seconds=float(diagnostic_seconds)
    if seconds<=0 or gap_seconds<=0:
        raise ValueError('Positive method and diagnostic budgets required')
    chosen=list(methods or METHODS)
    if not chosen or len(set(chosen))!=len(chosen) or any(v not in METHODS for v in chosen):
        raise ValueError('Methods must be a unique nonempty subset of METHODS')
    if core is not None:
        if not hasattr(os,'sched_setaffinity'):
            raise ValueError('CPU affinity is unavailable on this platform; omit --core')
        os.sched_setaffinity(0,{core})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    hashes=source_hashes()
    if previous and not resume:
        raise FileExistsError(f'Refusing to overwrite {out}')
    if previous:
        expected=dict(n=n,m=m,seed=seed,cap_seconds=seconds,requested_methods=chosen,source_hashes=hashes)
        for key,value in expected.items():
            if previous.get(key)!=value:
                raise RuntimeError(f'Resume identity mismatch for {key}; use a new output directory')
        if previous['state']=='complete' and not retry_invalid_diagnostics:
            x0=np.asarray(previous['initial_point'],dtype=float)
            for method in chosen:
                pair=validate_method_pair(out,method,previous,x0)
                if pair is None:
                    raise RuntimeError(f'{method}: complete manifest has missing method pair')
                saved,zs=pair
                rows=json.loads((out/f'{method}_diagnostics.json').read_text())
                validate_diagnostic_prefix(rows,saved,method,previous)
                if len(rows)!=len(zs):
                    raise RuntimeError(f'{method}: complete manifest has incomplete diagnostics')
            print(f'REUSE n={n} m={m} seed={seed}',flush=True)
            return
    with threadpool_limits(limits=1):
        started=perf_counter();next_setup_live=0.
        config=dict(n=n,m=m,seed=seed,cap_seconds=seconds,Lambda=1.,M0=1.,
                    L0=.5,radius=2.,scaling='none',source_hashes=hashes,
                    numpy=np.__version__,scipy=scipy.__version__,
                    cvxpy=cp.__version__,clarabel=clarabel.__version__,
                    python=platform.python_version(),platform=platform.platform(),
                    core=core,threads=1,epsilon=1e-5,
                    affinity='pinned' if core is not None else 'os_managed',
                    logical_cpus=os.cpu_count(),requested_methods=chosen,
                    gap_seconds=gap_seconds,diagnostic_seconds=gap_seconds,
                    diagnostic_budget_seconds=gap_seconds,max_conic_coefficients=MAX_CONIC_COEFFICIENTS,
                    factor_bytes=int(m*(n-n//2)*n*8),
                    thread_environment={key:os.environ.get(key) for key in
                        ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']},
                    state='initializing',methods={})
        if previous:
            config['methods']=dict(previous['methods'])
            config['resume_count']=previous.get('resume_count',0)+1
            if previous.get('gap_seconds')!=gap_seconds:
                config['diagnostic_budget_history']=previous.get('diagnostic_budget_history',[])+[
                    dict(previous_gap_seconds=previous.get('gap_seconds'),new_gap_seconds=gap_seconds)]
            elif 'diagnostic_budget_history' in previous:
                config['diagnostic_budget_history']=previous['diagnostic_budget_history']
        atomic_json(manifest,config)
        def initialize_progress(phase,completed,total):
            nonlocal next_setup_live
            elapsed=perf_counter()-started
            if elapsed>=next_setup_live or completed==total:
                atomic_json(out/'live.json',dict(method='initialization',k=0,elapsed=elapsed,
                    state='initializing',phase=phase,completed_constraints=completed,
                    total_constraints=total,factor_bytes=config['factor_bytes'],
                    rss_peak_bytes=peak_rss_bytes(),certificate=None,violation=None,
                    projection_calls=0))
                next_setup_live=elapsed+10.
        p=DenseProblem(n,m,seed,out,initialize_progress=initialize_progress)
        if previous and previous.get('instance_hash') not in (None,p.instance_hash):
            raise RuntimeError('Regenerated instance hash differs from saved timed instance')
        config.update(state='running',setup_seconds=perf_counter()-started,
                      instance_hash=p.instance_hash,true_M=p.M,initial_point=p.x0.tolist(),
                      min_initial_constraint=float(min(p.G(p.x0))),rss_peak_bytes=peak_rss_bytes())
        atomic_json(manifest,config)
        order=chosen[seed%len(chosen):]+chosen[:seed%len(chosen)]
        for method in order:
            pair=validate_method_pair(out,method,config,p.x0)
            if pair is not None:
                saved,zs=pair
                config['methods'][method]=dict(status=saved['result']['status'],
                    runtime=saved['result']['runtime'],checkpoints=len(zs),stage='timed_complete')
                atomic_json(manifest,config)
                continue
            samples=[];points=[];next_time=0.;next_live=10.;last=None
            def observe(k,t,value,info):
                nonlocal next_time,next_live,last
                if t>seconds:
                    return
                s=value if method=='proposed' else None
                z=s.average if s is not None else value
                last=(k,t,z.copy(),dict(info))
                if t>=next_live:
                    atomic_json(out/'live.json',dict(method=method,k=k,elapsed=t,
                        certificate=info.get('certificate'),violation=info.get('violation'),
                        projection_calls=info.get('projection_calls'),state='running',
                        rss_peak_bytes=peak_rss_bytes()))
                    next_live=t+10.
                if k==0 or t>=next_time:
                    samples.append(dict(info,k=k,elapsed=t));points.append(z.copy())
                    next_time=max(1e-4,next_time*1.4,t*1.4)
            print(f'START n={n} m={m} seed={seed} method={method}',flush=True)
            method_started=perf_counter()
            try:
                result=(run_proposed(p,seconds,observe) if method=='proposed'
                        else run_baseline(p,method,seconds,observe))
            except Exception as error:
                observed=last[3] if last is not None else {}
                counters=['F_calls','G_calls','J_calls','inner_steps','rejected_operator',
                          'trial_count','projection_calls','completed_projections','projection_seconds',
                          'projection_rounds','fallback_projections','max_projection_feasibility']
                result=dict(status='method_error',runtime=perf_counter()-method_started,
                    error=dict(exception_type=type(error).__name__,message=str(error)),
                    counts={key:observed.get(key) for key in counters},
                    counts_from_last_completed_output=True)
                if last is None:
                    info=dict(certificate=np.nan,violation=float(np.linalg.norm(np.maximum(p.G(p.x0),0.))))
                    last=(0,0.,p.x0.copy(),info)
            if result['status']=='certificate_stop' and (last is None or not last[3].get('certificate_stop',False)):
                result.update(status='time_limit',boundary_note='Certificate output completed after the deadline')
            if last is not None and (not samples or last[1]>samples[-1]['elapsed']):
                k,t,z,info=last;samples.append(dict(info,k=k,elapsed=t));points.append(z)
            if not points:
                raise RuntimeError(f'{method}: no initial or completed output available')
            pointfile=out/f'{method}_points.npz';atomic_points(pointfile,points)
            metadata={key:config[key] for key in ['n','m','seed','cap_seconds','instance_hash','source_hashes']}
            metadata['method']=method
            atomic_json(out/f'{method}_run.json',dict(result=result,checkpoints=samples,
                metadata=metadata,points_sha256=file_sha256(pointfile)))
            config['methods'][method]=dict(status=result['status'],runtime=result['runtime'],
                checkpoints=len(points),stage='timed_complete')
            config['rss_peak_bytes']=peak_rss_bytes();atomic_json(manifest,config)
            print(f'DONE n={n} m={m} seed={seed} method={method} status={result["status"]} runtime={result["runtime"]:.3f}',flush=True)
        config['state']='diagnostics';atomic_json(manifest,config)
        gap=FullSolver(p)
        for method in order:
            saved,zs=validate_method_pair(out,method,config,p.x0)
            diagfile=out/f'{method}_diagnostics.json'
            rows=json.loads(diagfile.read_text()) if diagfile.exists() else []
            validate_diagnostic_prefix(rows,saved,method,config)
            for i,(sample,z) in enumerate(zip(saved['checkpoints'],zs)):
                if i<len(rows) and (rows[i]['gap_valid'] or not retry_invalid_diagnostics):
                    continue
                clock=perf_counter();history=[];attempt=1
                if i<len(rows):
                    prior=rows[i];history=prior.get('attempt_history',[])+[
                        {key:prior.get(key) for key in ['gap_status','gap_valid','gap_seconds','diagnostic_seconds','diagnostic_error']}]
                    attempt=prior.get('diagnostic_attempt',1)+1
                atomic_json(out/'live.json',dict(method=method,state='diagnostics',checkpoint=i,
                    diagnostic_attempt=attempt,gap_seconds=gap_seconds,elapsed=sample['elapsed'],
                    k=sample['k'],certificate=sample.get('certificate'),violation=None,
                    rss_peak_bytes=peak_rss_bytes()))
                error_info=None
                try:
                    g,diag=gap.gap(z,seconds=gap_seconds)
                except Deadline:
                    g,diag=np.nan,dict(status='diagnostic_time_limit',valid=False)
                except Exception as error:
                    g,diag=np.nan,dict(status='diagnostic_exception',valid=False)
                    error_info=dict(exception_type=type(error).__name__,message=str(error))
                try:
                    v=float(np.linalg.norm(np.maximum(p.G(z),0.)))
                except Exception as error:
                    v=np.nan;diag['valid']=False
                    error_info=dict(exception_type=type(error).__name__,message=str(error),phase='violation_check')
                valid=bool(diag['valid'] and np.isfinite(g) and np.isfinite(v))
                row=dict(n=n,m=m,seed=seed,method=method,checkpoint=i,
                    elapsed=sample['elapsed'],k=sample['k'],gap=g,violation=v,
                    residual=max(g,v) if valid else np.nan,diagnostic_seconds=perf_counter()-clock,
                    gap_valid=valid,gap_status=str(diag['status']),gap_seconds=gap_seconds,
                    certificate=sample.get('certificate',np.nan),diagnostic_attempt=attempt,
                    attempt_history=history,diagnostic_error=error_info,solver_diagnostics=diag)
                if i<len(rows):rows[i]=row
                else:rows.append(row)
                atomic_json(diagfile,rows)
            config['methods'][method]['stage']='complete'
            config['methods'][method]['invalid_diagnostics']=sum(not row['gap_valid'] for row in rows)
            config['rss_peak_bytes']=peak_rss_bytes();atomic_json(manifest,config)
        all_rows=[row for method in chosen for row in json.loads((out/f'{method}_diagnostics.json').read_text())]
        config.update(state='complete',total_seconds=perf_counter()-started,rss_peak_bytes=peak_rss_bytes(),
            invalid_diagnostic_count=sum(not row['gap_valid'] for row in all_rows),
            diagnostic_status_counts=dict(Counter(row['gap_status'] for row in all_rows)))
        atomic_json(manifest,config)
        if p.factor_file is not None:
            factor_file=p.factor_file;del p.W;factor_file.unlink()
        print(f'COMPLETE n={n} m={m} seed={seed} invalid_diagnostics={config["invalid_diagnostic_count"]}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,required=True)
    ap.add_argument('--m',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--seconds',type=float,default=1000.)
    ap.add_argument('--out',required=True)
    ap.add_argument('--core',type=int)
    ap.add_argument('--methods',help='Comma-separated subset; default rotates all five')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--diagnostic-seconds','--gap-seconds',dest='diagnostic_seconds',type=float,default=300.)
    ap.add_argument('--retry-invalid-diagnostics',action='store_true')
    a=ap.parse_args()
    worker(a.n,a.m,a.seed,a.seconds,a.out,a.core,
           a.methods.split(',') if a.methods else None,a.resume,a.diagnostic_seconds,a.retry_invalid_diagnostics)
