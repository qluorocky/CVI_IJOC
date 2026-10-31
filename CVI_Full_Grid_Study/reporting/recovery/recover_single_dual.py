#!/usr/bin/env python3
"""Authorized second offline retry: two positive QCQP multipliers, m=1 only.

Uses one dense A (<=8 MB for the supplied single grid), never an A_i stack.
Acceptance is based on independent numerical primal/dual/KKT checks. Ordinary
floating-point brackets are not formal interval certificates.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls, root
from threadpoolctl import threadpool_limits
from constraint_generation_gap import (ReadOnlyProblem, completed_case_snapshot,
    assert_snapshot_unchanged, digest_file, vector_digest, TimedOut)

ROOT=Path(__file__).resolve().parents[2]
MODULE=Path(__file__).with_name("constraint_generation_gap.py")


def write(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=True)+"\n");temp.replace(path)


def evaluate_dual(matrix,h,b,c,linear,radius,log_parameters):
    multipliers=np.exp(log_parameters)
    if not np.all(np.isfinite(multipliers)) or min(multipliers)<=0:
        raise ValueError("Nonpositive/nonfinite dual trial")
    lam,mu=multipliers
    hessian=lam*matrix
    hessian[np.diag_indices_from(hessian)]+=2*h+2*mu
    factor=cho_factor(hessian,lower=True,overwrite_a=True,check_finite=False)
    rhs=linear-lam*b
    y=cho_solve(factor,rhs,check_finite=False)
    ay=matrix@y
    constraint=.5*float(y@ay)+float(b@y)+c
    ball=float(y@y-radius**2)
    gradients=np.column_stack([ay+b,2*y])
    inverse_gradients=cho_solve(factor,gradients,check_finite=False)
    jacobian=-(gradients.T@inverse_gradients)*multipliers[None,:]
    return dict(multipliers=multipliers,y=y,residual=np.array([constraint,ball]),
                jacobian=jacobian,linear=linear,
                dual_closed_form=-.5*float(rhs@y)+lam*c-mu*radius**2)


def solve(p,z,initial,seconds=300.):
    if p.m!=1 or p.n>1000:
        raise ValueError("This independently reviewed retry is restricted to m=1,n<=1000")
    started=perf_counter();deadline=started+seconds
    last=None;out=None;timed_out=False;error=None
    def check():
        if perf_counter()>=deadline:raise TimedOut()
    def alarm(signum,frame):raise TimedOut()
    previous=signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        matrix=p.W[0].T@p.W[0]
        linear=p.B.T@z+p.d
        initial_gradients=np.column_stack([matrix@initial+p.b[0],2*initial])
        seed,_=nnls(initial_gradients,-(2*p.h*initial-linear),maxiter=100)
        seed=np.maximum(seed,1e-8)
        cached=None
        def evaluated(t):
            nonlocal cached,last
            check()
            if cached is None or not np.array_equal(cached[0],t):
                last=evaluate_dual(matrix,p.h,p.b[0],p.c[0],linear,p.r,t)
                cached=(t.copy(),last)
            check();return cached[1]
        out=root(lambda t:evaluated(t)["residual"],np.log(seed),
                 jac=lambda t:evaluated(t)["jacobian"],method="hybr",
                 options=dict(xtol=1e-11,maxfev=100))
        last=evaluated(out.x)
        y=last["y"].copy();lam,mu=last["multipliers"]
        # A tiny segment toward strict-feasible x0 supplies an actual feasible
        # primal candidate even if root residuals round just above zero.
        blend=0.;feasible_y=y.copy();full_g=p.G(feasible_y)
        ball=float(feasible_y@feasible_y-p.r**2)
        for fraction in (0.,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8):
            feasible_y=(1-fraction)*y+fraction*p.x0
            full_g=p.G(feasible_y);ball=float(feasible_y@feasible_y-p.r**2)
            blend=fraction
            if np.max(full_g)<=0 and ball<=0:break
        check()
        objective=float(p.h@(feasible_y*feasible_y)-linear@feasible_y)
        root_objective=float(p.h@(y*y)-linear@y)
        root_g=p.G(y)
        root_ball=float(y@y-p.r**2)
        # Convexity of L with nonnegative multipliers yields a lower estimate
        # on the full minimum over ||v||<=r, regardless of primal feasibility.
        stationarity=2*p.h*y-linear+lam*(p.W[0].T@(p.W[0]@y)+p.b[0])+2*mu*y
        stationarity_l2=float(np.linalg.norm(stationarity))
        lagrangian=root_objective+lam*float(root_g[0])+mu*root_ball
        lower=float(lagrangian-(p.r+np.linalg.norm(y))*stationarity_l2)
        width=objective-lower
        dz=float(p.d@z)
        raw_gap=-objective-dz;gap_lower=max(0.,raw_gap);gap_upper=max(0.,-lower-dz)
        violation=max(0.,float(np.max(full_g)),ball)
        complementarity=float(max(abs(lam*root_g[0]),abs(mu*root_ball)))
        residual_inf=float(max(abs(root_g[0]),abs(root_ball)))
        finite=bool(np.all(np.isfinite(feasible_y)) and np.all(np.isfinite(y))
                    and np.all(np.isfinite(full_g)) and np.all(np.isfinite(root_g))
                    and np.all(np.isfinite([objective,lower,width,raw_gap,gap_lower,gap_upper,
                                            stationarity_l2,complementarity,residual_inf,lam,mu])))
        checks=dict(finite=finite,positive_multipliers=bool(lam>0 and mu>0),
                    full_feasible=bool(violation<=1e-10),
                    root_constraint_residual=bool(residual_inf<=1e-10),
                    stationarity=bool(stationarity_l2<=1e-9),
                    complementarity=bool(complementarity<=1e-9),
                    primal_dual_width=bool(-1e-10<=width<=1e-8))
        valid=all(checks.values())
        result=dict(gap=gap_lower if valid else None,gap_valid=valid,
            status="numerical_primal_dual_bracket_valid" if valid else "dual_quality_checks_unresolved",
            raw_objective=objective,raw_gap=raw_gap,
            full_feasibility=dict(constraint_max=float(np.max(full_g)),ball_excess=ball,
                                  max_violation=violation,tolerance=1e-10),
            numerical_quality=dict(objective_lower=lower,objective_upper=objective,
                primal_dual_width=float(width),max_width=1e-8,gap_lower=gap_lower,gap_upper=gap_upper,
                stationarity_l2=stationarity_l2,stationarity_inf=float(np.max(np.abs(stationarity))),
                root_constraint_residual_inf=residual_inf,complementarity_inf=complementarity,
                multipliers=[float(lam),float(mu)],multipliers_nonnegative=bool(lam>=0 and mu>=0),
                dual_closed_form=float(last["dual_closed_form"]),primal_feasible_blend_fraction=blend,
                checks=checks,disclaimer="ordinary floating-point numerical bounds; not a formal interval certificate"),
            solver=dict(name="two_multiplier_dual_root",success=bool(out.success),status=int(out.status),
                        message=str(out.message),nfev=int(out.nfev),njev=int(getattr(out,"njev",0)),
                        seed_multipliers=seed.tolist(),parameterization="log positive multipliers",
                        dense_A_bytes=int(matrix.nbytes),dense_A_count=1,
                        acceptance="independent finite/primal/dual/KKT checks, preserving optimizer status"))
        return result,feasible_y,y
    except Exception as exc:
        timed_out=isinstance(exc,TimedOut);error=f"{type(exc).__name__}: {exc}"
        result=dict(gap=None,gap_valid=False,status="dual_time_limit" if timed_out else "dual_exception",error=error,
                    solver=dict(name="two_multiplier_dual_root",success=bool(out.success) if out is not None else False,
                                status=int(out.status) if out is not None else None))
        return result,(last["y"] if last is not None else initial.copy()),(last["y"] if last is not None else initial.copy())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        # Returned dictionary is mutable; ensure all successes/failures retain time.
        if "result" in locals():result.update(seconds=perf_counter()-started,budget_seconds=seconds)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",type=Path,default=ROOT/"notes/single_delivery_inventory.json")
    ap.add_argument("--results",type=Path,default=ROOT/"results_single")
    ap.add_argument("--prior",type=Path,default=Path(__file__).parent/"single_delivery_v1")
    ap.add_argument("--out",type=Path,default=Path(__file__).parent/"single_delivery_v2")
    a=ap.parse_args();inventory=json.loads(a.inventory.read_text());rows=inventory["invalid_diagnostics"]
    if len(rows)!=8 or len({(r["case"],r["method"],r["checkpoint"]) for r in rows})!=8:
        raise ValueError("Authorization covers exactly eight unique second retries")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"solver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (a.out/"recovery_module_snapshot.py").write_bytes(MODULE.read_bytes())
    summary=dict(state="running",expected=8,diagnostic_budget_seconds=300,source_sha256=digest_file(__file__),
                 recovery_module_sha256=digest_file(MODULE),inventory_sha256=digest_file(a.inventory),
                 prior_summary_sha256=digest_file(a.prior/"summary.json"),timed_methods_run=0,
                 source_diagnostics_modified=False,automatic_retry=False,results=[])
    write(a.out/"summary.json",summary)
    with threadpool_limits(limits=1):
        for row in rows:
            case,method,checkpoint=row["case"],row["method"],row["checkpoint"]
            case_path=a.results/"verified"/case
            admission=completed_case_snapshot(case_path,require_quiescent=True,stable_single_snapshot=True)
            folder=a.out/f"{case}__{method}__cp{checkpoint}";folder.mkdir()
            prior_folder=a.prior/f"{case}__{method}"
            prior_file=prior_folder/"benchmark.json";prior=json.loads(prior_file.read_text())
            prior_result=next(r for r in prior["results"] if r["checkpoint"]==checkpoint)
            with np.load(prior_folder/f"checkpoint_{checkpoint}_solution.npz",allow_pickle=False) as saved:
                initial=saved["x"]
            pointfile=case_path/f"{method}_points.npz"
            with np.load(pointfile,allow_pickle=False) as saved:z=saved["x"][checkpoint]
            if vector_digest(z)!=row["point_sha256"] or vector_digest(z)!=prior_result["input_point_hash"]:
                raise ValueError("Input/inventory/prior attempt point identity mismatch")
            setup=perf_counter()
            with ReadOnlyProblem(case_path,reconstruct_factors=True) as p:
                setup_seconds=perf_counter()-setup
                result,feasible_y,root_y=solve(p,z,initial)
                result.update(case=case,n=p.n,m=p.m,seed=p.seed,method=method,checkpoint=checkpoint,
                    input_point_hash=vector_digest(z),instance_hash=p.instance_hash,
                    source_hashes=p.manifest["source_hashes"],points_file_sha256=digest_file(pointfile),
                    original_diagnostics_sha256=digest_file(case_path/f"{method}_diagnostics.json"),
                    recovery_source_sha256=digest_file(__file__),recovery_module_sha256=digest_file(MODULE),
                    setup_seconds=setup_seconds,completion_guard=admission,
                    original_diagnostic=json.loads((case_path/f"{method}_diagnostics.json").read_text())[checkpoint],
                    prior_attempt=dict(diagnostic=prior_result,benchmark_file=str(prior_file.resolve()),
                                       benchmark_sha256=digest_file(prior_file)),
                    feasible_candidate_hash=vector_digest(feasible_y),root_candidate_hash=vector_digest(root_y),
                    threads=1,adoption_authorized=False)
                try:assert_snapshot_unchanged(admission,require_quiescent=True,stable_single_snapshot=True)
                except Exception as error:
                    result.update(gap_valid=False,gap=None,status="input_snapshot_changed",admission_error=repr(error))
                with (folder/"candidates.npz").open("xb") as stream:
                    np.savez_compressed(stream,feasible=feasible_y,root=root_y)
                write(folder/"diagnostic.json",result)
                summary["results"].append(dict(case=case,method=method,checkpoint=checkpoint,
                    gap_valid=result["gap_valid"],status=result["status"],gap=result["gap"],
                    diagnostic_file=str((folder/"diagnostic.json").resolve()),
                    diagnostic_sha256=digest_file(folder/"diagnostic.json"),
                    primal_dual_width=result.get("numerical_quality",{}).get("primal_dual_width")))
                write(a.out/"summary.json",summary)
                print(json.dumps(summary["results"][-1]),flush=True)
    summary.update(state="attempts_finished",completed=len(summary["results"]),
                   valid=sum(r["gap_valid"] for r in summary["results"]))
    write(a.out/"summary.json",summary)


if __name__=="__main__":main()
