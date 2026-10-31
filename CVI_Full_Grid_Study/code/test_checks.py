"""Independent numerical and protocol checks; run: python test_checks.py."""
import json
import tempfile
from pathlib import Path
import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from algorithms import scalar_projection,relative_test,run_proposed
from study import DenseProblem,FullSolver
import study


def main():
    checks=[]
    def passed(name):
        checks.append(name)
        print('PASS',name,flush=True)
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as td:
        p=DenseProblem(10,1,0,td)
        assert min(p.G(p.x0)) < 0
        assert np.linalg.norm(p.x0)<p.r
        assert min(np.linalg.eigvalsh((p.B+p.B.T)/2))>-1e-12
        passed('dense_generator_monotonicity_strict_feasibility_and_no_scaling')
        rng=np.random.default_rng(982)
        for _ in range(8):
            y=rng.normal(size=p.n)
            delta=rng.normal(size=p.n)*1e-2
            explicit=p.G(y+delta)-p.G(y)-p.J(y)@delta
            np.testing.assert_allclose(p.remainder(delta),explicit,atol=1e-13)
        passed('stable_quadratic_remainder_matches_exact_identity')
        assert not relative_test(1e-24,1e-27)
        assert relative_test(0.,0.)
        passed('tiny_step_curvature_test_has_no_additive_floor')
        for _ in range(20):
            center=rng.normal(size=4)*3
            a=rng.normal(size=4)
            beta=float(rng.uniform(-.5,.5))
            coefficient=float(rng.uniform(1.,5.))
            x,lam=scalar_projection(center,2.,a,beta,coefficient)
            y=cp.Variable(4)
            aff=a@y<=beta
            prob=cp.Problem(cp.Minimize(coefficient/2*cp.sum_squares(y-center)),
                            [cp.sum_squares(y)<=4,aff])
            prob.solve(solver=cp.CLARABEL,tol_feas=1e-10,tol_gap_abs=1e-10,tol_gap_rel=1e-10)
            np.testing.assert_allclose(x,y.value,atol=2e-5)
            np.testing.assert_allclose(lam[0],aff.dual_value,atol=1e-4,rtol=1e-4)
        passed('PFAOS_QP_and_certificate_multiplier_match_independent_conic_solves')
        for n,m,seed in [(10,1,0),(12,4,0),(10,3,2)]:
            p=DenseProblem(n,m,seed,td)
            retained=[]
            def obs(k,t,state,info):
                if k:
                    retained.append((k,t,state.average.copy(),dict(info)))
            result=run_proposed(p,5.,obs,max_outer=20)
            assert len(retained)==20
            trace=result['trace']
            ups=[r for r in trace if r['phase']=='up']
            assert all(b['eta']==2*a['eta'] for a,b in zip(ups,ups[1:]))
            downs=[r for r in trace if r['phase']=='down']
            assert downs and all(b['eta']==a['eta']/2 for a,b in zip(downs,downs[1:]))
            rejected=[r for r in downs if not r['middle_pass']]
            assert rejected
            handoff=next(r for r in trace if r['k']==2)
            assert handoff['start_state_hash']==rejected[-1]['rollback_to_hash']
            assert result['max_balance_error']<1e-8
            evaluator=FullSolver(p)
            for k,t,z,info in retained[::4]+retained[-1:]:
                assert info['middle_lhs']<=info['middle_rhs']*(1+6e-12)
                assert info['operator_ratio']<=1+6e-12
                assert info['coupling_ratio']<=1+6e-12
                gap,diag=evaluator.gap(z,10.)
                assert diag['valid']
                assert gap<=info['certificate']+2e-6,(n,m,k,gap,info['certificate'])
                bound=(2 if m>1 else 1.5)*info['Gamma']*(p.r+np.linalg.norm(p.x0))**2
                assert info['certificate']<=bound+1e-6
            passed(f'paper_search_rollback_balance_certificate_and_curvature_n{n}_m{m}')
        p=DenseProblem(12,30,1,td)
        solver=FullSolver(p)
        center=rng.normal(size=12)*2
        x,info=solver.project(center,p.x0,float('inf'))
        y=cp.Variable(12)
        constraints=[cp.sum_squares(y)<=p.r**2]
        constraints += [.5*cp.sum_squares(p.W[i]@y)+p.b[i]@y+p.c[i]<=0 for i in range(p.m)]
        model=cp.Problem(cp.Minimize(.5*cp.sum_squares(y-center)),constraints)
        model.solve(solver=cp.CLARABEL,tol_feas=1e-9,tol_gap_abs=1e-9,tol_gap_rel=1e-9)
        assert max(p.G(x))<=solver.feasibility_tolerance
        np.testing.assert_allclose(.5*np.sum((x-center)**2),model.value,atol=1e-5,rtol=1e-6)
        passed('constraint_generation_matches_full_projection_and_checks_every_constraint')
        phi=(1+np.sqrt(5))/2
        alpha0=.7
        assert np.isclose(phi/(4*alpha0),phi**2/(4*(phi*alpha0)))
        assert not np.isclose(phi/(4*alpha0),phi**2/(4*alpha0))
        passed('Golden_Ratio_first_step_theta0_equivalence')
        z=np.array([[1.,0.],[0.,1.],[2.,3.]])
        a=np.array([.3,.7,.2])
        average=np.zeros(2);total=0.
        for ai,zi in zip(a,z):
            average=(total*average+ai*zi)/(total+ai);total+=ai
        np.testing.assert_allclose(average,np.sum(a[:,None]*z,axis=0)/sum(a))
        passed('Golden_Ratio_stepsize_weighted_ergodic_average_definition')
        p=DenseProblem(4,1,7,td)
        original=study.FullSolver
        class KnownProjection:
            def __init__(self,p):
                self.i=0
            def project(self,center,x0,deadline):
                self.i+=1
                return np.full(4,.03*self.i),dict(status='test',feasibility=0.,rounds=1,active_constraints=1)
        study.FullSolver=KnownProjection
        try:
            seen=[]
            result=study.run_baseline(p,'golden_ratio',10.,
                lambda k,t,x,d:seen.append((k,x.copy(),dict(d))),max_steps=3)
            weights=np.array([s[2]['alpha'] for s in seen[1:]])
            zs=np.stack([p.x0,np.full(4,.03),np.full(4,.06)])
            np.testing.assert_allclose(seen[-1][1],np.sum(weights[:,None]*zs,axis=0)/sum(weights))
            np.testing.assert_allclose(weights[0],phi*result['alpha0']/4,rtol=1e-10)
            seen=[]
            result=study.run_baseline(p,'oe_low',10.,
                lambda k,t,x,d:seen.append((k,x.copy(),dict(d))),max_steps=3)
            np.testing.assert_allclose(seen[-1][1],np.full(4,.06))
            assert all(np.isclose(s[2]['M_estimate'],.1*p.M) for s in seen[1:])
        finally:
            study.FullSolver=original
        passed('actual_baseline_code_uses_original_GR_first_step_and_correct_output_indices')
        passed('actual_fixed_OE_keeps_its_estimate_and_averages_completed_projections')
    out=dict(checks=checks,count=len(checks),all_passed=True)
    Path('checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':
    main()
