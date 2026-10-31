"""Regenerate measured tables, figures, and an evidence audit from saved runs.

This module never reads historical tables and never extrapolates runtimes.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

from study import THRESHOLDS

LABELS={'proposed':'PFAOS','oe_true':r'OE ($\widehat M=M$)',
        'golden_ratio':'Golden Ratio'}


def write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def first_hit(rows,metric,threshold,cap):
    times=[r['elapsed'] for r in rows if r['gap_valid'] and
           np.isfinite(r[metric]) and r['elapsed']<=cap and r[metric]<=threshold]
    return min(times) if times else np.nan


def number(v):
    return '--' if not np.isfinite(v) else (f'{v:.3f}' if v<1 else f'{v:.2f}')


def table(path,groups,study,metric,seeds,cap):
    ismulti=study=='multi'
    symbol='m' if ismulti else 'n'
    metric_tex=r'\widehat E' if metric=='residual' else r'\widehat{\operatorname{Gap}}'
    caption=('First observed time (seconds) to a numerically evaluated combined-error threshold'
             if metric=='residual' else 'First observed time (seconds) to a numerically evaluated weak-gap threshold')
    caption+=f' and final constraint violation over {seeds} measured instances per configuration, with a {cap:g}-second cap.'
    label='multi_constraint_comparative' if ismulti else 'single_constraint_comparative'
    if metric=='gap':
        label+='_gap_only'
    lines=[r'\begin{table*}[!htbp]',r'\centering',r'\caption{'+caption+'}',
        r'\label{tab:'+label+'}',r'\setlength{\tabcolsep}{4pt}',
        r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{clccccc}',r'\hline',
        '$'+symbol+r'$ & Method & '+ ' & '.join('$'+metric_tex+r'\le10^{'+str(e)+'}$' for e in [-2,-3,-4,-5])
        +r' & Final $V$ \\',r'\hline']
    for key,records in groups.items():
        n,m=key
        for method in ['proposed','oe_true','golden_ratio']:
            selected=[r for r in records if r['method']==method]
            assert len(selected)==seeds
            cells=[]
            for delta in THRESHOLDS:
                values=np.array([r[f'{metric}_{delta:g}'] for r in selected])
                successful=values[np.isfinite(values)]
                stat='/'.join(number(v) for v in [min(successful),np.mean(successful),max(successful)]) if len(successful) else '--'
                cells.append(r'\shortstack{$'+f'{len(successful)}/{seeds}'+r'$\\'+stat+'}')
            vv=np.array([r['final_violation'] for r in selected])
            vs='/'.join(f'{v:.2e}' for v in [min(vv),np.mean(vv),max(vv)])
            name='MC-PFAOS' if ismulti and method=='proposed' else LABELS[method]
            lines.append(f'{m if ismulti else n} & {name} & '+' & '.join(cells)+f' & {vs} '+r'\\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}}',r'\smallskip',r'\parbox{\textwidth}{\footnotesize\textit{Note:} '
        +('Fix $n=50$.' if ismulti else 'Fix $m=1$.')
        +r' Each threshold cell shows the observed success count, followed by best/mean/worst times over successful runs only. '
        +r'An entry -- means no observed successful crossing before termination. '
        +r'Final $V$ gives best/mean/worst over all runs. Threshold times use saved checkpoints; an earlier crossing between checkpoints may be missed. Times are unnormalized seconds; instance coefficients are unscaled. '
        +(r'The joint target requires both the numerical weak gap and $V$ to meet the threshold at the same averaged iterate. ' if metric=='residual' else r'These gap-only times do not establish simultaneous feasibility or certified accuracy. ')
        +r'OE uses the true operator norm; no trajectory selection or tuning sweep is performed. '
        +r'Failure and diagnostic-status counts are reported in the accompanying audit.}',r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n')


def protocol(cap,seeds,execution_policy=None):
    execution_policy=execution_policy or {}
    workers=execution_policy.get('workers',7)
    affinity=execution_policy.get('affinity','unspecified')
    if affinity=='pinned':
        scheduling=f'Up to {workers} independent instances execute on distinct pinned cores.'
    elif affinity=='os_managed':
        scheduling=(f'Up to {workers} independent instances execute concurrently; '
                    'the operating system assigns cores and no CPU affinity is set.')
    else:
        scheduling=(f'Up to {workers} independent instances execute concurrently; '
                    'the saved plan does not specify CPU affinity.')
    return r'''\paragraph{Instances and implementation.}
This is a fresh measured study, separate from the earlier 900-second tables.
We use $n=10,m=1$ and $n=50,m=10$, with '''+str(seeds)+r''' seeds per setting and
unscaled uniform coefficients on the ball $X=\{x:\|x\|_2\le2\}$.
The monotone operator is $F(x)=Bx-d$, where $B=R-R^\top+H$ and
$H$ is diagonal, positive semidefinite, and has $\lfloor n/2\rfloor$
zero eigenvalues. Each quadratic constraint has the same number of
zero eigenvalues and an independent random orthogonal eigenbasis.
Offsets give the common starting point $x_0=-d/\|d\|_2$ a strict
constraint margin of $0.02$. There is no scaling of displayed times,
constraint violations, or coefficients by dimension or constraint count.
PFAOS and MC-PFAOS use $\Lambda=1$, $M_0=1$, $L_0=1/2$, and initial
dual vector zero. These choices use no global Lipschitz constants.
Algorithms 3 and 4 replace the first outer iteration, including the
joint upward search, downward halving, and complete rollback of a
rejected trial. Quadratic Taylor remainders are evaluated as
$\tfrac12\Delta^\top A_i\Delta$, with a relative acceptance tolerance
of $5\times10^{-12}$ and no additive tolerance floor.

\paragraph{Subproblems and benchmark parameters.}
PFAOS solves its quadratic subproblem by a geometric projection onto
a ball intersected with one affine halfspace. MC-PFAOS uses ball and
nonnegative-orthant projections. The benchmarks project onto the full
original feasible set. CVXPY/Clarabel solves restricted convex models
and all original constraints are then checked; violated constraints
are added until the restricted solution is feasible for the full set.
In exact arithmetic this is the full projection, since the restricted
optimum is a lower bound and a full-feasible optimizer supplies the
matching upper bound. Numerical optimality, rather than an exact
finite-precision certificate, is claimed in the implementation.
Each model is built afresh, with warm starts disabled. Previously
identified constraint indices are retained, but models and
factorizations are not reused. Clarabel feasibility, absolute-gap,
and relative-gap tolerances are $10^{-8}$; the full-set feasibility
check uses $10^{-7}$. The declared fallback after a nonoptimal conic
solve is a full-constraint SLSQP solve with objective tolerance
$10^{-10}$ and the same feasibility check. Its use is logged and timed.
OE receives the reference norm $M=\|B\|_2$ and holds it fixed.
No parameter sweep, best-trajectory selection, or fallback selection
rule is used in the revised tables.

\paragraph{Golden Ratio initialization and outputs.}
For Golden Ratio, set $\phi=(1+\sqrt5)/2$, $\rho=1/\phi+1/\phi^2$,
$z_1=x_0$, $z_0=x_0+10^{-3}e_1$, $u_0=z_1$, and $\vartheta_0=1$.
Initialize $\alpha_0=\|z_1-z_0\|_2/\|F(z_1)-F(z_0)\|_2$ and use
$\alpha_{\max}=10^6$. If that denominator vanishes, use $\alpha_0=1$.
The original auxiliary-parameter update is
\[
 \alpha_k=\min\left\{\rho\alpha_{k-1},
 \frac{\phi\vartheta_{k-1}}{4\alpha_{k-1}}
 \frac{\|z_k-z_{k-1}\|_2^2}{\|F(z_k)-F(z_{k-1})\|_2^2},\alpha_{\max}\right\},
 \quad \vartheta_k=\frac{\phi\alpha_k}{\alpha_{k-1}},
\]
with $0/0=+\infty$. Its first step is specified by $\vartheta_0=1$.
The auxiliary recursion is $u_k=((\phi-1)z_k+u_{k-1})/\phi$ and
$z_{k+1}=\Pi_{\widetilde X}(u_k-\alpha_kF(z_k))$.
The reported Golden Ratio output is
$\bar x_K^{\mathrm{GR}}=\sum_{j=1}^K\alpha_jz_j/\sum_{j=1}^K\alpha_j$,
not $u_K$. The reported OE output is the arithmetic mean of completed
projected iterates. PFAOS and MC-PFAOS report their retained inverse-$M$
weighted averages; rejected trials and final unaveraged primal points
are not substituted into the reported averages.

\paragraph{Gap evaluation and stopping.}
We distinguish the exact weak gap
$\operatorname{Gap}(z)=[\max_{y\in\widetilde X}\langle F(y),z-y\rangle]_+$,
its numerically evaluated value $\widehat{\operatorname{Gap}}(z)$, and
the algorithmic certificate $\mathcal C_k$ of Section 6.
For the affine operator, the numerical gap is evaluated through
$p^*(z)=\min_{y\in\widetilde X}\{y^\top H y-(B^\top z+d)^\top y\}$
and $\operatorname{Gap}(z)=[-p^*(z)-d^\top z]_+$, using the objective
returned by a separate numerical solve. These objective values and
solver statuses are not claimed to be certified upper bounds.
Define $V(z)=\|[G(z)]_+\|_2$ and
$\widehat E(z)=\max\{\widehat{\operatorname{Gap}}(z),V(z)\}$.
The common numerical target is $\widehat E\le10^{-5}$ at the same
saved averaged iterate. Actual early stopping for the proposed
methods uses the Section 6 conditions $\mathcal C_k\le10^{-5}$ and
$V(\bar x_k)\le10^{-5}$ together. Baselines continue to their time
budget. Numerical gap evaluation occurs after all timed methods on
an instance finish and is excluded from solver time. The computed
certificate is recorded separately and is not relabeled as the
independently evaluated numerical gap.

\paragraph{Runtime and aggregation.}
Each method has a '''+f'{cap:g}'+r'''-second budget. Runs use one numerical thread
and methods on each instance execute sequentially, in an order
rotated across seeds. '''+scheduling+r''' Hardware, software versions, source hashes,
per-run clocks, and numerical statuses are recorded in the manifests.
Instance generation and the reference norm calculation are shared
setup outside the method clocks. Algorithm initialization, accepted
and rejected searches, model construction, all projection solves,
fallback work, and full-set feasibility checks are included in the
method clocks. Time spent on an unfinished final projection is
included in runtime, but a result completed after the deadline is
not added as a new iterate.
Tables report first observed crossings at saved checkpoints, without
interpolating threshold times. Checkpoints are spaced geometrically
in elapsed time (approximately a factor of $1.4$), with the initial
and final available averages also saved. Earlier crossings between
checkpoints may be missed, particularly for nonmonotone trajectories.
The figure uses log--log interpolation solely for visualization and
clips displayed residuals below $10^{-12}$ to that plotting floor;
its interpolated crossings and display floor are not used in tables. Every threshold
block gives the success count out of '''+str(seeds)+r''' and best/mean/worst times over
successful runs only. Final-violation statistics use all runs.
Failed solves are identified separately from timeouts. Main tables
use the joint numerical criterion; companion gap-only tables do not
claim feasibility at the gap-hitting time. Final feasibility never
implies feasibility at earlier checkpoints. All times are raw seconds.
'''


def figures(out,all_rows,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    styles={'proposed':dict(color='black',ls='-',lw=1.8),
            'oe_true':dict(color='.35',ls='-.',lw=1.4),
            'golden_ratio':dict(color='.65',ls='--',lw=1.4)}
    rng=np.random.default_rng(20261029)
    ids=rng.integers(0,len(plan['seeds']),(4000,len(plan['seeds'])))
    for ax,setting,panel in zip(axes,plan['settings'],['(a)','(b)']):
        n,m=setting['n'],setting['m']
        for method in plan['methods']:
            rs=[]
            for seed in plan['seeds']:
                rr=sorted([r for r in all_rows if r['n']==n and r['m']==m and r['seed']==seed
                           and r['method']==method and r['gap_valid'] and np.isfinite(r['residual'])],
                          key=lambda r:r['elapsed'])
                if len(rr)<2:
                    break
                rs.append(rr)
            if len(rs)!=len(plan['seeds']):
                continue
            common_end=min(r[-1]['elapsed'] for r in rs)
            ts=np.geomspace(1e-4,max(1.01e-4,common_end),180)
            values=[]
            for rr in rs:
                t=np.maximum([r['elapsed'] for r in rr],1e-4)
                y=np.maximum([r['residual'] for r in rr],1e-12)
                distinct=np.r_[True,np.diff(t)>0]
                values.append(np.exp(np.interp(np.log(ts),np.log(t[distinct]),np.log(y[distinct]))))
            values=np.array(values)
            median=np.median(values,axis=0)
            boot=np.median(values[ids],axis=1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.fill_between(ts,lo,hi,color=styles[method]['color'],alpha=.10,lw=0)
            ax.plot(ts,median,**styles[method])
        ax.axhline(1e-5,color='.35',lw=.8,ls='--')
        ax.set(xscale='log',yscale='log',xlim=(1e-4,plan['seconds']),
               xlabel='Solver time (s)',title=f'{panel} '+('PFAOS' if m==1 else 'MC-PFAOS')+f': n={n}, m={m}')
        ax.grid(True,alpha=.25)
    axes[0].set_ylabel('Combined error')
    handles=[Line2D([],[],label=label,**styles[method]) for method,label in
             [('proposed','PFAOS / MC-PFAOS'),('oe_true',r'OE: $\widehat M=M$'),('golden_ratio','Golden Ratio')]]
    handles.append(Line2D([],[],color='.35',ls='--',lw=.8,label=r'Target accuracy $10^{-5}$'))
    fig.legend(handles=handles,loc='center right',frameon=False)
    fig.tight_layout(rect=(0,0,.79,1))
    fig.savefig(out/'small_convergence.pdf')
    fig.savefig(out/'small_convergence.png',dpi=180)
    plt.close(fig)
    (out/'small_figure.tex').write_text(r'''\begin{figure*}[ht]
\centering
\includegraphics[width=\textwidth]{small_convergence.pdf}
\caption{Combined error $\widehat E(\bar x)=\max\{\widehat{\operatorname{Gap}}(\bar x),\|[G(\bar x)]_+\|_2\}$ versus solver time for (a) PFAOS at $n=10,m=1$ and (b) MC-PFAOS at $n=50,m=10$, compared with true-$M$ OE and the Golden Ratio Algorithm. Central curves are medians over '''+str(len(plan['seeds']))+r''' measured random instances; shaded bands are pointwise 95\% bootstrap confidence intervals for the medians. The dashed line marks the target accuracy $10^{-5}$. Each trajectory is interpolated linearly in log--log coordinates before aggregation; curves end at the earliest terminal checkpoint among seeds for each method, without extrapolation or monotonic fitting. Residuals below $10^{-12}$ are shown at that display floor. Interpolated curve crossings and the display floor do not affect the tables, which use saved checkpoints and unmodified numerical values.}
\label{fig:pfaos_small_measured}
\end{figure*}
''')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True)
    a=ap.parse_args();root=Path(a.results)
    plan=json.loads((root/'plan.json').read_text())
    rows=[];runs=[];failures=[]
    diagnostic_statuses=Counter()
    expected_cases={(s['n'],s['m'],seed) for s in plan['settings'] for seed in plan['seeds']}
    seen_cases=set()
    expected=len(plan['settings'])*len(plan['seeds'])
    manifests=sorted(root.glob('*/manifest.json'))
    assert len(manifests)==expected,(len(manifests),expected)
    for mf in manifests:
        meta=json.loads(mf.read_text());case=mf.parent
        assert meta['state']=='complete',case
        case_key=(meta['n'],meta['m'],meta['seed'])
        assert case_key in expected_cases and case_key not in seen_cases,case
        seen_cases.add(case_key)
        assert meta['source_hashes']==plan['source_hashes'],case
        for method in plan['methods']:
            data=json.loads((case/f'{method}_run.json').read_text())
            diag=json.loads((case/f'{method}_diagnostics.json').read_text())
            points=np.load(case/f'{method}_points.npz')['x']
            assert len(diag)==len(data['checkpoints'])==len(points)>0
            assert points.ndim==2 and points.shape[1]==meta['n'] and np.all(np.isfinite(points))
            assert all(np.isfinite(r[field]) for r in diag for field in ['elapsed','gap','violation','residual'])
            assert all(r['gap']>=0 and r['violation']>=0 for r in diag)
            assert all(np.isclose(r['residual'],max(r['gap'],r['violation']),rtol=1e-12,atol=1e-15) for r in diag)
            assert all((r['n'],r['m'],r['seed'],r['method'],r['checkpoint'])==(*case_key,method,i)
                       for i,r in enumerate(diag))
            assert all(r['elapsed']==c['elapsed'] and r['k']==c['k'] for r,c in zip(diag,data['checkpoints']))
            assert all(b['elapsed']>a['elapsed'] for a,b in zip(diag,diag[1:]))
            diagnostic_statuses.update(r['gap_status'] for r in diag)
            assert all(r['elapsed']<=plan['seconds'] for r in diag)
            assert all(r['gap_valid'] for r in diag),'Invalid diagnostics: '+str(case)
            result=data['result']
            if result['status']=='time_limit':
                assert result['runtime']>=plan['seconds']-.01
            elif result['status']=='certificate_stop':
                assert method=='proposed'
                stop=data['checkpoints'][-1]
                assert stop['certificate']<=1e-5 and stop['violation']<=1e-5
            else:
                failures.append(dict(case=case.name,method=method,status=result['status']))
            if method=='proposed':
                for tr in result['trace']:
                    if tr.get('retained'):
                        assert tr['middle_pass'] and tr['balance_error']<1e-8
                        assert tr['middle_lhs']<=tr['middle_rhs']*(1+6e-12)
                down=[r for r in result['trace'] if r['phase']=='down' and not r['middle_pass']]
                handoff=[r for r in result['trace'] if r['k']==2]
                if handoff:
                    assert down and down[-1]['rollback_to_hash']==handoff[0]['start_state_hash']
            if method=='oe_true':
                assert all(np.isclose(r['M_estimate'],meta['true_M'])
                           for r in data['checkpoints'] if r['k']>0)
            r=dict(n=meta['n'],m=meta['m'],seed=meta['seed'],method=method,
                   final_gap=diag[-1]['gap'],final_violation=diag[-1]['violation'],
                   final_residual=diag[-1]['residual'],runtime=result['runtime'],
                   status=result['status'],terminal_checkpoint_seconds=diag[-1]['elapsed'],
                   diagnostic_seconds=sum(x['diagnostic_seconds'] for x in diag),
                   checkpoints=len(diag),projection_calls=result['counts']['projection_calls'],
                   completed_projections=result['counts'].get('completed_projections',np.nan),
                   fallback_projections=result['counts'].get('fallback_projections',0),
                   projection_seconds=result['counts'].get('projection_seconds',np.nan),
                   max_projection_feasibility=result['counts'].get('max_projection_feasibility',np.nan))
            for delta in THRESHOLDS:
                for metric in ['gap','residual']:
                    r[f'{metric}_{delta:g}']=first_hit(diag,metric,delta,plan['seconds'])
                tg,tj=r[f'gap_{delta:g}'],r[f'residual_{delta:g}']
                assert not np.isfinite(tj) or np.isfinite(tg) and tg<=tj
            rows+=diag;runs.append(r)
    assert seen_cases==expected_cases
    out=root/'report';out.mkdir(exist_ok=True)
    write_csv(out/'checkpoints.csv',rows);write_csv(out/'per_run_summary.csv',runs)
    for study in ['single','multi']:
        selected=[r for r in runs if (r['m']==1)==(study=='single')]
        groups={key:[r for r in selected if (r['n'],r['m'])==key]
                for key in sorted({(r['n'],r['m']) for r in selected})}
        name='pfaos_table' if study=='single' else 'mc_table'
        table(out/f'{name}.tex',groups,study,'residual',len(plan['seeds']),plan['seconds'])
        table(out/f'{name}_gap_only.tex',groups,study,'gap',len(plan['seeds']),plan['seconds'])
    (out/'evaluation_protocol.tex').write_text(protocol(plan['seconds'],len(plan['seeds']),plan.get('execution_policy')))
    figures(out,rows,plan)
    narrative=[]
    for setting in plan['settings']:
        n,m=setting['n'],setting['m']
        narrative.append(f'At $n={n},m={m}$, the observed joint $10^{{-5}}$ target-attainment counts were ')
        pieces=[]
        for method in plan['methods']:
            rr=[r for r in runs if r['n']==n and r['m']==m and r['method']==method]
            vals=np.array([r['residual_1e-05'] for r in rr]);ok=vals[np.isfinite(vals)]
            name=('MC-PFAOS' if m>1 else 'PFAOS') if method=='proposed' else LABELS[method]
            part=f'{len(ok)}/{len(rr)} for {name}'
            if len(ok):part+=f' (median {np.median(ok):.2f} seconds among successful runs)'
            pieces.append(part)
        narrative.append(', '.join(pieces)+'. These are numerical joint-target outcomes at averaged outputs.\n\n')
    narrative.append('The final-violation measurements describe terminal averaged outputs only. '
                     'They do not establish feasibility at earlier gap-only threshold crossings. '
                     'These small-instance results do not establish performance on larger instances.\n')
    (out/'results_discussion.tex').write_text(''.join(narrative))
    checks=json.loads((Path(__file__).parent/'checks.json').read_text())
    proof=[
      ('Figure/table provenance','Fresh data only; new small-instance figure uses exactly the table checkpoints.','report/checkpoints.csv; manifests; small_figure.tex'),
      ('1000-second cap','Validate runtime and pre-deadline checkpoints; distinguish certificate stops and solver failures.','Each *_run.json; budget assertions in report.py'),
      ('Exact, numerical, and certified gaps','Define the exact maximization, its numerical convex solve, and the distinct Section 6 certificate. No independent-certification claim.','evaluation_protocol.tex; study.FullSolver.gap'),
      ('Upper-bound issue','No primal objective or solver status is represented as a valid upper bound.','evaluation_protocol.tex; numeric gap and certificate saved separately'),
      ('Simultaneous feasibility','Main tables use max(numerical gap, violation) at the SAME averaged checkpoint; gap-only tables are explicitly separate.','per_run_summary.csv; report.first_hit'),
      ('Conditional statistics and counts',f'Every threshold shows s/{len(plan["seeds"])}; times use only successful observed crossings.','pfaos_table.tex; mc_table.tex; raw per-run CSV'),
      ('Seconds and normalization','Raw seconds; no division by n or m. This fresh generator uses unscaled coefficients.','study.DenseProblem; evaluation_protocol.tex'),
      ('No-success wording','No observed crossing is distinguished from early solver failure; failure details are retained.','table notes; failure list in validation.json'),
      ('OE protocols and selection','Both revised tables use fixed true-M OE. The historical selected-path sweep is removed, so no undefined fallback remains.','study.run_baseline; fixed-M assertions in report.py'),
      ('Tuning and parallel timing','No sweep; reference norm/setup excluded explicitly. Methods sequential within an instance, independent workers under the recorded affinity policy, one numerical thread.','batch.py; per-instance manifests and measured setup_seconds'),
      ('Golden Ratio first step','Original theta_0=1 recurrence is implemented and checked against the actual baseline output.','test_checks.py; study.run_baseline'),
      ('Negative-index correction','If the simplified rule is extended to k=1, alpha_-1=phi*alpha_0. Equality to alpha_0 is not equivalent to theta_0=1.','Golden_Ratio_first_step_theta0_equivalence check; Malitsky Algorithm 1'),
      ('Reported averages','Retained inverse-M averages for proposed methods; arithmetic OE averages; alpha-weighted z_j Golden Ratio average.','actual-baseline index checks; evaluation_protocol.tex'),
      ('10M conclusion','No failed-convergence claim for overestimation is retained; the revised measured tables use true M.','results_discussion.tex; revised benchmark scope'),
      ('Zero gap from combined-error plot','No zero-gap inference is made from the combined-error curve. Components are separately saved.','checkpoints.csv; results_discussion.tex'),
      ('Unworkable-in-general claim','Remove it. Conclusions describe observed small-instance outcomes only.','results_discussion.tex'),
      ('Final versus pathwise feasibility','Final measurements are explicitly limited to terminal outputs.','evaluation_protocol.tex; results_discussion.tex'),
      ('Initial points and parameters','Common x0, zero initial dual, Lambda=1, M0=1, L0=1/2; Golden Ratio secant initialization recorded.','manifests; *_run.json; evaluation_protocol.tex'),
      ('Special first iteration and Section 6','Joint doubling, halving, full rollback, k=2 handoff, and joint certificate/feasibility stopping implemented.','algorithms.py; independent rollback and certificate checks'),
      ('PFAOS quadratic subproblem','Geometric ball/halfspace solution and multiplier match independent convex solves.','PFAOS_QP_and_certificate_multiplier check'),
      ('Solver tolerance, reuse, and runtime','Record conic and SLSQP tolerances, full-set checks, fresh models, index reuse, and timed fallback/model construction. Offline gap time is separate.','evaluation_protocol.tex; study.FullSolver; per-run timing/work counters'),
      ('Larger-instance estimates','They are not measured results and are not generated by this code. No estimated success counts or feasibility outcomes.','README.md; separate author-requested estimation artifact, if supplied')]
    lines=['# Comment-by-comment evidence audit','',
           f'Measured study: {expected} instances, {len(runs)} method runs; {checks["count"]} independent implementation checks passed.',
           f'Solver failures: {len(failures)}. All saved diagnostic records validated.',
           'Diagnostic solver statuses: '+json.dumps(dict(diagnostic_statuses),sort_keys=True)+'.',
           'Fallback projections: '+str(sum(r['fallback_projections'] for r in runs))+'.','',
           '| Comment | Correction | Inspectable evidence |','|---|---|---|']
    lines += [f'| {a} | {b} | {c} |' for a,b,c in proof]
    lines += ['', 'The audit verifies implementation checks and the measured small-instance dataset. It does not claim new large-instance measurements or mathematical guarantees for runtime extrapolations.',
      '', 'Primary references:',
      '- Malitsky, Algorithm 1 and Section 2.2: https://optimization-online.org/wp-content/uploads/2018/05/6598.pdf',
      '- Kotsalis, Lan, and Li, Theorem 2.6: https://arxiv.org/pdf/2011.02987']
    (out/'comment_by_comment_audit.md').write_text('\n'.join(lines)+'\n')
    summary=dict(instances=expected,method_runs=len(runs),checkpoints=len(rows),
                 all_saved_diagnostics_valid=True,failures=failures,checks=checks,
                 execution_policy=plan.get('execution_policy',{}),
                 termination_counts=dict(Counter(r['status'] for r in runs)),
                 diagnostic_status_counts=dict(diagnostic_statuses),
                 fallback_projection_count=sum(r['fallback_projections'] for r in runs),
                 timing_policy='wall-clock seconds; checkpoint crossings only; no threshold interpolation',
                 source_hashes=plan['source_hashes'],status='complete')
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
