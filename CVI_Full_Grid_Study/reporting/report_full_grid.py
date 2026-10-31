#!/usr/bin/env python3
"""Report the corrected full CVI grid from complete saved measurements only.

No historical inputs or runtime estimates are read. Invalid numerical diagnostics
remain explicit unresolved observations; they are never converted to failure-to-hit
or filled with the time cap. This module is outside the frozen timed source tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from priority_execution_context import load_priority_execution,summarize_priority_execution,priority_execution_text

THRESHOLDS=(1e-2,1e-3,1e-4,1e-5)
METHODS=('proposed','oe_true','oe_low','oe_high','golden_ratio')
TABLE_METHODS=('proposed','oe_true','golden_ratio')
SINGLE_N=(10,50,100,200,500,1000)
MULTI_M=(10,50,100,200,500)
MULTI_N=100
FULL_SETTINGS={*(('single',n,1) for n in SINGLE_N),*(('multi',MULTI_N,m) for m in MULTI_M)}
LABELS={'proposed':'PFAOS','oe_true':r'OE ($\widehat M=M$)',
        'oe_low':r'OE ($\widehat M=0.1M$)','oe_high':r'OE ($\widehat M=10M$)',
        'golden_ratio':'Golden Ratio'}
NORMAL_TERMINATIONS={'time_limit','certificate_stop'}
ADOPTED_GAP_STATUS='independent_dual_bracket_valid'


class IncompleteDataset(RuntimeError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def study_name(n,m):
    return 'single' if m==1 else 'multi'


def setting_identity(setting):
    if not isinstance(setting,dict):raise IncompleteDataset('Each setting must be an object')
    n,m=setting.get('n'),setting.get('m')
    if type(n) is not int or type(m) is not int or n<2 or m<1:
        raise IncompleteDataset('Setting dimensions must be positive integer dimensions with n >= 2')
    study=setting.get('study',study_name(n,m))
    if study!=study_name(n,m):raise IncompleteDataset('Setting study label disagrees with its constraint count')
    return study,n,m


def check_plan(plan,validation_subset=False,single_only=False):
    if not isinstance(plan,dict) or any(key not in plan for key in ('settings','methods','seeds','seconds','source_hashes')):
        raise IncompleteDataset('Plan requires settings, methods, seeds, seconds, and source_hashes')
    if not isinstance(plan['settings'],list) or not plan['settings']:raise IncompleteDataset('Empty or invalid settings list')
    identities=[setting_identity(s) for s in plan['settings']]
    if len(identities)!=len(set(identities)):raise IncompleteDataset('Duplicate settings in plan')
    if tuple(plan['methods']) not in ((METHODS,TABLE_METHODS) if validation_subset else (METHODS,)):
        raise IncompleteDataset('The full comparison must contain proposed, true-M/0.1M/10M OE, and Golden Ratio in that order')
    if not isinstance(plan['seeds'],list) or not plan['seeds'] or any(type(seed) is not int or seed<0 for seed in plan['seeds']) or len(plan['seeds'])!=len(set(plan['seeds'])):raise IncompleteDataset('Empty, invalid, or duplicate seed list')
    if type(plan['seconds']) not in (int,float) or not finite(plan['seconds']) or plan['seconds']<=0:raise IncompleteDataset('Invalid method cap')
    hashes=plan['source_hashes']
    if not isinstance(hashes,dict) or not hashes or any(not isinstance(name,str) or Path(name).name!=name or not name.endswith('.py') or not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for name,value in hashes.items()):raise IncompleteDataset('Invalid frozen source hash map')
    if validation_subset:
        if not plan.get('validation_fixture',False):raise IncompleteDataset('Subset reporting requires a copied, explicitly marked validation_fixture plan')
        if single_only and any(study!='single' for study,n,m in identities):raise IncompleteDataset('A single-only layout fixture may contain only single-constraint settings')
    else:
        expected={('single',n,1) for n in SINGLE_N} if single_only else FULL_SETTINGS
        if set(identities)!=expected:raise IncompleteDataset('Plan setting set differs from the explicit single-only release contract' if single_only else 'Plan must contain exactly the six single-constraint rows and five multi-constraint rows at n=100, ending at m=500')
        if single_only and plan.get('release_scope')!='single_only':raise IncompleteDataset('Single-only release requires release_scope=single_only in plan.json')
        if not single_only and plan.get('release_scope')=='single_only':raise IncompleteDataset('Single-only plan requires the explicit --single-only option')
        if list(plan['seeds'])!=list(range(10)):raise IncompleteDataset('Full-grid release requires seeds 0 through 9')
        if plan['seconds']!=1000:raise IncompleteDataset('Full-grid release requires the 1,000-second cap')
    return identities


def usable(row,metric):
    valid_gap=bool(row['gap_valid']) and finite(row.get('gap')) and row['gap']>=0
    if metric=='gap':return valid_gap
    return valid_gap and finite(row.get('violation')) and row['violation']>=0


def check_lineage(case,meta,planpath,file_hashes):
    """Bind carried-forward timed pairs without freezing later diagnostics."""
    path=case/'lineage.json'
    if not path.exists():return
    lineage=json.loads(path.read_text())
    plan=json.loads(planpath.read_text());origin=plan.get('inherited_single_provenance',{})
    if meta['m']!=1 or Path(lineage.get('source_case','')).name!=case.name or not lineage.get('source_results') or not lineage.get('source_plan_sha256'):
        raise IncompleteDataset(f'{case.name}: lineage does not describe same-study single-constraint inheritance')
    if any(origin.get(key) is not None and lineage.get(key)!=origin[key] for key in ('source_results','source_plan_sha256')):raise IncompleteDataset(f'{case.name}: lineage origin disagrees with amended-plan provenance')
    if lineage.get('new_plan_sha256')!=digest(planpath):raise IncompleteDataset(f'{case.name}: lineage amended-plan hash mismatch')
    copied=lineage.get('copied_artifact_sha256')
    if not isinstance(copied,dict) or not copied:raise IncompleteDataset(f'{case.name}: missing inherited artifact hashes')
    for name,value in copied.items():
        if not isinstance(name,str) or Path(name).name!=name or not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise IncompleteDataset(f'{case.name}: invalid inherited artifact identity')
        if name.endswith(('_run.json','_points.npz')) and (not (case/name).is_file() or digest(case/name)!=value):raise IncompleteDataset(f'{case.name}: inherited timed artifact changed: {name}')
    file_hashes[str(path)]=digest(path)


def outcome(rows,metric,threshold,cap):
    hits=[r['elapsed'] for r in rows if usable(r,metric) and r['elapsed']<=cap and
          (r['gap'] if metric=='gap' else max(r['gap'],r['violation']))<=threshold]
    hit=min(hits) if hits else math.nan
    invalid=[r for r in rows if not usable(r,metric)]
    state='observed_success' if hits else ('unresolved_diagnostics' if invalid else 'no_observed_crossing')
    prior=bool(hits and any(r['elapsed']<hit for r in invalid))
    return hit,state,prior


def load_complete(root,validation_subset=False,single_only=False):
    planpath=root/'plan.json'
    if not planpath.is_file():raise IncompleteDataset('Missing plan.json')
    plan=json.loads(planpath.read_text());identities=check_plan(plan,validation_subset,single_only)
    expected={(n,m,seed) for _,n,m in identities for seed in plan['seeds']}
    found={}
    for mf in sorted(root.glob('*/manifest.json')):
        meta=json.loads(mf.read_text());key=(meta['n'],meta['m'],meta['seed'])
        if key in found:raise IncompleteDataset(f'Duplicate case {key}')
        found[key]=(mf,meta)
    if set(found)!=expected:raise IncompleteDataset(f'Case set mismatch: missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}')
    loaded=[];file_hashes={str(planpath):digest(planpath)}
    for key in sorted(expected):
        mf,meta=found[key];case=mf.parent
        if meta['state']!='complete':raise IncompleteDataset(f'{case.name}: manifest is {meta["state"]}, not complete')
        if meta['source_hashes']!=plan['source_hashes']:raise IncompleteDataset(f'{case.name}: source hashes differ from plan')
        if meta['cap_seconds']!=plan['seconds']:raise IncompleteDataset(f'{case.name}: budget differs from plan')
        check_lineage(case,meta,planpath,file_hashes)
        try:priority_execution=load_priority_execution(case,meta,planpath,file_hashes)
        except ValueError as error:raise IncompleteDataset(f'{case.name}: {error}') from error
        file_hashes[str(mf)]=digest(mf)
        for method in plan['methods']:
            stage=meta.get('methods',{}).get(method,{}).get('stage')
            if stage!='complete':raise IncompleteDataset(f'{case.name}/{method}: stage is {stage}, not complete')
            paths=[case/f'{method}_{suffix}' for suffix in ('run.json','points.npz','diagnostics.json')]
            if not all(p.is_file() for p in paths):raise IncompleteDataset(f'{case.name}/{method}: missing run/points/diagnostic file')
            saved=json.loads(paths[0].read_text());diag=json.loads(paths[2].read_text());samples=saved['checkpoints']
            metadata=saved.get('metadata')
            if not metadata and not validation_subset:raise IncompleteDataset(f'{case.name}/{method}: missing saved-pair metadata')
            if metadata:
                for field in ('n','m','seed','cap_seconds','instance_hash','source_hashes'):
                    if metadata.get(field)!=meta[field]:raise IncompleteDataset(f'{case.name}/{method}: saved-pair {field} mismatch')
                if metadata.get('method')!=method:raise IncompleteDataset(f'{case.name}/{method}: saved-pair method mismatch')
            if saved.get('points_sha256'):
                if saved['points_sha256']!=digest(paths[1]):raise IncompleteDataset(f'{case.name}/{method}: saved-point checksum mismatch')
            elif not validation_subset:raise IncompleteDataset(f'{case.name}/{method}: missing point checksum')
            with np.load(paths[1],allow_pickle=False) as pointsfile:points=pointsfile['x']
            if points.shape!=(len(samples),key[0]) or len(diag)!=len(samples) or not samples:raise IncompleteDataset(f'{case.name}/{method}: record lengths or point dimensions disagree')
            if not np.all(np.isfinite(points)):raise IncompleteDataset(f'{case.name}/{method}: nonfinite saved averaged point')
            for i,(sample,row) in enumerate(zip(samples,diag)):
                identity=(row['n'],row['m'],row['seed'],row['method'],row['checkpoint'])
                if identity!=(*key,method,i):raise IncompleteDataset(f'{case.name}/{method}/{i}: diagnostic identity mismatch')
                if row['elapsed']!=sample['elapsed'] or row['k']!=sample['k']:raise IncompleteDataset(f'{case.name}/{method}/{i}: checkpoint identity mismatch')
                if not finite(row['elapsed']) or not 0<=row['elapsed']<=plan['seconds']:raise IncompleteDataset(f'{case.name}/{method}/{i}: timestamp outside cap')
                if i and row['elapsed']<=diag[i-1]['elapsed']:raise IncompleteDataset(f'{case.name}/{method}: timestamps not strictly ordered')
                if row['gap_valid']:
                    if not usable(row,'gap'):raise IncompleteDataset(f'{case.name}/{method}/{i}: gap marked valid but nonfinite/negative')
                    if not finite(row.get('violation')) or row['violation']<0:raise IncompleteDataset(f'{case.name}/{method}/{i}: valid diagnostic has unavailable/negative violation')
                    if usable(row,'residual'):
                        expected_residual=max(row['gap'],row['violation'])
                        if not finite(row.get('residual')) or not math.isclose(row['residual'],expected_residual,rel_tol=1e-12,abs_tol=1e-15):raise IncompleteDataset(f'{case.name}/{method}/{i}: joint residual inconsistent')
                elif not row.get('gap_status'):raise IncompleteDataset(f'{case.name}/{method}/{i}: invalid gap lacks status reason')
            result=saved['result'];status=result['status']
            if not finite(result['runtime']) or result['runtime']<0:raise IncompleteDataset(f'{case.name}/{method}: invalid runtime')
            if status=='time_limit' and result['runtime']<plan['seconds']-.01:raise IncompleteDataset(f'{case.name}/{method}: timeout clock below cap')
            if status=='certificate_stop':
                last=samples[-1]
                if method!='proposed' or not finite(last.get('certificate')) or last['certificate']>meta['epsilon'] or not finite(last.get('violation')) or last['violation']>meta['epsilon']:raise IncompleteDataset(f'{case.name}/{method}: invalid certificate stop')
            if method=='proposed':
                for trace in result.get('trace',[]):
                    if trace.get('retained') and (not trace.get('middle_pass') or trace.get('balance_error',math.inf)>=1e-8):raise IncompleteDataset(f'{case.name}/{method}: retained-trial invariant failed')
            if method in ('oe_true','oe_low','oe_high'):
                factor={'oe_true':1.,'oe_low':.1,'oe_high':10.}[method]
                if any(not finite(s.get('M_estimate')) or not math.isclose(s['M_estimate'],factor*meta['true_M'],rel_tol=1e-10,abs_tol=1e-12) for s in samples if s['k']>0):raise IncompleteDataset(f'{case.name}: fixed OE estimate changed or differs from its declared norm multiple')
            for path in paths:file_hashes[str(path)]=digest(path)
            loaded.append(dict(case=case.name,meta=meta,method=method,result=result,checkpoints=diag,samples=samples,priority_execution=priority_execution))
    adoption_path=root/'diagnostic_adoption_manifest.json'
    adopted=[row for item in loaded for row in item['checkpoints'] if row.get('gap_status')==ADOPTED_GAP_STATUS]
    if adopted or adoption_path.exists():
        if not adoption_path.is_file():raise IncompleteDataset('Adopted independent gap rows require the explicit adoption manifest')
        adoption=json.loads(adoption_path.read_text())
        if adoption.get('state')!='applied' or adoption.get('rows')!=len(adopted) or adoption.get('quality_reviewed_by_explicit_cli') is not True or adoption.get('timed_pairs_unchanged') is not True:raise IncompleteDataset('Invalid independent diagnostic adoption manifest')
        for path,value in adoption.get('input_sha256_after',{}).items():
            if not Path(path).is_file() or digest(path)!=value:raise IncompleteDataset(f'Adoption input identity changed: {path}')
            file_hashes[path]=value
        snapshot=Path(adoption['helper_snapshot_file'])
        if digest(snapshot)!=adoption['helper_snapshot_sha256'] or adoption['helper_snapshot_sha256']!=adoption['helper_sha256']:raise IncompleteDataset('Adoption helper snapshot hash mismatch')
        file_hashes[str(snapshot)]=digest(snapshot);file_hashes[str(adoption_path)]=digest(adoption_path)
        for row in adopted:
            evidence=row.get('diagnostic_adoption',{})
            if evidence.get('kind')!='independent_single_constraint_dual_v2' or evidence.get('quality_passed') is not True or evidence.get('quality_reviewed_by_explicit_cli') is not True:raise IncompleteDataset('Adopted numerical gap lacks explicit posterior-quality/review provenance')
    return plan,loaded,file_hashes


def summarize(plan,loaded):
    rows=[];runs=[];diagnostic_statuses=Counter();terminations=Counter();failures=[]
    for item in loaded:
        meta=item['meta'];method=item['method'];rr=item['result'];diag=item['checkpoints'];last=diag[-1]
        diagnostic_statuses.update(r.get('gap_status','unspecified') for r in diag);terminations[rr['status']]+=1
        if rr['status'] not in NORMAL_TERMINATIONS:failures.append(dict(case=item['case'],method=method,status=rr['status']))
        counts=rr.get('counts') or {}
        def count(key,default=math.nan):
            value=counts.get(key,default)
            return value if finite(value) else math.nan
        record=dict(n=meta['n'],m=meta['m'],seed=meta['seed'],method=method,
            final_gap=last['gap'] if usable(last,'gap') else math.nan,
            final_violation=last.get('violation',math.nan),
            final_residual=max(last['gap'],last['violation']) if usable(last,'residual') else math.nan,
            final_gap_valid=usable(last,'gap'),final_joint_valid=usable(last,'residual'),
            runtime=rr['runtime'],status=rr['status'],terminal_checkpoint_seconds=last['elapsed'],
            diagnostic_seconds=sum(r.get('diagnostic_seconds',0.) for r in diag),checkpoints=len(diag),
            valid_gap_checkpoints=sum(usable(r,'gap') for r in diag),invalid_gap_checkpoints=sum(not usable(r,'gap') for r in diag),
            valid_joint_checkpoints=sum(usable(r,'residual') for r in diag),invalid_joint_checkpoints=sum(not usable(r,'residual') for r in diag),
            projection_calls=count('projection_calls'),completed_projections=count('completed_projections'),
            fallback_projections=count('fallback_projections',0 if method=='proposed' else math.nan),projection_seconds=count('projection_seconds'),
            memory_guard_fallback_projections=count('memory_guard_fallback_projections',0 if method=='proposed' else math.nan),
            projection_rounds=count('projection_rounds'),F_calls=count('F_calls'),G_calls=count('G_calls'),J_calls=count('J_calls'),
            inner_steps=count('inner_steps'),trial_count=count('trial_count'),
            max_projection_feasibility=count('max_projection_feasibility'),
            counts_from_last_completed_output=bool(rr.get('counts_from_last_completed_output',False)))
        for threshold in THRESHOLDS:
            for metric in ('gap','residual'):
                hit,state,prior=outcome(diag,metric,threshold,plan['seconds'])
                record[f'{metric}_{threshold:g}']=hit;record[f'{metric}_{threshold:g}_outcome']=state
                record[f'{metric}_{threshold:g}_invalid_before_hit']=prior
            tg,tj=record[f'gap_{threshold:g}'],record[f'residual_{threshold:g}']
            if finite(tj) and (not finite(tg) or tg>tj):raise ValueError('Joint crossing lacks earlier/equal gap crossing')
        for row in diag:
            rows.append(dict(row,algorithm_termination=rr['status'],gap_usable=usable(row,'gap'),joint_usable=usable(row,'residual')))
        runs.append(record)
    return rows,runs,diagnostic_statuses,terminations,failures


def time_number(value):
    return f'{value:.3f}' if value<1 else f'{value:.2f}'


def table(path,plan,runs,study,metric):
    settings=sorted((s for s in plan['settings'] if setting_identity(s)[0]==study),key=lambda s:s['m'] if study=='multi' else s['n'])
    multi=study=='multi';N=len(plan['seeds']);label='multi_constraint_comparative' if multi else 'single_constraint_comparative'
    if metric=='gap':label+='_gap_only'
    symbol=r'\widehat E' if metric=='residual' else r'\widehat{\operatorname{Gap}}'
    target='joint numerical-error' if metric=='residual' else 'numerical weak-gap'
    fixed=sorted({s['n'] if multi else s['m'] for s in settings})
    fixed_text=('n='+str(fixed[0]) if multi else 'm='+str(fixed[0])) if len(fixed)==1 else 'fixed dimensions as specified'
    caption=f'First observed valid {target} threshold times (seconds) and terminal constraint violation ({fixed_text}) over {N} attempted runs per configuration, with a {plan["seconds"]:g}-second method cap.'
    if plan.get('validation_fixture'):caption='VALIDATION FIXTURE ONLY. '+caption
    lines=[r'\begin{table*}[!htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',
        r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{clccccc}',r'\hline',
        ('$m$' if multi else '$n$')+r' & Method & '+' & '.join('$'+symbol+r'\le10^{'+str(e)+'}$' for e in (-2,-3,-4,-5))+r' & Final $V$ \\',r'\hline']
    for setting in settings:
        n,m=setting['n'],setting['m']
        for method in TABLE_METHODS:
            chosen=[r for r in runs if (r['n'],r['m'],r['method'])==(n,m,method)]
            if len(chosen)!=N:raise ValueError('Table group does not contain every planned seed')
            cells=[]
            for threshold in THRESHOLDS:
                vals=[r[f'{metric}_{threshold:g}'] for r in chosen if finite(r[f'{metric}_{threshold:g}'])]
                unknown=sum(r[f'{metric}_{threshold:g}_outcome']=='unresolved_diagnostics' for r in chosen)
                count=f'{len(vals)}/{N}'+(r',\ u='+str(unknown) if unknown else '')
                stats='/'.join(time_number(x) for x in (min(vals),statistics.fmean(vals),max(vals))) if vals else '--'
                cells.append(r'\shortstack{$'+count+r'$\\'+stats+'}')
            v=[r['final_violation'] for r in chosen]
            vf='/'.join(f'{x:.2e}' for x in (min(v),statistics.fmean(v),max(v))) if all(finite(x) and x>=0 for x in v) else r'\textit{unresolved}'
            name='MC-PFAOS' if multi and method=='proposed' else LABELS[method]
            lines.append(f'{m if multi else n} & {name} & '+' & '.join(cells)+' & '+vf+r' \\')
        lines.append(r'\hline')
    fixed=sorted({s['n'] for s in settings}) if multi else sorted({s['m'] for s in settings})
    fixed_note=('Fix $n='+str(fixed[0])+'$.' if multi else 'Fix $m='+str(fixed[0])+'$.') if len(fixed)==1 else 'The fixed dimensions are stated in the plan.'
    lines += [r'\end{tabular}}',r'\smallskip',r'\parbox{\textwidth}{\footnotesize\textit{Note:} '+fixed_note+
        r' Each threshold cell shows observed valid successes $s/N$, then best/mean/worst times over those successful runs only. '
        r'Where present, $u$ counts additional runs with no observed crossing and unresolved diagnostics; such runs are not classified as nonattainment. '
        r'-- means no observed valid crossing before the recorded termination; algorithm failures are separately enumerated in the audit. '
        r'Final $V$ gives best/mean/worst over all terminal reported outputs, or is marked unresolved if any terminal violation is unavailable. '
        r'Times are raw seconds, coefficients are unscaled, and threshold times are saved-checkpoint observations without interpolation. '+
        (r'The joint criterion requires numerical gap and violation at the same averaged point. ' if metric=='residual' else r'Gap-only times do not establish simultaneous feasibility or rigorous accuracy. ')+
        r'The OE table row uses its fixed true-$M$ trajectory. The $0.1M$ and $10M$ trajectories are separate sensitivity runs shown in the figures; no best-trajectory selection is performed.}',r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n')


def protocol(plan,manifest_policy=None):
    singles=sorted(s['n'] for s in plan['settings'] if setting_identity(s)[0]=='single')
    multiples=sorted(s['m'] for s in plan['settings'] if setting_identity(s)[0]=='multi')
    N=len(plan['seeds']);policy=plan.get('execution_policy',{})
    five=all(method in plan['methods'] for method in ('oe_low','oe_high'))
    method_text=r'fixed OE with $\widehat M\in\{M,0.1M,10M\}$' if five else r'fixed true-$M$ OE'
    figure_text=('The figures show all five trajectories. Main and companion tables retain the three primary method rows: proposed, true-$M$ OE, and Golden Ratio. No OE trajectory is selected as ``best.\'\' ' if five else 'The validation figures show the three trajectories in this fixture. ')
    multi_settings=[s for s in plan['settings'] if setting_identity(s)[0]=='multi']
    multi_text=('The multiple-constraint study fixes $n='+str(multi_settings[0]['n'])+r'$ and uses $m\in\{'+','.join(map(str,multiples))+r'\}$.') if multi_settings else 'This release contains only the single-constraint experiments. Multi-constraint measurements and comparisons are deferred.'
    proposed_text='PFAOS/MC-PFAOS' if multi_settings else 'PFAOS'
    text=r'''\paragraph{Study grid and measurements.}
The single-constraint study fixes $m=1$ and uses $n\in\{'''+','.join(map(str,singles))+r'''\}$.
'''+multi_text+r'''
Each configuration uses '''+str(N)+r''' paired random seeds. '''+proposed_text+r''', '''+method_text+r''', and adaptive Golden Ratio each receive a '''+f'{plan["seconds"]:g}'+r'''-second wall-clock budget. '''+figure_text+r'''The coefficients are dense and unscaled on a radius-two ball, with common strictly feasible $x_0=-d/\|d\|_2$, constraint margin $0.02$, and proposed-method parameters $\Lambda=1$, $M_0=1$, and $L_0=1/2$. All reported values are measurements from this corrected study, including unchanged single-constraint cases carried forward with their original source and point-file provenance; no historical table value or runtime estimate is substituted.

\paragraph{Outputs, accuracy, and timing.}
The reported outputs are retained inverse-$M$ averages for proposed methods, arithmetic averages of completed projected iterates for OE, and $\sum_j\alpha_jz_j/\sum_j\alpha_j$ for Golden Ratio, with its original $\vartheta_0=1$ initialization. Define the exact weak gap $\operatorname{Gap}(z)=[\max_{y\in\widetilde X}\langle F(y),z-y\rangle]_+$, its numerical objective estimate $\widehat{\operatorname{Gap}}(z)$, and $V(z)=\|[G(z)]_+\|_2$. Main tables use $\widehat E(z)=\max\{\widehat{\operatorname{Gap}}(z),V(z)\}$ at the same saved average; companion tables use gap alone. Numerical gap objectives and solver labels are not rigorous upper bounds. The separately computed algorithmic certificate is not relabeled as the numerical gap. Proposed methods stop by their joint computed certificate/feasibility test; baselines continue to the cap unless an algorithmic/solver failure is recorded.

Shared instance setup and the reference norm are outside method clocks. Method initialization, rejected and accepted searches, generic projection model construction/solves/full feasibility checks/fallback work, and unfinished terminal work are timed. Offline gap diagnostics have separate clocks. One numerical thread is used per worker; methods on an instance execute sequentially in a seed-rotated order. The resource-aware admission policy, concurrency, and affinity are recorded in the plan and manifests, so reported times describe that workload.

\paragraph{Aggregation and unresolved numerical records.}
Threshold times are first observed valid saved-checkpoint crossings, not interpolated estimates of an earlier crossing. Each cell reports $s/N$ observed successes and conditional best/mean/worst times. A count $u$ denotes additional runs with unresolved diagnostics and no observed crossing; these are not silently treated as nonattainment or assigned the cap as a hitting time. An observed success can still have an earlier unresolved diagnostic; that limitation is preserved per run. Final-violation summaries describe terminal outputs only and do not imply pathwise feasibility. Algorithmic failures, diagnostic statuses, coverage, and fallbacks are reported separately. Aggregate figures require full valid diagnostic coverage for every seed of a method, retain a fixed seed set, and end at the earliest terminal checkpoint; otherwise the method is identified as unavailable. Log--log interpolation and a $10^{-12}$ plotting floor affect visualization only.
'''
    manifest_policy=manifest_policy or {}
    caps=manifest_policy.get('max_conic_coefficients',[]);budgets=manifest_policy.get('observed_gap_seconds',[])
    if caps or budgets:
        text+='\n'+r'\paragraph{Recorded numerical-resource limits.}'+'\n'
        if caps:text+=r'The recorded conic dense-coefficient limit is '+', '.join(f'{v:,}' for v in caps)+r'; larger restricted models use the declared full-set SLSQP backend. '
        if budgets:text+='Offline gap call budgets recorded in the saved diagnostics are '+', '.join(f'{v:g}' for v in budgets)+' seconds; retries and statuses are retained per checkpoint. '
        text+='\n'
    text+='\n'+priority_execution_text(manifest_policy.get('priority_execution_overrides',{}))
    adopted=manifest_policy.get('selected_independent_gap_evaluations',0)
    if adopted:
        text+='\n'+r'\paragraph{Selected independent numerical gap refinements.}'+'\n'+str(adopted)+' initially invalid single-constraint gap diagnostics were re-evaluated by an independent two-multiplier dual formulation. Original primary diagnostics and failed constraint-generation attempts are preserved in full attempt histories. Adopted values passed full original feasibility, stationarity, and numerical primal--dual-width checks at $10^{-8}$ and received separate explicit quality review. '+str(manifest_policy.get('selected_dual_optimizer_non_success',0))+' root-solver calls retained a non-success termination flag despite passing these posterior checks; their raw flags are preserved rather than relabeled. These selected ordinary-floating-point evaluations are not formal interval certificates and do not imply that all reported gaps were independently re-solved. The adoption manifest binds point, candidate, instance, source, and backup identities.\n'
    if plan.get('validation_fixture'):text=r'\textbf{Validation fixture only; not paper results.}\par'+'\n'+text
    return text


def figures(out,plan,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10.7})
    styles={'proposed':dict(color='black',ls='-',lw=1.5),
            'oe_true':dict(color='#1b5280',ls='--',lw=1.1),
            'oe_low':dict(color='#aa531d',ls='-.',lw=1.1),
            'oe_high':dict(color='#75508b',ls=':',lw=1.3),
            'golden_ratio':dict(color='#34723e',ls=(0,(5,2,1,2)),lw=1.1)}
    ids=np.random.default_rng(20261029).integers(0,len(plan['seeds']),(4000,len(plan['seeds'])))
    provenance=[]
    for study,filename in (('single','pfaos_full_grid'),('multi','mc_full_grid')):
        settings=sorted((s for s in plan['settings'] if setting_identity(s)[0]==study),key=lambda s:s['n'] if study=='single' else s['m'])
        if not settings:continue
        columns=min(2,len(settings));nrows=math.ceil(len(settings)/columns)
        fig,axes=plt.subplots(nrows,columns,figsize=(8.6,2.65*nrows+1.4),squeeze=False,sharey=True)
        for index,(ax,setting) in enumerate(zip(axes.flat,settings)):
            n,m=setting['n'],setting['m'];unavailable=[]
            for method in plan['methods']:
                trajectories=[];reason=None
                for seed in plan['seeds']:
                    rr=sorted((r for r in rows if (r['n'],r['m'],r['seed'],r['method'])==(n,m,seed,method)),key=lambda r:r['elapsed'])
                    if len(rr)<2:reason='fewer than two available checkpoints';break
                    if not all(usable(r,'residual') for r in rr):reason='incomplete valid diagnostic coverage';break
                    trajectories.append(rr)
                evidence=dict(study=study,n=n,m=m,method=method,seeds=len(plan['seeds']),plotted=reason is None,reason=reason)
                if reason:
                    unavailable.append(('MC-PFAOS' if study=='multi' and method=='proposed' else LABELS[method])+': unavailable');provenance.append(evidence);continue
                endpoint=min(rr[-1]['elapsed'] for rr in trajectories);ts=np.geomspace(1e-4,max(1.01e-4,endpoint),180);values=[]
                for rr in trajectories:
                    tx=np.maximum([r['elapsed'] for r in rr],1e-4);yy=np.maximum([max(r['gap'],r['violation']) for r in rr],1e-12);distinct=np.r_[True,np.diff(tx)>0]
                    values.append(np.exp(np.interp(np.log(ts),np.log(tx[distinct]),np.log(yy[distinct]))))
                values=np.array(values);median=np.median(values,axis=0);boot=np.median(values[ids],axis=1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
                ax.fill_between(ts,lo,hi,color=styles[method]['color'],alpha=.10,lw=0);ax.plot(ts,median,**styles[method]);evidence['common_endpoint_seconds']=endpoint;provenance.append(evidence)
            ax.axhline(1e-5,color='.3',ls=':',lw=.8);ax.set(xscale='log',yscale='log',xlim=(1e-4,plan['seconds']),xlabel='Solver time (s)',title=f'({chr(97+index)}) n={n}, m={m}')
            ax.set_xticks([1e-3,1e-1,1e1,1e3]);ax.grid(True,alpha=.2)
            if unavailable:ax.text(.02,.03,'\n'.join(unavailable),transform=ax.transAxes,fontsize=8,ha='left',va='bottom')
        for ax in list(axes.flat)[len(settings):]:ax.set_visible(False)
        for row in axes:row[0].set_ylabel('Combined error')
        handles=[Line2D([],[],label=('MC-PFAOS' if study=='multi' else 'PFAOS') if k=='proposed' else LABELS[k],**styles[k]) for k in plan['methods']]
        handles.append(Line2D([],[],color='.3',ls=':',lw=.8,label=r'Target $10^{-5}$'))
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005));fig.tight_layout(rect=(0,.08,1,1))
        fig.savefig(out/f'{filename}.pdf');fig.savefig(out/f'{filename}.png',dpi=180);plt.close(fig)
        fixed_dims=sorted({s['m'] if study=='single' else s['n'] for s in settings})
        dims=(('$m=' if study=='single' else '$n=')+str(fixed_dims[0])+'$; ' if len(fixed_dims)==1 else '')+', '.join((r'$n='+str(s['n'])+'$') if study=='single' else (r'$m='+str(s['m'])+'$') for s in settings)
        caption=r'Combined numerical error over the '+('single-constraint' if study=='single' else 'multiple-constraint')+r' grid ('+dims+r'). Curves are medians over '+str(len(plan['seeds']))+r' seeds, with pointwise 95\% bootstrap confidence intervals for each median. Each method uses a fixed complete seed set and ends at the earliest observed terminal checkpoint. Aggregation requires at least two saved points and complete valid gap coverage for every seed; unavailable methods are explicitly identified. Log--log interpolation and a $10^{-12}$ display floor are for visualization only; tables use unmodified values and saved checkpoint times. The dotted line is $10^{-5}$. Later per-seed attainments may lie beyond the common aggregate horizon.'
        if plan.get('validation_fixture'):caption='VALIDATION FIXTURE ONLY. '+caption
        (out/f'{filename}_figure.tex').write_text(r'\begin{figure*}[!htbp]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=\textwidth]{'+filename+r'.pdf}'+'\n'+r'\caption{'+caption+'}\n'+r'\label{fig:'+filename+'}\n'+r'\end{figure*}'+'\n')
    write_json(out/'full_grid_figure_provenance.json',dict(source='checkpoints.csv',source_sha256=digest(out/'checkpoints.csv'),settings=provenance,bootstrap_replicates=4000,bootstrap_seed=20261029,plot_floor=1e-12))


def discussion(plan,runs):
    paragraphs=[]
    for study in ('single','multi'):
        multi=study=='multi';settings=sorted((s for s in plan['settings'] if setting_identity(s)[0]==study),key=lambda s:s['m'] if multi else s['n']);parts=[]
        if not settings:continue
        for method in TABLE_METHODS:
            results=[]
            for s in settings:
                selected=[r for r in runs if (r['n'],r['m'],r['method'])==(s['n'],s['m'],method)];success=sum(finite(r['residual_1e-05']) for r in selected);unknown=sum(r['residual_1e-05_outcome']=='unresolved_diagnostics' for r in selected)
                label=('$m='+str(s['m'])+'$') if multi else ('$n='+str(s['n'])+'$');results.append(label+f': {success}/{len(selected)}'+(f' ({unknown} unresolved)' if unknown else ''))
            name='MC-PFAOS' if multi and method=='proposed' else LABELS[method];parts.append(name+' ['+'; '.join(results)+']')
        fixed=sorted({s['n'] if multi else s['m'] for s in settings})
        setting_note=(' at $n='+str(fixed[0])+'$' if multi else ' at $m='+str(fixed[0])+'$') if len(fixed)==1 else ''
        paragraphs.append(('For the multiple-constraint grid' if multi else 'For the single-constraint grid')+setting_note+', the observed valid joint $10^{-5}$ target counts were '+', '.join(parts)+'. Timing entries in Table~\\ref{tab:'+('multi_constraint_comparative' if multi else 'single_constraint_comparative')+'} are conditioned on observed successful runs.\n')
    other_terminations=sum(r['status'] not in NORMAL_TERMINATIONS for r in runs)
    unresolved=sum(r['invalid_gap_checkpoints']>0 for r in runs)
    if all(method in plan['methods'] for method in ('oe_low','oe_high')):
        sensitivity=[]
        for method in ('oe_low','oe_high'):
            chosen=[r for r in runs if r['method']==method]
            success=sum(finite(r['residual_1e-05']) for r in chosen);unknown=sum(r['residual_1e-05_outcome']=='unresolved_diagnostics' for r in chosen)
            sensitivity.append(LABELS[method]+f': {success}/{len(chosen)}'+(f' ({unknown} unresolved)' if unknown else ''))
        paragraphs.append('The fixed-step OE sensitivity trajectories shown in the figures had observed joint $10^{-5}$ counts of '+', '.join(sensitivity)+'. These variants are separate measured trajectories, not candidates in a selected best-OE row. Slower progress with an overestimate is not interpreted as failed convergence.\n')
    paragraphs.append(f'Across all measured trajectories, {other_terminations} method runs had a recorded termination other than the prescribed cap or computed-certificate stop, and {unresolved} method runs contained unresolved gap diagnostics. '+'These are numerical gap-and-feasibility outcomes at the same averaged checkpoints. Unresolved diagnostics and algorithmic failures are distinguished from recorded nonattainment; they cannot establish that a method fails to converge. Terminal violation values describe final reported outputs, not feasibility throughout a run or at an earlier gap-only crossing. The comparisons concern the stated dense instances, initialization, resource policy, generic projection implementation, and time cap; they do not show that projection methods are unworkable for other feasible sets or solver implementations.\n')
    if plan.get('validation_fixture'):paragraphs.insert(0,r'\textbf{Validation fixture only; not experimental conclusions.}'+'\n')
    return '\n'.join(paragraphs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',required=True,type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--validation-subset-plan',action='store_true');parser.add_argument('--single-only',action='store_true',help='Require the explicit 60-instance/300-run single-constraint release');parser.add_argument('--skip-figures',action='store_true');parser.add_argument('--strict-diagnostics',action='store_true');args=parser.parse_args()
    root=args.results.resolve();plan,loaded,hashes=load_complete(root,args.validation_subset_plan,args.single_only);rows,runs,statuses,terminations,failures=summarize(plan,loaded)
    out=(args.out or root/'report').resolve();out.mkdir(parents=True,exist_ok=True)
    if args.single_only and any(out.glob('mc_*')):raise IncompleteDataset('Single-only output directory contains stale MC artifacts; choose a separate output directory')
    write_csv(out/'checkpoints.csv',rows);write_csv(out/'per_run_summary.csv',runs)
    for study,filename in (('single','pfaos_table'),('multi','mc_table')):
        if args.single_only and study=='multi':continue
        for metric,suffix in (('residual',''),('gap','_gap_only')):table(out/f'{filename}{suffix}.tex',plan,runs,study,metric)
    manifest_policy=dict(max_conic_coefficients=sorted({item['meta']['max_conic_coefficients'] for item in loaded if item['meta'].get('max_conic_coefficients') is not None}),
        configured_gap_seconds=sorted({item['meta']['gap_seconds'] for item in loaded if item['meta'].get('gap_seconds') is not None}),
        observed_gap_seconds=sorted({row['gap_seconds'] for row in rows if row.get('gap_seconds') is not None}))
    adopted=[row for row in rows if row.get('gap_status')==ADOPTED_GAP_STATUS]
    manifest_policy.update(selected_independent_gap_evaluations=len(adopted),selected_dual_optimizer_non_success=sum(row.get('solver_diagnostics',{}).get('raw_optimizer',{}).get('success') is False for row in adopted))
    contexts={item['case']:item['priority_execution'] for item in loaded if item.get('priority_execution')}
    manifest_policy['priority_execution_overrides']=summarize_priority_execution(contexts.values())
    (out/'evaluation_protocol.tex').write_text(protocol(plan,manifest_policy));(out/'results_discussion.tex').write_text(discussion(plan,runs))
    if not args.skip_figures:figures(out,plan,rows)
    unresolved_gap=sum(r['invalid_gap_checkpoints']>0 for r in runs);unresolved_joint=sum(r['invalid_joint_checkpoints']>0 for r in runs)
    all_valid=unresolved_gap==unresolved_joint==0
    status='validation_fixture' if plan.get('validation_fixture') else ('complete' if all_valid and not failures else 'complete_with_exceptions')
    summary=dict(status=status,release_scope='validation_fixture' if args.validation_subset_plan else 'single_only' if args.single_only else 'full_grid',plan_snapshot=plan,completed_at_utc=datetime.now(timezone.utc).isoformat(),instances=len(plan['settings'])*len(plan['seeds']),method_runs=len(runs),checkpoints=len(rows),all_saved_diagnostics_valid=all_valid,diagnostic_status_counts=dict(statuses),termination_counts=dict(terminations),failures=failures,runs_with_invalid_gap_diagnostics=unresolved_gap,runs_with_invalid_joint_diagnostics=unresolved_joint,fallback_projection_count=sum(r['fallback_projections'] for r in runs if finite(r['fallback_projections'])),runs_with_unknown_fallback_count=sum(not finite(r['fallback_projections']) for r in runs),memory_guard_fallback_projection_count=sum(r['memory_guard_fallback_projections'] for r in runs if finite(r['memory_guard_fallback_projections'])),runs_with_unknown_memory_guard_count=sum(not finite(r['memory_guard_fallback_projections']) for r in runs),execution_policy=plan.get('execution_policy',{}),projection_policy=plan.get('projection_policy',{}),diagnostic_policy=plan.get('diagnostic_policy',{}),recorded_manifest_policy=manifest_policy,table_methods=list(TABLE_METHODS),figure_methods=plan['methods'],source_hashes=plan['source_hashes'],reporter_sha256=digest(__file__),input_sha256=hashes,threshold_policy='first observed valid checkpoint; exact recorded threshold comparisons; no time interpolation or cap imputation',unresolved_policy='invalid no-hit diagnostics remain unresolved, not nonattainment; valid observed hits remain reported with invalid-prior flags',large_instance_estimates=False)
    write_json(out/'validation.json',summary)
    audit=['# Full-grid measured-data reporting audit','',f'**Status: {status}.**',f'{summary["instances"]} instances; {len(runs)} method runs; {len(rows)} saved checkpoints.','',f'Algorithm/solver failures: {len(failures)}. Runs with invalid gap diagnostics: {unresolved_gap}.',f'Diagnostic statuses: {dict(statuses)}.',f'Terminations: {dict(terminations)}.',f'Projection fallbacks: {summary["fallback_projection_count"]}.','', 'Every planned setting, seed, method, saved point, and diagnostic record was checked for identity and completeness before these tables were generated. Missing data are refused. Invalid numerical gaps remain explicit unresolved observations; no runtime estimate or old table entry is used.','', 'The first valid observed threshold times, success counts and conditional time summaries should also be independently audited. This reporting audit is not an independent numerical gap solve and is not a formal finite-precision accuracy certificate.']
    (out/'reporting_audit.md').write_text('\n'.join(audit)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','input_sha256')},indent=2))
    return 2 if args.strict_diagnostics and not all_valid else 0


if __name__=='__main__':
    raise SystemExit(main())
