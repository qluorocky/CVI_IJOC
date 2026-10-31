#!/usr/bin/env python3
"""Replot two audited settings in the user's original two-panel grayscale format.

Only completed saved averaged-point measurements are read. A missing setting or
audit is refused before output creation. Historical PDF curve values are never
read; only its page geometry, typography, strokes and legend format were used.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from priority_execution_context import load_priority_execution,summarize_priority_execution,priority_execution_text

ROOT=Path(__file__).resolve().parents[1]
METHODS=('proposed','golden_ratio','oe_true','oe_low','oe_high')
PLAN_METHODS=('proposed','oe_true','oe_low','oe_high','golden_ratio')
REFERENCE_SHA256='ab8c6b098aa9ff516f7a58d28521b97523793c1654ddf36de61fcf01034d1f50'
STYLE=dict(figure_inches=[11.,4.8],page_points=[792.,345.6],font_family='STIXGeneral',mathtext_fontset='stix',font_size=11.,title_size=11.5,legend_size=10.3,left=.075,right=.77,bottom=.19,top=.89,wspace=.16,legend_anchor=[.785,.52],legend_labelspacing=.9,legend_handlelength=3.,line_width=1.65,marker_size=3.2,marker_edge_width=.7,legend_marker_edge_width=1.,band_alpha=.10,legend_band_alpha=.15,xlim=[1e-4,1e3],ylim=[1e-6,1e2],major_grid_color='#dddddd',major_grid_width=.5,target=1e-5,target_width=.85,target_dashes=[4,3],display_floor=1e-12,base_grid_points=401)
STYLES={
    'proposed':dict(color='#111111',linestyle='-'),
    'golden_ratio':dict(color='#666666',linestyle='--',marker='s'),
    'oe_true':dict(color='#3c3c3c',linestyle='-.',marker='o'),
    'oe_low':dict(color='#888888',linestyle=':',marker='^'),
    'oe_high':dict(color='#aaaaaa',linestyle=(0,(7,2)),marker='D')}
LABELS={'proposed':'PFAOS / MC-PFAOS','golden_ratio':'Golden ratio (adaptive)','oe_true':r'Fixed OE: $\widehat M=M$','oe_low':r'Fixed OE: $\widehat M=0.1M$','oe_high':r'Fixed OE: $\widehat M=10M$'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def point_sha(point):return hashlib.sha256(np.ascontiguousarray(point,dtype=np.float64).tobytes()).hexdigest()
def usable(row):return row.get('gap_valid') is True and finite(row.get('gap')) and row['gap']>=0 and finite(row.get('violation')) and row['violation']>=0
def checked_file(path,hashes):hashes[str(path.resolve())]=sha(path)


def audit_binds(path,audit_hashes):
    """Allow a bit-identical selected snapshot audited at a different root."""
    path=Path(path).resolve();actual=sha(path)
    if audit_hashes.get(str(path))==actual:return
    suffix='/'+path.parent.name+'/'+path.name
    matches=[value for source,value in audit_hashes.items() if source.endswith(suffix)]
    require(matches and all(value==actual for value in matches),f'Structural audit does not bind the selected file: {path}')


def load_setting(results,n,m,audit_path,code):
    results=Path(results).resolve();audit_path=Path(audit_path).resolve();code=Path(code).resolve()
    require(type(n) is int and type(m) is int and n>=2 and m>=1,'Setting dimensions must be explicit integers')
    plan_path=results/'plan.json';require(plan_path.is_file(),f'Missing dataset plan: {plan_path}')
    plan=read(plan_path)
    require(plan.get('validation_fixture') is not True,'Production plot cannot use a validation fixture')
    require(plan.get('seeds')==list(range(10)) and tuple(plan.get('methods',[]))==PLAN_METHODS and plan.get('seconds')==1000,'Plot requires seeds 0..9, all five ordered methods and 1000-second caps')
    settings=[s for s in plan.get('settings',[]) if (s.get('n'),s.get('m'))==(n,m)]
    require(len(settings)==1,f'Requested setting (n,m)=({n},{m}) is absent or duplicated; no fallback is permitted')
    source_hashes={path.name:sha(path) for path in sorted(code.glob('*.py'))}
    require(source_hashes and source_hashes==plan.get('source_hashes'),'Frozen worker source identity differs from the dataset plan')
    require(audit_path.is_file(),f'Missing completed structural audit: {audit_path}')
    audit=read(audit_path)
    require(audit.get('status')=='structural_pass' and audit.get('coverage_complete') is True and audit.get('preflight_only') is False,'A completed factorwise structural pass is required; preflight/incomplete audits do not qualify')
    require(audit.get('auditor_sha256')==sha(Path(__file__).with_name('audit_saved_records.py')),'Structural auditor source version differs; repeat the selected audit')
    require(audit.get('source_identity',{}).get('recorded_hashes')==source_hashes and audit.get('source_identity',{}).get('exact_root_Python_file_set_match') is True,'Audit does not establish the recorded frozen worker identity')
    audited_items=[item for item in audit.get('instances',[]) if (item.get('n'),item.get('m'))==(n,m)]
    audited={item['seed']:item for item in audited_items}
    require(len(audited_items)==10 and set(audited)==set(range(10)),f'Audit does not cover exactly ten selected seeds for ({n},{m})')
    found={}
    for path in sorted(results.glob('*/manifest.json')):
        meta=read(path)
        if (meta.get('n'),meta.get('m'))==(n,m):
            require(meta['seed'] not in found,'Duplicate selected seed manifest');found[meta['seed']]=(path,meta)
    require(set(found)==set(range(10)),f'Missing completed selected cases for ({n},{m}); required seeds 0..9')
    hashes={};checked_file(plan_path,hashes);checked_file(audit_path,hashes)
    for path in code.glob('*.py'):checked_file(path,hashes)
    records={method:[] for method in METHODS};exceptions=[];refinements=[];execution_contexts=[]
    for seed in range(10):
        manifest,meta=found[seed];case=manifest.parent
        require(case.name==('single' if m==1 else 'multi')+f'_n{n}_m{m}_seed{seed}','Selected directory/manifest identity mismatch')
        require(meta.get('state')=='complete' and meta.get('source_hashes')==source_hashes and meta.get('cap_seconds')==1000,'Selected case is incomplete or belongs to another protocol')
        context=load_priority_execution(case,meta,plan_path,hashes,allow_subset_plan=True)
        if context:execution_contexts.append(context)
        evidence=audited[seed]
        require(evidence.get('exact_instance_digest_match') is True and evidence.get('recorded_instance_hash')==evidence.get('independent_instance_hash')==meta.get('instance_hash'),'Independent selected instance hash does not match')
        audit_binds(manifest,audit['file_sha256']);checked_file(manifest,hashes)
        for method in METHODS:
            require(meta.get('methods',{}).get(method,{}).get('stage')=='complete',f'{case.name}/{method} is not complete')
            paths=[case/f'{method}_{suffix}' for suffix in ('run.json','points.npz','diagnostics.json')]
            require(all(path.is_file() for path in paths),f'{case.name}/{method}: missing saved run/points/diagnostics')
            for path in paths:audit_binds(path,audit['file_sha256']);checked_file(path,hashes)
            saved=read(paths[0]);rows=read(paths[2]);samples=saved['checkpoints'];result=saved['result']
            expected={key:meta[key] for key in ('n','m','seed','cap_seconds','instance_hash','source_hashes')};expected['method']=method
            require(saved.get('metadata')==expected,f'{case.name}/{method}: saved pair metadata mismatch')
            require(saved.get('points_sha256')==sha(paths[1]),f'{case.name}/{method}: saved point-file checksum mismatch')
            with np.load(paths[1],allow_pickle=False) as archive:points=archive['x']
            require(points.shape==(len(samples),n) and points.dtype==np.dtype('float64') and np.all(np.isfinite(points)) and len(rows)==len(samples)>0,'Saved averaged-point/checkpoint dimensions or finite values disagree')
            for index,(row,sample) in enumerate(zip(rows,samples)):
                require((row['n'],row['m'],row['seed'],row['method'],row['checkpoint'])==(n,m,seed,method,index),'Diagnostic checkpoint identity mismatch')
                require(row['elapsed']==sample['elapsed'] and row['k']==sample['k'] and finite(row['elapsed']) and 0<=row['elapsed']<=1000 and (not index or row['elapsed']>rows[index-1]['elapsed']),'Checkpoint clock/iteration identity or order mismatch')
                if usable(row):require(finite(row.get('residual')) and math.isclose(row['residual'],max(row['gap'],row['violation']),rel_tol=1e-12,abs_tol=1e-15),'Combined numerical residual is inconsistent')
                elif row.get('gap_valid') is True:require(False,'Diagnostic marked valid has a nonfinite/negative numerical gap or violation')
                else:require(bool(row.get('gap_status')),'Invalid diagnostic lacks a reason/status')
                if row.get('gap_status')=='independent_dual_bracket_valid':
                    adopted=row.get('diagnostic_adoption',{})
                    require(adopted.get('quality_reviewed_by_explicit_cli') is True and adopted.get('input_point_hash')==point_sha(points[index]) and adopted.get('instance_hash')==meta['instance_hash'],'Selected refined gap lacks its reviewed point identity')
                    raw_optimizer=row.get('solver_diagnostics',{}).get('raw_optimizer',{})
                    refinements.append(dict(seed=seed,method=method,checkpoint=index,sidecar_sha256=adopted.get('sidecar_sha256'),raw_optimizer_success=raw_optimizer.get('success'),raw_optimizer_status=raw_optimizer.get('status'),formal_interval_certificate=False))
                if method.startswith('oe_') and sample['k']>0:
                    factor={'oe_true':1.,'oe_low':.1,'oe_high':10.}[method]
                    require(finite(sample.get('M_estimate')) and math.isclose(sample['M_estimate'],factor*meta['true_M'],rel_tol=1e-10,abs_tol=1e-12),'Fixed OE norm estimate differs from its label')
            require(finite(result.get('runtime')) and result['runtime']>=0,'Invalid timed method runtime')
            if result['status']=='time_limit':require(result['runtime']>=999.99,'Time-limit runtime is below its cap')
            if result['status']=='certificate_stop':require(method=='proposed' and finite(samples[-1].get('certificate')) and samples[-1]['certificate']<=meta['epsilon'] and finite(samples[-1].get('violation')) and samples[-1]['violation']<=meta['epsilon'],'Invalid proposed certificate stop')
            if result['status'] not in ('time_limit','certificate_stop'):exceptions.append(dict(case=case.name,seed=seed,method=method,kind='method_termination',status=result['status']))
            invalid=[index for index,row in enumerate(rows) if not usable(row)]
            if invalid:exceptions.append(dict(case=case.name,seed=seed,method=method,kind='unresolved_diagnostics',checkpoints=invalid))
            if len(rows)<2:exceptions.append(dict(case=case.name,seed=seed,method=method,kind='insufficient_saved_trajectory',checkpoints=len(rows)))
            records[method].append(dict(seed=seed,rows=rows,termination=result['status'],instance_hash=meta['instance_hash']))
    adoption=results/'diagnostic_adoption_manifest.json'
    if any(row.get('gap_status')=='independent_dual_bracket_valid' for group in records.values() for item in group for row in item['rows']):
        require(adoption.is_file() and audit['file_sha256'].get(str(adoption))==sha(adoption),'Selected adopted gap rows require the audit-bound adoption manifest');checked_file(adoption,hashes)
    require(all(sha(path)==value for path,value in hashes.items()),'Selected input changed during preflight')
    return dict(n=n,m=m,seeds=list(range(10)),records=records,exceptions=exceptions,selected_independent_gap_refinements=refinements,priority_execution_overrides=summarize_priority_execution(execution_contexts),source_results=str(results),source_code=str(code),structural_audit=str(audit_path),input_sha256=hashes,fixture=False)


def aggregate(panel,bootstrap_replicates=4000,bootstrap_seed=20261029):
    require(bootstrap_replicates>=100,'At least 100 bootstrap replicates are required')
    ids=np.random.default_rng(bootstrap_seed).integers(0,10,(bootstrap_replicates,10));output={}
    for method in METHODS:
        trajectories=panel['records'][method]
        require(len(trajectories)==10 and {item['seed'] for item in trajectories}==set(range(10)),'Every aggregate retains the same complete ten-seed set')
        if not all(len(item['rows'])>=2 and all(usable(row) for row in item['rows']) for item in trajectories):output[method]=dict(available=False,reason='complete valid ten-seed diagnostic coverage is unavailable');continue
        endpoint=min(item['rows'][-1]['elapsed'] for item in trajectories)
        require(endpoint>1e-4,'Common recorded horizon does not extend beyond the displayed starting time')
        # Preserve observed knots in addition to the common logarithmic display grid.
        knots=[row['elapsed'] for item in trajectories for row in item['rows'] if 1e-4<=row['elapsed']<=endpoint]
        grid=np.unique(np.r_[np.geomspace(1e-4,endpoint,STYLE['base_grid_points']),knots]);values=[]
        for item in trajectories:
            times=np.maximum([row['elapsed'] for row in item['rows']],1e-4);errors=np.maximum([max(row['gap'],row['violation']) for row in item['rows']],STYLE['display_floor']);distinct=np.r_[True,np.diff(times)>0]
            values.append(np.exp(np.interp(np.log(grid),np.log(times[distinct]),np.log(errors[distinct]))))
        values=np.asarray(values);median=np.median(values,axis=0);boot=np.median(values[ids],axis=1);low,high=np.quantile(boot,[.025,.975],axis=0)
        output[method]=dict(available=True,time=grid,median=median,lower=low,upper=high,common_endpoint_seconds=endpoint,seed_count=10)
    return output


def make_figure(panels,aggregates,fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator,FuncFormatter
    with plt.rc_context({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'axes.titlesize':11.5,'axes.labelsize':11,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42}):
        figure,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
        figure.subplots_adjust(left=.075,right=.77,bottom=.19,top=.89,wspace=.16)
        for index,(axis,panel,curves) in enumerate(zip(axes,panels,aggregates)):
            unavailable=[]
            for method in METHODS:
                curve=curves[method]
                if not curve['available']:unavailable.append(LABELS[method]+': unavailable');continue
                style=STYLES[method];axis.fill_between(curve['time'],curve['lower'],curve['upper'],color=style['color'],alpha=.10)
                axis.plot(curve['time'],curve['median'],**style,linewidth=1.65,markersize=3.2,markerfacecolor='white',markeredgewidth=.7,markevery=31)
            axis.set(xscale='log',yscale='log',xlim=(1e-4,1000),ylim=(1e-6,100),xlabel='Wall-clock time (s)',title=f'({"ab"[index]}) '+('PFAOS' if panel['m']==1 else 'MC-PFAOS')+f': $n={panel["n"]},\\,m={panel["m"]}$')
            axis.set_xticks(10.**np.arange(-4,4));axis.xaxis.set_major_formatter(FuncFormatter(lambda value,position:'1000' if np.isclose(value,1000) else rf'$10^{{{int(round(np.log10(value)))}}}$'))
            axis.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100));axis.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100))
            axis.grid(which='major',color='#dddddd',linewidth=.5);axis.spines[['top','right']].set_visible(False);axis.tick_params(which='major',length=3.5,width=.8);axis.tick_params(which='minor',length=2,width=.6)
            axis.axhline(1e-5,color='#555555',linewidth=.85,linestyle=(0,(4,3)))
            if unavailable:axis.text(.02,.03,'\n'.join(unavailable),transform=axis.transAxes,fontsize=7.5,ha='left',va='bottom')
            if panel['exceptions']:axis.text(.02,.98,f'{len(panel["exceptions"])} recorded exception(s); see provenance',transform=axis.transAxes,fontsize=7.5,ha='left',va='top')
        # The requested PDF's y-label was separately reset; match its baseline x.
        axes[0].set_ylabel('Combined error',labelpad=4.90625)
        handles=[Line2D([],[],label=LABELS[method],**STYLES[method],linewidth=1.65,markersize=3.2,markerfacecolor='white',markeredgewidth=1.) for method in METHODS]
        handles.extend([Line2D([],[],label=r'Target $\varepsilon=10^{-5}$',color='#555555',linewidth=.85,linestyle=(0,(4,3))),Patch(facecolor='#888888',edgecolor='none',alpha=.15,label='95% bootstrap CI (10 seeds)')])
        figure.legend(handles=handles,loc='center left',bbox_to_anchor=(.785,.52),frameon=False,fontsize=10.3,handlelength=3.,labelspacing=.9)
        if fixture:figure.text(.40,.025,'SYNTHETIC STYLE FIXTURE — NOT UPDATED EXPERIMENTAL RESULTS',ha='center',fontsize=8,fontweight='bold',color='#7a0000')
        return figure,axes


def synthetic_panels():
    """Deterministic, prominently labeled style fixtures; never production input."""
    panels=[]
    for n,m,scale in ((1000,1,30.),(100,100,4.)):
        records={method:[] for method in METHODS}
        for method,rate in zip(METHODS,(1.7,.65,.8,.05,.45)):
            for seed in range(10):
                rows=[dict(elapsed=float(t),gap=float(scale*(1.+.02*(seed-4.5))/(1.+(t/.04)**rate)),violation=0.,gap_valid=True) for t in np.r_[0.,np.geomspace(1e-4,1000,35)]]
                records[method].append(dict(seed=seed,rows=rows,termination='SYNTHETIC_FIXTURE'))
        panels.append(dict(n=n,m=m,seeds=list(range(10)),records=records,exceptions=[],input_sha256={},fixture=True))
    return panels


def write_outputs(out,panels,bootstrap_replicates,bootstrap_seed,fixture=False):
    out=Path(out).resolve()
    protected=[ROOT/'code',ROOT/'results',ROOT/'results_single',ROOT/'results_n100',ROOT/'manuscript_single_ready',ROOT.parent/'CVI_Single_Constraint_LaTeX']
    protected.extend(Path(panel[key]) for panel in panels for key in ('source_results','source_code') if panel.get(key))
    require(not any(out.is_relative_to(path.resolve()) for path in protected),'Output must be separate from frozen data/code and the delivered single paper')
    require(not out.exists(),'Refusing to overwrite an existing figure output directory')
    if fixture:require(out.is_relative_to(Path('/private/tmp').resolve()) or out.is_relative_to(Path('/tmp').resolve()),'Synthetic style preview output must be in a temporary directory')
    aggregates=[aggregate(panel,bootstrap_replicates,bootstrap_seed) for panel in panels]
    figure,_=make_figure(panels,aggregates,fixture);out.mkdir(parents=True)
    stem='sensitivity_fixture_preview' if fixture else 'sensitivity_residual'
    figure.savefig(out/(stem+'.pdf'),facecolor='white',metadata={'Title':'SYNTHETIC STYLE FIXTURE ONLY' if fixture else 'Corrected two-instance sensitivity comparison'})
    figure.savefig(out/(stem+'.png'),dpi=300,facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(figure)
    with (out/'sensitivity_plot_data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['panel','n','m','method','solver_seconds','median','CI_lower','CI_upper']);writer.writeheader()
        for index,(panel,curves) in enumerate(zip(panels,aggregates)):
            for method,curve in curves.items():
                if curve['available']:
                    for time,median,low,high in zip(curve['time'],curve['median'],curve['lower'],curve['upper']):writer.writerow(dict(panel='ab'[index],n=panel['n'],m=panel['m'],method=method,solver_seconds=time,median=median,CI_lower=low,CI_upper=high))
    caption=r'\caption{'+('SYNTHETIC STYLE FIXTURE ONLY; NOT EXPERIMENTAL RESULTS. ' if fixture else '')+r'Combined numerical error $\widehat E(\bar x)=\max\{\widehat{\operatorname{Gap}}(\bar x),\|[G(\bar x)]_+\|_2\}$ versus recorded solver wall-clock time for (a) PFAOS at $n='+str(panels[0]['n'])+',m='+str(panels[0]['m'])+r'$ and (b) MC-PFAOS at $n='+str(panels[1]['n'])+',m='+str(panels[1]['m'])+r'$, compared with adaptive Golden Ratio and separate fixed OE trajectories at $\widehat M=M,0.1M,10M$. Curves are medians over all ten seeds, with pointwise 95\% bootstrap confidence intervals. Each method ends at the earliest terminal checkpoint across its fixed ten-seed set; unavailable diagnostic coverage is explicitly identified rather than dropping seeds. Log--log interpolation and a $10^{-12}$ display floor affect visualization only. The dashed line marks $10^{-5}$. Numerical objectives are not formal upper-gap certificates.}'+'\n'
    execution_contexts=[context for panel in panels for context in panel.get('priority_execution_overrides',{}).get('case_contexts',[])]
    execution_summary=summarize_priority_execution(execution_contexts)
    qualification=priority_execution_text(execution_summary,paragraph=False)
    if qualification:caption=caption.rstrip()[:-1]+' '+qualification.strip()+'}\n'
    (out/'sensitivity_residual_caption.tex').write_text(caption)
    inputs={key:value for panel in panels for key,value in panel['input_sha256'].items()}
    require(all(sha(path)==value for path,value in inputs.items()),'Selected measurements/audits changed while plotting')
    import matplotlib
    provenance=dict(status='synthetic_style_fixture_only' if fixture else 'measured_two_instance_plot_with_exceptions' if any(panel['exceptions'] for panel in panels) else 'measured_two_instance_plot',created_utc=datetime.now(timezone.utc).isoformat(),style_reference_pdf_sha256=REFERENCE_SHA256,style_reconstruction='PDF page geometry, STIX font sizes, axis/legend positions, gray levels, dashes, markers and alpha; no historical curve data used',style=STYLE,method_styles=STYLES,method_order=list(METHODS),bootstrap_replicates=bootstrap_replicates,bootstrap_seed=bootstrap_seed,aggregation='same complete ten-seed set; per-method earliest terminal horizon; piecewise log-log interpolation with observed knots; pointwise percentile bootstrap of seedwise median; no extrapolation or monotone fitting',input_sha256=inputs,panels=[dict(n=panel['n'],m=panel['m'],seeds=panel['seeds'],source_results=panel.get('source_results'),structural_audit=panel.get('structural_audit'),exceptions=panel['exceptions'],selected_independent_gap_refinements=panel.get('selected_independent_gap_refinements',[]),methods={method:{key:value for key,value in curve.items() if key not in ('time','median','lower','upper')} for method,curve in curves.items()}) for panel,curves in zip(panels,aggregates)],matplotlib_version=matplotlib.__version__,plotter_sha256=sha(__file__),figure_pdf_sha256=sha(out/(stem+'.pdf')),figure_png_sha256=sha(out/(stem+'.png')),new_solver_calls=0,historical_values_used=False,delivered_single_paper_modified=False)
    provenance['priority_execution_overrides']=execution_summary
    provenance['matched_concurrency_claimed']=False
    (out/'sensitivity_plot_provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    return dict(status=provenance['status'],pdf=str(out/(stem+'.pdf')),png=str(out/(stem+'.png')),settings=[dict(n=panel['n'],m=panel['m']) for panel in panels])


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_subparsers(dest='mode',required=True)
    plot=modes.add_parser('plot');plot.add_argument('--left-results',type=Path,default=ROOT/'results_single/verified');plot.add_argument('--left-n',type=int,default=1000);plot.add_argument('--left-m',type=int,default=1);plot.add_argument('--left-audit',type=Path);plot.add_argument('--right-results',type=Path,default=ROOT/'results_n100/verified');plot.add_argument('--right-n',type=int,default=100);plot.add_argument('--right-m',type=int,default=100);plot.add_argument('--right-audit',type=Path);plot.add_argument('--code',type=Path,default=ROOT/'code');plot.add_argument('--allow-exceptions',action='store_true')
    preview=modes.add_parser('fixture-preview')
    for command in (plot,preview):command.add_argument('--out',type=Path,required=True);command.add_argument('--bootstrap-replicates',type=int,default=4000);command.add_argument('--bootstrap-seed',type=int,default=20261029)
    args=parser.parse_args()
    try:
        fixture=args.mode=='fixture-preview'
        if fixture:panels=synthetic_panels()
        else:
            require(args.left_m==1 and args.right_m>1,'The two-panel comparison requires a single-constraint left panel and multi-constraint right panel')
            panels=[load_setting(args.left_results,args.left_n,args.left_m,args.left_audit or args.left_results/'report/structural/structural_audit.json',args.code),load_setting(args.right_results,args.right_n,args.right_m,args.right_audit or args.right_results/'report/structural/structural_audit.json',args.code)]
            require(args.allow_exceptions or not any(panel['exceptions'] for panel in panels),'Recorded method/diagnostic exceptions require explicit --allow-exceptions; no seed may be silently removed')
        result=write_outputs(args.out,panels,args.bootstrap_replicates,args.bootstrap_seed,fixture)
    except (ValueError,KeyError,OSError) as error:print(json.dumps(dict(status='refused',error=str(error),figure_created=False)),file=sys.stderr);return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':sys.exit(main())
