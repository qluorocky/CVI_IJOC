#!/usr/bin/env python3
"""Assemble a full-grid article from complete measured outputs, without editing theory.

The destination must not exist. No partial experiment values or historical tables
are copied into a final article. --validation-fixture permits only explicitly
marked fixture reports for isolated layout checks outside delivery directories.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from report_full_grid import check_plan

THEORY=('sec_PFAOS.tex','sec_MC_PFAOS.tex','special_first_iteration_strategy.tex','sec_stopping_criteria.tex','sec_appendix.tex')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_between(text,start,end,replacement):
    if start not in text or end not in text:raise RuntimeError(f'Template anchor missing: {start!r} / {end!r}')
    a=text.index(start);b=text.index(end,a)
    return text[:a]+replacement+text[b:]


def settings_text(plan):
    singles=sorted(s['n'] for s in plan['settings'] if s.get('study','single' if s['m']==1 else 'multi')=='single')
    multi=sorted((s for s in plan['settings'] if s.get('study','single' if s['m']==1 else 'multi')=='multi'),key=lambda s:s['m'])
    return singles,multi


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--template',required=True,type=Path);ap.add_argument('--results',required=True,type=Path);ap.add_argument('--destination',required=True,type=Path);ap.add_argument('--allow-unresolved',action='store_true');ap.add_argument('--validation-fixture',action='store_true');ap.add_argument('--single-only',action='store_true',help='Assemble the explicit single-constraint release and leave an editable MC placeholder');a=ap.parse_args()
    template=a.template.resolve();results=a.results.resolve();report=results/'report';dest=a.destination.resolve()
    plan=json.loads((results/'plan.json').read_text());check_plan(plan,a.validation_fixture,a.single_only)
    validation=json.loads((report/'validation.json').read_text())
    if validation.get('plan_snapshot')!=plan or any(not Path(path).is_file() or sha(path)!=value for path,value in validation.get('input_sha256',{}).items()):raise RuntimeError('Reporting plan or measured input hashes are stale; regenerate the report before assembly')
    table_audit_path=report/'table_statistics_audit.json'
    table_audit=json.loads(table_audit_path.read_text()) if table_audit_path.exists() else None
    table_names=('pfaos_table.tex','pfaos_table_gap_only.tex') if a.single_only else ('pfaos_table.tex','mc_table.tex','pfaos_table_gap_only.tex','mc_table_gap_only.tex')
    audit_inputs={str(report/name) for name in ('per_run_summary.csv',)+table_names}
    audit_current=bool(table_audit and table_audit.get('status')=='passed' and
        table_audit.get('scope')==('declared_plan_validation' if a.validation_fixture else 'single_constraint_release' if a.single_only else 'full_requested_grid') and
        table_audit.get('results')==str(results) and table_audit.get('report')==str(report) and
        table_audit.get('expected_instances')==len(plan['settings'])*len(plan['seeds']) and
        table_audit.get('expected_method_runs')==len(plan['settings'])*len(plan['seeds'])*len(plan['methods']) and
        table_audit.get('auditor_sha256')==sha(Path(__file__).with_name('audit_table_statistics.py')) and
        audit_inputs.issubset(table_audit.get('file_sha256',{})) and
        all(Path(path).is_file() and sha(path)==value for path,value in table_audit.get('file_sha256',{}).items()))
    audit_run_by_assembler=False
    if not audit_current:
        command=[sys.executable,str(Path(__file__).with_name('audit_table_statistics.py')),'--results',str(results)]
        if a.validation_fixture:command.append('--plan-scope')
        if a.single_only:command.append('--single-only')
        subprocess.run(command,check=True)
        table_audit=json.loads(table_audit_path.read_text());audit_run_by_assembler=True
    if table_audit['status']!='passed':raise RuntimeError('A passed independent CSV/table-statistics audit is required before assembly')
    if a.validation_fixture:
        if validation['status']!='validation_fixture' or not plan.get('validation_fixture'):raise RuntimeError('Explicitly marked validation data required')
    elif validation['status']!='complete' and not (a.allow_unresolved and validation['status']=='complete_with_exceptions'):
        raise RuntimeError('Final assembly requires complete validated reporting, or explicit --allow-unresolved for a visibly qualified draft')
    structural_path=report/'structural/structural_audit.json'
    structural=json.loads(structural_path.read_text()) if structural_path.exists() else None
    if not a.validation_fixture and (not structural or structural['status']!='structural_pass'):
        raise RuntimeError('A passed full-plan factorwise structural audit is required before final assembly')
    if not a.validation_fixture and structural.get('scope')!=('single_constraint_release' if a.single_only else 'requested_full_grid'):raise RuntimeError('Structural audit scope differs from the explicitly selected release')
    if structural and any(not Path(path).is_file() or sha(path)!=value for path,value in structural.get('file_sha256',{}).items()):
        raise RuntimeError('Measured records changed after the structural audit; repeat it before assembly')
    required=('pfaos_table.tex','mc_table.tex','pfaos_table_gap_only.tex','mc_table_gap_only.tex','results_discussion.tex','evaluation_protocol.tex','pfaos_full_grid_figure.tex','mc_full_grid_figure.tex','pfaos_full_grid.pdf','mc_full_grid.pdf')
    if a.single_only:required=tuple(name for name in required if not name.startswith('mc_'))
    for name in required:
        if not (report/name).is_file():raise RuntimeError(f'Missing final report artifact: {name}')
    if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
    if not (template/'main.tex').exists():raise RuntimeError('Template root main.tex is required')
    # Read actual manifest policy, rather than guessing the large-model budget.
    metas=[json.loads(p.read_text()) for p in results.glob('*/manifest.json')]
    caps=sorted({m.get('max_conic_coefficients') for m in metas if m.get('max_conic_coefficients') is not None})
    if not caps and not a.validation_fixture:raise RuntimeError('Actual conic coefficient cap is not recorded in manifests')
    captext=(f'{caps[0]:,}' if len(caps)==1 else ', '.join(f'{c:,}' for c in caps)) if caps else 'recorded'
    gapbudgets=sorted({row.get('gap_seconds') for p in results.glob('*/*_diagnostics.json') for row in json.loads(p.read_text()) if row.get('gap_seconds') is not None})
    gaptext=(', '.join(f'{v:g}' for v in gapbudgets)+' seconds') if gapbudgets else 'separately recorded limits'
    policy=plan.get('execution_policy',{});workers=policy.get('workers',7);affinity=policy.get('affinity','os_managed')
    scheduling=(f'Up to {workers} independent instances run on pinned cores, subject to memory-aware admission.' if affinity=='pinned' else f'Up to {workers} independent instances run under operating-system scheduling without CPU affinity, subject to memory-aware admission.')
    if validation.get('recorded_manifest_policy',{}).get('priority_execution_overrides',{}).get('case_count'):
        scheduling='In regular-grid phases, '+scheduling[0].lower()+scheduling[1:]
    shutil.copytree(template,dest,ignore=shutil.ignore_patterns('*.pdf','*.aux','*.log','*.out','*.toc','*.bbl','*.blg','*.fls','*.fdb_latexmk','build_validation.json','assembly_manifest.json'))
    # Restore supplied class/logo PDFs that the author master needs.
    for p in (template/'JoC_submission').glob('*.pdf'):
        if p.name.startswith(('informs_Logo','Sample-Figure')):shutil.copy2(p,dest/'JoC_submission'/p.name)
    relreport=Path(os.path.relpath(report,dest)).as_posix()
    master=dest/'JoC_submission/INFORMS-IJOC.tex';text=master.read_text()
    text,count=re.subn(r'\\providecommand\{\\ResultsDir\}\{[^}]+\}',lambda m:r'\providecommand{\ResultsDir}{'+relreport+'}',text)
    if count!=1:raise RuntimeError('Cannot identify unique ResultsDir definition')
    text=text.replace('small dense monotone affine VIs with convex quadratic constraints','dense monotone affine VIs across the specified dimension and constraint-count grids with convex quadratic constraints')
    if a.single_only:
        text=text.replace('Numerical experiments compare the proposed methods with','Numerical experiments compare PFAOS with').replace('across the specified dimension and constraint-count grids','across the specified single-constraint dimension grid')
    if a.validation_fixture:text=text.replace(r'\ABSTRACT{%',r'\ABSTRACT{%'+ '\n'+r'\textbf{VALIDATION FIXTURE ONLY; NOT EXPERIMENTAL RESULTS.} ')
    master.write_text(text)
    section=dest/'sec_numerical_experiments_new.tex';text=section.read_text();singles,multi=settings_text(plan);N=len(plan['seeds'])
    multi_grid=(r'; the multiple-constraint study fixes $n='+str(multi[0]['n'])+r'$ and uses $m\in\{'+','.join(str(s['m']) for s in multi)+r'\}$.') if multi else '. This release reports only the single-constraint experiments; the multi-constraint results will be added separately.'
    opening=r'''\section{Numerical experiments}\label{sec:numerical_experiments}

We compare '''+('PFAOS' if a.single_only else 'PFAOS and MC-PFAOS')+r''' with fixed-step operator extrapolation
\citep{kotsalis2020simplea} and the adaptive Golden Ratio Algorithm
\citep{malitsky2020golden}. The single-constraint study fixes $m=1$ and uses
$n\in\{'''+','.join(map(str,singles))+r'''\}$'''+multi_grid+r'''
Each setting uses '''+str(N)+r''' paired random seeds and a '''+f'{plan["seconds"]:g}'+r'''-second budget for each trajectory. The main tables retain three method rows:
the proposed method, true-$M$ OE, and Golden Ratio. The figures additionally
show independently timed OE trajectories with $\widehat M=0.1M$ and $10M$;
no trajectory is selected as ``best.''
All methods start from the same primal point on each realized instance.
The comparison assesses the implemented methods and projection policy
across the specified rows; it does not establish asymptotic runtime laws.

'''
    if a.validation_fixture:opening+=r'\textbf{Validation fixture only; not experimental conclusions.}\par'+'\n'
    text=replace_between(text,r'\section{Numerical experiments}',r'\subsection{Test instances}',opening)
    projections=r'''\paragraph{Full feasible-set projections.}
OE and Golden Ratio solve
\[
 \min_{y\in\widetilde X}\tfrac12\|y-z\|_2^2.
\]
Fresh CVXPY/Clarabel models use feasibility, absolute-gap, and relative-gap
tolerances of $10^{-8}$, with warm starts disabled. For larger constraint
sets, the implementation solves restricted models and checks the ball and
\emph{every} original constraint after each solve; violated constraints are
added as needed. A numerical solution is accepted only after the full-set
feasibility check passes at tolerance $10^{-7}$. Models and factorizations
are not reused.

Conic model construction is bounded by the recorded dense-coefficient
budget ('''+captext+r''' coefficients in these runs). If the next restricted
model would exceed that limit, or a conic solve returns a nonoptimal status
or raises a solver exception, the declared fallback is a full-constraint
SLSQP solve with objective tolerance $10^{-10}$ and the same full-set
feasibility check. The memory guard changes the numerical backend, not the
original feasible set. Model construction, restricted solves, all checks,
backend transitions, and fallback work are included in the method clock.
Actual fallback counts and solver statuses are retained in the run audit.

'''
    text=replace_between(text,r'\paragraph{Full feasible-set projections.}',r'\subsection{Performance measures and protocol}',projections)
    old='within each instance, with their order rotated across seeds. Up to seven\nindependent instances execute concurrently under macOS scheduling,\nwithout CPU affinity.'
    new='within each instance, with their order rotated across seeds. '+scheduling+' Heavy factor/model cases reserve capacity; their actual admission policy is recorded in the plan and manifests.'
    if old not in text:raise RuntimeError('Timing-policy template anchor changed')
    text=text.replace(old,new)
    text=text.replace(r'\frac{2F(x_{k-1})-F(x_{k-2})}{2M}',r'\frac{2F(x_{k-1})-F(x_{k-2})}{2\widehat M}')
    text=text.replace('For the reference norm $M=\\|B\\|_2$, initialize $x_{-1}=x_0$ and apply', 'For the reference norm $M=\\|B\\|_2$, each OE trajectory fixes $\\widehat M\\in\\{M,0.1M,10M\\}$, initializes $x_{-1}=x_0$, and applies')
    oe_start='The same true-$M$ configuration is used in both tables.'
    oe_end=r'\paragraph{Golden Ratio initialization and average.}'
    oe_policy=r'''Both tables use the true-$M$ OE trajectory. The figures also show separate
fixed trajectories at $0.1M$ and $10M$, each with its own full method budget;
no best-trajectory selection is made. Computing the reference norm $M$ is
shared setup outside method clocks and is recorded separately. These OE
configurations use the reference norm, whereas the proposed searches and
Golden Ratio adaptation use no global norm information.

'''
    text=replace_between(text,oe_start,oe_end,oe_policy)
    text=text.replace('and its cost is recorded separately. Methods execute sequentially','and its cost is recorded separately. Diagnostic calls use '+gaptext+r'; retries and numerical failures are recorded per checkpoint. Methods execute sequentially')
    text=text.replace('Thus the reported times describe this concurrent OS-scheduled workload.','Thus the reported times describe this resource-scheduled workload.')
    text=text.replace('Each threshold cell gives $s/10$, followed by best, mean, and worst','Each threshold cell gives $s/'+str(N)+'$, followed by best, mean, and worst')
    text=text.replace('statistics are best, mean, and worst values over all ten terminal','statistics are best, mean, and worst values over all '+str(N)+' terminal')
    # Explain unavailable diagnostics in the main article, before interpreting counts.
    insertion=r'''Runs with unresolved numerical diagnostics and no observed crossing are
shown separately as $u$ in the tables; they are not classified as
nonattainment. A first valid observed crossing can follow an unresolved
earlier checkpoint, as recorded in the per-run audit. Missing terminal
violation values are marked unresolved rather than summarized over a
silently reduced subset. Algorithm failures and diagnostic failures are
separate from time-limit censoring.

'''
    protocol_text=(report/'evaluation_protocol.tex').read_text()
    for marker in (r'\paragraph{Priority execution phase.}',r'\paragraph{Selected independent numerical gap refinements.}'):
        if marker in protocol_text:
            start=protocol_text.index(marker);end=protocol_text.find(r'\paragraph{',start+len(marker));insertion+=protocol_text[start:end if end>=0 else None]+'\n'
    text=text.replace(r'\subsection{Numerical results}',insertion+r'\subsection{Numerical results}')
    start=r'Figure~\ref{fig:pfaos_small_measured}'
    if start not in text:raise RuntimeError('Old figure anchor missing')
    text=text[:text.index(start)]+r'''Figures~\ref{fig:pfaos_full_grid} and~\ref{fig:mc_full_grid} use the same
averaged points and joint numerical residual as the main tables. Each panel
corresponds to one table configuration. Central curves are seedwise medians
with pointwise 95\% bootstrap intervals. Log--log interpolation and a
$10^{-12}$ display floor are visual conventions, not threshold estimators.
A method is aggregated only when every seed has complete valid diagnostic
coverage, and its curve ends at the earliest terminal checkpoint; later
per-seed crossings can therefore appear in the tables beyond that common
horizon. Missing diagnostic coverage is identified, without shrinking the
seed denominator or extrapolating a trajectory.
\input{\ResultsDir/pfaos_full_grid_figure.tex}
\input{\ResultsDir/mc_full_grid_figure.tex}

The conclusions concern the stated instances, numerical backend and
resource policy, initialization, and budgets. They do not establish that
projection methods are unworkable for other feasible sets or specialized
projection implementations. The additional fixed-$M$ variants characterize
observed stepsize sensitivity under the same recorded budgets; slower
progress with an overestimate is not treated as failed convergence.
'''
    if a.single_only:
        text=text.replace(r'Tables~\ref{tab:single_constraint_comparative}'+'\n'+r'and~\ref{tab:multi_constraint_comparative} use',r'Table~\ref{tab:single_constraint_comparative} uses')
        text=text.replace('Their companion gap-only tables are provided separately','Its companion gap-only table is provided separately')
        text=text.replace(r'\input{\ResultsDir/mc_table.tex}'+'\n','').replace(r'\input{\ResultsDir/mc_full_grid_figure.tex}'+'\n','')
        text=text.replace(r'Figures~\ref{fig:pfaos_full_grid} and~\ref{fig:mc_full_grid} use',r'Figure~\ref{fig:pfaos_full_grid} uses')
        text+=r'\input{sec_multi_constraint_results_pending.tex}'+'\n'
        (dest/'sec_multi_constraint_results_pending.tex').write_text(r'''\subsection{Multi-constraint experiments: results to be added}

The present numerical release covers the single-constraint study only.
The multi-constraint table and discussion will be inserted here after those
measurements are completed and checked. No multi-constraint performance
claim is made from the current numerical release.
% Future insertion: \input{JoC_submission/results/mc_table}
''')
    section.write_text(text)
    intro=dest/'sec_intro.tex';text=intro.read_text().replace('experiments on small dense monotone affine VIs with convex quadratic constraints,','experiments on dense monotone affine VIs over the stated dimension and constraint-count grids, with convex quadratic constraints,');intro.write_text(text)
    conclusion=dest/'sec_conclusion.tex';text=conclusion.read_text();marker='The numerical study compares both methods'
    if marker not in text:raise RuntimeError('Conclusion empirical-scope anchor missing')
    conclusion.write_text(text[:text.index(marker)]+r'''The numerical study compares both methods with fixed OE at the three
specified norm estimates and the adaptive Golden Ratio Algorithm over the dimension and constraint-count
grids, using the same averaged outputs for tables and figures. It reports
observed joint numerical gap-and-feasibility crossings, success counts,
unresolved diagnostic counts, and terminal violations under the stated budget.
These measurements assess the recorded implementations and resource policy;
they do not establish universal runtime scaling or the theoretical multiplier
condition for the experimental value $\Lambda=1$. Additional constraint
geometries and specialized projection implementations remain directions for
future empirical work.
''')
    if a.single_only:
        text=conclusion.read_text().replace('The numerical study compares both methods','The numerical study compares PFAOS').replace('over the dimension and constraint-count\ngrids','over the single-constraint dimension\ngrid')
        conclusion.write_text(text+'\nMulti-constraint measurements are deferred to a later numerical update.\n')
        text=intro.read_text().replace('over the stated dimension and constraint-count grids','over the stated single-constraint dimension grid');intro.write_text(text)
    (dest/'sec_numerical_supplement.tex').write_text(r'''\section{Supplementary numerical results}\label{app:numerical_supplement}

The companion tables use the same instances, averaged outputs, and method
budgets as Tables~\ref{tab:single_constraint_comparative} and
\ref{tab:multi_constraint_comparative}. They require only a valid numerical
weak-gap crossing, so they do not establish feasibility at the same checkpoint.
Unresolved diagnostic counts remain explicit; timings are conditioned on
observed successful runs and do not use cap imputation.
\input{\ResultsDir/pfaos_table_gap_only.tex}
\input{\ResultsDir/mc_table_gap_only.tex}
''')
    (dest/'numerical_review.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,graphicx}
\newcommand{\ResultsDir}{'''+relreport+r'''}
\graphicspath{{\ResultsDir/}}
\title{CVI full-grid numerical report}
\author{}\date{}
\begin{document}\maketitle
\input{\ResultsDir/evaluation_protocol.tex}
\clearpage
\input{\ResultsDir/pfaos_table.tex}
\input{\ResultsDir/mc_table.tex}
\input{\ResultsDir/results_discussion.tex}
\clearpage
\input{\ResultsDir/pfaos_full_grid_figure.tex}
\clearpage
\input{\ResultsDir/mc_full_grid_figure.tex}
\clearpage
\section*{Companion gap-only tables}
\input{\ResultsDir/pfaos_table_gap_only.tex}
\input{\ResultsDir/mc_table_gap_only.tex}
\end{document}
''')
    if a.single_only:
        supplement=dest/'sec_numerical_supplement.tex';text=supplement.read_text().replace(r'Tables~\ref{tab:single_constraint_comparative} and'+'\n'+r'\ref{tab:multi_constraint_comparative}',r'Table~\ref{tab:single_constraint_comparative}').replace(r'\input{\ResultsDir/mc_table_gap_only.tex}'+'\n','');supplement.write_text(text)
        review=dest/'numerical_review.tex';text=review.read_text().replace('CVI full-grid numerical report','CVI single-constraint numerical report')
        for name in ('mc_table.tex','mc_table_gap_only.tex','mc_full_grid_figure.tex'):text=text.replace(r'\input{\ResultsDir/'+name+'}'+'\n','')
        review.write_text(text)
    # Keep the author's traditional input paths current as well as ResultsDir.
    traditional=dest/'JoC_submission/results';traditional.mkdir(parents=True,exist_ok=True)
    dropin_copies=[]
    for name in table_names:
        target=traditional/name;shutil.copy2(report/name,target)
        dropin_copies.append(dict(source=str(report/name),destination=str(target.relative_to(dest)),source_sha256=sha(report/name),copied_sha256=sha(target),transformation='none; exact measured table fragment'))
    for source_name,target_name,pdf_name in (('pfaos_full_grid_figure.tex','pfaos_figures.tex','pfaos_full_grid.pdf'),('mc_full_grid_figure.tex','mc_figures.tex','mc_full_grid.pdf')):
        if a.single_only and source_name.startswith('mc_'):continue
        fragment=(report/source_name).read_text().replace('{'+pdf_name+'}',r'{\ResultsDir/'+pdf_name+'}')
        target=traditional/target_name;target.write_text(fragment)
        dropin_copies.append(dict(source=str(report/source_name),destination=str(target.relative_to(dest)),source_sha256=sha(report/source_name),copied_sha256=sha(target),transformation='only graphic reference prefixed with ResultsDir; numerical caption/labels unchanged'))
    if a.single_only:
        for name in ('mc_table.tex','mc_table_gap_only.tex','mc_figures.tex'):
            target=traditional/name;target.write_text('% Multi-constraint results are pending; insert the checked measured fragment here.\n% No historical numerical values are included in this single-constraint release.\n')
            dropin_copies.append(dict(source=None,destination=str(target.relative_to(dest)),copied_sha256=sha(target),transformation='explicit empty MC insertion placeholder; no numerical values'))
    preservation=[dict(file=name,template_sha256=sha(template/name),assembled_sha256=sha(dest/name),unchanged=sha(template/name)==sha(dest/name)) for name in THEORY]
    if not all(row['unchanged'] for row in preservation):raise RuntimeError('Theory source changed during assembly')
    manifest=dict(status='validation_fixture_only' if a.validation_fixture else ('assembled_with_explicit_unresolved_diagnostics' if validation['status']!='complete' else 'assembled_from_complete_valid_measurements'),created_utc=datetime.now(timezone.utc).isoformat(),compiled=False,template=str(template),results=str(results),plan_sha256=sha(results/'plan.json'),plan_snapshot=plan,report_validation_status=validation['status'],structural_audit_status=structural['status'] if structural else 'not_required_for_fixture',table_statistics_audit_status=table_audit['status'],table_statistics_audit_run_by_assembler=audit_run_by_assembler,independent_gap_validation=structural.get('gap_validation','not_independently_checked') if structural else 'not_performed',theory_source_preservation=preservation,report_inputs_sha256={name:sha(report/name) for name in required},traditional_input_copies=dropin_copies,max_conic_coefficients=caps,gap_call_seconds_observed=gapbudgets,figures=['fig:pfaos_full_grid','fig:mc_full_grid'],table_labels=['tab:single_constraint_comparative','tab:multi_constraint_comparative','tab:single_constraint_comparative_gap_only','tab:multi_constraint_comparative_gap_only'])
    manifest['release_scope']='single_only' if a.single_only else 'full_grid'
    manifest['selected_independent_gap_evaluations']=validation.get('recorded_manifest_policy',{}).get('selected_independent_gap_evaluations',0)
    adoption_path=results/'diagnostic_adoption_manifest.json'
    manifest['diagnostic_adoption_manifest_sha256']=sha(adoption_path) if adoption_path.exists() else None
    if a.single_only:
        manifest['figures']=['fig:pfaos_full_grid'];manifest['table_labels']=['tab:single_constraint_comparative','tab:single_constraint_comparative_gap_only'];manifest['multi_constraint_results']='pending_author_insertion; no numerical claims'
    (dest/'assembly_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'README.md').write_text('''# Full-grid measured manuscript

`main.tex` is the article entry point. Run `sh build.sh` from this directory
only after the complete measured report and audits have passed. `numerical_review.tex`
provides a separate protocol/tables/figures review document and compiles with
pdfLaTeX twice. The default `ResultsDir` points to the actual report relative to
this directory; preserve that relationship when packaging.

`assembly_manifest.json` records actual report hashes, audit status, numerical
limits, and byte-identical theory sources. Assembly is not compilation or visual
review. Those steps must be recorded before treating a PDF as final. Invalid
diagnostics, if explicitly allowed during assembly, remain marked unresolved in
the tables and discussion; no independent gap verification is inferred from a
structural audit. Historical table values are not imported.

The traditional paths `JoC_submission/results/pfaos_table.tex` and
`mc_table.tex` also contain the new measured table fragments, alongside their
gap-only companions. `pfaos_figures.tex` and `mc_figures.tex` now refer to the
new full-grid figures through `ResultsDir`. Exact copy/transformation hashes
are recorded in the assembly manifest.
''')
    if a.single_only:
        readme=dest/'README.md';text=readme.read_text().replace('# Full-grid measured manuscript','# Single-constraint measured manuscript')
        text=text[:text.index('The traditional paths')]+'''The traditional paths `JoC_submission/results/pfaos_table.tex` and its
gap-only companion contain the new single-constraint measurements.
`pfaos_figures.tex` refers to the six-panel, five-method figure through
`ResultsDir`. MC table/figure fragments are empty insertion placeholders.
Edit `sec_multi_constraint_results_pending.tex` to add the later MC results
and discussion. Exact copy and placeholder hashes are in the assembly manifest.
''';readme.write_text(text)
    print(json.dumps({k:manifest[k] for k in ('status','results','report_validation_status','structural_audit_status','compiled')},indent=2))


if __name__=='__main__':main()
