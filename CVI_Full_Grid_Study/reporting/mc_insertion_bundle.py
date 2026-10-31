#!/usr/bin/env python3
"""Prepare a separate, reviewed MC insertion bundle from complete audited data.

No numerical experiment or resampling is performed. Existing tables/figure are
copied, MC CSV rows and discussion are extracted, and standalone LaTeX reviews
may be compiled. The delivered single-constraint paper/ZIP are never edited.
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
import re
import shutil
import subprocess
import sys
import uuid
import zipfile

import report_full_grid as reporting
import audit_table_statistics as statistics_audit

ROOT=Path(__file__).resolve().parents[1]
DESTINATION=ROOT.parent/'CVI_MC_Insertion_Bundle'
ARCHIVE=ROOT.parent/'CVI_MC_Insertion_Bundle.zip'
MC_M=(10,50,100,200,500)
METHODS=('proposed','oe_true','oe_low','oe_high','golden_ratio')
JUNK=('.aux','.log','.out','.toc','.blg','.fls','.fdb_latexmk','.synctex.gz','.pyc','.tmp')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(reporting.clean(value),indent=2,allow_nan=False)+'\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def current_inputs(values):return all(Path(path).is_file() and sha(path)==value for path,value in values.items())


def select_mc(plan,loaded):
    """The full release contract and the MC subset are separately enforced."""
    reporting.check_plan(plan)
    require(plan.get('validation_fixture') is not True,'Production bundle cannot consume a validation fixture')
    selected=[item for item in loaded if item['meta']['m']>1]
    expected={(100,m,seed,method) for m in MC_M for seed in range(10) for method in METHODS}
    identities=[(item['meta']['n'],item['meta']['m'],item['meta']['seed'],item['method']) for item in selected]
    require(len(identities)==250 and set(identities)==expected,'MC coverage must be exactly n=100, five m rows, ten seeds and five methods (50 instances/250 trajectories)')
    return selected


def mc_inventory(plan,selected):
    rows,runs,statuses,terminations,failures=reporting.summarize(plan,selected)
    invalid=[dict(case=item['case'],n=item['meta']['n'],m=item['meta']['m'],seed=item['meta']['seed'],method=item['method'],checkpoint=row['checkpoint'],gap_status=row.get('gap_status'),gap_valid=row.get('gap_valid'),joint_valid=reporting.usable(row,'residual'),elapsed=row.get('elapsed'),diagnostic_error=row.get('diagnostic_error')) for item in selected for row in item['checkpoints'] if not reporting.usable(row,'residual')]
    unavailable=[]
    for m in MC_M:
        for method in METHODS:
            trajectories=[item['checkpoints'] for item in selected if item['meta']['m']==m and item['method']==method]
            if not all(len(points)>=2 and all(reporting.usable(point,'residual') for point in points) for points in trajectories):unavailable.append(dict(n=100,m=m,method=method))
    inventory=dict(scope='new_mc_n100_insertion_only',instances=50,method_runs=250,n=100,constraint_counts=list(MC_M),seeds=list(range(10)),methods=list(METHODS),seconds_per_method=1000,table_method_rows=15,figure_panels=5,figure_methods_per_panel=5,checkpoints=len(rows),diagnostic_status_counts=dict(statuses),termination_counts=dict(terminations),algorithm_failures=failures,unresolved_joint_diagnostics=invalid,unavailable_aggregate_groups=unavailable,mc_has_exceptions=bool(invalid or failures),status='mc_complete_with_exceptions' if invalid or failures else 'mc_complete',single_constraint_diagnostics_used_for_mc_metrics=False,execution_policy=plan.get('execution_policy',{}),independent_gap_solves_by_this_helper=0)
    return inventory,rows,runs


def check_audits(plan,validation,structural,statistics,expected_reporter,expected_structural,expected_statistics):
    require(validation.get('status') in ('complete','complete_with_exceptions') and validation.get('instances')==110 and validation.get('method_runs')==550,'Complete full 110-instance/550-run record coverage is required')
    require(validation.get('plan_snapshot')==plan and validation.get('reporter_sha256')==expected_reporter,'Full report plan/reporter identity mismatch')
    require(structural.get('status')=='structural_pass' and structural.get('scope')=='requested_full_grid' and structural.get('expected_instances')==structural.get('audited_instances')==110 and structural.get('expected_method_runs')==structural.get('audited_method_runs')==550 and structural.get('coverage_complete') is True and structural.get('auditor_sha256')==expected_structural,'A passed complete full-grid structural audit is required')
    require(statistics.get('status')=='passed' and statistics.get('scope')=='full_requested_grid' and statistics.get('expected_instances')==110 and statistics.get('expected_method_runs')==550 and statistics.get('auditor_sha256')==expected_statistics,'A passed complete independent full-grid statistics audit is required')


def extract_csv(source,target):
    with Path(source).open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=[row for row in reader if int(row['m'])>1]
    require(all(int(row['n'])==100 and int(row['m']) in MC_M for row in rows),'CSV contains an unexpected MC setting')
    with Path(target).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return len(rows)


def mc_protocol(inventory):
    return r'''\paragraph{Multi-constraint experimental protocol.}
The MC experiment fixes $n=100$ and uses $m\in\{10,50,100,200,500\}$,
with seeds $0,\ldots,9$ and a 1,000-second method budget for each of
MC-PFAOS, fixed OE at $\widehat M\in\{M,0.1M,10M\}$, and Golden Ratio.
The primary and companion tables retain MC-PFAOS, true-$M$ OE, and Golden Ratio;
the five-panel figure displays all five measured trajectories when complete
valid seed coverage permits. No best-OE selection or new resampling is performed.

All entries concern the original saved averaged points. The joint criterion is
$\widehat E(z)=\max\{\widehat{\operatorname{Gap}}(z),V(z)\}$, with
$V(z)=\|[G(z)]_+\|_2$; gap-only crossings do not establish simultaneous
feasibility. Counts use all ten seeds; times condition on observed valid
crossings and are not filled with the cap. Unresolved diagnostics remain
explicit, and unavailable aggregate curves do not silently drop seeds.
The copied figure retains its original interpolation conventions and bootstrap
bands; this extraction performs no bootstrap or other resampling.
Numerical gaps and solver labels are not formal finite-precision certificates.
Shared instance setup and the reference norm are outside method clocks;
initialization, projection construction/solves, feasibility checks, fallbacks,
and unfinished terminal work are timed. Gap diagnostics have separate clocks.
Worker concurrency and resource admission are retained in the source plan;
these measurements describe that recorded workload.

This MC subset contains '''+str(len(inventory['unresolved_joint_diagnostics']))+r''' unresolved
joint diagnostics and '''+str(len(inventory['algorithm_failures']))+r''' method terminations
outside the prescribed cap/certificate statuses. Their exact identities are in
\texttt{mc\_diagnostic\_inventory.json}; actual exceptions are retained in the
tables, discussion, and figure availability. Single-constraint diagnostic
exceptions do not enter these MC counts or determine this subset's status.
'''


def figure_fragment(text):
    require(text.count('{mc_full_grid.pdf}')==1,'Expected one source MC figure asset')
    return r'\providecommand{\MCResultsDir}{MC_results}'+'\n'+text.replace('{mc_full_grid.pdf}',r'{\MCResultsDir/mc_full_grid.pdf}')


def write_entries(destination):
    preamble=r'''\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,graphicx}
\newcommand{\MCResultsDir}{MC_results}
\begin{document}
'''
    (destination/'mc_review.tex').write_text(preamble+r'''\section*{MC insertion-bundle numerical review}
\input{\MCResultsDir/mc_protocol.tex}
\input{\MCResultsDir/mc_table.tex}
\input{\MCResultsDir/mc_discussion.tex}
\clearpage
\input{\MCResultsDir/mc_full_grid_figure.tex}
\clearpage
\section*{Companion gap-only table}
\input{\MCResultsDir/mc_table_gap_only.tex}
\end{document}
''')
    (destination/'mc_table_figure.tex').write_text(preamble+r'''\input{\MCResultsDir/mc_table.tex}
\clearpage
\input{\MCResultsDir/mc_full_grid_figure.tex}
\end{document}
''')
    (destination/'mc_insert.tex').write_text(r'''% Copy MC_results/ beside the article's main.tex, then input this fragment.
\providecommand{\MCResultsDir}{MC_results}
\input{\MCResultsDir/mc_protocol.tex}
\input{\MCResultsDir/mc_table.tex}
\input{\MCResultsDir/mc_discussion.tex}
\input{\MCResultsDir/mc_full_grid_figure.tex}
% Put the companion table in the numerical supplement if desired:
% \input{\MCResultsDir/mc_table_gap_only.tex}
''')
    (destination/'build.sh').write_text('''#!/bin/sh
set -eu
cd "$(dirname "$0")"
for entry in mc_review mc_table_figure; do
  pdflatex -interaction=nonstopmode -halt-on-error "$entry.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "$entry.tex"
done
''')


def compile_reviews(destination):
    for entry in ('mc_review','mc_table_figure'):
        for iteration in range(2):
            with (destination/f'{entry}_build_{iteration+1}.log').open('w') as output:subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',entry+'.tex'],cwd=destination,stdout=output,stderr=subprocess.STDOUT,check=True)
        log=(destination/(entry+'.log')).read_text(errors='replace')
        require(re.search(r'(?:Citation|Reference) `[^\n]*undefined|There were undefined references|LaTeX Error|Undefined control sequence',log) is None,f'Unresolved build/reference issue in {entry}')


def copy_file(source,target,destination,records,group):
    source=Path(source).resolve();target.parent.mkdir(parents=True,exist_ok=True);before=sha(source);shutil.copy2(source,target)
    require(sha(target)==before and sha(source)==before,f'Input changed during MC extraction: {source}')
    records.append(dict(group=group,source=str(source),path=target.relative_to(destination).as_posix(),sha256=before))


def package_files(destination):
    return [path for path in sorted(destination.rglob('*')) if path.is_file() and not path.name.endswith(JUNK) and '__pycache__' not in path.parts and not path.name.startswith('.')]


def checksum_file(destination):
    paths=[path for path in package_files(destination) if path.name!='CHECKSUMS.sha256']
    (destination/'CHECKSUMS.sha256').write_text('\n'.join(sha(path)+'  '+path.relative_to(destination).as_posix() for path in paths)+'\n')
    return paths


def prepare(results,destination,allow_unresolved=False,compile_pdf=True):
    results=Path(results).resolve();destination=Path(destination).resolve();report=results/'report'
    require(results==(ROOT/'results_n100/verified').resolve(),'Only the new results_n100/verified release may supply this insertion bundle')
    require(destination==DESTINATION.resolve() and not DESTINATION.is_symlink(),'Use the separate workspace/CVI_MC_Insertion_Bundle destination')
    require(not destination.exists(),'Refusing to overwrite an existing MC bundle')
    plan,loaded,raw_hashes=reporting.load_complete(results);selected=select_mc(plan,loaded)
    require({path.name:sha(path) for path in (ROOT/'code').glob('*.py')}==plan['source_hashes'],'Original frozen worker source identity changed')
    validation=read(report/'validation.json');structural=read(report/'structural/structural_audit.json');stats=read(report/'table_statistics_audit.json')
    check_audits(plan,validation,structural,stats,sha(Path(reporting.__file__)),sha(Path(__file__).with_name('audit_saved_records.py')),sha(Path(statistics_audit.__file__)))
    for evidence in (validation.get('input_sha256',{}),structural.get('file_sha256',{}),stats.get('file_sha256',{})):require(evidence and current_inputs(evidence),'Full reporting/audit input hashes are stale')
    inventory,rows,runs=mc_inventory(plan,selected)
    require(allow_unresolved or not inventory['mc_has_exceptions'],'MC-specific exceptions require explicit --allow-unresolved; single-only exceptions never trigger this gate')
    figure_provenance=read(report/'full_grid_figure_provenance.json');require(figure_provenance.get('source_sha256')==sha(report/'checkpoints.csv'),'Full figure provenance is stale')
    groups=[group for group in figure_provenance['settings'] if group['study']=='multi']
    require(len(groups)==25 and {(g['n'],g['m'],g['method']) for g in groups}=={(100,m,method) for m in MC_M for method in METHODS} and all(g['seeds']==10 for g in groups),'MC figure must contain five settings and five methods with all ten seeds')
    unavailable={(g['m'],g['method']) for g in inventory['unavailable_aggregate_groups']}
    require(all(bool(group['plotted'])==((group['m'],group['method']) not in unavailable) for group in groups),'MC aggregate availability disagrees with saved diagnostics')
    for group in groups:
        if group['plotted']:
            endpoint=min(item['checkpoints'][-1]['elapsed'] for item in selected if item['meta']['m']==group['m'] and item['method']==group['method'])
            require(group.get('common_endpoint_seconds')==endpoint,'Copied MC figure horizon disagrees with the full fixed seed set')
    destination.mkdir();assets=destination/'MC_results';assets.mkdir();provenance=destination/'provenance';provenance.mkdir();copies=[]
    try:
        for name in ('mc_table.tex','mc_table_gap_only.tex','mc_full_grid.pdf','mc_full_grid.png'):copy_file(report/name,assets/name,destination,copies,'existing_mc_report')
        for name in ('checkpoints.csv','per_run_summary.csv'):extract_csv(report/name,assets/name)
        with (assets/'per_run_summary.csv').open(newline='') as stream:require(sum(1 for _ in csv.DictReader(stream))==250,'MC summary must have exactly 250 trajectories')
        (assets/'mc_full_grid_figure.tex').write_text(figure_fragment((report/'mc_full_grid_figure.tex').read_text()))
        mc_plan=dict(plan,settings=[setting for setting in plan['settings'] if setting['study']=='multi'])
        (assets/'mc_discussion.tex').write_text(reporting.discussion(mc_plan,runs));(assets/'mc_protocol.tex').write_text(mc_protocol(inventory));write(assets/'mc_diagnostic_inventory.json',inventory)
        independent,issues=statistics_audit.derive_records([((item['meta']['n'],item['meta']['m'],item['meta']['seed']),item['method'],item['checkpoints']) for item in selected])
        csvcheck=statistics_audit.compare_csv(assets/'per_run_summary.csv',independent)
        tables=[statistics_audit.compare_table(assets/name,independent,mc_plan['settings'],reporting.TABLE_METHODS,metric,10,True) for name,metric in (('mc_table.tex','residual'),('mc_table_gap_only.tex','gap'))]
        require(not issues and not csvcheck['issues'] and all(not table['issues'] and table['rows']==15 for table in tables),'Independent MC-only extraction statistics check failed')
        write(provenance/'mc_statistics_check.json',dict(status='passed',records=250,source='independent raw-diagnostic derivation, no reporter import in statistics module',csv=csvcheck,tables=tables,issues=issues))
        write(provenance/'mc_figure_provenance.json',dict(source_full_provenance_sha256=sha(report/'full_grid_figure_provenance.json'),source_full_csv_sha256=sha(report/'checkpoints.csv'),local_mc_csv_sha256=sha(assets/'checkpoints.csv'),copied_figure_sha256=sha(assets/'mc_full_grid.pdf'),groups=groups,extraction_resampling_performed=False))
        copy_file(results/'plan.json',provenance/'source_full_plan.json',destination,copies,'full_plan_metadata')
        for source in (Path(__file__),Path(reporting.__file__),Path(statistics_audit.__file__)):
            copy_file(source,provenance/'reporting_sources'/source.name,destination,copies,'untimed_reporting_source')
        write(provenance/'full_audit_receipts.json',{name:dict(source_file=str(path),sha256=sha(path),status=value['status'],scope=value.get('scope'),instances=value.get('expected_instances',value.get('instances')),method_runs=value.get('expected_method_runs',value.get('method_runs'))) for name,path,value in (('report',report/'validation.json',validation),('structural',report/'structural/structural_audit.json',structural),('statistics',report/'table_statistics_audit.json',stats))})
        data=destination/'data';data.mkdir();write(data/'extraction_plan.json',dict(release_scope='mc_insertion_only',source_full_plan_sha256=sha(results/'plan.json'),settings=mc_plan['settings'],seeds=plan['seeds'],methods=plan['methods'],seconds=plan['seconds'],source_hashes=plan['source_hashes'],new_measurements_generated=False))
        for case in sorted({item['case'] for item in selected}):
            require(case.startswith('multi_n100_'),'Unexpected MC source case')
            for path in sorted((results/case).iterdir()):
                if path.is_file() and path.suffix in ('.json','.npz'):copy_file(path,data/case/path.name,destination,copies,'mc_saved_records')
        require(len(list(data.glob('*/manifest.json')))==50,'Bundle must contain exactly fifty MC case records')
        write_entries(destination)
        manifest=dict(state='prepared_with_mc_exceptions' if inventory['mc_has_exceptions'] else 'prepared_mc_complete',created_utc=datetime.now(timezone.utc).isoformat(),source_results=str(results),source_full_report_status=validation['status'],source_full_report_sha256=sha(report/'validation.json'),mc_inventory=inventory,allow_unresolved=bool(allow_unresolved),copied_files=copies,helper_sha256=sha(__file__),all_single_constraint_records_excluded=True,delivered_single_paper_touched=False,new_measurements_generated=False,resampling_performed=False,compiled=False,selected_mc_raw_input_sha256={path:value for path,value in raw_hashes.items() if '/multi_n100_' in path})
        write(destination/'bundle_manifest.json',manifest)
        (destination/'README.md').write_text('''# MC insertion bundle

This separate bundle contains only the new n=100 multi-constraint measurements:
m=10,50,100,200,500; ten seeds; five methods; 50 instances and 250 trajectories.
The main table has 15 method rows and the copied figure has five panels with
five methods where complete valid seed coverage permits. No experiment or
resampling was performed by extraction. Actual MC exceptions, if any, remain
explicit in MC_results/mc_diagnostic_inventory.json and the measured discussion.

Build both standalone review PDFs with `sh build.sh`. Preparation builds them
unless --no-compile was explicitly requested. Visual review remains required.

To insert manually, copy MC_results/ beside your article's main.tex, then insert
the contents of mc_insert.tex at your chosen MC results location. Place the
gap-only companion in the numerical supplement if desired. MCResultsDir is
independent of the single paper's ResultsDir. No delivered single paper or ZIP
is edited by this helper. The table/figure labels retain the manuscript's MC
labels, so insert each only once.

data/ contains only the fifty new MC saved cases. Full audit receipts and exact
source hashes establish the complete 110-instance coverage gate; full single
measurement records are not copied. Single-constraint diagnostic exceptions do
not block this MC subset. A qualified bundle requires explicit acceptance of
its MC exceptions at final review. Numerical diagnostics are not formal interval
certificates, and the structural auditor itself solves no gap problem.
''')
        if compile_pdf:
            compile_reviews(destination);manifest['compiled']=True;manifest['review_pdf_sha256']={name:sha(destination/(name+'.pdf')) for name in ('mc_review','mc_table_figure')};write(destination/'bundle_manifest.json',manifest)
        checksum_file(destination)
        return dict(state=manifest['state'],folder=str(destination),instances=50,method_runs=250,mc_has_exceptions=inventory['mc_has_exceptions'],compiled=manifest['compiled'],finalized=False)
    except BaseException:
        write(destination/'PREPARATION_FAILED.json',dict(state='incomplete_mc_preparation',inspect_before_removing=True));raise


def check_final_review(review,manifest,pdf_hashes):
    require(review.get('decision')=='approved_for_mc_insertion_bundle','MC bundle requires its own approved final review')
    require(review.get('mc_review_pdf_sha256')==pdf_hashes['mc_review'] and review.get('mc_table_figure_pdf_sha256')==pdf_hashes['mc_table_figure'],'Review must identify both actual standalone MC PDFs')
    require(review.get('references_resolved') is True and all(review.get('checks',{}).get(key) is True for key in ('coverage','statistics','structural','visual_table','visual_figure','qualifications')),'MC final review checks are incomplete')
    if manifest['mc_inventory']['mc_has_exceptions']:require(review.get('acknowledges_mc_exceptions') is True,'Qualified MC bundle requires explicit acknowledgement of its actual exceptions')


def finalize(destination,review_path,archive):
    destination=Path(destination).resolve();archive=Path(archive).resolve();review_path=Path(review_path).resolve()
    require(destination==DESTINATION.resolve() and archive==ARCHIVE.resolve() and not DESTINATION.is_symlink() and not ARCHIVE.is_symlink(),'Use only the separate MC bundle/ZIP paths')
    require(not archive.exists() and not archive.with_suffix('.zip.sha256').exists(),'Refusing to overwrite an existing MC archive/checksum')
    manifest=read(destination/'bundle_manifest.json');require(manifest.get('state') in ('prepared_mc_complete','prepared_with_mc_exceptions','reviewed_mc_complete','reviewed_with_mc_exceptions'),'A prepared MC-only bundle is required')
    require(not (destination/'PREPARATION_FAILED.json').exists(),'Incomplete preparation cannot be finalized')
    for item in manifest['copied_files']:require(sha(destination/item['path'])==item['sha256'],f'Copied MC measurement/asset changed: {item["path"]}')
    # All generated fragments and extracted CSVs must still match the prepared checksums.
    for line in (destination/'CHECKSUMS.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1)
        if relative.endswith(('.pdf',)) and '/' not in relative:continue
        require(sha(destination/relative)==expected,f'MC bundle content changed since preparation: {relative}')
    for entry in ('mc_review','mc_table_figure'):
        pdf=destination/(entry+'.pdf');log=destination/(entry+'.log')
        require(pdf.is_file() and pdf.read_bytes()[:5]==b'%PDF-' and log.is_file(),'Build both standalone PDFs before final review')
        require(re.search(r'(?:Citation|Reference) `[^\n]*undefined|There were undefined references|LaTeX Error|Undefined control sequence',log.read_text(errors='replace')) is None,'Standalone MC review contains unresolved compilation/references')
    pdf_hashes={name:sha(destination/(name+'.pdf')) for name in ('mc_review','mc_table_figure')};review=read(review_path);check_final_review(review,manifest,pdf_hashes)
    (destination/'provenance').mkdir(exist_ok=True);shutil.copy2(review_path,destination/'provenance/mc_insertion_review.json')
    manifest.update(state='reviewed_with_mc_exceptions' if manifest['mc_inventory']['mc_has_exceptions'] else 'reviewed_mc_complete',finalized_utc=datetime.now(timezone.utc).isoformat(),compiled=True,review_pdf_sha256=pdf_hashes,final_review_sha256=sha(review_path),final_checks=review['checks']);write(destination/'bundle_manifest.json',manifest)
    paths=checksum_file(destination);temporary=archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as output:
            for path in paths+[destination/'CHECKSUMS.sha256']:output.write(path,arcname=destination.name+'/'+path.relative_to(destination).as_posix())
        with zipfile.ZipFile(temporary) as check:require(check.testzip() is None,'MC archive integrity check failed')
        temporary.replace(archive)
    finally:
        if temporary.exists():temporary.unlink()
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    return dict(state=manifest['state'],archive=str(archive),archive_sha256=digest,instances=50,method_runs=250,delivered_single_paper_touched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_subparsers(dest='mode',required=True)
    stage=modes.add_parser('prepare');stage.add_argument('--results',type=Path,default=ROOT/'results_n100/verified');stage.add_argument('--destination',type=Path,default=DESTINATION);stage.add_argument('--allow-unresolved',action='store_true');stage.add_argument('--no-compile',action='store_true')
    final=modes.add_parser('finalize');final.add_argument('--destination',type=Path,default=DESTINATION);final.add_argument('--review',type=Path,default=ROOT/'notes/mc_insertion_review.json');final.add_argument('--zip',type=Path,default=ARCHIVE)
    args=parser.parse_args()
    try:result=prepare(args.results,args.destination,args.allow_unresolved,not args.no_compile) if args.mode=='prepare' else finalize(args.destination,args.review,args.zip)
    except (ValueError,KeyError,OSError,reporting.IncompleteDataset,subprocess.CalledProcessError) as error:print(json.dumps(dict(state='refused',error=str(error))),file=sys.stderr);return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':sys.exit(main())
