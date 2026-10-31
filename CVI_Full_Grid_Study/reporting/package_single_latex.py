#!/usr/bin/env python3
"""Prepare a portable single-constraint LaTeX folder; archive only after review.

This helper never runs a solver, compiler, or timed method. Prepare copies only
when explicitly invoked. Finalize requires the approved review of the actual
portable main.pdf, checks unchanged sources/data, and creates the requested ZIP.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid
import zipfile

from report_full_grid import load_complete,IncompleteDataset

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
DESTINATION=WORKSPACE/'CVI_Single_Constraint_LaTeX'
ARCHIVE=WORKSPACE/'CVI_Single_Constraint_LaTeX.zip'
THEORY=('sec_PFAOS.tex','sec_MC_PFAOS.tex','special_first_iteration_strategy.tex','sec_stopping_criteria.tex','sec_appendix.tex')
JUNK_SUFFIXES=('.aux','.log','.out','.toc','.lof','.lot','.fls','.fdb_latexmk','.synctex.gz','.blg','.pyc','.tmp')
GENERATED_BUILD={'main.pdf','numerical_review.pdf','main.bbl','numerical_review.bbl'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def included(path):
    return not any(part in ('__pycache__','.git','.DS_Store') for part in path.parts) and not path.name.startswith('.') and not path.name.endswith(JUNK_SUFFIXES)
def files(root):return [p for p in sorted(Path(root).rglob('*')) if p.is_file() and included(p.relative_to(root))]


def normalize_wrapper(text,entry):
    """Change only ResultsDir in the two permitted build-entry wrappers."""
    pattern=r'(\\(?:providecommand|newcommand)\{\\ResultsDir\}\{)[^{}]*(\})'
    normalized,count=re.subn(pattern,lambda m:m[1]+'measured_results'+m[2],text)
    if not count:
        normalized,count=re.subn(r'(\\def\\ResultsDir\{)[^{}]*(\})',lambda m:m[1]+'measured_results'+m[2],text)
    if count:
        require(count==1,f'{entry}: ResultsDir must have one unambiguous wrapper definition')
        return normalized,'only ResultsDir path replaced with measured_results'
    require(entry=='main.tex' and r'\input{JoC_submission/INFORMS-IJOC.tex}' in text,'Missing ResultsDir definition in numerical_review.tex or unsupported main wrapper')
    return r'\providecommand{\ResultsDir}{measured_results}'+'\n'+text,'only ResultsDir wrapper override added before unchanged submission master'


def check_review(review,main_pdf_sha):
    require(review.get('decision')=='approved_for_portable_release','Final archive requires an approved_for_portable_release review')
    require(review.get('package_main_pdf_sha256')==main_pdf_sha,'Review does not identify the actual portable main.pdf')
    require(all(review.get(key) is True for key in ('references_resolved','citations_resolved','visual_inspection_passed')),'Explicit reference/citation/visual review is incomplete')
    checks=review.get('checks',{})
    require(all(checks.get(key) is True for key in ('references','citations','visual_tables','visual_figure','proofs_unchanged')),'Required portable review checks are not all approved')


def copy_one(source,destination,root,group,records):
    source=Path(source).resolve();destination=Path(destination)
    require(source.is_file(),f'Missing package input: {source}')
    destination.parent.mkdir(parents=True,exist_ok=True);before=sha(source);shutil.copy2(source,destination)
    require(sha(destination)==before and sha(source)==before,f'Copy/input changed during packaging: {source}')
    records.append(dict(group=group,source=str(source),path=destination.relative_to(root).as_posix(),source_sha256=before,copied_sha256=before))


def copy_tree(source,destination,root,group,records):
    require(Path(source).is_dir(),f'Missing package input directory: {source}')
    for path in files(source):copy_one(path,Path(destination)/path.relative_to(source),root,group,records)


def write_readme(destination,reviewed=False):
    (destination/'README.md').write_text('''# CVI single-constraint LaTeX release

'''+('The portable PDF and source package have passed the recorded final review.\n' if reviewed else 'Prepared for build and visual review; this staging folder is not a final release.\n')+'''
`main.tex` is the article entry at this folder's root. Build the article with:

```sh
sh build.sh
```

This runs pdfLaTeX, BibTeX, and the required additional pdfLaTeX passes. Build the
standalone numerical review with pdfLaTeX twice on `numerical_review.tex`.
Both wrappers use the local `measured_results/` folder. The submission master,
all proof/theory sources, table values, figures, and measured records retain
their recorded source bytes; only the two wrappers' ResultsDir paths change.

The empirical release contains six single-constraint dimensions, ten paired
seeds and five timed methods (60 instances / 300 runs). The primary table has
three method rows; its six-panel figure shows all five trajectories. MC empirical
results are pending. Edit `sec_multi_constraint_results_pending.tex` and the
empty `JoC_submission/results/mc_table.tex` insertion point when adding them.

`data/` contains the new single-constraint plan and all 60 saved case records;
`code/` contains the original frozen worker source and environment specifications.
`recovery/` contains the eight selected v1/v2 refinement sidecars, source snapshots,
and primary backups. `audit_notes/` and `measured_results/` contain review/audit
evidence. No old multi-constraint measurement directory is included.

The eight selected independent numerical refinements preserve original and v1
failures, raw root termination flags, candidate identities and review evidence.
Their ordinary-float posterior bounds are not formal interval certificates.
Structural auditing itself performs no independent gap solves.

`package_manifest.json` records original and copied hashes, both wrapper changes,
and unchanged proof sources. `CHECKSUMS.sha256` covers included package files
except itself. Original absolute provenance paths remain unedited in raw records;
the source-to-package copy records locate their bit-identical local counterparts.
''')


def checksums(destination):
    included_files=[p for p in files(destination) if p.name!='CHECKSUMS.sha256']
    lines=[]
    for path in included_files:
        relative=path.relative_to(destination).as_posix();require('\n' not in relative,'Unsupported newline in packaged filename');lines.append(sha(path)+'  '+relative)
    (destination/'CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n')
    return included_files


def prepare(manuscript,destination):
    manuscript=Path(manuscript).resolve();destination=Path(destination).resolve();results=ROOT/'results_single/verified';report=results/'report'
    require(not DESTINATION.is_symlink(),'Portable destination must not be a symlink')
    require(destination==DESTINATION.resolve(),'Use the requested workspace/CVI_Single_Constraint_LaTeX destination')
    require(not destination.exists(),'Refusing to overwrite an existing portable folder')
    plan,loaded,_=load_complete(results,single_only=True)
    validation=read(report/'validation.json');assembly=read(manuscript/'assembly_manifest.json')
    require(validation.get('status')=='complete' and validation.get('instances')==60 and validation.get('method_runs')==300,'Complete 60-instance/300-run measured reporting is required')
    require(assembly.get('release_scope')=='single_only' and assembly.get('status')=='assembled_from_complete_valid_measurements','Expected the checked single-only article assembly')
    require(read(report/'structural/structural_audit.json').get('status')=='structural_pass' and read(report/'table_statistics_audit.json').get('status')=='passed','Passed structural and independent table-statistics audits are required')
    require(all(row.get('unchanged') is True for row in assembly['theory_source_preservation']),'Assembly theory preservation evidence is incomplete')
    for name,value in assembly['report_inputs_sha256'].items():require(sha(report/name)==value,f'Article/report input changed: {name}')
    require(not any(row['m']!=1 for row in plan['settings']),'Package data must be exclusively the single-constraint release')
    source_hashes={p.name:sha(p) for p in (ROOT/'code').glob('*.py')};require(source_hashes==plan['source_hashes'],'Original frozen worker source changed')
    records=[];normalizations=[]
    try:
        destination.mkdir()
        # Flatten the editable manuscript at package root; retain the author's README separately.
        for path in files(manuscript):
            relative=path.relative_to(manuscript)
            target=destination/('provenance/manuscript_README_original.md' if relative.as_posix()=='README.md' else relative)
            copy_one(path,target,destination,'manuscript',records)
        for name in ('main.tex','numerical_review.tex'):
            path=destination/name;before=sha(path);text,description=normalize_wrapper(path.read_text(),name);path.write_text(text)
            normalizations.append(dict(file=name,source_sha256=before,portable_sha256=sha(path),transformation=description))
        copy_tree(report,destination/'measured_results',destination,'measured_report',records)
        for path in sorted(results.glob('*.json')):copy_one(path,destination/'data'/path.name,destination,'measured_data',records)
        cases=sorted(results.glob('*/manifest.json'));require(len(cases)==60,'Expected exactly 60 saved case directories')
        for manifest in cases:
            meta=read(manifest);require(meta['m']==1 and meta['state']=='complete','Only complete single-constraint cases may be packaged')
            for path in files(manifest.parent):
                if path.suffix in ('.json','.npz'):copy_one(path,destination/'data'/manifest.parent.name/path.name,destination,'measured_data',records)
        copy_tree(ROOT/'code',destination/'code',destination,'frozen_worker_code',records)
        for path in files(Path(__file__).parent):
            if path.parent==Path(__file__).parent and path.suffix in ('.py','.md','.json'):copy_one(path,destination/'reproducibility/reporting'/path.name,destination,'reporting_source',records)
        recovery=ROOT/'reporting/recovery'
        for name in ('single_delivery_v1','single_delivery_v2'):copy_tree(recovery/name,destination/'recovery'/name,destination,'selected_recovery_evidence',records)
        for name in ('constraint_generation_gap.py','recover_single_dual.py','bind_single_dual_candidates.py','run_single_delivery_recovery.py','README.md'):
            copy_one(recovery/name,destination/'recovery'/name,destination,'recovery_source',records)
        backups=ROOT/'results_single/diagnostic_adoption_backups'
        require(backups.is_dir(),'Missing original primary-diagnostic adoption backups')
        copy_tree(backups,destination/'recovery/primary_backups',destination,'primary_diagnostic_backups',records)
        for path in sorted((ROOT/'notes').glob('single*.json')):
            if path.name!='single_delivery_review.json':copy_one(path,destination/'audit_notes'/path.name,destination,'audit_notes',records)
        proofs=[]
        for name in THEORY:
            before=sha(manuscript/name);after=sha(destination/name);require(before==after,f'Proof source changed: {name}');proofs.append(dict(file=name,source_sha256=before,portable_sha256=after,unchanged=True))
        write_readme(destination)
        manifest=dict(state='prepared_pending_portable_build_and_review',created_utc=datetime.now(timezone.utc).isoformat(),source_manuscript=str(manuscript),source_results=str(results),portable_root=str(destination),instances=60,method_runs=300,release_scope='single_only',methods=plan['methods'],source_hashes=source_hashes,path_normalizations=normalizations,theory_source_preservation=proofs,copy_records=records,old_multi_constraint_data_included=False,measurement_values_changed=False,packaging_helper_sha256=sha(__file__),final_review_required=str(ROOT/'notes/single_delivery_review.json'))
        write(destination/'package_manifest.json',manifest);checksums(destination)
        return dict(state=manifest['state'],folder=str(destination),copied_files=len(records),zip_created=False,main_entry=str(destination/'main.tex'),review_entry=str(destination/'numerical_review.tex'))
    except BaseException:
        # Keep an incomplete staging folder inspectable; never claim it is ready.
        if destination.exists():write(destination/'PREPARATION_FAILED.json',dict(state='incomplete_preparation',inspect_before_removing=True))
        raise


def finalize(destination,review_path,archive):
    destination=Path(destination).resolve();review_path=Path(review_path).resolve();archive=Path(archive).resolve()
    require(not DESTINATION.is_symlink() and not ARCHIVE.is_symlink(),'Portable output paths must not be symlinks')
    require(destination==DESTINATION.resolve() and archive==ARCHIVE.resolve(),'Use the requested portable folder and workspace ZIP filename')
    require(not archive.exists() and not archive.with_suffix('.zip.sha256').exists(),'Refusing to overwrite an existing release archive/checksum')
    manifest=read(destination/'package_manifest.json')
    require(manifest.get('state') in ('prepared_pending_portable_build_and_review','reviewed_portable_release'),'A successfully prepared portable folder is required')
    for name in ('main.pdf','numerical_review.pdf'):
        path=destination/name;require(path.is_file() and path.read_bytes()[:5]==b'%PDF-',f'Build the portable {name} before finalization')
    review=read(review_path);check_review(review,sha(destination/'main.pdf'))
    for name in ('main.log','numerical_review.log'):
        path=destination/name;require(path.is_file(),f'Missing portable build log: {name}')
        log=path.read_text(errors='replace')
        require(re.search(r'(?:Citation|Reference) `[^\n]*undefined|There were undefined references|LaTeX Error|Undefined control sequence',log) is None,f'Unresolved compilation/reference issue in {name}')
    normalized={row['file']:row['portable_sha256'] for row in manifest['path_normalizations']}
    for record in manifest['copy_records']:
        relative=record['path'];path=destination/relative
        if relative in GENERATED_BUILD:continue
        expected=normalized.get(relative,record['copied_sha256']);require(path.is_file() and sha(path)==expected,f'Portable source/data changed since preparation: {relative}')
    for proof in manifest['theory_source_preservation']:require(sha(destination/proof['file'])==proof['source_sha256'],f'Proof source changed: {proof["file"]}')
    review_target=destination/'provenance/single_delivery_review.json';review_target.parent.mkdir(exist_ok=True);shutil.copy2(review_path,review_target)
    manifest.update(state='reviewed_portable_release',reviewed_utc=datetime.now(timezone.utc).isoformat(),final_review_file=str(review_path),final_review_sha256=sha(review_path),package_main_pdf_sha256=sha(destination/'main.pdf'),package_numerical_review_pdf_sha256=sha(destination/'numerical_review.pdf'),final_checks=review['checks'],archive_filename=archive.name)
    write(destination/'package_manifest.json',manifest);write_readme(destination,reviewed=True);included_files=checksums(destination)
    temporary=archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as output:
            for path in included_files+[destination/'CHECKSUMS.sha256']:output.write(path,arcname=destination.name+'/'+path.relative_to(destination).as_posix())
        with zipfile.ZipFile(temporary) as check:require(check.testzip() is None,'Archive integrity check failed')
        temporary.replace(archive)
    finally:
        if temporary.exists():temporary.unlink()
    archive_sha=sha(archive);archive.with_suffix('.zip.sha256').write_text(archive_sha+'  '+archive.name+'\n')
    return dict(state=manifest['state'],folder=str(destination),archive=str(archive),archive_sha256=archive_sha,archive_bytes=archive.stat().st_size,files=len(included_files)+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    stage=commands.add_parser('prepare');stage.add_argument('--manuscript',type=Path,default=ROOT/'manuscript_single_ready');stage.add_argument('--destination',type=Path,default=DESTINATION)
    final=commands.add_parser('finalize');final.add_argument('--destination',type=Path,default=DESTINATION);final.add_argument('--review',type=Path,default=ROOT/'notes/single_delivery_review.json');final.add_argument('--zip',type=Path,default=ARCHIVE)
    args=parser.parse_args()
    try:result=prepare(args.manuscript,args.destination) if args.mode=='prepare' else finalize(args.destination,args.review,args.zip)
    except (ValueError,KeyError,OSError,IncompleteDataset) as error:print(json.dumps(dict(state='refused',error=str(error))),file=sys.stderr);return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':sys.exit(main())
