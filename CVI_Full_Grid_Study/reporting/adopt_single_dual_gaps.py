#!/usr/bin/env python3
"""Review eight independent m=1 dual-gap sidecars; dry-run unless --apply.

Only the copied results_single/verified diagnostics and diagnostic counters may
change. No solver runs here. Ordinary-float primal/dual bounds are not formal
interval certificates. --quality-reviewed records a separate explicit review;
it is never inferred from an optimizer success flag or posterior checks.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
from report_full_grid import load_complete,IncompleteDataset

ROOT=Path(__file__).resolve().parents[1]
ADOPTED_STATUS='independent_dual_bracket_valid'
TOLERANCE=1e-8


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,indent=2,allow_nan=True)+'\n').encode()
def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=True)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def require(condition,message):
    if not condition:raise ValueError(message)
def point_hash(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()


def bound_file(path,expected,inputs):
    path=Path(path).resolve()
    require(path.is_file() and sha(path)==expected,f'File identity mismatch: {path}')
    inputs[str(path)]=expected
    return path


def check_quality(record):
    require(record.get('gap_valid') is True and finite(record.get('gap')) and record['gap']>=0,'Sidecar has no valid finite nonnegative numerical gap')
    require(record.get('status')=='numerical_primal_dual_bracket_valid','Unexpected independent dual sidecar status')
    feasibility=record['full_feasibility'];quality=record['numerical_quality']
    for key in ('max_violation','constraint_max','ball_excess'):require(finite(feasibility.get(key)),f'Nonfinite full feasibility {key}')
    require(0<=feasibility['max_violation']<=TOLERANCE,'Full original feasibility exceeds 1e-8')
    require(abs(feasibility['max_violation']-max(0.,feasibility['constraint_max'],feasibility['ball_excess']))<=1e-12,'Full feasibility arithmetic disagrees')
    for key in ('objective_lower','objective_upper','primal_dual_width','gap_lower','gap_upper','stationarity_l2','stationarity_inf','root_constraint_residual_inf','complementarity_inf'):
        require(finite(quality.get(key)),f'Nonfinite numerical quality {key}')
    width=quality['objective_upper']-quality['objective_lower']
    require(0<=width<=TOLERANCE and 0<=quality['primal_dual_width']<=TOLERANCE and abs(width-quality['primal_dual_width'])<=1e-12,'Primal-dual width exceeds 1e-8 or arithmetic disagrees')
    require(0<=quality['stationarity_l2']<=TOLERANCE,'Stationarity L2 exceeds 1e-8')
    require(0<=quality['stationarity_inf']<=TOLERANCE and 0<=quality['root_constraint_residual_inf']<=TOLERANCE and 0<=quality['complementarity_inf']<=TOLERANCE,'Posterior KKT residual exceeds 1e-8')
    require(quality.get('multipliers_nonnegative') is True and len(quality.get('multipliers',[]))==2 and all(finite(x) and x>=0 for x in quality['multipliers']),'Dual multipliers are not two finite nonnegative values')
    checks=quality.get('checks',{})
    for key in ('finite','positive_multipliers','full_feasible','root_constraint_residual','stationarity','complementarity','primal_dual_width'):require(checks.get(key) is True,f'Posterior check did not pass: {key}')
    require(0<=quality['gap_lower']<=record['gap']+1e-12 and record['gap']<=quality['gap_upper']+1e-12 and quality['gap_upper']-quality['gap_lower']<=TOLERANCE,'Gap is inconsistent with its numerical primal-dual interval')
    require(finite(record.get('raw_gap')) and abs(max(0.,record['raw_gap'])-record['gap'])<=1e-12,'Raw and clipped gap disagree')
    require(finite(record.get('raw_objective')) and abs(record['raw_objective']-quality['objective_upper'])<=1e-12,'Primal objective disagrees with quality bound')
    for target in (1e-2,1e-3,1e-4,1e-5):require(not quality['gap_lower']<=target<quality['gap_upper'],f'Numerical gap interval straddles paper target {target}')
    require(record.get('solver',{}).get('name')=='two_multiplier_dual_root','Unexpected independent solver')
    require(finite(record.get('seconds')) and record['seconds']>=0 and record.get('budget_seconds')==300,'Invalid independent diagnostic duration/budget')
    return dict(full_feasibility=feasibility['max_violation'],primal_dual_width=quality['primal_dual_width'],stationarity_l2=quality['stationarity_l2'],gap_lower=quality['gap_lower'],gap_upper=quality['gap_upper'],quality_passed=True,optimizer_success=record['solver'].get('success'),optimizer_status=record['solver'].get('status'))


def prepare(results,inventory,v2_summary,quality_reviewed=False,review_file=None):
    """Return an exact reviewed write set without changing any primary file."""
    results=Path(results).resolve();inventory=Path(inventory).resolve();v2_summary=Path(v2_summary).resolve()
    require(not (ROOT/'results_single').is_symlink() and not (ROOT/'results_single/verified').is_symlink(),'Copied release directories must not be symlinks to original data')
    require(results==(ROOT/'results_single/verified').resolve(),'Target must be this study\'s copied results_single/verified directory')
    require(not (results/'diagnostic_adoption_manifest.json').exists(),'An adoption manifest already exists; inspect it rather than silently re-adopting')
    plan,loaded,inputs=load_complete(results,single_only=True)
    code=ROOT/'code';actual_source={p.name:sha(p) for p in sorted(code.glob('*.py'))}
    require(actual_source==plan['source_hashes'],'Original frozen worker source hash set changed')
    inputs.update({str(p.resolve()):sha(p) for p in code.glob('*.py')})
    inv=read(inventory);summary=read(v2_summary);inputs[str(inventory)]=sha(inventory);inputs[str(v2_summary)]=sha(v2_summary)
    allowed={(r['case'],r['method'],r['checkpoint']):r for r in inv['invalid_diagnostics']}
    require(len(allowed)==len(inv['invalid_diagnostics'])==8 and not any(r['gap_valid'] for r in allowed.values()),'Authorization inventory must contain exactly eight distinct invalid rows')
    invalid={(item['case'],item['method'],r['checkpoint']) for item in loaded for r in item['checkpoints'] if not r['gap_valid']}
    require(invalid==set(allowed),'Current invalid rows differ from the eight authorized inventory rows')
    require(summary.get('state')=='attempts_finished' and summary.get('completed')==summary.get('expected')==summary.get('valid')==8 and summary.get('timed_methods_run')==0 and summary.get('source_diagnostics_modified') is False,'V2 summary is not the complete eight-row sidecar-only attempt')
    require(summary.get('inventory_sha256')==sha(inventory),'V2 inventory hash mismatch')
    entries={(e['case'],e['method'],e['checkpoint']):e for e in summary['results']}
    require(len(entries)==len(summary['results'])==8 and set(entries)==set(allowed),'V2 sidecar row set differs from authorized inventory')
    if review_file:
        review_file=Path(review_file).resolve();external_review=read(review_file)
        require(external_review.get('decision')=='approved_for_copied_single_release_only' and external_review.get('expected_records')==8 and external_review.get('summary_sha256')==sha(v2_summary),'External quality review does not approve this exact copied-release summary')
        reviewed={(r['case'],r['method'],r['checkpoint']):r for r in external_review['records']}
        require(len(reviewed)==8 and set(reviewed)==set(entries) and all(reviewed[key]['diagnostic_sha256']==entries[key]['diagnostic_sha256'] for key in entries),'External quality review sidecar identities differ')
        inputs[str(review_file)]=sha(review_file)
    original_bytes={};documents={};changes=[]
    for key in sorted(allowed):
        case_name,method,index=key;case=results/case_name;meta=read(case/'manifest.json');run=read(case/f'{method}_run.json')
        require(not case.is_symlink() and case.resolve().parent==results,'Copied case must not resolve outside results_single/verified')
        diagpath=case/f'{method}_diagnostics.json';pointpath=case/f'{method}_points.npz'
        require(not diagpath.is_symlink() and not (case/'manifest.json').is_symlink(),'Adopted diagnostic/manifest must be copied regular files')
        if diagpath not in documents:original_bytes[diagpath]=diagpath.read_bytes();documents[diagpath]=read(diagpath)
        old=deepcopy(documents[diagpath][index]);entry=entries[key]
        sidepath=bound_file(entry['diagnostic_file'],entry['diagnostic_sha256'],inputs);record=read(sidepath)
        require((record.get('case'),record.get('method'),record.get('checkpoint'))==key and (record.get('n'),record.get('m'),record.get('seed'))==(meta['n'],1,meta['seed']),'Sidecar case/checkpoint identity mismatch')
        require(record.get('instance_hash_verified') is True and record.get('instance_hash')==meta['instance_hash'] and record.get('source_hashes')==meta['source_hashes']==plan['source_hashes'],'Sidecar instance/source identity mismatch')
        require(record.get('points_file_sha256')==sha(pointpath)==run['points_sha256'] and record.get('original_diagnostics_sha256')==sha(diagpath),'Sidecar saved-point/original-diagnostic file identity mismatch')
        with np.load(pointpath,allow_pickle=False) as archive:input_hash=point_hash(archive['x'][index])
        require(input_hash==record.get('input_point_hash')==allowed[key]['point_sha256'],'Saved input point identity mismatch')
        require(canonical(record.get('original_diagnostic'))==canonical(old),'Sidecar did not preserve the exact original primary diagnostic')
        guard=record.get('completion_guard',{})
        require(guard.get('plan_sha256')==sha(results/'plan.json') and guard.get('verified_snapshot') is True and guard.get('release_scope')=='single_only','Sidecar completion/plan guard mismatch')
        for name,value in guard.get('manifest_and_data_sha256',{}).items():bound_file(case/name,value,inputs)
        require(record.get('recovery_source_sha256')==summary['source_sha256'] and record.get('recovery_module_sha256')==summary['recovery_module_sha256'],'V2 source summary hashes disagree')
        for stem in ('recovery_source','recovery_module','original_solver_output','candidate_binding_source'):
            bound_file(record[stem+'_path'] if stem+'_path' in record else record[stem+'_file'],record[stem+'_sha256'],inputs)
        candidate=bound_file(record['candidate_file'],record['candidate_file_sha256'],inputs)
        with np.load(candidate,allow_pickle=False) as archive:
            require(set(archive.files)=={'x'} and archive['x'].shape==(meta['n'],) and np.all(np.isfinite(archive['x'])),'Invalid independent optimizer candidate archive')
            require(point_hash(archive['x'])==record['solution_hash']==record['feasible_candidate_hash'],'Independent optimizer candidate hash mismatch')
        prior=record['prior_attempt'];priorpath=bound_file(prior['benchmark_file'],prior['benchmark_sha256'],inputs);benchmark=read(priorpath)
        prior_rows=[r for r in benchmark['results'] if r.get('checkpoint')==index]
        require(len(prior_rows)==1 and canonical(prior_rows[0])==canonical(prior['diagnostic']) and prior_rows[0].get('valid') is False and prior_rows[0].get('status')=='restricted_slsqp_status_8','Full failed v1 attempt is missing or changed')
        require(finite(prior['diagnostic'].get('seconds')) and prior['diagnostic']['seconds']>=0,'Invalid v1 duration')
        metrics=check_quality(record)
        provenance=dict(kind='independent_single_constraint_dual_v2',sidecar_file=str(sidepath),sidecar_sha256=sha(sidepath),summary_file=str(v2_summary),summary_sha256=sha(v2_summary),candidate_file=str(candidate),candidate_file_sha256=sha(candidate),solution_hash=record['solution_hash'],input_point_hash=input_hash,instance_hash=meta['instance_hash'],source_hashes=plan['source_hashes'],recovery_source_sha256=record['recovery_source_sha256'],recovery_module_sha256=record['recovery_module_sha256'],quality_passed=True,quality_reviewed_by_explicit_cli=bool(quality_reviewed),quality_limits=dict(full_feasibility=TOLERANCE,primal_dual_width=TOLERANCE,stationarity_l2=TOLERANCE),numerical_quality=metrics,formal_interval_certificate=False)
        new=deepcopy(old);history=deepcopy(old.get('attempt_history',[]))
        history.extend([dict(attempt_kind='original_primary',full_diagnostic=old),dict(attempt_kind='constraint_generation_v1_failed',benchmark_file=str(priorpath),benchmark_sha256=sha(priorpath),full_diagnostic=prior['diagnostic']),dict(attempt_kind='independent_dual_v2',sidecar_file=str(sidepath),sidecar_sha256=sha(sidepath),full_diagnostic=record)])
        new.update(gap=record['gap'],gap_valid=True,gap_status=ADOPTED_STATUS,residual=max(record['gap'],old['violation']),diagnostic_error=None,diagnostic_attempt=old.get('diagnostic_attempt',1)+2,attempt_history=history,diagnostic_adoption=provenance,diagnostic_seconds=old['diagnostic_seconds']+prior['diagnostic']['seconds']+record['seconds'],diagnostic_seconds_policy='sum of original primary, failed v1, and accepted v2 diagnostic solve durations; setup recorded separately in sidecars',solver_diagnostics=dict(status=ADOPTED_STATUS,valid=True,backend='independent_two_multiplier_dual_root_posterior_bracket',raw_optimizer=record['solver'],raw_objective=record['raw_objective'],full_feasibility=record['full_feasibility'],numerical_quality=record['numerical_quality']))
        require(finite(old.get('violation')) and old['violation']>=0,'Original saved output violation is unavailable')
        documents[diagpath][index]=new
        changes.append(dict(case=case_name,method=method,checkpoint=index,old_status=old['gap_status'],new_status=ADOPTED_STATUS,gap=record['gap'],quality=metrics,provenance=provenance))
    # Only diagnostic counters change inside each affected original manifest.
    for case_name in sorted({c['case'] for c in changes}):
        case=results/case_name;path=case/'manifest.json';original_bytes[path]=path.read_bytes();meta=read(path);all_rows=[]
        for method in plan['methods']:
            rows=documents.get(case/f'{method}_diagnostics.json')
            if rows is None:rows=read(case/f'{method}_diagnostics.json')
            meta['methods'][method]['invalid_diagnostics']=sum(not row['gap_valid'] for row in rows);all_rows.extend(rows)
        meta['invalid_diagnostic_count']=sum(not row['gap_valid'] for row in all_rows);meta['diagnostic_status_counts']=dict(Counter(row['gap_status'] for row in all_rows));documents[path]=meta
    writes={path:encoded(value) for path,value in documents.items()}
    for path in documents:
        if path.name.endswith('_diagnostics.json'):
            before=json.loads(original_bytes[path]);after=documents[path];method=path.name.removesuffix('_diagnostics.json')
            changed_indices={index for index,(old,new) in enumerate(zip(before,after)) if canonical(old)!=canonical(new)}
            expected_indices={index for case_name,method_name,index in allowed if case_name==path.parent.name and method_name==method}
            require(len(before)==len(after) and changed_indices==expected_indices,'Write set changes an unapproved diagnostic row')
    review=dict(state='dry_run_ready',created_utc=datetime.now(timezone.utc).isoformat(),target=str(results),quality_passed=True,quality_reviewed_by_explicit_cli=bool(quality_reviewed),apply_eligible=bool(quality_reviewed),rows=len(changes),changes=changes,write_set=[dict(path=str(path),before_sha256=sha(path),after_sha256=hashlib.sha256(value).hexdigest()) for path,value in sorted(writes.items())],all_other_rows_and_timed_pairs_unchanged=True,plan_unchanged=True,inventory_file=str(inventory),v2_summary=str(v2_summary),helper_sha256=sha(__file__),input_sha256=inputs,formal_interval_certificate=False,limitation='Only these eight numerical gaps receive selected independent dual-gap evaluation; the remaining gap values are not independently re-solved by this procedure.')
    if review_file:review.update(external_quality_review_file=str(review_file),external_quality_review_sha256=sha(review_file))
    require(all(Path(path).is_file() and sha(path)==value for path,value in inputs.items()),'Input changed during dry-run review')
    return review,writes,original_bytes


def atomic_bytes(path,data):
    temporary=path.with_name(path.name+'.adoption-'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def apply(review,writes,original_bytes):
    require(review['quality_reviewed_by_explicit_cli'],'--apply requires the separate explicit --quality-reviewed flag')
    results=Path(review['target']);lock=results/'.diagnostic_adoption.lock'
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(descriptor)
    backup=results.parent/'diagnostic_adoption_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    changed=[]
    try:
        require(all(sha(path)==value for path,value in review['input_sha256'].items()),'Reviewed input changed before apply')
        backup.mkdir(parents=True,exist_ok=False)
        (backup/'adoption_helper_snapshot.py').write_bytes(Path(__file__).read_bytes())
        for path,data in original_bytes.items():
            destination=backup/path.relative_to(results);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        (backup/'dry_run_review.json').write_bytes(encoded(review))
        for path,data in sorted(writes.items()):atomic_bytes(path,data);changed.append(path)
        protected={path:value for path,value in review['input_sha256'].items() if Path(path) not in writes}
        require(all(sha(path)==value for path,value in protected.items()),'Protected source/timed/other-diagnostic input changed during apply')
        require(all(sha(path)==hashlib.sha256(data).hexdigest() for path,data in writes.items()),'Written diagnostic/manifest verification failed')
        result=dict(review,state='applied',applied_utc=datetime.now(timezone.utc).isoformat(),backup_directory=str(backup),helper_snapshot_file=str(backup/'adoption_helper_snapshot.py'),helper_snapshot_sha256=sha(backup/'adoption_helper_snapshot.py'),input_sha256_before=review['input_sha256'],input_sha256_after={path:sha(path) for path in review['input_sha256']},timed_pairs_unchanged=True,all_other_valid_rows_unchanged=True)
        result.pop('input_sha256',None)
        atomic_bytes(results/'diagnostic_adoption_manifest.json',encoded(result))
        return result
    except BaseException:
        for path in reversed(changed):atomic_bytes(path,original_bytes[path])
        raise
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results_single/verified')
    parser.add_argument('--inventory',type=Path,default=ROOT/'notes/single_delivery_inventory.json')
    parser.add_argument('--v2-summary',type=Path,default=ROOT/'reporting/recovery/single_delivery_v2/summary.json')
    parser.add_argument('--report',type=Path,help='Optional new dry-run/apply review JSON; refuses to overwrite')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--quality-reviewed',action='store_true',help='Explicit root review authorization, separate from computed quality checks')
    parser.add_argument('--review-file',type=Path,help='Optional hash-bound independent/root review JSON for the exact eight sidecars')
    args=parser.parse_args()
    committed=False
    try:
        if args.report:
            require(not args.report.exists(),'Refusing to overwrite review output')
            reportpath=args.report.resolve()
            require(not any(reportpath.is_relative_to(ROOT/name) for name in ('results','code')),'Review output must stay outside original data and frozen code')
        if args.apply:require(args.quality_reviewed,'--apply requires --quality-reviewed after independent/root review')
        review,writes,originals=prepare(args.results,args.inventory,args.v2_summary,args.quality_reviewed,args.review_file)
        if args.apply:review=apply(review,writes,originals);committed=True
        if args.report:args.report.write_bytes(encoded(review))
        print(json.dumps({key:review[key] for key in ('state','target','rows','quality_passed','quality_reviewed_by_explicit_cli','apply_eligible','changes')},indent=2,allow_nan=False))
        return 0
    except (ValueError,KeyError,OSError,IncompleteDataset) as error:
        print(json.dumps(dict(state='refused_after_commit' if committed else 'refused',error=str(error),primary_files_modified=committed)),file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
