"""Read and bind temporary scheduling overrides without changing worker plans."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

METHODS={'proposed','oe_true','oe_low','oe_high','golden_ratio'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def bind(path,expected,hashes):
    path=Path(path).resolve();require(path.is_file() and sha(path)==expected,f'Priority execution evidence hash mismatch: {path}');hashes[str(path)]=expected;return path
def digest_string(value):return isinstance(value,str) and len(value)==64 and all(character in '0123456789abcdef' for character in value)


def load_priority_execution(case,meta,plan_path,hashes,allow_subset_plan=False):
    path=Path(case)/'priority_execution.json'
    if not path.exists():return None
    record=json.loads(path.read_text());plan_path=Path(plan_path).resolve()
    require(record.get('schema_version')==1,'Unsupported priority_execution sidecar schema')
    setting=record.get('setting')
    require(isinstance(setting,dict) and setting.get('n')==meta['n'] and setting.get('m')==meta['m'] and set(setting).issubset({'study','n','m'}) and setting.get('study','multi')=='multi' and (meta['n'],meta['m'])==(100,100),'Priority override setting differs from the selected n100/m100 case')
    require(record.get('original_worker_limit')==7 and record.get('priority_worker_limit')==10 and record.get('numerical_threads_per_worker')==1,'Priority sidecar must record the seven-to-ten single-threaded-worker override')
    require(type(record.get('logical_cpu_count')) is int and record['logical_cpu_count']>=10,'Priority logical CPU count is unavailable or below the configured pool')
    for key in ('memory_budget_gib','modeled_case_resident_gib'):require(type(record.get(key)) in (int,float) and math.isfinite(record[key]) and record[key]>0,f'Invalid priority resource field: {key}')
    require(record.get('configured_pool_semantics') in {'maximum admitted workers, not simultaneous-CPU guarantee','Maximum admitted workers; not a simultaneous-CPU or matched-panel workload guarantee'},'Priority admission/CPU semantics must remain explicit')
    for key in ('observed_peak_priority_workers','observed_active_at_launch'):require(type(record.get(key)) is int and 0<=record[key]<=10,f'Invalid observed priority worker count: {key}')
    require(record['observed_peak_priority_workers']>=record['observed_active_at_launch'],'Observed priority launch count exceeds recorded phase peak')
    phase_path=bind(record['phase_plan_file'],record['phase_plan_sha256'],hashes);phase=json.loads(phase_path.read_text())
    require(digest_string(record.get('original_plan_sha256')) and digest_string(record.get('priority_source_sha256')),'Priority base-plan/source SHA256 evidence is invalid')
    require(phase.get('original_plan_sha256')==record['original_plan_sha256'] and phase.get('priority_source_sha256')==record['priority_source_sha256'],'Phase plan and case sidecar source/base-plan hashes disagree')
    original_path=phase.get('original_plan_file')
    if original_path:bind(original_path,record['original_plan_sha256'],hashes)
    if sha(plan_path)!=record['original_plan_sha256']:
        selected_plan=json.loads(plan_path.read_text())
        linked=selected_plan.get('source_full_plan_sha256')==record['original_plan_sha256'] or selected_plan.get('source_plan_sha256')==record['original_plan_sha256']
        require(allow_subset_plan and (original_path or linked),'Priority base plan differs from the unchanged full dataset plan')
    source_path=phase.get('priority_source_file') or phase.get('priority_source_path')
    require(source_path is not None,'Phase plan must identify the frozen priority source file')
    bind(source_path,record['priority_source_sha256'],hashes)
    committed=record.get('committed_before_launch_methods');scheduled=record.get('methods_scheduled_in_priority')
    require(isinstance(committed,list) and isinstance(scheduled,list) and len(committed)==len(set(committed)) and len(scheduled)==len(set(scheduled)),'Priority method lists are absent or duplicated')
    require(not set(committed)&set(scheduled) and set(committed)|set(scheduled)==METHODS,'Prior-committed and priority-scheduled method sets must be disjoint and cover all five methods')
    pairs=record.get('timed_pair_sha256',{});require(isinstance(pairs,dict) and set(scheduled).issubset(pairs) and set(pairs).issubset(METHODS),'Priority execution sidecar does not bind its scheduled timed pairs')
    for method,pair in pairs.items():
        require(isinstance(pair,dict),'Invalid priority timed-pair hash mapping')
        bind(Path(case)/(method+'_run.json'),pair.get('run',pair.get('run_sha256')),hashes)
        bind(Path(case)/(method+'_points.npz'),pair.get('points',pair.get('points_sha256')),hashes)
    hashes[str(path.resolve())]=sha(path);hashes[str(Path(__file__).resolve())]=sha(__file__)
    return dict(case=Path(case).name,n=meta['n'],m=meta['m'],seed=meta['seed'],sidecar_file=str(path.resolve()),sidecar_sha256=sha(path),phase_plan_file=str(phase_path),phase_plan_sha256=record['phase_plan_sha256'],original_plan_sha256=record['original_plan_sha256'],priority_source_sha256=record['priority_source_sha256'],original_worker_limit=7,priority_worker_limit=10,numerical_threads_per_worker=1,observed_peak_priority_workers=record['observed_peak_priority_workers'],observed_active_at_launch=record['observed_active_at_launch'],committed_before_launch_methods=committed,methods_scheduled_in_priority=scheduled,configured_pool_semantics=record['configured_pool_semantics'],prior_execution=record.get('prior_execution'),recorded_sidecar=record)


def summarize_priority_execution(contexts):
    contexts=[context for context in contexts if context]
    return dict(case_count=len(contexts),priority_method_runs=sum(len(context['methods_scheduled_in_priority']) for context in contexts),previously_committed_method_runs=sum(len(context['committed_before_launch_methods']) for context in contexts),original_worker_limits=sorted({context['original_worker_limit'] for context in contexts}),priority_worker_limits=sorted({context['priority_worker_limit'] for context in contexts}),observed_peak_priority_workers=sorted({context['observed_peak_priority_workers'] for context in contexts}),numerical_threads_per_worker=sorted({context['numerical_threads_per_worker'] for context in contexts}),case_contexts=contexts,admission_is_not_a_simultaneous_cpu_guarantee=True)


def priority_execution_text(summary,paragraph=True):
    if not summary.get('case_count'):return ''
    observed=', '.join(map(str,summary['observed_peak_priority_workers']))
    text=(str(summary['priority_method_runs'])+' completed method trajectories in the $n=100,m=100$ priority phase were scheduled with a temporary admission ceiling of ten workers, overriding the original seven-worker ceiling. Each worker uses one numerical thread. The sidecars record observed phase-pool peak counts of '+observed+' active worker processes; admission limits do not guarantee simultaneous CPU execution. '+str(summary['previously_committed_method_runs'])+' previously committed trajectories retain their earlier execution context. The original plan is unchanged, and its regular-grid ceiling remains seven. Per-case sidecars preserve the scheduled-method split, earlier context, phase/source hashes, and timed-pair identities.\n')
    return (r'\paragraph{Priority execution phase.}'+'\n' if paragraph else '')+text
