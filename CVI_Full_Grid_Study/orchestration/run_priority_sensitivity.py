#!/usr/bin/env python3
"""Untimed priority phase for n=100,m=100; original timed source stays frozen.

This runner changes queue order only. It uses the original case paths/budgets,
exports completed cases, runs an explicitly supplied audit/plot command, then
executes the unchanged original wrapper (which adds controller --resume).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep
from types import SimpleNamespace

import batch_n100 as original

ROOT=Path(__file__).resolve().parents[1]
SETTING=dict(study="multi",n=100,m=100)
TARGETS={original.name(SETTING,seed):seed for seed in range(10)}
CURRENT_STATE=None


def require(condition,message):
    if not condition:raise RuntimeError(message)


def priority_admission(admission,workers):
    require(1<=workers<=min(10,os.cpu_count() or 1),"Priority pool exceeds the ten cases or available CPU count")
    require(workers*original.weight(SETTING)["resident_gib"]<=admission.memory_budget_gib,
            "Priority pool exceeds the unchanged 8-GiB modeled RAM allowance")
    copied=SimpleNamespace(**vars(admission));copied.workers=workers
    return copied


def phase_records(note):
    controller=note.get("original_controller",{})
    wrapper=note.get("original_wrapper",{})
    if not controller:controller=dict(pid=note.get("original_controller_pid",note.get("controller_pid")))
    if not wrapper:wrapper=dict(pid=note.get("original_wrapper_pid",note.get("wrapper_pid")))
    workers=note.get("interrupted_workers",note.get("stopped_workers",note.get("owned_worker_records")))
    if workers is None:
        workers=list(note.get("original_progress_snapshot",{}).get("worker_identity_records",{}).values())
    if isinstance(workers,dict):workers=list(workers.values())
    require(controller.get("pid") and wrapper.get("pid"),"Transition must identify original wrapper/controller PIDs")
    require(isinstance(workers,list),"Transition must identify every interrupted original worker")
    return controller,wrapper,workers


def assert_quiescent(note,results):
    require(note.get("original_queue_quiescent",note.get("quiescent",False)),
            "Root has not confirmed controlled original-queue interruption")
    controller,wrapper,workers=phase_records(note)
    for role,record in [("controller",controller),("wrapper",wrapper)]+[("worker",r) for r in workers]:
        require(record.get("pid"),f"Missing {role} PID")
        state=original.process_state(record["pid"],record)
        require(not state["alive"],f"Original {role} {record['pid']} remains alive/unknown: {state}")
    current=original.load(results/"progress.json") if (results/"progress.json").exists() else {}
    current_pid=current.get("controller_pid")
    if current_pid and current_pid!=controller["pid"]:
        require(not original.process_state(current_pid)["alive"],"A newer original controller is active/unknown")
    # Include records written just before the controlled stop, even if a root
    # snapshot omitted a late admission. Never start over a live old actor.
    for record in original.load(results/"launch_ledger.json").values():
        if record.get("priority_phase"):
            continue
        if record.get("state") in {"launching","running"}:
            require(record.get("pid") is not None,"Original queue has an unresolved PID-less launch intent")
            require(not original.process_state(record["pid"],record)["alive"],
                    "An original queue worker is still active/unknown")


def assert_original_compatibility(results,note):
    plan=original.load(results/"plan.json")
    expected_hash=note.get("original_plan_sha256",note.get("plan_sha256"))
    require(expected_hash==original.digest(results/"plan.json"),"Original plan checksum differs from phase authorization")
    require(plan["source_hashes"]==original.source_hashes(),"Frozen timed source changed")
    require(plan.get("orchestrator_hashes",{}).get("batch_n100.py")==original.digest(original.__file__),
            "Original controller changed")
    require(plan["methods"]==original.METHODS and plan["seeds"]==list(range(10))
            and plan["seconds"]==1000. and plan["diagnostic_seconds"]==300.,"Original measurement contract differs")
    require(SETTING in plan["settings"],"Priority target is absent from the original plan")
    grid_note=Path(plan["plan_amendment"]["transition_file"])
    require(original.digest(grid_note)==plan["plan_amendment"]["transition_sha256"],"Frozen grid-transition note changed")
    transition,source,oldplan=original.validate_source(grid_note)
    policy=plan["execution_policy"]
    args=SimpleNamespace(workers=plan["workers"],memory_budget_gib=policy["memory_budget_gib"],
                         min_free_disk_gib=policy["min_free_disk_gib"],heavy_factor_gib=policy["heavy_factor_threshold_gib"])
    require(original.make_plan(args,grid_note,transition,source,oldplan)==plan,
            "Unchanged original controller --resume would reject the plan")
    wrapper=ROOT/"run_n100_study.py"
    require(note.get("original_wrapper_sha256",note.get("wrapper_sha256"))==original.digest(wrapper),
            "Original wrapper checksum differs from root authorization")
    return plan,args,wrapper


def input_fingerprints(left,plan):
    leftplan=original.load(left/"plan.json")
    require(leftplan.get("release_scope")=="single_only","Expected refined single-only left data")
    require(leftplan["source_hashes"]==plan["source_hashes"] and leftplan["seconds"]==1000.,
            "Left data source/budget differs")
    out={}
    for seed in range(10):
        case=left/f"single_n1000_m1_seed{seed}"
        _,files=original.verify_case(case,leftplan,dict(study="single",n=1000,m=1),seed)
        out[case.name]=files
    return out


def make_phase_plan(args,note,plan,wrapper):
    render=note.get("render_command")
    require(isinstance(render,list) and render and all(isinstance(v,str) for v in render),
            "Transition requires an explicit audit/plot render_command argv")
    figure=Path(note["figure_output"]).resolve()
    return dict(phase="priority_two_panel_sensitivity_v1",target_setting=SETTING,seeds=list(range(10)),
        methods=original.METHODS,method_seconds=1000.,diagnostic_seconds=300.,target_cases=10,target_method_runs=50,
        original_results=str(args.results.resolve()),original_plan_file=str((args.results/"plan.json").resolve()),
        original_plan_sha256=original.digest(args.results/"plan.json"),
        original_source_hashes=plan["source_hashes"],original_orchestrator_hashes=plan["orchestrator_hashes"],
        original_wrapper=str(wrapper),original_wrapper_sha256=original.digest(wrapper),
        priority_source_file=str(args.state_dir.resolve()/"runner_snapshot.py"),
        priority_source_sha256=original.digest(__file__),transition_file=str(args.transition.resolve()),
        transition_sha256=original.digest(args.transition),left_results=str(args.left_results.resolve()),
        left_case_artifact_sha256=input_fingerprints(args.left_results,plan),render_command=render,
        figure_output=str(figure),auto_resume_original=True,
        execution_policy=dict(original_worker_limit=plan["workers"],priority_worker_limit=args.priority_workers,
            restored_original_worker_limit=plan["workers"],logical_cpu_count=os.cpu_count(),
            numerical_threads_per_worker=1,memory_budget_gib=plan["execution_policy"]["memory_budget_gib"],
            modeled_case_resident_gib=original.weight(SETTING)["resident_gib"],
            modeled_priority_pool_gib=args.priority_workers*original.weight(SETTING)["resident_gib"],
            semantics="Priority-only maximum pool override; original plan/source and resumed seven-worker queue unchanged"),
        interruption_policy="Queue interruptions are not method failures; preserve committed pairs and all observed errors")


def write_ledgers(state,results,own,record):
    """Both journals are durable before spawn; original --resume sees orphans."""
    own[record["case"]]=dict(record)
    original.atomic_json(state/"launch_ledger.json",own)
    compatible=original.load(results/"launch_ledger.json")
    compatible[record["case"]]=dict(record)
    original.atomic_json(results/"launch_ledger.json",compatible)


def reconcile_ledgers(own,compatible):
    for case,record in list(own.items()):
        require(case in TARGETS,"Priority ledger contains a nontarget case")
        if record.get("state")=="launching" and record.get("pid") is None:
            other=compatible.get(case,{})
            if other.get("intent_id")==record.get("intent_id") and other.get("pid") is not None:
                own[case]=dict(other)
            else:
                raise RuntimeError(f"Unresolved launch intent for {case}; reconcile actor identity before any resume")
    return own


def committed_snapshot(case,plan,seed):
    if not (case/"manifest.json").exists():return dict(pairs={},diagnostic_prefix={})
    original.verify_case(case,plan,SETTING,seed,complete=False)
    pairs={};prefix={}
    for method in original.METHODS:
        run,points=case/f"{method}_run.json",case/f"{method}_points.npz"
        if run.exists() and points.exists():pairs[method]={run.name:original.digest(run),points.name:original.digest(points)}
        diag=case/f"{method}_diagnostics.json"
        if diag.exists():prefix[method]=original.load(diag)
    return dict(pairs=pairs,diagnostic_prefix=prefix)


def verify_preserved(case,before):
    for hashes in before["pairs"].values():
        require(all(original.digest(case/f)==h for f,h in hashes.items()),"A committed timed pair changed on resume")
    for method,rows in before["diagnostic_prefix"].items():
        now=original.load(case/f"{method}_diagnostics.json")[:len(rows)]
        require(json.dumps(now,sort_keys=True)==json.dumps(rows,sort_keys=True),"A saved diagnostic attempt/prefix changed")


def update(state,phase,**data):
    original.atomic_json(state/"progress.json",dict(state=phase,updated_utc=original.utc(),
        priority_pid=os.getpid(),target_cases=10,target_method_runs=50,**data))


def execution_sidecar(case,phase_file,phase_plan,before,peak,active_at_launch=None,reused_complete=False):
    file=case/"priority_execution.json"
    if active_at_launch is not None or not file.exists():
        previous=original.load(file) if file.exists() else None
        value=dict(schema_version=1,phase_plan_file=str(phase_file),phase_plan_sha256=original.digest(phase_file),
            original_plan_sha256=phase_plan["original_plan_sha256"],priority_source_sha256=phase_plan["priority_source_sha256"],
            original_plan_file=phase_plan["original_plan_file"],priority_source_file=phase_plan["priority_source_file"],
            setting=SETTING,original_worker_limit=phase_plan["execution_policy"]["original_worker_limit"],
            priority_worker_limit=phase_plan["execution_policy"]["priority_worker_limit"],
            logical_cpu_count=os.cpu_count(),numerical_threads_per_worker=1,
            memory_budget_gib=phase_plan["execution_policy"]["memory_budget_gib"],
            modeled_case_resident_gib=original.weight(SETTING)["resident_gib"],
            configured_pool_semantics="Maximum admitted workers; not a simultaneous-CPU or matched-panel workload guarantee",
            observed_active_at_launch=None,planned_pool_size_at_launch=active_at_launch,
            committed_before_launch_methods=sorted(before["pairs"]),
            methods_scheduled_in_priority=[] if reused_complete else [m for m in original.METHODS if m not in before["pairs"]],
            measurement_origin="completed_before_priority_phase" if reused_complete else "priority_phase_with_committed_pairs_reused",
            prior_execution=previous,method_seconds=1000.,diagnostic_seconds=300.,
            left_pfaos_data_rerun=False,updated_utc=original.utc())
    else:value=original.load(file)
    value["observed_peak_priority_workers"]=max(value.get("observed_peak_priority_workers",0),peak)
    value["observed_peak_semantics"]="Largest observed active priority pool by this case's export; not per-method CPU occupancy"
    value["timed_pair_sha256"]={}
    for method in original.METHODS:
        run,points=case/f"{method}_run.json",case/f"{method}_points.npz"
        if run.exists() and points.exists():
            value["timed_pair_sha256"][method]={"run":original.digest(run),"points":original.digest(points)}
    value["updated_utc"]=original.utc()
    original.atomic_json(file,value)


def finish_export(case,results,plan,seed,before,phase_file,phase_plan,peak):
    original.verify_case(case,plan,SETTING,seed)
    verify_preserved(case,before)
    execution_sidecar(case,phase_file,phase_plan,before,peak,reused_complete=not (case/"priority_execution.json").exists())
    original.copy_complete(case,results/"verified"/case.name,plan,SETTING,seed)


def main():
    global CURRENT_STATE
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,default=ROOT/"results_n100")
    ap.add_argument("--left-results",type=Path,default=ROOT/"results_single/verified")
    ap.add_argument("--transition",type=Path,default=ROOT/"notes/priority_sensitivity_transition.json")
    ap.add_argument("--state-dir",type=Path,default=ROOT/"priority_sensitivity")
    ap.add_argument("--priority-workers",type=int,default=10)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--validate-only",action="store_true")
    args=ap.parse_args();args.results=args.results.resolve();args.left_results=args.left_results.resolve()
    note=original.load(args.transition)
    require(note.get("priority_workers",args.priority_workers)==args.priority_workers,
            "Priority pool differs from explicit phase authorization")
    plan,admission,wrapper=assert_original_compatibility(args.results,note)
    # Override only this independent priority admission object; never the plan
    # or original controller's seven-worker configuration.
    admission=priority_admission(admission,args.priority_workers)
    phase_plan=make_phase_plan(args,note,plan,wrapper)
    if args.validate_only:
        assert_quiescent(note,args.results)
        print(json.dumps(dict(validation="passed",target_cases=10,target_method_runs=50,processes_launched=0,
              original_resume_plan_compatible=True,source_hashes_unchanged=True)),flush=True);return
    state=args.state_dir.resolve();state.mkdir(parents=True,exist_ok=True)
    phase_lock=(state/"controller.lock").open("a+")
    try:fcntl.flock(phase_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("Another priority controller is active")
    CURRENT_STATE=state
    phase_file=state/"plan.json"
    if phase_file.exists():
        require(args.resume,"Use --resume for an existing priority phase")
        require(original.load(phase_file)==phase_plan,"Priority/source/authorization identity changed")
    snapshot=Path(phase_plan["priority_source_file"])
    if snapshot.exists():require(original.digest(snapshot)==phase_plan["priority_source_sha256"],"Priority source snapshot changed")
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    original.atomic_json(phase_file,phase_plan)
    own=original.load(state/"launch_ledger.json") if (state/"launch_ledger.json").exists() else {}
    own=reconcile_ledgers(own,original.load(args.results/"launch_ledger.json"))
    # Orphans inherit the original controller-lock descriptor. A resumed priority
    # runner waits read-only for their exit; it cannot admit duplicate actors.
    original_lock=(args.results/"controller.lock").open("a+")
    while True:
        assert_quiescent(note,args.results)
        try:fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            alive=[r for r in own.values() if r.get("pid") and r.get("state") in {"running","launching"}
                   and original.process_state(r["pid"],r)["alive"]]
            require(alive,"Original controller lock is held by an unknown actor")
            update(state,"waiting_for_priority_orphans",active=[r["case"] for r in alive]);sleep(2)
    assert_quiescent(note,args.results)
    pending={};done=set();running={};failures=[]
    previous_progress=original.load(state/"progress.json") if (state/"progress.json").exists() else {}
    peak=previous_progress.get("observed_peak_priority_workers",0)
    for case_name,seed in TARGETS.items():
        case=args.results/"working"/case_name;target=args.results/"verified"/case_name
        if target.exists():original.verify_case(target,plan,SETTING,seed);done.add(case_name);continue
        record=own.get(case_name)
        if record and record.get("pid") and record.get("state") in {"running","launching"}:
            observation=original.process_state(record["pid"],record)
            require(not observation["alive"],"Priority orphan still alive after lock acquisition")
            record=dict(record,state="exited",returncode=None,observed_exit_utc=original.utc(),
                        exit_observation=observation,exitcode_note="Prior owner unavailable; no method-failure inference")
            write_ledgers(state,args.results,own,record)
        before=committed_snapshot(case,plan,seed)
        if (case/"manifest.json").exists() and original.load(case/"manifest.json")["state"]=="complete":
            finish_export(case,args.results,plan,seed,before,phase_file,phase_plan,peak);done.add(case_name)
        else:pending[case_name]=dict(case=case,seed=seed,before=before)
    interrupted=False;env=dict(os.environ,**original.THREAD_ENV)
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while pending or running:
        if interrupted:
            for e in running.values():e["process"].terminate()
            for case_name,e in running.items():
                try:e["process"].wait(timeout=15)
                except subprocess.TimeoutExpired:e["process"].kill();e["process"].wait()
                e["log"].close();record=dict(own[case_name],state="exited",returncode=e["process"].returncode,
                    interruption="priority_controller_interruption_not_method_failure",observed_exit_utc=original.utc())
                write_ledgers(state,args.results,own,record)
            update(state,"interrupted",completed=len(done),interruption="not_a_method_failure")
            raise SystemExit(130)
        for case_name,e in list(running.items()):
            if e["process"].poll() is None:continue
            e["log"].close()
            record=dict(own[case_name],state="exited",returncode=e["process"].returncode,observed_exit_utc=original.utc())
            write_ledgers(state,args.results,own,record)
            try:
                require(e["process"].returncode==0,f"Priority worker process exited {e['process'].returncode}")
                finish_export(e["case"],args.results,plan,e["seed"],e["before"],phase_file,phase_plan,peak);done.add(case_name)
            except Exception as error:failures.append(dict(case=case_name,error=repr(error)))
            del running[case_name]
        for case_name in list(pending):
            active={k:dict(setting=SETTING) for k in running}
            if not original.admissible(SETTING,active,admission):break
            if shutil.disk_usage(args.results).free/original.GIB < admission.min_free_disk_gib+original.weight(SETTING)["factor_gib"]:break
            assert_quiescent(note,args.results)
            require(original.source_hashes()==plan["source_hashes"],"Frozen worker source changed before priority admission")
            e=pending[case_name];case=e["case"];case.mkdir(parents=True,exist_ok=True)
            resume=(case/"manifest.json").exists()
            if resume:original.verify_case(case,plan,SETTING,e["seed"],complete=False)
            command=[sys.executable,str(original.CODE/"study.py"),"--n","100","--m","100",
                "--seed",str(e["seed"]),"--seconds","1000.0","--diagnostic-seconds","300.0",
                "--out",str(case),"--methods",",".join(original.METHODS)]
            if resume:command.append("--resume")
            execution_sidecar(case,phase_file,phase_plan,e["before"],peak,
                              active_at_launch=len(running)+1)
            record=dict(case=case_name,path=str(case),pid=None,state="launching",intent_id=secrets.token_hex(16),
                intent_utc=original.utc(),command=command,priority_phase=str(phase_file),
                priority_plan_sha256=original.digest(phase_file),committed_before=e["before"])
            write_ledgers(state,args.results,own,record)
            log=(case/"worker.log").open("a",buffering=1)
            try:
                process=subprocess.Popen(command,cwd=original.CODE,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         pass_fds=(original_lock.fileno(),))
            except Exception as error:
                log.close();write_ledgers(state,args.results,own,dict(record,state="launch_failed",error=repr(error)));raise
            record.update(pid=process.pid,state="running")
            # Original ledger first after spawn: unchanged --resume must see it
            # even if this priority process dies before its own journal updates.
            compatible=original.load(args.results/"launch_ledger.json");compatible[case_name]=dict(record)
            original.atomic_json(args.results/"launch_ledger.json",compatible)
            own[case_name]=dict(record);original.atomic_json(state/"launch_ledger.json",own)
            identity=original.process_state(process.pid,record)
            if identity.get("process_start"):record["process_start"]=identity["process_start"]
            write_ledgers(state,args.results,own,record)
            running[case_name]=dict(**e,process=process,log=log);del pending[case_name]
            observed_active=sum(actor["process"].poll() is None for actor in running.values())
            peak=max(peak,observed_active)
            execution=original.load(case/"priority_execution.json")
            execution.update(observed_active_at_launch=observed_active,observed_peak_priority_workers=peak)
            original.atomic_json(case/"priority_execution.json",execution)
        update(state,"running_priority_targets",completed=len(done),pending=len(pending),active=list(running),
               worker_pids={k:e["process"].pid for k,e in running.items()},failures=failures,
               priority_worker_limit=admission.workers,original_worker_limit=plan["workers"],
               observed_peak_priority_workers=peak,
               modeled_resident_gib=len(running)*original.weight(SETTING)["resident_gib"])
        if failures:raise RuntimeError("Priority worker/export failure retained; no automatic resampling")
        if pending and not running:raise RuntimeError("Priority jobs cannot fit configured RAM/disk admission")
        if pending or running:sleep(2)
    require(len(done)==10,"Priority target count differs")
    for case_name,seed in TARGETS.items():original.verify_case(args.results/"verified"/case_name,plan,SETTING,seed)
    require(input_fingerprints(args.left_results,plan)==phase_plan["left_case_artifact_sha256"],"Refined left data changed")
    assert_original_compatibility(args.results,note)
    update(state,"rendering_requested_figure",completed=10,method_runs=50,render_command=phase_plan["render_command"],
           observed_peak_priority_workers=peak,priority_worker_limit=admission.workers,original_worker_limit=plan["workers"])
    with (state/"render.log").open("a",buffering=1) as log:
        subprocess.run(phase_plan["render_command"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                       pass_fds=(original_lock.fileno(),),check=True)
    figure=Path(phase_plan["figure_output"])
    artifacts={}
    for filename in ("sensitivity_residual.pdf","sensitivity_residual.png"):
        file=figure/filename;require(file.is_file(),f"Requested figure output missing: {file}")
        artifacts[str(file)]=original.digest(file)
    update(state,"priority_complete_resuming_original_queue",completed=10,method_runs=50,
           observed_peak_priority_workers=peak,priority_worker_limit=admission.workers,original_worker_limit=plan["workers"],
           figure_artifact_sha256=artifacts,original_wrapper=str(wrapper),original_wrapper_sha256=original.digest(wrapper),
           note="Unchanged wrapper automatically passes --resume; verified priority targets are skipped")
    assert_original_compatibility(args.results,note);assert_quiescent(note,args.results)
    # All target children have exited and been reaped. Close only now, then
    # replace this process with the original wrapper, avoiding a wrapper orphan.
    original_lock.close();phase_lock.close()
    os.execve(sys.executable,[sys.executable,str(wrapper)],env)


if __name__=="__main__":
    try:main()
    except Exception as error:
        if CURRENT_STATE is not None:
            progress=CURRENT_STATE/"progress.json"
            value=original.load(progress) if progress.exists() else {}
            value.update(state="needs_attention",updated_utc=original.utc(),priority_pid=os.getpid(),
                         error=f"{type(error).__name__}: {error}",
                         note="Existing actors/pairs retained; no automatic resampling or original-queue resume on failure")
            original.atomic_json(progress,value)
        print(f"PRIORITY_PHASE_ERROR {type(error).__name__}: {error}",file=sys.stderr,flush=True)
        raise
