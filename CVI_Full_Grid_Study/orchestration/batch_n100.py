#!/usr/bin/env python3
"""Amended n=100 MC orchestration; the original timed worker remains frozen.

Only single-constraint cases from the original NEW full-grid study are reusable.
Inherited workers are observed read-only until their original PID has exited.
"""
import argparse
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
GIB = 1024**3
COEF_CAP = 5_000_000
METHODS = ["proposed", "oe_true", "oe_low", "oe_high", "golden_ratio"]
SINGLE_N = (10, 50, 100, 200, 500, 1000)
ORIGINAL_MULTI_M = (10, 50, 100, 200, 500, 1000)
MULTI_M = (10, 50, 100, 200, 500)
SETTINGS = ([dict(study="single", n=n, m=1) for n in SINGLE_N]
            + [dict(study="multi", n=100, m=m) for m in MULTI_M])
PLANNED_INSTANCES = len(SETTINGS)*10
PLANNED_METHODS = PLANNED_INSTANCES*len(METHODS)
THREAD_ENV = {k:"1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREAD_ENV)


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True)+"\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {p.name:digest(p) for p in sorted(CODE.glob("*.py"))}


def name(setting, seed):
    return f"{setting['study']}_n{setting['n']}_m{setting['m']}_seed{seed}"


def weight(setting):
    count = setting["m"]*(setting["n"]-setting["n"]//2)*setting["n"]
    return dict(factor_gib=8*count/GIB, conic_reserve_gib=320*min(count, COEF_CAP)/GIB,
                resident_gib=.35+8*count/GIB+320*min(count, COEF_CAP)/GIB)


def admissible(setting, running, args):
    others = [e["setting"] for e in running.values()]
    if len(others) >= args.workers:
        return False
    if weight(setting)["resident_gib"]+sum(weight(s)["resident_gib"] for s in others) > args.memory_budget_gib:
        return False
    # Dimension-aware replacement for the old m>=500 rule. In the n=100
    # grid even m=1000 has only 40 MB factors and is not a heavy-factor case.
    heavy = lambda s: weight(s)["factor_gib"] >= args.heavy_factor_gib
    if heavy(setting) and any(heavy(s) for s in others):
        return False
    return True


def process_start_from_record(record):
    if record.get("process_start"):
        return " ".join(record["process_start"].split())
    text = record.get("process_identity", "")
    match = re.match(r"\s*\d+\s+\d+\s+(\w{3}\s+\w{3}\s+\d+\s+\d\d:\d\d:\d\d\s+\d{4})\s+", text)
    return " ".join(match.group(1).split()) if match else None


def process_state(pid, record=None):
    """Conservative liveness: EPERM or unavailable ps remains alive/unknown.

    A recorded start-time mismatch proves PID reuse and the original worker's
    exit. This function never sends a signal except the read-only kill(pid,0).
    """
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return dict(alive=False, evidence="pid_absent")
    except PermissionError:
        return dict(alive=True, evidence="permission_denied_liveness_unknown")
    except OSError as error:
        if error.errno == errno.ESRCH:
            return dict(alive=False, evidence="pid_absent")
        return dict(alive=True, evidence=f"liveness_unknown_errno_{error.errno}")
    try:
        probe = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "command="],
                               text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return dict(alive=True, evidence="pid_exists_identity_unavailable")
    fields = probe.stdout.strip().split(None, 6)
    if len(fields) != 7:
        return dict(alive=True, evidence="pid_exists_identity_unavailable")
    if fields[0].upper().startswith("Z"):
        return dict(alive=False, evidence="worker_exited_zombie")
    start, command = " ".join(fields[1:6]), fields[6]
    expected = process_start_from_record(record or {})
    if expected and start != expected:
        return dict(alive=False, evidence="recorded_worker_exited_pid_reused", observed_start=start)
    if record and record.get("path"):
        if str(CODE / "study.py") not in command or str(Path(record["path"]).resolve()) not in command:
            raise RuntimeError(f"Live PID {pid} no longer matches inherited worker command")
    return dict(alive=True, evidence="pid_and_start_match" if expected else "pid_exists",
                process_start=start, command=command)


def validate_source(transition_file):
    transition = load(transition_file)
    if not transition.get("old_controller_stopped"):
        raise RuntimeError("Transition has not confirmed the old controller stopped")
    source = Path(transition["source_results"]).resolve()
    source_plan = load(source / "plan.json")
    if digest(source / "plan.json") != transition["source_plan_sha256"]:
        raise RuntimeError("Original plan checksum mismatch")
    expected_settings = {("single", n, 1) for n in SINGLE_N} | {("multi", 1000, m) for m in ORIGINAL_MULTI_M}
    actual_settings = {(s["study"], s["n"], s["m"]) for s in source_plan["settings"]}
    if (source_plan.get("validation_fixture") or source_plan.get("prior_runs_reused")
            or actual_settings != expected_settings or source_plan.get("seeds") != list(range(10))
            or source_plan.get("methods") != METHODS or source_plan.get("seconds") != 1000.
            or source_plan.get("diagnostic_seconds") != 300.):
        raise RuntimeError("Reuse source must be the NEW original full-grid study, not a small/historical study")
    if source_plan["source_hashes"] != source_hashes():
        raise RuntimeError("Frozen worker source differs from original plan")
    if transition.get("source_hashes_verified", source_plan["source_hashes"]) != source_plan["source_hashes"]:
        raise RuntimeError("Transition source checksum mismatch")
    return transition, source, source_plan


def verify_case(case, plan, setting, seed, complete=True):
    """Strict identity/pair/prefix validation, without regenerating the instance."""
    import numpy as np
    case = Path(case)
    meta = load(case / "manifest.json")
    expected = dict(n=setting["n"], m=setting["m"], seed=seed, cap_seconds=plan["seconds"],
                    source_hashes=plan["source_hashes"], requested_methods=METHODS)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"{case.name}: manifest identity mismatch for {key}")
    if meta.get("gap_seconds") != plan["diagnostic_seconds"]:
        raise RuntimeError(f"{case.name}: original diagnostic budget differs")
    if complete and meta.get("state") != "complete":
        raise RuntimeError(f"{case.name}: not complete")
    # An interrupted initialization may not yet have initial_point/instance_hash.
    x0 = np.asarray(meta.get("initial_point", []), dtype=float)
    artifacts = {"manifest.json":digest(case / "manifest.json")}
    for method in METHODS:
        runfile, pointfile = case/f"{method}_run.json", case/f"{method}_points.npz"
        diagfile = case/f"{method}_diagnostics.json"
        if not (runfile.exists() and pointfile.exists()):
            if complete:
                raise RuntimeError(f"{case.name}/{method}: missing timed pair")
            # Preserve partial atomic-write artifacts; frozen resume decides
            # whether an interrupted, uncommitted method needs its full budget.
            for f in (runfile, pointfile, diagfile):
                if f.exists(): artifacts[f.name] = digest(f)
            continue
        saved = load(runfile)
        identity = saved.get("metadata", {})
        for key in ("n", "m", "seed", "cap_seconds", "source_hashes", "instance_hash"):
            if identity.get(key) != meta.get(key):
                raise RuntimeError(f"{case.name}/{method}: timed identity mismatch for {key}")
        if identity.get("method") != method or saved.get("points_sha256") != digest(pointfile):
            raise RuntimeError(f"{case.name}/{method}: method/pair checksum mismatch")
        with np.load(pointfile, allow_pickle=False) as archive:
            points = archive["x"]
        checkpoints = saved["checkpoints"]
        if (points.shape != (len(checkpoints), setting["n"]) or not len(checkpoints)
                or not np.all(np.isfinite(points)) or x0.shape != (setting["n"],)
                or not np.array_equal(points[0], x0)):
            raise RuntimeError(f"{case.name}/{method}: checkpoint values/initial identity mismatch")
        times = np.asarray([cp["elapsed"] for cp in checkpoints], dtype=float)
        runtime = saved["result"].get("runtime")
        if (not np.all(np.isfinite(times)) or times[0] < 0 or times[-1] > plan["seconds"]
                or np.any(np.diff(times) <= 0) or runtime is None
                or not np.isfinite(runtime) or runtime < times[-1]):
            raise RuntimeError(f"{case.name}/{method}: checkpoint/runtime timing mismatch")
        ks = [cp["k"] for cp in checkpoints]
        if any(not np.isfinite(k) or int(k) != k or k < 0 for k in ks) or any(a>b for a,b in zip(ks,ks[1:])):
            raise RuntimeError(f"{case.name}/{method}: invalid checkpoint iteration indices")
        rows = load(diagfile) if diagfile.exists() else []
        if len(rows) > len(checkpoints) or (complete and len(rows) != len(checkpoints)):
            raise RuntimeError(f"{case.name}/{method}: diagnostic count mismatch")
        for i, (row, cp) in enumerate(zip(rows, checkpoints)):
            if ((row.get("n"),row.get("m"),row.get("seed"),row.get("method"),row.get("checkpoint"))
                    != (setting["n"],setting["m"],seed,method,i)
                    or row.get("elapsed") != cp["elapsed"] or row.get("k") != cp["k"]):
                raise RuntimeError(f"{case.name}/{method}: diagnostic/checkpoint identity mismatch")
        stage = meta.get("methods", {}).get(method, {})
        if complete and (stage.get("stage") != "complete" or stage.get("status") != saved["result"]["status"]):
            raise RuntimeError(f"{case.name}/{method}: terminal manifest/method status mismatch")
        for f in (runfile, pointfile, diagfile):
            if f.exists(): artifacts[f.name] = digest(f)
    return meta, artifacts


def make_plan(args, transition_file, transition, source, oldplan):
    return dict(study_version="full_grid_corrected_mc_n100_v1", validation_fixture=False,
                purpose="full_grid_measured", settings=SETTINGS, seeds=list(range(10)),
                methods=METHODS, seconds=1000., diagnostic_seconds=300., workers=args.workers,
                planned_instances=PLANNED_INSTANCES, planned_methods=PLANNED_METHODS, source_hashes=oldplan["source_hashes"],
                prior_runs_reused=True,
                orchestrator_hashes={"batch_n100.py":digest(__file__)},
                plan_amendment=dict(original_multi_dimension=1000, amended_multi_dimension=100,
                    single_grid_unchanged=True, constraint_counts=list(MULTI_M),
                    removed_constraint_counts=[1000],
                    reason=transition.get("reason"), transition_file=str(Path(transition_file).resolve()),
                    transition_sha256=digest(transition_file),
                    scheduling_amendment="Heavy classification uses >=1 GiB factor footprint, not constraint count; seven workers and 8 GiB modeled RAM retained"),
                inherited_single_provenance=dict(source_results=str(source),
                    source_plan_sha256=digest(source/"plan.json"),
                    eligible="Only m=1 single cases from this NEW original full-grid study",
                    excluded="All original multi cases and all small/historical studies",
                    inherited_workers=transition.get("inherited_single_workers", []),
                    adoption="After worker exit, strict identity/checksum validation and per-case lineage"),
                execution_policy=dict(workers=args.workers, affinity="os_managed", logical_cpus=os.cpu_count(),
                    hardware="Apple M4,10cores,16GiB", platform=platform.platform(), machine=platform.machine(),
                    numerical_threads=1, thread_environment=THREAD_ENV,
                    memory_budget_gib=args.memory_budget_gib, min_free_disk_gib=args.min_free_disk_gib,
                    model_reserve_bytes_per_coefficient=320, heavy_factor_threshold_gib=args.heavy_factor_gib,
                    max_simultaneous_heavy_factor_cases=1, inherited_workers_count_toward_slots_and_ram=True,
                    admission="First pending admissible case under factor+conic reserve; dimension-aware heavy footprint"),
                projection_policy=oldplan["projection_policy"])


def copy_complete(case, target, plan, setting, seed, lineage=None):
    """Atomic export with source/copy checksums; never rewrites source files."""
    meta, original = verify_case(case, plan, setting, seed)
    copied_files = {f.name:digest(f) for f in Path(case).iterdir()
                    if f.is_file() and f.suffix in {".json", ".jsonl", ".npz", ".log"}}
    if target.exists():
        verify_case(target, plan, setting, seed)
        existing = {f: digest(target/f) for f in original}
        if existing != original:
            raise RuntimeError(f"Refusing conflicting existing verified case {target}")
        return meta
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for f in sorted(Path(case).iterdir()):
            if f.is_file() and f.suffix in {".json", ".jsonl", ".npz", ".log"}:
                shutil.copy2(f, stage/f.name)
        _, copied = verify_case(stage, plan, setting, seed)
        if (original != copied or any(digest(Path(case)/f) != h or digest(stage/f) != h
                                     for f,h in copied_files.items())):
            raise RuntimeError("Case changed during adoption/export")
        if lineage is not None:
            atomic_json(stage/"lineage.json", dict(**lineage, copied_utc=utc(),
                        source_case=str(Path(case).resolve()), copied_artifact_sha256=copied_files,
                        original_failures_and_invalid_diagnostics_preserved=True))
        stage.replace(target)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return meta


def event(out, kind, **info):
    with (out/"controller_events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(updated_utc=utc(), event=kind, **info))+"\n")


def main():
    ap = argparse.ArgumentParser()
    for flag, typ, default in (("seconds",float,1000.),("diagnostic-seconds",float,300.),
            ("seeds",int,10),("workers",int,7),("memory-budget-gib",float,8.),
            ("min-free-disk-gib",float,12.),("heavy-factor-gib",float,1.)):
        ap.add_argument("--"+flag, type=typ, default=default)
    ap.add_argument("--out",type=Path,default=ROOT/"results_n100")
    ap.add_argument("--transition",type=Path,default=ROOT/"notes/grid_transition.json")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--skip-report",action="store_true")
    ap.add_argument("--validate-only",action="store_true")
    args = ap.parse_args()
    if ((args.seconds,args.diagnostic_seconds,args.seeds) != (1000.,300.,10)
            or not 1<=args.workers<=7 or not 0<args.memory_budget_gib<=8
            or args.min_free_disk_gib<12 or args.heavy_factor_gib<=0):
        raise ValueError("Amended measured scope is fixed: 10 seeds, 1000/300 s, <=7 workers, <=8 GiB, >=12 GiB disk reserve")
    transition, source, oldplan = validate_source(args.transition)
    out = args.out.resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("New results must be separate from original results")
    plan = make_plan(args,args.transition,transition,source,oldplan)
    inherited = {}
    for record in transition.get("inherited_single_workers", []):
        record = dict(record)
        record["path"] = str(Path(record.get("path",record.get("case_path"))).resolve())
        record["case"] = record.get("case",record.get("name"))
        allowed = {name(s,seed):(s,seed) for s in SETTINGS if s["study"]=="single" for seed in range(10)}
        if record["case"] not in allowed or Path(record["path"]) != source/"working"/record["case"]:
            raise ValueError("Inherited record is not an eligible original single case")
        if record["case"] in inherited:
            raise ValueError("Duplicate inherited case")
        inherited[record["case"]] = record
    if args.validate_only:
        complete = 0
        for s in SETTINGS[:6]:
            for seed in range(10):
                case_name=name(s,seed)
                if case_name in inherited: continue
                candidates=[source/"verified"/case_name,source/"working"/case_name]
                case=next((c for c in candidates if (c/"manifest.json").exists()),None)
                if case is not None and load(case/"manifest.json").get("state")=="complete":
                    verify_case(case,oldplan,s,seed);complete+=1
        print(json.dumps(dict(validation="passed", planned_instances=PLANNED_INSTANCES, planned_methods=PLANNED_METHODS,
              reusable_complete_single_cases=complete,inherited_single_cases=len(inherited),
              original_source_hashes_verified=True, processes_launched=0)))
        return
    old_pid=transition.get("old_controller_pid")
    if old_pid and process_state(old_pid)["alive"]:
        raise RuntimeError("Old controller PID is still alive/unknown; refuse competing controller")
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"controller.lock").open("a+")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Another amended controller already holds the lock")
    planfile=out/"plan.json"
    if planfile.exists():
        if not args.resume: raise FileExistsError("Use --resume for existing amended results")
        if load(planfile)!=plan: raise RuntimeError("Amended plan/orchestrator/source identity mismatch")
    previous_progress = load(out/"progress.json") if (out/"progress.json").exists() else {}
    ledger_file=out/"launch_ledger.json"
    ledger=load(ledger_file) if ledger_file.exists() else {}
    for case_name,record in ledger.items():
        if record["state"]=="launching" and record.get("pid") is None:
            raise RuntimeError(f"Unresolved launch intent for {case_name}; coordinate process reconciliation before resume")
        if record["state"] in {"launching","running"} and process_state(record["pid"],record)["alive"]:
            raise RuntimeError(f"Previous owned worker {case_name}/{record['pid']} is alive/unknown; no duplicate resume")
    for case_name,pid in previous_progress.get("worker_pids",{}).items():
        if case_name in ledger:
            continue
        if case_name in inherited and pid==inherited[case_name]["pid"]:
            continue
        record=previous_progress.get("worker_identity_records",{}).get(case_name,{})
        if process_state(pid,record)["alive"]:
            raise RuntimeError(f"Previous controller worker {case_name}/{pid} is alive/unknown; coordinate handoff before resume")
    atomic_json(planfile,plan)
    working,verified=out/"working",out/"verified"
    working.mkdir(exist_ok=True);verified.mkdir(exist_ok=True);atomic_json(verified/"plan.json",plan)
    running,pending,done,failed={},{},set(),[]
    lineage_base=dict(reused_from_new_original_single_study=True,
        source_results=str(source),source_plan_sha256=digest(source/"plan.json"),
        transition_sha256=digest(args.transition),new_plan_sha256=digest(planfile))
    for s in SETTINGS:
        for seed in range(10):
            case_name=name(s,seed);target=verified/case_name
            if target.exists():
                verify_case(target,plan,s,seed);done.add(case_name);continue
            if s["study"]=="single" and case_name in inherited:
                record=inherited[case_name]
                running[case_name]=dict(name=case_name,setting=s,seed=seed,case=Path(record["path"]),
                    external=True,pid=record["pid"],record=record,resumed_original=False)
                continue
            if s["study"]=="single":
                candidates=[source/"verified"/case_name,source/"working"/case_name]
                case=next((c for c in candidates if (c/"manifest.json").exists()),None)
                if case is not None:
                    meta,_=verify_case(case,oldplan,s,seed,complete=False)
                    if meta["state"]=="complete":
                        copy_complete(case,target,plan,s,seed,lineage_base)
                        done.add(case_name);event(out,"adopted_complete_original_single",case=case_name);continue
                    # The stopped controller's exhaustive transition snapshot
                    # records all inherited single workers. Never assume an
                    # unrecorded old PID is gone if snapshot says otherwise.
                    old_workers=transition.get("original_progress_snapshot",{}).get("worker_pids",{})
                    if case_name in old_workers:
                        record=dict(case=case_name,path=str(case),pid=old_workers[case_name])
                        running[case_name]=dict(name=case_name,setting=s,seed=seed,case=case,
                            external=True,pid=record["pid"],record=record,resumed_original=False)
                        continue
                    pending[case_name]=dict(setting=s,seed=seed,case=case,original_single=True)
                    continue
            case=working/case_name
            if (case/"manifest.json").exists():verify_case(case,plan,s,seed,complete=False)
            pending[case_name]=dict(setting=s,seed=seed,case=case,original_single=False)
    start,last,interrupted=monotonic(),-60.,False
    locations={k:e["case"] for k,e in {**running,**pending}.items()}
    locations.update({k:verified/k for k in done})
    env=dict(os.environ,**THREAD_ENV)
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while pending or running:
        if interrupted:
            # Owned workers stop normally; externally inherited workers keep
            # their original clocks and are never signaled by this controller.
            for e in running.values():
                if not e["external"]:e["process"].terminate()
            for e in running.values():
                if not e["external"]:
                    try:e["process"].wait(timeout=15)
                    except subprocess.TimeoutExpired:e["process"].kill();e["process"].wait()
                    e["log"].close()
            atomic_json(out/"controller_status.json",dict(state="interrupted",controller_pid=os.getpid(),updated_utc=utc(),
                inherited_workers_preserved=[e["name"] for e in running.values() if e["external"]]))
            raise SystemExit(130)
        # Reap/adopt before admission so inherited workers consume real slots.
        for case_name,e in list(running.items()):
            if e["external"]:
                state=process_state(e["pid"],e["record"])
                e["last_process_state"]=state
                if state["alive"]:continue
                event(out,"inherited_worker_exited",case=case_name,pid=e["pid"],evidence=state)
                try:
                    meta,_=verify_case(e["case"],oldplan,e["setting"],e["seed"],complete=False)
                    if meta["state"]=="complete":
                        copy_complete(e["case"],verified/case_name,plan,e["setting"],e["seed"],lineage_base)
                        done.add(case_name)
                        locations[case_name]=verified/case_name
                    else:
                        pending[case_name]=dict(setting=e["setting"],seed=e["seed"],case=e["case"],original_single=True)
                        event(out,"inherited_incomplete_resume_pending",case=case_name,
                              note="Frozen worker resume preserves every committed pair and observed failure")
                except Exception as error:failed.append(dict(case=case_name,reason=repr(error)))
            else:
                if e["process"].poll() is None:continue
                e["log"].close()
                ledger[case_name].update(state="exited",returncode=e["process"].returncode,observed_exit_utc=utc())
                atomic_json(ledger_file,ledger)
                if e["process"].returncode==0:
                    try:
                        copy_complete(e["case"],verified/case_name,plan,e["setting"],e["seed"],
                                      lineage_base if e["original_single"] else None)
                        done.add(case_name)
                        locations[case_name]=verified/case_name
                    except Exception as error:failed.append(dict(case=case_name,reason=repr(error)))
                else:
                    failure=dict(case=case_name,exit_code=e["process"].returncode)
                    failed.append(failure);event(out,"worker_failed",**failure)
            del running[case_name]
        order=sorted(pending,key=lambda k:(-weight(pending[k]["setting"])["factor_gib"],pending[k]["seed"],k))
        for case_name in order:
            job=pending[case_name];s=job["setting"]
            if not admissible(s,running,args):continue
            if shutil.disk_usage(out).free/GIB < args.min_free_disk_gib+weight(s)["factor_gib"]:continue
            if source_hashes()!=plan["source_hashes"]:raise RuntimeError("Frozen worker source changed; admission stopped")
            case=job["case"];case.mkdir(parents=True,exist_ok=True)
            resume=(case/"manifest.json").exists()
            if resume:verify_case(case,plan,s,job["seed"],complete=False)
            log=(case/"worker.log").open("a",buffering=1)
            cmd=[sys.executable,str(CODE/"study.py"),"--n",str(s["n"]),"--m",str(s["m"]),
                 "--seed",str(job["seed"]),"--seconds","1000.0","--diagnostic-seconds","300.0",
                 "--out",str(case),"--methods",",".join(METHODS)]
            if resume:cmd.append("--resume")
            # A durable PID-less intent precedes Popen. If the controller dies
            # between spawning and recording the PID, resume blocks rather
            # than risk rerunning an orphan's timed method.
            identity=dict(case=case_name,path=str(case),pid=None,state="launching",intent_utc=utc(),command=cmd)
            ledger[case_name]=identity.copy();atomic_json(ledger_file,ledger)
            try:
                process=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            except Exception as error:
                ledger[case_name].update(state="launch_failed",error=repr(error));atomic_json(ledger_file,ledger)
                log.close();raise
            identity.update(pid=process.pid,state="running")
            ledger[case_name]=identity.copy();atomic_json(ledger_file,ledger)
            observed=process_state(process.pid,identity)
            if observed.get("process_start"):identity["process_start"]=observed["process_start"]
            ledger[case_name]=identity.copy();atomic_json(ledger_file,ledger)
            running[case_name]=dict(**job,name=case_name,external=False,process=process,pid=process.pid,log=log,record=identity)
            del pending[case_name]
            event(out,"worker_launched",case=case_name,pid=process.pid,resume=resume,
                  original_single=job["original_single"],command=cmd)
        timed=complete=invalid=0;stages={};method_errors=[];live={}
        for case_name,case in sorted(locations.items()):
            if not (case/"manifest.json").exists():continue
            meta=load(case/"manifest.json");stages[case_name]=meta["state"]
            invalid+=meta.get("invalid_diagnostic_count",sum(v.get("invalid_diagnostics",0) for v in meta.get("methods",{}).values()))
            for method,v in meta.get("methods",{}).items():
                timed+=v["stage"] in {"timed_complete","complete"};complete+=v["stage"]=="complete"
                if v.get("status") not in {"time_limit","certificate_stop"}:method_errors.append(dict(case=case_name,method=method,status=v.get("status")))
            if case_name in running and (case/"live.json").exists():live[case_name]=load(case/"live.json")
        state="running" if pending or running else ("failed" if failed else "complete")
        progress=dict(state=state,controller_pid=os.getpid(),updated_utc=utc(),elapsed_seconds=monotonic()-start,
            planned_instances=PLANNED_INSTANCES,planned_methods=PLANNED_METHODS,instances_finished=len(done),methods_timed=int(timed),
            methods_complete=int(complete),pending_instances=len(pending),active=list(running),
            worker_pids={k:e["pid"] for k,e in running.items()},
            worker_identity_records={k:e["record"] for k,e in running.items()},
            inherited_active=[k for k,e in running.items() if e["external"]],
            active_modeled_resident_gib=sum(weight(e["setting"])["resident_gib"] for e in running.values()),
            invalid_diagnostics=invalid,failed=failed,method_errors=method_errors,stages=stages,live=live,
            prior_runs_reused=True,adopted_single_cases=sum(k.startswith("single_") for k in done))
        atomic_json(out/"progress.json",progress)
        atomic_json(out/"controller_status.json",{k:progress[k] for k in ("state","controller_pid","updated_utc","instances_finished","pending_instances","active","inherited_active")})
        if monotonic()-start-last>=30 or state!="running":
            print(json.dumps({k:v for k,v in progress.items() if k not in {"stages","live"}}),flush=True);last=monotonic()-start
        if pending and not running and not any(weight(j["setting"])["resident_gib"]<=args.memory_budget_gib and shutil.disk_usage(out).free/GIB>=args.min_free_disk_gib+weight(j["setting"])["factor_gib"] for j in pending.values()):
            raise RuntimeError("Pending cases cannot fit configured RAM/disk admission")
        if pending or running:sleep(2)
    if len(done)!=PLANNED_INSTANCES or failed:
        raise SystemExit("Worker/export failures recorded; no failed methods were resampled")
    if not args.skip_report:
        subprocess.run([sys.executable,str(ROOT/"reporting/report_full_grid.py"),"--results",str(verified)],cwd=ROOT,env=env,check=True)
    atomic_json(out/"controller_status.json",dict(state="complete",controller_pid=os.getpid(),updated_utc=utc(),instances_finished=PLANNED_INSTANCES))


if __name__ == "__main__":
    main()
