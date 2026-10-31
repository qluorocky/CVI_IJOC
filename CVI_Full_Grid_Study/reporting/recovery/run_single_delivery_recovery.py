#!/usr/bin/env python3
"""One explicitly authorized 300-second attempt for each of eight saved gaps.

Sequential offline sidecars only; no timed method, adoption, or automatic retry.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOLVER=Path(__file__).with_name("constraint_generation_gap.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=True)+"\n");temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inventory",type=Path,default=ROOT/"notes/single_delivery_inventory.json")
    p.add_argument("--results",type=Path,default=ROOT/"results_single")
    p.add_argument("--out",type=Path,default=Path(__file__).parent/"single_delivery_v1")
    a=p.parse_args()
    inv=json.loads(a.inventory.read_text());rows=inv["invalid_diagnostics"]
    keys=[(r["case"],r["method"],r["checkpoint"]) for r in rows]
    if len(rows)!=8 or len(set(keys))!=8 or any(r["gap_valid"] for r in rows):
        raise ValueError("Authorization covers exactly eight unique existing invalid diagnostics")
    if json.loads((a.results/"plan.json").read_text()).get("release_scope")!="single_only":
        raise ValueError("Expected explicit single-only release")
    a.out.mkdir(parents=True,exist_ok=False)
    # Save the exact solver used before launching any authorized attempt.
    (a.out/"solver_snapshot.py").write_bytes(SOLVER.read_bytes())
    groups=OrderedDict()
    for row in rows:groups.setdefault((row["case"],row["method"]),[]).append(row)
    report=dict(scope="eight original invalid single-only diagnostics; one 300-second attempt each",
                inventory_sha256=sha(a.inventory), solver_sha256=sha(SOLVER),
                runner_sha256=sha(__file__),expected_evaluations=8,
                timed_methods_run=0,original_diagnostics_modified=False,automatic_retry=False,
                state="running",attempts=[])
    write(a.out/"summary.json",report)
    for (case,method),missing in groups.items():
        path=a.results/"verified"/case
        with np.load(path/f"{method}_points.npz",allow_pickle=False) as archive:
            points=archive["x"]
        for row in missing:
            actual=hashlib.sha256(np.ascontiguousarray(points[row["checkpoint"]]).tobytes()).hexdigest()
            if actual!=row["point_sha256"]:raise ValueError("Inventory/saved input point identity mismatch")
        output=a.out/f"{case}__{method}"
        cmd=[sys.executable,str(SOLVER),"--case",str(path),"--method",method,
             "--checkpoints",*[str(r["checkpoint"]) for r in missing],"--seconds","300",
             "--reconstruct-factors","--reconstruction-seconds","300",
             "--stable-single-snapshot","--attempt-label","authorized_single_delivery_300s",
             "--output",str(output)]
        entry=dict(case=case,method=method,checkpoints=[r["checkpoint"] for r in missing],
                   output=str(output.resolve()),command=cmd,results=[])
        start=perf_counter()
        try:
            completed=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,
                                     timeout=330+330*len(missing))
            entry.update(returncode=completed.returncode,wall_seconds=perf_counter()-start)
            (a.out/f"{case}__{method}.log").write_text(completed.stdout+completed.stderr)
        except subprocess.TimeoutExpired as error:
            entry.update(returncode=None,wall_seconds=perf_counter()-start,
                         runner_failure=f"TimeoutExpired: {error}")
        if (output/"benchmark.json").exists():
            benchmark=json.loads((output/"benchmark.json").read_text())
            for result in benchmark["results"]:
                q=result.get("numerical_optimality_check",{})
                lo,hi=q.get("numerical_gap_lower"),q.get("numerical_gap_upper")
                classification={}
                for threshold in (1e-2,1e-3,1e-4,1e-5):
                    classification[str(threshold)]=("numerically_below" if hi is not None and hi<=threshold
                        else "numerically_above" if lo is not None and lo>threshold else "unresolved")
                entry["results"].append(dict(checkpoint=result["checkpoint"],valid=result["valid"],
                    status=result["status"],gap=result["gap"],seconds=result["seconds"],
                    all_original_violation=result["all_original_violation"],
                    numerical_quality=q,threshold_classification=classification,
                    adoption_authorized=False))
        if (output/"failure.json").exists():entry["failure_record"]=json.loads((output/"failure.json").read_text())
        report["attempts"].append(entry);write(a.out/"summary.json",report)
        print(json.dumps(dict(case=case,method=method,returncode=entry["returncode"],
              results=[{k:r[k] for k in ("checkpoint","valid","status","gap")} for r in entry["results"]])),flush=True)
    report["state"]="attempts_finished"
    report["completed_evaluations"]=sum(len(e["results"]) for e in report["attempts"])
    report["numerically_valid_evaluations"]=sum(r["valid"] for e in report["attempts"] for r in e["results"])
    write(a.out/"summary.json",report)


if __name__=="__main__":main()
