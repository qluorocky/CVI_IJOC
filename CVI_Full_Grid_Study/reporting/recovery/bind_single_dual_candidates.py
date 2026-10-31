#!/usr/bin/env python3
"""Bind existing v2 candidates/source snapshots; no optimization or adoption."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).parent/"single_delivery_v2"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def vector_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    summary=json.loads((BASE/"summary.json").read_text())
    for entry in summary["results"]:
        file=Path(entry["diagnostic_file"]);row=json.loads(file.read_text())
        original=file.with_name("diagnostic.pre_candidate_binding.json")
        if original.exists():raise FileExistsError("Candidate binding is single-use")
        original.write_bytes(file.read_bytes())
        with np.load(file.parent/"candidates.npz",allow_pickle=False) as saved:
            candidate=saved["feasible"]
        assert vector_sha(candidate)==row["feasible_candidate_hash"]
        solution=file.parent/"solution.npz"
        with solution.open("xb") as stream:np.savez_compressed(stream,x=candidate)
        solver=BASE/"solver_snapshot.py";module=BASE/"recovery_module_snapshot.py"
        assert sha(solver)==row["recovery_source_sha256"]
        assert sha(module)==row["recovery_module_sha256"]
        row.update(instance_hash_verified=True,candidate_file=str(solution.resolve()),
                   candidate_file_sha256=sha(solution),solution_hash=vector_sha(candidate),
                   recovery_source_path=str(solver.resolve()),recovery_module_path=str(module.resolve()),
                   original_solver_output_file=str(original.resolve()),original_solver_output_sha256=sha(original),
                   candidate_binding_source_path=str(Path(__file__).resolve()),
                   candidate_binding_source_sha256=sha(__file__),
                   candidate_binding_note="Metadata/source/NPZ binding only; numerical fields and original solver output preserved")
        file.write_text(json.dumps(row,indent=2,allow_nan=True)+"\n")
        entry["diagnostic_sha256"]=sha(file)
    summary["candidate_binding_source_sha256"]=sha(__file__)
    (BASE/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Bound eight existing feasible candidates and immutable source snapshots; no solves or adoption.")


if __name__=="__main__":main()
