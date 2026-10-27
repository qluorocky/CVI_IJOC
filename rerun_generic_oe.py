"""Rerun projected OE after changing its solver, retaining valid proposed runs.

Use this when the OE benchmark changes but PFAOS/MC-PFAOS and the generated
instances do not. Full independent reruns remain available through the two
main experiment drivers. Write to a NEW output file, never the input file.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from cvx_benchmark import WeakGap, projected_oe
from experiment_common import _checkpoints, append_rows
from game_problem import make_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")

    groups = defaultdict(list)
    with args.input.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] in {"PFAOS", "MC-PFAOS"}:
                groups[(int(row["n"]), int(row["m"]), int(row["seed"]))].append(row)
    if not groups:
        parser.error("input has no proposed-method checkpoints")

    completed = set()
    if args.output.exists():
        with args.output.open(newline="") as f:
            completed = {(int(r["n"]), int(r["m"]), int(r["seed"]))
                         for r in csv.DictReader(f) if r["method"] == "Projected OE"}
    for key, proposed_rows in sorted(groups.items()):
        if key in completed:
            continue
        n, m, seed = key
        if len([r for r in proposed_rows if int(r["k"]) == 0]) != 1:
            raise ValueError(f"Missing or duplicate initial checkpoint: {key}")
        problem = make_game(seed, n, m)
        oe = projected_oe(problem, args.seconds)
        diagnostic = WeakGap(problem)
        rows = [dict(row) for row in proposed_rows]
        template = proposed_rows[0]
        for i in _checkpoints(len(oe["history"]) - 1):
            sample = oe["history"][i]
            clock = perf_counter()
            gap, gap_status = diagnostic.evaluate(sample["x"])
            violation = float(np.linalg.norm(np.maximum(problem.G(sample["x"]), 0.)))
            rows.append(dict(template, method="Projected OE", k=sample["k"],
                             elapsed=round(sample["elapsed"], 9),
                             F_calls=sample["F_calls"], grad_calls=0,
                             projection_calls=sample["projection_calls"],
                             projection_time=sample["projection_time"],
                             gap=gap, violation=violation,
                             residual=max(gap, violation),
                             gap_status=gap_status,
                             diagnostic_seconds=perf_counter()-clock,
                             solver_status=oe["status"], L0="", multiplier_max=""))
        append_rows(args.output, rows)
        print(f"n={n} m={m} seed={seed}: OE {len(oe['history'])-1} iter, "
              f"{oe['projection_calls']} generic solves, "
              f"gap={rows[-1]['gap']:.4g}, status={oe['status']}", flush=True)


if __name__ == "__main__":
    main()
