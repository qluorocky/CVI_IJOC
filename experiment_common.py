"""Shared measurement and reporting code for the two separate experiment drivers."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cvx_benchmark import OE_SETTINGS, WeakGap, operator_lipschitz, projected_oe
from initialization import (estimate_initial_constraint_curvature,
                            estimate_initial_vector_constraint_curvature)
from mc_pfaos import MCPFAOSOptions, mc_pfaos
from pfaos import PFAOSOptions, pfaos


def _checkpoints(total):
    if total < 1:
        return [0]
    grid = {0, total}
    k = 1
    while k < total:
        grid.add(k)
        k = max(k + 1, int(np.ceil(1.5 * k)))
    return sorted(grid)


def _proposed(problem, seconds, multi, seed, max_iterations):
    clock = perf_counter()
    estimate = (estimate_initial_vector_constraint_curvature if multi
                else estimate_initial_constraint_curvature)
    L0, _ = estimate(problem, 4, 0.5, seed=seed + 10000)
    calibration_time = perf_counter() - clock
    remaining = max(0.001, seconds - calibration_time)
    if multi:
        _, history = mc_pfaos(problem, MCPFAOSOptions(
            N=max_iterations, lambda_bound=100., L0=L0,
            x0=problem.feasible_point, verbose=False, time_limit=remaining))
        grad_label = "J_calls"
        grad_counts = history["J_evaluations"]
    else:
        _, history = pfaos(problem, PFAOSOptions(
            N=max_iterations, lambda_bound=100., L0=L0,
            x0=problem.feasible_point, verbose=False, time_limit=remaining))
        grad_label = "grad_g_calls"
        grad_counts = history["grad_g_evaluations"]
    # Calibration makes 4 J and 12 G evaluations (or 4 grad g, 12 g).
    points = []
    for k in _checkpoints(history["completed_iterations"]):
        points.append(dict(k=k, elapsed=float(history["elapsed"][k]+calibration_time),
                           F_calls=int(history["F_evaluations"][k]),
                           grad_calls=int(grad_counts[k] + 4),
                           projection_calls=0,
                           projection_time=0.0,
                           x=history["x_bar"][:, k].copy()))
    status = "Time limit" if history["timed_out"] else "Iteration cap"
    max_multiplier = (float(np.nanmax(history["lambda_norm"][:history["completed_iterations"]]))
                      if multi and history["completed_iterations"] else 0.0)
    return points, status, grad_label, L0, max_multiplier


def run_pair(problem, seed, seconds, target, max_iterations=10000):
    multi = problem.m > 1
    proposed_name = "MC-PFAOS" if multi else "PFAOS"
    M_true = operator_lipschitz(problem)
    points, proposed_status, gradient_label, L0, max_multiplier = _proposed(
        problem, seconds, multi, seed, max_iterations)
    runs = [(proposed_name, points, proposed_status, None)]
    for factor, name in OE_SETTINGS:
        oe = projected_oe(problem, seconds, max_iterations, M0=factor*M_true)
        oe_points = [oe["history"][i] for i in _checkpoints(len(oe["history"])-1)]
        runs.append((name, oe_points, oe["status"], factor))
    gap = WeakGap(problem)
    rows = []
    for name, samples, status, factor in runs:
        for sample in samples:
            begin = perf_counter()
            value, gap_status = gap.evaluate(sample["x"])
            violation = float(np.linalg.norm(np.maximum(problem.G(sample["x"]), 0.)))
            rows.append(dict(seed=seed, n=problem.n, m=problem.m, method=name,
                             k=sample["k"], elapsed=round(sample["elapsed"], 9),
                             F_calls=sample["F_calls"],
                             grad_calls=sample.get("grad_calls", 0),
                             projection_calls=sample["projection_calls"],
                             projection_time=sample["projection_time"],
                             gap=value, violation=violation,
                             residual=max(value, violation),
                             gap_status=gap_status,
                             diagnostic_seconds=perf_counter()-begin,
                             solver_status=status, target=target,
                             L0=L0 if name == proposed_name else "",
                             multiplier_max=max_multiplier if name == proposed_name else "",
                             gradient_label=gradient_label,
                             M_true=M_true, M0=factor*M_true if factor else "",
                             M0_factor=factor if factor else ""))
    return rows


def append_rows(destination, rows):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def write_config(path, args):
    path = Path(path)
    if path.exists() or path.with_name("checkpoints.csv").exists():
        raise FileExistsError(
            f"{path.parent} already contains experiment output; choose a new --out directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        file.write(json.dumps(dict(vars(args), oe_M0_factors=[s[0] for s in OE_SETTINGS],
                                   oe_M_definition="spectral norm of realized operator matrix"),
                              indent=2, default=str) + "\n")
