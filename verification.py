"""Numerical checks for the synthetic problem assumptions."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .problem import QuadraticCVIProblem
except ImportError:
    from problem import QuadraticCVIProblem


def verify_synthetic_problem(
    problem: QuadraticCVIProblem, verbose: bool = True
) -> dict[str, Any]:
    if problem.m != 1:
        raise ValueError("Single-constraint verification requires m=1.")
    symmetric_operator = 0.5 * (problem.M + problem.M.T)
    constraint_matrix = 0.5 * (problem.A[0] + problem.A[0].T)
    report = {
        "minimum_operator_symmetric_eigenvalue": float(
            np.min(np.linalg.eigvalsh(symmetric_operator))
        ),
        "minimum_constraint_eigenvalue": float(
            np.min(np.linalg.eigvalsh(constraint_matrix))
        ),
        "feasible_point_constraint": problem.g(problem.feasible_point),
        "feasible_point_ball_residual": float(
            np.linalg.norm(problem.feasible_point) - problem.r
        ),
        "initial_point_ball_residual": float(
            np.linalg.norm(problem.old_experiment_x0) - problem.r
        ),
        "true_operator_lipschitz": problem.true_operator_lipschitz,
        "true_constraint_smoothness": float(
            problem.true_constraint_smoothness[0]
        ),
    }
    tolerance = 1e-9
    report["matches_pfaos_assumptions"] = bool(
        report["minimum_operator_symmetric_eigenvalue"] >= -tolerance
        and report["minimum_constraint_eigenvalue"] >= -tolerance
        and report["feasible_point_constraint"] <= tolerance
        and report["feasible_point_ball_residual"] <= tolerance
        and report["initial_point_ball_residual"] <= tolerance
    )
    if verbose:
        print("Synthetic-problem verification")
        print(
            "  min eig((M+M.T)/2): "
            f"{report['minimum_operator_symmetric_eigenvalue']:.3e}"
        )
        print(
            "  min eig(A):          "
            f"{report['minimum_constraint_eigenvalue']:.3e}"
        )
        print(
            "  g(feasible point):   "
            f"{report['feasible_point_constraint']:.3e}"
        )
        print(
            "  assumptions matched: "
            f"{int(report['matches_pfaos_assumptions'])}"
        )
    if not report["matches_pfaos_assumptions"]:
        raise RuntimeError("Generated problem failed a PFAOS assumption check.")
    return report


def verify_synthetic_mcvi_problem(
    problem: QuadraticCVIProblem, verbose: bool = True
) -> dict[str, Any]:
    symmetric_operator = 0.5 * (problem.M + problem.M.T)
    minimum_constraint_eigenvalue = np.inf
    maximum_constraint_rank = 0
    for constraint_index in range(problem.m):
        matrix = 0.5 * (
            problem.A[constraint_index] + problem.A[constraint_index].T
        )
        minimum_constraint_eigenvalue = min(
            minimum_constraint_eigenvalue,
            float(np.min(np.linalg.eigvalsh(matrix))),
        )
        maximum_constraint_rank = max(
            maximum_constraint_rank,
            int(np.linalg.matrix_rank(matrix, tol=1e-9)),
        )

    minimum_operator_eigenvalue = float(
        np.min(np.linalg.eigvalsh(symmetric_operator))
    )
    report = {
        "minimum_operator_symmetric_eigenvalue": minimum_operator_eigenvalue,
        "minimum_constraint_eigenvalue": minimum_constraint_eigenvalue,
        "maximum_constraint_rank": maximum_constraint_rank,
        "maximum_feasible_constraint": float(
            np.max(problem.G(problem.feasible_point))
        ),
        "feasible_point_ball_residual": float(
            np.linalg.norm(problem.feasible_point) - problem.r
        ),
        "operator_is_not_strongly_monotone": minimum_operator_eigenvalue <= 1e-9,
        "constraints_are_not_strongly_convex": maximum_constraint_rank
        < problem.n,
    }
    tolerance = 1e-8
    report["matches_mc_pfaos_assumptions"] = bool(
        minimum_operator_eigenvalue >= -tolerance
        and minimum_constraint_eigenvalue >= -tolerance
        and report["maximum_feasible_constraint"] <= tolerance
        and report["feasible_point_ball_residual"] <= tolerance
        and report["operator_is_not_strongly_monotone"]
        and report["constraints_are_not_strongly_convex"]
    )
    if verbose:
        print("Synthetic MCVI verification")
        print(f"  min eig((M+M.T)/2): {minimum_operator_eigenvalue:.3e}")
        print(f"  min eig(A_i):        {minimum_constraint_eigenvalue:.3e}")
        print(
            "  max G(x_feasible):   "
            f"{report['maximum_feasible_constraint']:.3e}"
        )
        print(
            "  assumptions matched: "
            f"{int(report['matches_mc_pfaos_assumptions'])}"
        )
    if not report["matches_mc_pfaos_assumptions"]:
        raise RuntimeError("Generated problem failed an MC-PFAOS assumption check.")
    return report
