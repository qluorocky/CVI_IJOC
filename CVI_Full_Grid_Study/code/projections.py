"""Projection routines shared by PFAOS and projected OE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from problem import QuadraticCVIProblem


Array = NDArray[np.float64]


@dataclass
class BallHalfspaceProjectionInfo:
    case: int
    description: str
    ball_residual: float
    halfspace_residual: float


@dataclass
class PreparedConstraint:
    eigenvectors: Array
    eigenvalues: Array
    b_in_basis: Array
    c: float


@dataclass
class PreparedProjection:
    constraints: list[PreparedConstraint]


@dataclass
class ProjectionOptions:
    tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-8
    max_cycles: int = 1000
    max_bisection_iterations: int = 80
    max_bracket_expansions: int = 80
    fail_on_nonconvergence: bool = True


@dataclass
class IntersectionProjectionInfo:
    converged: bool
    cycles: int
    constraint_projection_calls: int
    bisection_iterations: int
    max_violation: float


def project_ball(z: Array, radius: float) -> Array:
    z = np.asarray(z, dtype=float)
    z_norm = np.linalg.norm(z)
    return z.copy() if z_norm <= radius else radius * z / z_norm


def project_ball_halfspace(
    z: Array,
    radius: float,
    normal: Array,
    beta: float,
    tolerance: float = 1e-12,
) -> tuple[Array, BallHalfspaceProjectionInfo]:
    """Project onto ``{x: ||x|| <= radius, normal.T @ x <= beta}``."""

    z = np.asarray(z, dtype=float).reshape(-1)
    normal = np.asarray(normal, dtype=float).reshape(-1)
    normal_norm_squared = float(normal @ normal)
    z_norm = float(np.linalg.norm(z))
    scale = 1.0 + radius + z_norm + abs(beta)
    absolute_tolerance = tolerance * scale

    if normal_norm_squared <= absolute_tolerance**2:
        if beta < -absolute_tolerance:
            raise RuntimeError(
                "Linearized constraint is infeasible: zero gradient with "
                "positive offset."
            )
        x = project_ball(z, radius)
        return x, BallHalfspaceProjectionInfo(
            1,
            "redundant half-space",
            float(np.linalg.norm(x) - radius),
            float(normal @ x - beta),
        )

    normal_norm = np.sqrt(normal_norm_squared)
    if beta < -radius * normal_norm - absolute_tolerance:
        raise RuntimeError("The half-space and ball do not intersect.")

    if z_norm <= radius + absolute_tolerance and normal @ z <= beta + absolute_tolerance:
        x = z
        active_case = 1
        description = "unconstrained point feasible"
    else:
        x_ball = project_ball(z, radius)
        if normal @ x_ball <= beta + absolute_tolerance:
            x = x_ball
            active_case = 2
            description = "ball active"
        else:
            halfspace_violation = float(normal @ z - beta)
            x_halfspace = z - (halfspace_violation / normal_norm_squared) * normal
            if np.linalg.norm(x_halfspace) <= radius + absolute_tolerance:
                x = x_halfspace
                active_case = 3
                description = "half-space active"
            else:
                plane_center = (beta / normal_norm_squared) * normal
                radius_squared = radius**2 - beta**2 / normal_norm_squared
                if radius_squared < -absolute_tolerance:
                    raise RuntimeError(
                        "Numerically inconsistent ball/half-space intersection."
                    )
                circle_radius = np.sqrt(max(radius_squared, 0.0))
                tangent = z - (normal @ z / normal_norm_squared) * normal
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm <= absolute_tolerance:
                    coordinate = int(np.argmin(np.abs(normal)))
                    tangent = np.zeros_like(normal)
                    tangent[coordinate] = 1.0
                    tangent -= normal * (normal @ tangent / normal_norm_squared)
                    tangent_norm = np.linalg.norm(tangent)
                    if (
                        tangent_norm <= absolute_tolerance
                        and circle_radius > absolute_tolerance
                    ):
                        raise RuntimeError("Unable to construct a tangent direction.")
                x = (
                    plane_center
                    if circle_radius <= absolute_tolerance
                    else plane_center + circle_radius * tangent / tangent_norm
                )
                active_case = 4
                description = "ball and half-space active"

    ball_residual = float(np.linalg.norm(x) - radius)
    halfspace_residual = float(normal @ x - beta)
    if (
        ball_residual > 10.0 * absolute_tolerance
        or halfspace_residual > 10.0 * absolute_tolerance
    ):
        raise RuntimeError("Projection failed its feasibility check.")
    return x, BallHalfspaceProjectionInfo(
        active_case,
        description,
        ball_residual,
        halfspace_residual,
    )


def prepare_quadratic_projection(problem: QuadraticCVIProblem) -> PreparedProjection:
    constraints: list[PreparedConstraint] = []
    for constraint_index in range(problem.m):
        matrix = 0.5 * (
            problem.A[constraint_index] + problem.A[constraint_index].T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        negative_tolerance = 1e-10 * max(1.0, np.linalg.norm(matrix, 2))
        if float(np.min(eigenvalues)) < -negative_tolerance:
            raise ValueError(
                f"Constraint matrix {constraint_index} is not positive semidefinite."
            )
        constraints.append(
            PreparedConstraint(
                eigenvectors=eigenvectors,
                eigenvalues=np.maximum(eigenvalues, 0.0),
                b_in_basis=eigenvectors.T @ problem.b[constraint_index],
                c=float(problem.c[constraint_index]),
            )
        )
    return PreparedProjection(constraints)


def _constraint_value(coordinates: Array, constraint: PreparedConstraint) -> float:
    return float(
        0.5 * np.sum(constraint.eigenvalues * coordinates**2)
        + constraint.b_in_basis @ coordinates
        + constraint.c
    )


def _multiplier_solution(
    z_coordinates: Array, multiplier: float, constraint: PreparedConstraint
) -> Array:
    return (z_coordinates - multiplier * constraint.b_in_basis) / (
        1.0 + multiplier * constraint.eigenvalues
    )


def _project_one_constraint(
    z: Array,
    constraint: PreparedConstraint,
    tolerance: float,
    max_bisection_iterations: int,
    max_bracket_expansions: int,
) -> tuple[Array, int]:
    coordinates = constraint.eigenvectors.T @ z
    value_at_zero = _constraint_value(coordinates, constraint)
    if value_at_zero <= tolerance:
        return z.copy(), 0

    lower_multiplier = 0.0
    upper_multiplier = 1.0
    upper_coordinates = _multiplier_solution(
        coordinates, upper_multiplier, constraint
    )
    for _ in range(max_bracket_expansions):
        if _constraint_value(upper_coordinates, constraint) <= 0.0:
            break
        upper_multiplier *= 2.0
        upper_coordinates = _multiplier_solution(
            coordinates, upper_multiplier, constraint
        )
    else:
        raise RuntimeError("Unable to bracket the projection multiplier.")

    best_coordinates = upper_coordinates
    scale = 1.0 + abs(value_at_zero)
    iterations = 0
    for iterations in range(1, max_bisection_iterations + 1):
        trial_multiplier = 0.5 * (lower_multiplier + upper_multiplier)
        trial_coordinates = _multiplier_solution(
            coordinates, trial_multiplier, constraint
        )
        trial_value = _constraint_value(trial_coordinates, constraint)
        if trial_value > 0.0:
            lower_multiplier = trial_multiplier
        else:
            upper_multiplier = trial_multiplier
            best_coordinates = trial_coordinates
        if abs(trial_value) <= tolerance * scale:
            break
    return constraint.eigenvectors @ best_coordinates, iterations


def project_quadratic_intersection(
    z: Array,
    problem: QuadraticCVIProblem,
    options: ProjectionOptions | None = None,
    prepared: PreparedProjection | None = None,
) -> tuple[Array, IntersectionProjectionInfo]:
    """Dykstra projection onto the ball and all original quadratic sets."""

    options = options or ProjectionOptions()
    prepared = prepared or prepare_quadratic_projection(problem)
    if options.max_cycles < 1:
        raise ValueError("max_cycles must be positive.")

    z = np.asarray(z, dtype=float).reshape(-1)
    initial_values = problem.G(z)
    initial_max_violation = float(
        max(np.linalg.norm(z) - problem.r, np.max(initial_values))
    )
    if initial_max_violation <= options.feasibility_tolerance:
        return z.copy(), IntersectionProjectionInfo(
            True, 0, 0, 0, initial_max_violation
        )

    corrections = np.zeros((problem.m + 1, problem.n), dtype=float)
    x = z.copy()
    total_constraint_calls = 0
    total_bisection_iterations = 0
    converged = False
    max_violation = initial_max_violation

    for cycle in range(1, options.max_cycles + 1):
        cycle_start = x.copy()
        shifted = x + corrections[0]
        x = project_ball(shifted, problem.r)
        corrections[0] = shifted - x

        for constraint_index, constraint in enumerate(prepared.constraints):
            shifted = x + corrections[constraint_index + 1]
            x, bisection_iterations = _project_one_constraint(
                shifted,
                constraint,
                options.tolerance,
                options.max_bisection_iterations,
                options.max_bracket_expansions,
            )
            corrections[constraint_index + 1] = shifted - x
            total_constraint_calls += 1
            total_bisection_iterations += bisection_iterations

        max_violation = float(
            max(np.linalg.norm(x) - problem.r, np.max(problem.G(x)))
        )
        relative_change = np.linalg.norm(x - cycle_start) / (1.0 + np.linalg.norm(x))
        if (
            relative_change <= options.tolerance
            and max_violation <= options.feasibility_tolerance
        ):
            converged = True
            break

    info = IntersectionProjectionInfo(
        converged,
        cycle,
        total_constraint_calls,
        total_bisection_iterations,
        max_violation,
    )
    if not converged and options.fail_on_nonconvergence:
        raise RuntimeError(
            "Dykstra projection did not converge in "
            f"{options.max_cycles} cycles; maximum feasibility violation is "
            f"{max_violation:.3e}."
        )
    return x, info
