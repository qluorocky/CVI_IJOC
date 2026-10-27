"""Adaptive projected operator-extrapolation benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .problem import QuadraticCVIProblem
    from .projections import (
        ProjectionOptions,
        prepare_quadratic_projection,
        project_quadratic_intersection,
    )
except ImportError:
    from problem import QuadraticCVIProblem
    from projections import (
        ProjectionOptions,
        prepare_quadratic_projection,
        project_quadratic_intersection,
    )


Array = NDArray[np.float64]


@dataclass
class ProjectedOEOptions:
    N: int
    M0: float
    x0: Array
    verbose: bool = True
    operator_tolerance: float = 1e-12
    max_operator_backtracks: int = 60
    projection_options: ProjectionOptions = field(default_factory=ProjectionOptions)


def projected_oe(
    problem: QuadraticCVIProblem, options: ProjectedOEOptions
) -> tuple[Array, dict[str, Any]]:
    """Run adaptive OE with direct projection onto the original feasible set."""

    if options.N < 1:
        raise ValueError("N must be positive.")
    if options.M0 <= 0.0:
        raise ValueError("M0 must be positive.")
    if options.operator_tolerance < 0.0:
        raise ValueError("operator_tolerance must be nonnegative.")
    if options.max_operator_backtracks < 0:
        raise ValueError("max_operator_backtracks must be nonnegative.")
    x0 = np.asarray(options.x0, dtype=float).reshape(-1)
    if x0.size != problem.n:
        raise ValueError("x0 has the wrong dimension.")

    preparation_start = perf_counter()
    prepared = prepare_quadratic_projection(problem)
    preparation_time = perf_counter() - preparation_start
    projection_start = perf_counter()
    x_previous, initial_projection = project_quadratic_intersection(
        x0, problem, options.projection_options, prepared
    )
    total_projection_time = preparation_time + perf_counter() - projection_start

    x_previous2 = x_previous.copy()
    F_previous = problem.F(x_previous)
    F_previous2 = F_previous.copy()
    F_evaluations = 1
    M_previous = options.M0
    total_operator_backtracks = 0
    x_bar = x_previous.copy()
    N = options.N

    history: dict[str, Any] = {
        "x": np.full((problem.n, N + 1), np.nan),
        "x_bar": np.full((problem.n, N + 1), np.nan),
        "max_constraint_violation": np.full(N + 1, np.nan),
        "step_norm": np.full(N, np.nan),
        "step_size": np.full(N, np.nan),
        "operator_estimate": np.full(N, np.nan),
        "line_search_trials": np.full(N, np.nan),
        "operator_backtracks": np.full(N, np.nan),
        "operator_test_violation": np.full(N, np.nan),
        "projection_cycles": np.full(N, np.nan),
        "projection_constraint_calls": np.full(N, np.nan),
        "projection_bisection_iterations": np.full(N, np.nan),
        "projection_time": np.full(N, np.nan),
        "initial_projection": initial_projection,
        "initial_operator_estimate": options.M0,
    }
    history["x"][:, 0] = x_previous
    history["x_bar"][:, 0] = x_bar
    history["max_constraint_violation"][0] = max(
        float(np.max(problem.G(x_bar))), 0.0
    )

    for outer_index in range(N):
        k = outer_index + 1
        q = 0
        iteration_projection_time = 0.0
        iteration_projection_cycles = 0
        iteration_constraint_calls = 0
        iteration_bisection_iterations = 0

        while True:
            if q > options.max_operator_backtracks:
                raise RuntimeError(
                    "Projected-OE line search exceeded "
                    f"{options.max_operator_backtracks} doublings at k={k}."
                )
            M_trial = 2.0**q * M_previous
            theta = 1.0 if k == 1 else M_trial / M_previous
            step_size = 1.0 / (2.0 * M_trial)
            extrapolated_operator = F_previous + theta * (
                F_previous - F_previous2
            )
            projection_input = x_previous - step_size * extrapolated_operator

            projection_start = perf_counter()
            x_candidate, projection_info = project_quadratic_intersection(
                projection_input,
                problem,
                options.projection_options,
                prepared,
            )
            projection_time = perf_counter() - projection_start
            iteration_projection_time += projection_time
            iteration_projection_cycles += projection_info.cycles
            iteration_constraint_calls += projection_info.constraint_projection_calls
            iteration_bisection_iterations += projection_info.bisection_iterations

            F_candidate = problem.F(x_candidate)
            F_evaluations += 1
            operator_left = np.linalg.norm(F_candidate - F_previous)
            operator_right = M_trial * np.linalg.norm(x_candidate - x_previous)
            operator_slack = options.operator_tolerance * (
                1.0 + operator_left + operator_right
            )
            operator_test_violation = (
                operator_left - operator_right - operator_slack
            )
            if operator_test_violation <= 0.0:
                break
            q += 1
            total_operator_backtracks += 1

        x_current = x_candidate
        F_current = F_candidate
        total_projection_time += iteration_projection_time
        x_bar = outer_index / k * x_bar + x_current / k

        history["x"][:, outer_index + 1] = x_current
        history["x_bar"][:, outer_index + 1] = x_bar
        history["max_constraint_violation"][outer_index + 1] = max(
            float(np.max(problem.G(x_bar))), 0.0
        )
        history["step_norm"][outer_index] = np.linalg.norm(
            x_current - x_previous
        )
        history["step_size"][outer_index] = step_size
        history["operator_estimate"][outer_index] = M_trial
        history["line_search_trials"][outer_index] = q + 1
        history["operator_backtracks"][outer_index] = q
        history["operator_test_violation"][outer_index] = operator_test_violation
        history["projection_cycles"][outer_index] = iteration_projection_cycles
        history["projection_constraint_calls"][outer_index] = (
            iteration_constraint_calls
        )
        history["projection_bisection_iterations"][outer_index] = (
            iteration_bisection_iterations
        )
        history["projection_time"][outer_index] = iteration_projection_time

        x_previous2 = x_previous
        x_previous = x_current
        F_previous2 = F_previous
        F_previous = F_current
        M_previous = M_trial

        if options.verbose and (
            k == 1 or k == N or k % max(1, N // 10) == 0
        ):
            print(
                f"Projected OE k={k}/{N}, max G_+="
                f"{history['max_constraint_violation'][outer_index + 1]:.3e}, "
                f"M={M_trial:.3e}, trials={q + 1}, projection cycles="
                f"{iteration_projection_cycles}, time={iteration_projection_time:.3f}s"
            )

    history["total_F_evaluations"] = F_evaluations
    history["total_operator_backtracks"] = total_operator_backtracks
    history["total_projection_time"] = total_projection_time
    history["projection_preparation_time"] = preparation_time
    history["total_projection_cycles"] = int(
        np.sum(history["projection_cycles"]) + initial_projection.cycles
    )
    history["total_constraint_projection_calls"] = int(
        np.sum(history["projection_constraint_calls"])
        + initial_projection.constraint_projection_calls
    )
    history["total_projection_bisection_iterations"] = int(
        np.sum(history["projection_bisection_iterations"])
        + initial_projection.bisection_iterations
    )
    history["options"] = options
    return x_bar, history
