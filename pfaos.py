"""Parameter-free accelerated operator sliding for one constraint."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .problem import QuadraticCVIProblem
    from .projections import project_ball_halfspace
except ImportError:
    from problem import QuadraticCVIProblem
    from projections import project_ball_halfspace


Array = NDArray[np.float64]


@dataclass
class PFAOSOptions:
    N: int
    lambda_bound: float
    L0: float
    x0: Array
    verbose: bool = True
    projection_tolerance: float = 1e-12
    operator_tolerance: float = 1e-12
    middle_tolerance: float = 1e-12
    integer_tolerance: float = 1e-10
    max_middle_trials: int = 60
    max_operator_backtracks: int = 60
    max_inner_iterations: int = 10_000_000
    time_limit: float | None = None


def _validate(problem: QuadraticCVIProblem, options: PFAOSOptions) -> Array:
    if problem.m != 1:
        raise ValueError("PFAOS requires exactly one constraint.")
    if options.N < 1:
        raise ValueError("N must be positive.")
    if options.lambda_bound <= 0.0 or options.L0 <= 0.0:
        raise ValueError("lambda_bound and L0 must be positive.")
    x0 = np.asarray(options.x0, dtype=float).reshape(-1)
    if x0.size != problem.n or np.linalg.norm(x0) > problem.r * (1.0 + 1e-12):
        raise ValueError("x0 must belong to the ball X.")
    return x0


def pfaos(
    problem: QuadraticCVIProblem, options: PFAOSOptions
) -> tuple[Array, dict[str, Any]]:
    """Implement Algorithm PFAOS from ``paper_2/sec_PFAOS.tex``."""

    x0 = _validate(problem, options)
    start = perf_counter()
    deadline = start + options.time_limit if options.time_limit is not None else float("inf")
    timed_out = False
    N = options.N
    lambda_bound = options.lambda_bound
    eta_previous = 2.0 * lambda_bound * options.L0
    M_previous_outer = eta_previous
    gamma_weight_previous = np.nan

    x_previous_outer = x0.copy()
    x_previous_penultimate = x0.copy()
    x_bar_previous = x0.copy()
    F_previous_outer = problem.F(x0)
    F_previous_penultimate = F_previous_outer.copy()

    F_evaluations = 1
    g_evaluations = 1
    grad_g_evaluations = 0
    total_operator_backtracks = 0
    projection_case_counts = np.zeros(4, dtype=int)

    history: dict[str, Any] = {
        # Checkpoints are columns; contiguous columns avoid touching unused pages.
        "x_bar": np.empty((problem.n, N + 1), order="F"),
        "x_lower": np.empty((problem.n, N), order="F"),
        "g_value": np.full(N + 1, np.nan),
        "gamma": np.full(N, np.nan),
        "eta": np.full(N, np.nan),
        "M": np.full(N, np.nan),
        "middle_trials": np.full(N, np.nan),
        "inner_iterations": np.full(N, np.nan),
        "completed_inner_steps_all_trials": np.full(N, np.nan),
        "operator_backtracks": np.full(N, np.nan),
        "balance_invariant_error": np.full(N, np.nan),
        "accepted_middle_test_violation": np.full(N, np.nan),
        "max_operator_test_violation": np.full(N, np.nan),
        "F_evaluations": np.full(N + 1, np.nan),
        "g_evaluations": np.full(N + 1, np.nan),
        "grad_g_evaluations": np.full(N + 1, np.nan),
        "elapsed": np.full(N + 1, np.nan),
    }
    history["x_bar"][:, 0] = x0
    history["g_value"][0] = problem.g(x0)
    history["F_evaluations"][0] = F_evaluations
    history["g_evaluations"][0] = g_evaluations
    history["grad_g_evaluations"][0] = grad_g_evaluations
    history["elapsed"][0] = 0.0

    for outer_index in range(N):
        if perf_counter() >= deadline:
            timed_out = True
            break
        k = outer_index + 1
        middle_trial = 1
        M_previous_middle_trial = M_previous_outer
        outer_operator_backtracks = 0
        outer_completed_inner_steps = 0

        while True:
            if middle_trial > options.max_middle_trials:
                raise RuntimeError(
                    f"Middle line search exceeded {options.max_middle_trials} "
                    f"trials at k={k}."
                )

            eta = 2.0 ** (middle_trial - 2) * eta_previous
            gamma = (
                1.0
                if k == 1
                else gamma_weight_previous / (gamma_weight_previous + eta)
            )
            x_lower = (1.0 - gamma) * x_bar_previous + gamma * x_previous_outer
            g_lower = problem.g(x_lower)
            grad_lower = problem.grad_g(x_lower)
            g_evaluations += 1
            grad_g_evaluations += 1
            beta = float(grad_lower @ x_lower - g_lower)

            M_zero = max(eta, M_previous_middle_trial)
            initial_horizon_ratio = M_zero / eta
            dynamic_horizon = int(round(initial_horizon_ratio))
            if (
                dynamic_horizon < 1
                or abs(dynamic_horizon - initial_horizon_ratio)
                > options.integer_tolerance * max(1.0, initial_horizon_ratio)
            ):
                raise RuntimeError(
                    "M_k^(0,s)/eta_k^s is not an integer within tolerance."
                )

            t = 1
            x_inner_previous = x_previous_outer.copy()
            x_inner_previous2 = x_previous_penultimate.copy()
            F_inner_previous = F_previous_outer.copy()
            F_inner_previous2 = F_previous_penultimate.copy()
            M_inner_previous = M_zero
            inverse_M_sum = 0.0
            weighted_x_sum = np.zeros(problem.n)
            max_operator_violation = -np.inf

            while True:
                if perf_counter() >= deadline:
                    timed_out = True
                    break
                if t > options.max_inner_iterations:
                    raise RuntimeError(
                        f"Inner loop exceeded {options.max_inner_iterations} "
                        f"iterations at k={k}, s={middle_trial}."
                    )

                q = 0
                while True:
                    if perf_counter() >= deadline:
                        timed_out = True
                        break
                    if q > options.max_operator_backtracks:
                        raise RuntimeError(
                            "Operator line search exceeded "
                            f"{options.max_operator_backtracks} doublings at "
                            f"k={k}, s={middle_trial}, t={t}."
                        )
                    M_trial = 2.0**q * M_inner_previous
                    if k == 1 and t == 1:
                        theta = 1.0
                    elif t == 1:
                        theta = M_trial / M_previous_outer
                    else:
                        theta = M_trial / M_inner_previous

                    p = 2.0 * M_trial + eta * M_trial * inverse_M_sum
                    extrapolated_operator = F_inner_previous + theta * (
                        F_inner_previous - F_inner_previous2
                    )
                    unconstrained_center = (
                        eta * x_previous_outer
                        + p * x_inner_previous
                        - extrapolated_operator
                    ) / (eta + p)
                    x_candidate, projection_info = project_ball_halfspace(
                        unconstrained_center,
                        problem.r,
                        grad_lower,
                        beta,
                        options.projection_tolerance,
                    )
                    projection_case_counts[projection_info.case - 1] += 1

                    F_candidate = problem.F(x_candidate)
                    F_evaluations += 1
                    operator_left = np.linalg.norm(
                        F_candidate - F_inner_previous
                    )
                    operator_right = M_trial * np.linalg.norm(
                        x_candidate - x_inner_previous
                    )
                    operator_slack = options.operator_tolerance * (
                        1.0 + operator_left + operator_right
                    )
                    operator_violation = (
                        operator_left - operator_right - operator_slack
                    )
                    if operator_violation <= 0.0:
                        max_operator_violation = max(
                            max_operator_violation, operator_violation
                        )
                        break
                    q += 1
                    outer_operator_backtracks += 1
                    total_operator_backtracks += 1

                if timed_out:
                    break

                new_horizon = t - 1 + 2**q * (dynamic_horizon - t + 1)
                if new_horizon < t:
                    raise RuntimeError("Dynamic inner-loop horizon became invalid.")
                inverse_M_sum += 1.0 / M_trial
                weighted_x_sum += x_candidate / M_trial
                outer_completed_inner_steps += 1

                if t < new_horizon:
                    x_inner_previous2 = x_inner_previous
                    x_inner_previous = x_candidate
                    F_inner_previous2 = F_inner_previous
                    F_inner_previous = F_candidate
                    M_inner_previous = M_trial
                    dynamic_horizon = new_horizon
                    t += 1
                else:
                    x_trial = x_candidate
                    F_trial = F_candidate
                    x_trial_penultimate = x_inner_previous
                    F_trial_penultimate = F_inner_previous
                    M_trial_final = M_trial
                    T_trial = new_horizon
                    break

            if timed_out:
                break

            balance_error = abs(eta * inverse_M_sum - 1.0)
            if balance_error > 100.0 * options.integer_tolerance:
                raise RuntimeError("PFAOS balance invariant failed.")

            x_hat = weighted_x_sum / inverse_M_sum
            x_bar_candidate = (1.0 - gamma) * x_bar_previous + gamma * x_hat
            g_bar_candidate = problem.g(x_bar_candidate)
            g_evaluations += 1
            model_value = g_lower + grad_lower @ (x_bar_candidate - x_lower)
            model_upper_bound = model_value + eta / (
                2.0 * lambda_bound * gamma
            ) * np.linalg.norm(x_bar_candidate - x_lower) ** 2
            middle_slack = options.middle_tolerance * (
                1.0 + abs(g_bar_candidate) + abs(model_upper_bound)
            )
            middle_violation = g_bar_candidate - model_upper_bound - middle_slack

            if perf_counter() >= deadline:
                timed_out = True
                break

            if middle_violation <= 0.0:
                x_previous_outer = x_trial
                F_previous_outer = F_trial
                x_previous_penultimate = x_trial_penultimate
                F_previous_penultimate = F_trial_penultimate
                x_bar_previous = x_bar_candidate
                eta_previous = eta
                M_previous_outer = M_trial_final
                gamma_weight_previous = eta * gamma

                history["x_bar"][:, outer_index + 1] = x_bar_candidate
                history["x_lower"][:, outer_index] = x_lower
                history["g_value"][outer_index + 1] = g_bar_candidate
                history["gamma"][outer_index] = gamma
                history["eta"][outer_index] = eta
                history["M"][outer_index] = M_trial_final
                history["middle_trials"][outer_index] = middle_trial
                history["inner_iterations"][outer_index] = T_trial
                history["completed_inner_steps_all_trials"][outer_index] = (
                    outer_completed_inner_steps
                )
                history["operator_backtracks"][outer_index] = (
                    outer_operator_backtracks
                )
                history["balance_invariant_error"][outer_index] = balance_error
                history["accepted_middle_test_violation"][outer_index] = (
                    middle_violation
                )
                history["max_operator_test_violation"][outer_index] = (
                    max_operator_violation
                )
                history["F_evaluations"][outer_index + 1] = F_evaluations
                history["g_evaluations"][outer_index + 1] = g_evaluations
                history["grad_g_evaluations"][outer_index + 1] = (
                    grad_g_evaluations
                )
                history["elapsed"][outer_index + 1] = perf_counter() - start

                if options.verbose and (
                    k == 1 or k == N or k % max(1, N // 10) == 0
                ):
                    print(
                        f"k={k}/{N}, g(xbar)={g_bar_candidate:.3e}, "
                        f"S_k={middle_trial}, T_k={T_trial}, eta={eta:.3e}, "
                        f"M={M_trial_final:.3e}"
                    )
                break

            M_previous_middle_trial = M_trial_final
            middle_trial += 1

        if timed_out:
            break

    history["constraint_violation"] = np.maximum(history["g_value"], 0.0)
    history["total_F_evaluations"] = F_evaluations
    history["total_g_evaluations"] = g_evaluations
    history["total_grad_g_evaluations"] = grad_g_evaluations
    history["total_operator_backtracks"] = total_operator_backtracks
    history["projection_case_counts"] = projection_case_counts
    history["options"] = options
    history["completed_iterations"] = int(np.count_nonzero(np.isfinite(history["elapsed"])) - 1)
    history["timed_out"] = timed_out
    history["run_time"] = perf_counter() - start
    return x_bar_previous, history
