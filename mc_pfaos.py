"""Multi-constraint parameter-free accelerated operator sliding."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .problem import QuadraticCVIProblem
    from .projections import project_ball
except ImportError:
    from problem import QuadraticCVIProblem
    from projections import project_ball


Array = NDArray[np.float64]


@dataclass
class MCPFAOSOptions:
    N: int
    lambda_bound: float
    L0: float
    x0: Array
    verbose: bool = True
    operator_tolerance: float = 1e-12
    coupling_tolerance: float = 1e-12
    middle_tolerance: float = 1e-12
    integer_tolerance: float = 1e-10
    max_middle_trials: int = 60
    max_operator_backtracks: int = 60
    max_inner_iterations: int = 10_000_000
    time_limit: float | None = None


def _validate(problem: QuadraticCVIProblem, options: MCPFAOSOptions) -> Array:
    if problem.m < 2:
        raise ValueError("MC-PFAOS requires at least two constraints.")
    if options.N < 1:
        raise ValueError("N must be positive.")
    if options.lambda_bound <= 0.0 or options.L0 <= 0.0:
        raise ValueError("lambda_bound and L0 must be positive.")
    x0 = np.asarray(options.x0, dtype=float).reshape(-1)
    if x0.size != problem.n or np.linalg.norm(x0) > problem.r * (1.0 + 1e-12):
        raise ValueError("x0 must belong to the ball X.")
    return x0


def mc_pfaos(
    problem: QuadraticCVIProblem, options: MCPFAOSOptions
) -> tuple[Array, dict[str, Any]]:
    """Implement Algorithm MC-PFAOS from ``paper_2/sec_MC_PFAOS.tex``."""

    x0 = _validate(problem, options)
    start = perf_counter()
    deadline = start + options.time_limit if options.time_limit is not None else float("inf")
    timed_out = False
    N = options.N
    lambda_bound = options.lambda_bound
    lambda0 = np.zeros(problem.m)
    eta_previous = 2.0 * lambda_bound * options.L0
    M_previous_outer = eta_previous
    gamma_weight_previous = np.nan

    x_previous_outer = x0.copy()
    x_previous_penultimate = x0.copy()
    lambda_previous_outer = lambda0.copy()
    lambda_previous_penultimate = lambda0.copy()
    x_bar_previous = x0.copy()
    F_previous_outer = problem.F(x0)
    F_previous_penultimate = F_previous_outer.copy()
    J_previous_lower = problem.J(x0)
    initial_G = problem.G(x0)

    F_evaluations = 1
    G_evaluations = 1
    J_evaluations = 1
    total_backtracks = 0

    history: dict[str, Any] = {
        "x_bar": np.full((problem.n, N + 1), np.nan),
        "max_constraint_violation": np.full(N + 1, np.nan),
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
        "max_coupling_test_violation": np.full(N, np.nan),
        "lambda_norm": np.full(N, np.nan),
        "F_evaluations": np.full(N + 1, np.nan),
        "G_evaluations": np.full(N + 1, np.nan),
        "J_evaluations": np.full(N + 1, np.nan),
        "elapsed": np.full(N + 1, np.nan),
    }
    history["x_bar"][:, 0] = x0
    history["max_constraint_violation"][0] = max(float(np.max(initial_G)), 0.0)
    history["F_evaluations"][0] = F_evaluations
    history["G_evaluations"][0] = G_evaluations
    history["J_evaluations"][0] = J_evaluations
    history["elapsed"][0] = 0.0

    for outer_index in range(N):
        if perf_counter() >= deadline:
            timed_out = True
            break
        k = outer_index + 1
        middle_trial = 1
        M_previous_middle_trial = M_previous_outer
        outer_backtracks = 0
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
            G_lower = problem.G(x_lower)
            J_lower = problem.J(x_lower)
            G_evaluations += 1
            J_evaluations += 1

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
            lambda_inner_previous = lambda_previous_outer.copy()
            lambda_inner_previous2 = lambda_previous_penultimate.copy()
            F_inner_previous = F_previous_outer.copy()
            F_inner_previous2 = F_previous_penultimate.copy()
            M_inner_previous = M_zero
            inverse_M_sum = 0.0
            weighted_x_sum = np.zeros(problem.n)
            max_operator_violation = -np.inf
            max_coupling_violation = -np.inf

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
                            "MC-PFAOS line search exceeded "
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
                    p = 3.0 * M_trial + eta * M_trial * inverse_M_sum

                    extrapolated_operator = F_inner_previous + theta * (
                        F_inner_previous - F_inner_previous2
                    )
                    if t == 1:
                        dual_coupling = J_lower.T @ lambda_inner_previous + theta * (
                            J_previous_lower.T
                            @ (lambda_inner_previous - lambda_inner_previous2)
                        )
                    else:
                        extrapolated_lambda = lambda_inner_previous + theta * (
                            lambda_inner_previous - lambda_inner_previous2
                        )
                        dual_coupling = J_lower.T @ extrapolated_lambda

                    primal_center = (
                        eta * x_previous_outer
                        + p * x_inner_previous
                        - extrapolated_operator
                        - dual_coupling
                    ) / (eta + p)
                    x_candidate = project_ball(primal_center, problem.r)
                    F_candidate = problem.F(x_candidate)
                    F_evaluations += 1

                    affine_constraint = G_lower + J_lower @ (
                        x_candidate - x_lower
                    )
                    dual_center = (
                        eta * lambda_previous_outer
                        + p * lambda_inner_previous
                        + affine_constraint
                    ) / (eta + p)
                    lambda_candidate = np.maximum(dual_center, 0.0)

                    operator_left = np.linalg.norm(
                        F_candidate - F_inner_previous
                    )
                    operator_right = M_trial * np.linalg.norm(
                        x_candidate - x_inner_previous
                    )
                    operator_slack = options.operator_tolerance * (
                        1.0 + operator_left + operator_right
                    )
                    lambda_difference = lambda_candidate - lambda_inner_previous
                    coupling_left = np.linalg.norm(J_lower.T @ lambda_difference)
                    coupling_right = M_trial * np.linalg.norm(lambda_difference)
                    coupling_slack = options.coupling_tolerance * (
                        1.0 + coupling_left + coupling_right
                    )
                    operator_violation = (
                        operator_left - operator_right - operator_slack
                    )
                    coupling_violation = (
                        coupling_left - coupling_right - coupling_slack
                    )
                    if operator_violation <= 0.0 and coupling_violation <= 0.0:
                        max_operator_violation = max(
                            max_operator_violation, operator_violation
                        )
                        max_coupling_violation = max(
                            max_coupling_violation, coupling_violation
                        )
                        break
                    q += 1
                    outer_backtracks += 1
                    total_backtracks += 1

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
                    lambda_inner_previous2 = lambda_inner_previous
                    lambda_inner_previous = lambda_candidate
                    F_inner_previous2 = F_inner_previous
                    F_inner_previous = F_candidate
                    M_inner_previous = M_trial
                    dynamic_horizon = new_horizon
                    t += 1
                else:
                    x_trial = x_candidate
                    x_trial_penultimate = x_inner_previous
                    lambda_trial = lambda_candidate
                    lambda_trial_penultimate = lambda_inner_previous
                    F_trial = F_candidate
                    F_trial_penultimate = F_inner_previous
                    M_trial_final = M_trial
                    T_trial = new_horizon
                    break

            if timed_out:
                break

            balance_error = abs(eta * inverse_M_sum - 1.0)
            if balance_error > 100.0 * options.integer_tolerance:
                raise RuntimeError("MC-PFAOS balance invariant failed.")

            x_hat = weighted_x_sum / inverse_M_sum
            x_bar_candidate = (1.0 - gamma) * x_bar_previous + gamma * x_hat
            G_bar_candidate = problem.G(x_bar_candidate)
            G_evaluations += 1
            remainder = (
                G_bar_candidate
                - G_lower
                - J_lower @ (x_bar_candidate - x_lower)
            )
            middle_left = np.linalg.norm(remainder)
            middle_right = eta / (2.0 * lambda_bound * gamma) * np.linalg.norm(
                x_bar_candidate - x_lower
            ) ** 2
            middle_slack = options.middle_tolerance * (
                1.0 + middle_left + middle_right
            )
            middle_violation = middle_left - middle_right - middle_slack

            if perf_counter() >= deadline:
                timed_out = True
                break

            if middle_violation <= 0.0:
                x_previous_outer = x_trial
                x_previous_penultimate = x_trial_penultimate
                lambda_previous_outer = lambda_trial
                lambda_previous_penultimate = lambda_trial_penultimate
                F_previous_outer = F_trial
                F_previous_penultimate = F_trial_penultimate
                x_bar_previous = x_bar_candidate
                J_previous_lower = J_lower
                eta_previous = eta
                M_previous_outer = M_trial_final
                gamma_weight_previous = eta * gamma

                history["x_bar"][:, outer_index + 1] = x_bar_candidate
                history["max_constraint_violation"][outer_index + 1] = max(
                    float(np.max(G_bar_candidate)), 0.0
                )
                history["gamma"][outer_index] = gamma
                history["eta"][outer_index] = eta
                history["M"][outer_index] = M_trial_final
                history["middle_trials"][outer_index] = middle_trial
                history["inner_iterations"][outer_index] = T_trial
                history["completed_inner_steps_all_trials"][outer_index] = (
                    outer_completed_inner_steps
                )
                history["operator_backtracks"][outer_index] = outer_backtracks
                history["balance_invariant_error"][outer_index] = balance_error
                history["accepted_middle_test_violation"][outer_index] = (
                    middle_violation
                )
                history["max_operator_test_violation"][outer_index] = (
                    max_operator_violation
                )
                history["max_coupling_test_violation"][outer_index] = (
                    max_coupling_violation
                )
                history["lambda_norm"][outer_index] = np.linalg.norm(lambda_trial)
                history["F_evaluations"][outer_index + 1] = F_evaluations
                history["G_evaluations"][outer_index + 1] = G_evaluations
                history["J_evaluations"][outer_index + 1] = J_evaluations
                history["elapsed"][outer_index + 1] = perf_counter() - start

                if options.verbose and (
                    k == 1 or k == N or k % max(1, N // 10) == 0
                ):
                    print(
                        f"MC-PFAOS k={k}/{N}, max G_+="
                        f"{history['max_constraint_violation'][outer_index + 1]:.3e}, "
                        f"S_k={middle_trial}, T_k={T_trial}, eta={eta:.3e}, "
                        f"M={M_trial_final:.3e}"
                    )
                break

            M_previous_middle_trial = M_trial_final
            middle_trial += 1

        if timed_out:
            break

    history["total_F_evaluations"] = F_evaluations
    history["total_G_evaluations"] = G_evaluations
    history["total_J_evaluations"] = J_evaluations
    history["total_operator_backtracks"] = total_backtracks
    history["options"] = options
    history["completed_iterations"] = int(np.count_nonzero(np.isfinite(history["elapsed"])) - 1)
    history["timed_out"] = timed_out
    history["run_time"] = perf_counter() - start
    return x_bar_previous, history
