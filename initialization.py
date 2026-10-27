"""Local initial curvature estimates used by PFAOS and MC-PFAOS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from .problem import QuadraticCVIProblem
except ImportError:
    from problem import QuadraticCVIProblem


Array = NDArray[np.float64]


@dataclass
class CurvatureEstimateDetails:
    quotients: Array
    sampled_curvature: float
    shrink_factor: float


def _sample_points(
    problem: QuadraticCVIProblem,
    number_of_trials: int,
    rng: np.random.Generator,
) -> list[tuple[Array, Array]]:
    pairs: list[tuple[Array, Array]] = []
    for _ in range(number_of_trials):
        direction_u = rng.standard_normal(problem.n)
        direction_v = rng.standard_normal(problem.n)
        direction_u /= np.linalg.norm(direction_u)
        direction_v /= np.linalg.norm(direction_v)
        pairs.append(
            (0.25 * problem.r * direction_u, 0.75 * problem.r * direction_v)
        )
    return pairs


def _validate_inputs(number_of_trials: int, shrink_factor: float) -> None:
    if number_of_trials < 1:
        raise ValueError("number_of_trials must be positive.")
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("shrink_factor must lie strictly between zero and one.")


def estimate_initial_constraint_curvature(
    problem: QuadraticCVIProblem,
    number_of_trials: int = 4,
    shrink_factor: float = 0.5,
    seed: int | None = None,
) -> tuple[float, CurvatureEstimateDetails]:
    """Estimate single-constraint curvature from local first-order data."""

    if problem.m != 1:
        raise ValueError("This estimator requires a single constraint.")
    _validate_inputs(number_of_trials, shrink_factor)
    rng = np.random.default_rng(seed)
    quotients = np.zeros(number_of_trials, dtype=float)
    for trial, (u, v) in enumerate(_sample_points(problem, number_of_trials, rng)):
        difference = v - u
        numerator = 2.0 * (
            problem.g(v) - problem.g(u) - problem.grad_g(u) @ difference
        )
        quotients[trial] = max(numerator / float(difference @ difference), 0.0)

    sampled_curvature = float(np.max(quotients))
    if sampled_curvature <= 100.0 * np.finfo(float).eps:
        raise RuntimeError("All sampled constraint-curvature quotients were zero.")
    estimate = shrink_factor * sampled_curvature
    return estimate, CurvatureEstimateDetails(
        quotients, sampled_curvature, shrink_factor
    )


def estimate_initial_vector_constraint_curvature(
    problem: QuadraticCVIProblem,
    number_of_trials: int = 4,
    shrink_factor: float = 0.5,
    seed: int | None = None,
) -> tuple[float, CurvatureEstimateDetails]:
    """Estimate vector-constraint curvature from local first-order data."""

    _validate_inputs(number_of_trials, shrink_factor)
    rng = np.random.default_rng(seed)
    quotients = np.zeros(number_of_trials, dtype=float)
    for trial, (u, v) in enumerate(_sample_points(problem, number_of_trials, rng)):
        difference = v - u
        remainder = problem.G(v) - problem.G(u) - problem.J(u) @ difference
        quotients[trial] = (
            2.0 * np.linalg.norm(remainder) / float(difference @ difference)
        )

    sampled_curvature = float(np.max(quotients))
    if sampled_curvature <= 100.0 * np.finfo(float).eps:
        raise RuntimeError("All sampled vector-curvature quotients were zero.")
    estimate = shrink_factor * sampled_curvature
    return estimate, CurvatureEstimateDetails(
        quotients, sampled_curvature, shrink_factor
    )
