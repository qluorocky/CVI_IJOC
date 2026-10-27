from __future__ import annotations

import numpy as np

from initialization import estimate_initial_constraint_curvature
from pfaos import PFAOSOptions, pfaos
from problem import generate_old_style_synthetic_cvi
from projections import project_ball_halfspace
from verification import verify_synthetic_problem


def test_pfaos_small() -> None:
    problem = generate_old_style_synthetic_cvi(7, 12, 12.0)
    report = verify_synthetic_problem(problem, verbose=False)
    L0, _ = estimate_initial_constraint_curvature(
        problem, 3, 0.5, seed=7001
    )
    options = PFAOSOptions(
        N=8,
        lambda_bound=100.0,
        L0=L0,
        x0=problem.old_experiment_x0,
        verbose=False,
    )
    x_bar, history = pfaos(problem, options)

    assert report["matches_pfaos_assumptions"]
    assert np.all(np.isfinite(x_bar))
    assert np.linalg.norm(x_bar) <= problem.r * (1.0 + 1e-9)
    assert np.all(np.linalg.norm(history["x_bar"], axis=0) <= problem.r * (1.0 + 1e-9))
    assert np.max(history["balance_invariant_error"]) <= 1e-8
    assert np.max(history["accepted_middle_test_violation"]) <= 1e-12
    assert np.max(history["max_operator_test_violation"]) <= 1e-12
    assert history["total_F_evaluations"] >= options.N + 1
    assert history["total_grad_g_evaluations"] >= options.N


def test_simultaneous_ball_halfspace_projection() -> None:
    projected, info = project_ball_halfspace(
        np.array([2.0, 2.0]), 1.0, np.array([1.0, 0.0]), 0.5
    )
    assert info.case == 4
    assert abs(np.linalg.norm(projected) - 1.0) <= 1e-10
    assert abs(projected[0] - 0.5) <= 1e-10
