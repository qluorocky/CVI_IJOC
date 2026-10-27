from __future__ import annotations

import numpy as np

from initialization import estimate_initial_vector_constraint_curvature
from mc_pfaos import MCPFAOSOptions, mc_pfaos
from problem import generate_old_style_synthetic_mcvi
from projected_oe import ProjectedOEOptions, projected_oe
from verification import verify_synthetic_mcvi_problem


def test_mc_pfaos_and_projected_oe_small() -> None:
    problem = generate_old_style_synthetic_mcvi(13, 8, 3, 10.0)
    report = verify_synthetic_mcvi_problem(problem, verbose=False)
    L0, _ = estimate_initial_vector_constraint_curvature(
        problem, 3, 0.5, seed=13001
    )
    x_bar, history = mc_pfaos(
        problem,
        MCPFAOSOptions(
            N=5,
            lambda_bound=100.0,
            L0=L0,
            x0=problem.feasible_point,
            verbose=False,
        ),
    )
    assert report["matches_mc_pfaos_assumptions"]
    assert np.all(np.isfinite(x_bar))
    assert np.linalg.norm(x_bar) <= problem.r * (1.0 + 1e-9)
    assert np.max(history["balance_invariant_error"]) <= 1e-8
    assert np.max(history["accepted_middle_test_violation"]) <= 1e-12
    assert np.max(history["max_operator_test_violation"]) <= 1e-12
    assert np.max(history["max_coupling_test_violation"]) <= 1e-12

    x_oe, oe_history = projected_oe(
        problem,
        ProjectedOEOptions(
            N=3,
            M0=1.0,
            x0=problem.feasible_point,
            verbose=False,
        ),
    )
    assert np.all(np.isfinite(x_oe))
    assert np.linalg.norm(x_oe) <= problem.r * (1.0 + 1e-8)
    assert np.max(problem.G(x_oe)) <= 1e-8
    assert np.all(oe_history["operator_test_violation"] <= 1e-12)
    assert oe_history["total_F_evaluations"] == 1 + int(
        np.sum(oe_history["line_search_trials"])
    )
