from __future__ import annotations

import numpy as np

from problem import generate_old_style_synthetic_cvi
from projected_oe import ProjectedOEOptions, projected_oe
from projections import (
    ProjectionOptions,
    prepare_quadratic_projection,
    project_quadratic_intersection,
)


def test_projected_oe_and_direct_projection() -> None:
    problem = generate_old_style_synthetic_cvi(11, 8, 10.0)
    projection_options = ProjectionOptions(max_cycles=2000)
    prepared = prepare_quadratic_projection(problem)
    projected, projection_info = project_quadratic_intersection(
        problem.old_experiment_x0,
        problem,
        projection_options,
        prepared,
    )
    assert projection_info.converged
    assert np.linalg.norm(projected) <= problem.r * (1.0 + 1e-8)
    assert np.max(problem.G(projected)) <= 1e-8

    options = ProjectedOEOptions(
        N=5,
        M0=1.0,
        x0=problem.feasible_point,
        verbose=False,
        projection_options=projection_options,
    )
    x_bar, history = projected_oe(problem, options)
    assert np.all(np.isfinite(x_bar))
    assert np.linalg.norm(x_bar) <= problem.r * (1.0 + 1e-8)
    assert np.max(problem.G(x_bar)) <= 1e-8
    assert np.all(history["projection_cycles"] >= 0)
    assert history["total_F_evaluations"] == 1 + int(
        np.sum(history["line_search_trials"])
    )
    assert np.all(history["operator_test_violation"] <= 1e-12)
    assert np.all(np.diff(history["operator_estimate"]) >= 0)
    assert history["total_operator_backtracks"] == int(
        np.sum(history["operator_backtracks"])
    )
