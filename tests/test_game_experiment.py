"""One independent check of the generated model and the measured OE benchmark."""

import numpy as np

from cvx_benchmark import (OE_SETTINGS, GenericProjection, WeakGap,
                           operator_lipschitz, projected_oe)
from game_problem import make_game


def test_game_and_generic_benchmark():
    p = make_game(3, 12, 5)
    rng = np.random.default_rng(40)
    x, y = rng.normal(size=(2, p.n))
    assert (p.F(x) - p.F(y)) @ (x-y) >= -1e-12
    assert np.count_nonzero(p.h == 0) > 0
    assert np.max(p.G(p.feasible_point)) < 0
    step = y-x
    assert np.allclose(p.G(y)-p.G(x)-p.J(x)@step, 0.5*p.a@(step*step))
    assert np.linalg.norm(p.M.toarray(), 2) <= 1.4
    z = p.feasible_point + 1.3*p.d
    projected, called = GenericProjection(p).project(z, 2.)
    assert called and np.max(p.G(projected)) <= 2e-6
    assert np.linalg.norm(projected) <= p.r + 2e-6
    feasible, called = GenericProjection(p).project(p.feasible_point, 2.)
    assert called and np.allclose(feasible, p.feasible_point, atol=2e-6)
    gap, status = WeakGap(p).evaluate(p.feasible_point)
    assert status == "optimal" and gap >= -1e-12
    for candidate in [p.feasible_point, 0.8*p.feasible_point]:
        if max(p.G(candidate)) <= 0:
            assert gap + 1e-6 >= p.F(candidate) @ (p.feasible_point-candidate)
    oe = projected_oe(p, 0.3, 5)
    assert len(oe["history"]) >= 2
    assert np.max(p.G(oe["history"][-1]["x"])) <= 2e-6
    assert oe["history"][-1]["F_calls"] >= len(oe["history"])
    assert oe["projection_calls"] >= len(oe["history"]) - 1


def test_oe_initial_operator_estimates():
    p = make_game(3, 12, 5)
    M = operator_lipschitz(p)
    np.testing.assert_allclose(M, np.linalg.norm(p.M.toarray(), 2), rtol=1e-10)
    for factor, _ in OE_SETTINGS:
        run = projected_oe(p, 2., max_iterations=1, M0=factor*M)
        assert len(run["history"]) == 2
        assert run["initial_M"] == factor*M
        if factor >= 1:
            # A valid global bound must accept the first trial.
            assert run["projection_calls"] == 1
            assert run["final_M"] == factor*M
        else:
            assert run["projection_calls"] > 1
            assert run["final_M"] > factor*M
