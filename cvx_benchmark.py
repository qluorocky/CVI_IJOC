"""Projected operator extrapolation and independent weak-gap postprocessing."""

from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.sparse.linalg import svds


SOLVER_TOL = 1e-7
OE_SETTINGS = ((1.0, "Projected OE (M)"),
               (0.1, "Projected OE (0.1M)"),
               (10.0, "Projected OE (10M)"))


def operator_lipschitz(problem):
    """Numerical spectral norm of the realized operator, outside solver timing."""
    return float(svds(problem.M, k=1, which="LM", tol=1e-12,
                      v0=np.ones(problem.n), return_singular_vectors=False)[0])


def _constraints(problem, variable):
    quadratic = 0.5 * (problem.a @ cp.square(variable))
    return [cp.norm(variable, 2) <= problem.r,
            quadratic + problem.b @ variable + problem.c <= 0]


def _solve(model, remaining=None, warm_start=False):
    kwargs = dict(solver=cp.CLARABEL, verbose=False, warm_start=warm_start,
                  tol_feas=SOLVER_TOL, tol_gap_abs=SOLVER_TOL,
                  tol_gap_rel=SOLVER_TOL)
    if remaining is not None:
        kwargs["time_limit"] = max(remaining, 1e-4)
    model.solve(**kwargs)
    return model.status == cp.OPTIMAL


class GenericProjection:
    def __init__(self, problem):
        self.problem = problem

    def project(self, z, remaining):
        # Construct and solve a fresh generic model on every OE trial.
        # Model construction and canonicalization are included in OE's clock.
        variable = cp.Variable(self.problem.n)
        model = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(variable - z)),
            _constraints(self.problem, variable),
        )
        if not _solve(model, remaining, warm_start=False) or variable.value is None:
            raise RuntimeError(f"Projection solver returned {model.status}.")
        x = np.asarray(variable.value).reshape(-1)
        if np.linalg.norm(x) > self.problem.r + 2e-6 or np.max(self.problem.G(x)) > 2e-6:
            raise RuntimeError("Projected point failed independent feasibility check.")
        return x, True


class WeakGap:
    def __init__(self, problem):
        self.problem = problem
        y = cp.Variable(problem.n)
        self.linear = cp.Parameter(problem.n)
        self.y = y
        self.model = cp.Problem(
            cp.Minimize(cp.sum(cp.multiply(problem.h, cp.square(y))) - self.linear @ y),
            _constraints(problem, y),
        )

    def evaluate(self, x):
        self.linear.value = np.asarray(self.problem.M.T @ x + self.problem.d).reshape(-1)
        # Diagnostic postprocessing is outside the timed OE and proposed runs.
        if not _solve(self.model, warm_start=True) or self.y.value is None:
            return float("nan"), str(self.model.status)
        raw = -float(self.model.value) - self.problem.d @ x
        return max(raw, 0.0), str(self.model.status)


def projected_oe(problem, seconds, max_iterations=10000, M0=1.0):
    """Original adaptive OE recursion; full nonlinear projection is generic CVX."""
    if not np.isfinite(M0) or M0 <= 0:
        raise ValueError("M0 must be finite and positive.")
    start = perf_counter()
    deadline = start + seconds
    projection = GenericProjection(problem)
    x_prev = problem.feasible_point.copy()
    x_prev2 = x_prev.copy()
    f_prev = problem.F(x_prev)
    f_prev2 = f_prev.copy()
    avg = x_prev.copy()
    M_prev = float(M0)
    F_calls = 1
    projection_calls = 0
    projection_time = 0.0
    status = "Time limit"
    history = [dict(k=0, elapsed=0.0, F_calls=1, projection_calls=0,
                    projection_time=0.0, x=avg.copy())]
    for k in range(1, max_iterations + 1):
        if perf_counter() >= deadline:
            break
        for q in range(61):
            if perf_counter() >= deadline:
                break
            M = 2.0**q * M_prev
            theta = 1.0 if k == 1 else M / M_prev
            z = x_prev - (f_prev + theta * (f_prev - f_prev2)) / (2 * M)
            clock = perf_counter()
            try:
                x_new, called = projection.project(z, deadline - clock)
            except Exception as exc:
                status = ("Time limit" if perf_counter() >= deadline - 0.01
                          else f"Projection failure: {exc}")
                break
            projection_time += perf_counter() - clock
            projection_calls += int(called)
            if perf_counter() >= deadline:
                break
            f_new = problem.F(x_new)
            F_calls += 1
            lhs = np.linalg.norm(f_new - f_prev)
            rhs = M * np.linalg.norm(x_new - x_prev)
            if lhs <= rhs + 1e-12 * (1 + lhs + rhs):
                break
        else:
            status = "Operator line search failed"
            break
        if status.startswith("Projection failure") or perf_counter() >= deadline:
            break
        avg += (x_new - avg) / k
        history.append(dict(k=k, elapsed=perf_counter() - start, F_calls=F_calls,
                            projection_calls=projection_calls,
                            projection_time=projection_time, x=avg.copy()))
        x_prev2, x_prev = x_prev, x_new
        f_prev2, f_prev = f_prev, f_new
        M_prev = M
    else:
        status = "Iteration cap"
    return dict(history=history, status=status, runtime=perf_counter()-start,
                initial_M=float(M0), final_M=M_prev,
                F_calls=F_calls, projection_calls=projection_calls,
                projection_time=projection_time)
