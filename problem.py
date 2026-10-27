"""Synthetic constrained variational-inequality test problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space


Array = NDArray[np.float64]


@dataclass
class QuadraticCVIProblem:
    """Affine monotone VI with convex quadratic constraints on a ball."""

    n: int
    m: int
    r: float
    M: Array
    d: Array
    A: Array
    b: Array
    c: Array
    monotone_part: Array
    feasible_point: Array
    old_experiment_x0: Array
    true_operator_lipschitz: float
    true_constraint_smoothness: Array

    def F(self, x: Array) -> Array:
        return self.M @ np.asarray(x, dtype=float) - self.d

    def G(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        quadratic = 0.5 * np.einsum("i,kij,j->k", x, self.A, x)
        return quadratic + self.b @ x + self.c

    def J(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        return np.einsum("kij,j->ki", self.A, x) + self.b

    def g(self, x: Array) -> float:
        if self.m != 1:
            raise ValueError("g is available only for a single-constraint problem.")
        return float(self.G(x)[0])

    def grad_g(self, x: Array) -> Array:
        if self.m != 1:
            raise ValueError(
                "grad_g is available only for a single-constraint problem."
            )
        return self.J(x)[0]

    @property
    def true_vector_constraint_smoothness(self) -> float:
        return float(np.linalg.norm(self.true_constraint_smoothness))


def _orthogonal_from_uniform(rng: np.random.Generator, n: int) -> Array:
    basis, _ = np.linalg.qr(rng.random((n, n)))
    return basis


def _singular_psd_matrix(
    rng: np.random.Generator, n: int, zeros_at_end: bool = True
) -> tuple[Array, Array]:
    number_of_zeros = n // 2
    positive = rng.random(n - number_of_zeros)
    if zeros_at_end:
        eigenvalues = np.concatenate((positive, np.zeros(number_of_zeros)))
    else:
        eigenvalues = np.concatenate((np.zeros(number_of_zeros), positive))
    basis = _orthogonal_from_uniform(rng, n)
    matrix = basis @ np.diag(eigenvalues) @ basis.T
    return 0.5 * (matrix + matrix.T), eigenvalues


def generate_old_style_synthetic_cvi(
    seed: int, n: int, r: float
) -> QuadraticCVIProblem:
    """Generate the MATLAB folder's single-constraint problem class."""

    if n < 2:
        raise ValueError("n must be at least two.")
    if r <= 0:
        raise ValueError("r must be positive.")

    rng = np.random.default_rng(seed)
    raw = rng.random((n, n))
    skew_part = raw - raw.T
    monotone_part, _ = _singular_psd_matrix(rng, n)
    operator = skew_part + monotone_part
    d = rng.random(n)

    c_scalar = 0.1 * r
    y_base = -c_scalar * d / float(d @ d)
    null_d = null_space(d.reshape(1, -1))
    feasible_point = y_base + null_d[:, 0]
    if np.linalg.norm(feasible_point) > r:
        raise ValueError("Constructed feasible point is outside X; increase r.")

    first_direction = feasible_point / np.linalg.norm(feasible_point)
    constraint_basis = np.column_stack(
        (first_direction, null_space(first_direction.reshape(1, -1)))
    )
    number_of_zeros = n // 2
    constraint_eigenvalues = np.concatenate(
        (np.zeros(number_of_zeros), rng.random(n - number_of_zeros))
    )
    constraint_matrix = (
        constraint_basis
        @ np.diag(constraint_eigenvalues)
        @ constraint_basis.T
    )
    constraint_matrix = 0.5 * (constraint_matrix + constraint_matrix.T)

    A = constraint_matrix[np.newaxis, :, :]
    b = d[np.newaxis, :]
    c = np.array([c_scalar], dtype=float)
    return QuadraticCVIProblem(
        n=n,
        m=1,
        r=float(r),
        M=operator,
        d=d,
        A=A,
        b=b,
        c=c,
        monotone_part=monotone_part,
        feasible_point=feasible_point,
        old_experiment_x0=r * d / np.linalg.norm(d),
        true_operator_lipschitz=float(np.linalg.norm(operator, 2)),
        true_constraint_smoothness=np.array(
            [np.linalg.norm(constraint_matrix, 2)], dtype=float
        ),
    )


def generate_old_style_synthetic_mcvi(
    seed: int, n: int, m: int, r: float
) -> QuadraticCVIProblem:
    """Generate the MATLAB folder's strictly feasible multi-constraint class."""

    if n < 2:
        raise ValueError("n must be at least two.")
    if m < 2:
        raise ValueError("m must be at least two.")
    if r <= 0:
        raise ValueError("r must be positive.")

    rng = np.random.default_rng(seed)
    raw = rng.random((n, n))
    skew_part = raw - raw.T
    monotone_part, _ = _singular_psd_matrix(rng, n)
    operator = skew_part + monotone_part
    d = rng.random(n)
    feasible_point = -(r / 2.0) * d / np.linalg.norm(d)

    A = np.zeros((m, n, n), dtype=float)
    b = rng.random((m, n))
    c = np.zeros(m, dtype=float)
    smoothness = np.zeros(m, dtype=float)
    strict_margin = 0.01 * r

    for constraint_index in range(m - 1):
        A_i, _ = _singular_psd_matrix(rng, n)
        A[constraint_index] = A_i
        b_i = b[constraint_index]
        c[constraint_index] = (
            -0.5 * feasible_point @ A_i @ feasible_point
            - b_i @ feasible_point
            - strict_margin
        )
        smoothness[constraint_index] = np.linalg.norm(A_i, 2)

    b[-1] = d
    c[-1] = -0.01 * float(d @ feasible_point)

    return QuadraticCVIProblem(
        n=n,
        m=m,
        r=float(r),
        M=operator,
        d=d,
        A=A,
        b=b,
        c=c,
        monotone_part=monotone_part,
        feasible_point=feasible_point,
        old_experiment_x0=r * d / np.linalg.norm(d),
        true_operator_lipschitz=float(np.linalg.norm(operator, 2)),
        true_constraint_smoothness=smoothness,
    )
