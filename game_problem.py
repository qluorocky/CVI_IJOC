"""Normalized sparse zero-sum saddle games with shared quadratic constraints."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class GameProblem:
    n: int
    m: int
    r: float
    M: sparse.csr_matrix
    h: np.ndarray
    d: np.ndarray
    a: np.ndarray  # Diagonal quadratic coefficients, one row per constraint.
    b: np.ndarray
    c: np.ndarray
    feasible_point: np.ndarray

    def F(self, x):
        return np.asarray(self.M @ x - self.d)

    def G(self, x):
        return 0.5 * self.a @ (x * x) + self.b @ x + self.c

    def J(self, x):
        return self.a * x[None, :] + self.b

    def g(self, x):
        assert self.m == 1
        return float(self.G(x)[0])

    def grad_g(self, x):
        assert self.m == 1
        return self.J(x)[0]


def make_game(seed: int, n: int, m: int) -> GameProblem:
    if n < 4 or m < 1:
        raise ValueError("Require n >= 4 and m >= 1.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    p = n // 2
    q = n - p
    # C is a sparse payoff matrix with norm at most 1; diag(H) is singular.
    C = sparse.diags([np.full(p, 0.7), np.full(min(p, q - 1), 0.15),
                      np.full(p - 1, -0.15)], [0, 1, -1], shape=(p, q), format="csr")
    S = sparse.bmat([[None, C], [-C.T, None]], format="csr")
    h = rng.uniform(0.1, 0.4, n)
    h[np.arange(n) % 3 == 0] = 0.0
    M = S + sparse.diags(h, format="csr")
    d = rng.normal(size=n)
    d /= np.linalg.norm(d)
    x_f = -0.35 * d

    a = np.zeros((m, n))
    b = np.zeros((m, n))
    c = np.zeros(m)
    for i in range(m):
        local_rng = np.random.default_rng(np.random.SeedSequence([seed, i + 1]))
        # Five resource types, each with distinct slightly perturbed limits.
        family_rng = np.random.default_rng(np.random.SeedSequence([seed, 100 + i % 5]))
        mask = family_rng.random(n) < 0.5
        a[i] = mask * family_rng.uniform(0.1, 0.4, n)
        a[i] *= 1 + local_rng.uniform(-0.02, 0.02, n)
        direction = family_rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        noise = local_rng.normal(size=n)
        noise /= np.linalg.norm(noise)
        b[i] = 0.7 * d + 0.5 * direction + 0.01 * noise
        c[i] = -0.5 * a[i] @ (x_f * x_f) - b[i] @ x_f - 0.04
    # Rescaling leaves the feasible set fixed and controls the vector norm of G.
    scale = 1 / np.sqrt(m)
    game = GameProblem(n, m, 1.0, M, h, d, a * scale, b * scale, c * scale, x_f)
    assert np.linalg.norm(game.feasible_point) < game.r
    assert np.all(game.G(game.feasible_point) < 0)
    return game
