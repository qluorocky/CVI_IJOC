#!/usr/bin/env python3
"""Read-only, bounded-memory offline weak-gap recovery experiment.

This file is outside the frozen measured-method source. It never edits a case.
Its results are numerical optimization evidence, not formal gap certificates.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import time

import numpy as np
import scipy
from scipy.optimize import brentq, minimize, nnls
from threadpoolctl import threadpool_limits


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def vector_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class ReadOnlyProblem:
    """Independent generator, using existing C-order factors when available."""
    def __init__(self, case):
        self.case = Path(case).resolve()
        self.manifest = json.loads((self.case / "manifest.json").read_text())
        self.n, self.m, self.seed = (self.manifest[k] for k in ("n", "m", "seed"))
        self.r = float(self.manifest["radius"])
        n, m, rank = self.n, self.m, self.n - self.n // 2
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, n, 101]))
        raw = rng.uniform(0., 1., (n, n))
        self.h = np.r_[rng.uniform(0., 1., rank), np.zeros(n // 2)]
        self.B = raw - raw.T + np.diag(self.h)
        del raw
        self.d = rng.uniform(0., 1., n)
        self.x0 = -self.d / np.linalg.norm(self.d)
        factor_file = self.case / "quadratic_factors.dat"
        self.memmap = factor_file.exists()
        self.factor_bytes = m * rank * n * 8
        if self.memmap:
            if factor_file.stat().st_size != self.factor_bytes:
                raise ValueError("Factor file size does not match manifest dimensions")
            self.W = np.memmap(factor_file, dtype="float64", mode="r", shape=(m, rank, n))
        else:
            if self.factor_bytes > 128 * 1024**2:
                raise ValueError("Large reconstruction requires the existing read-only factor file")
            self.W = np.empty((m, rank, n), order="C")
        self.b, self.c = np.empty((m, n)), np.empty(m)
        for i in range(m):
            local = np.random.default_rng(np.random.SeedSequence([self.seed, n, i, 202]))
            # Consume exactly the generator's QR input, but do no QR for memmap cases.
            raw = local.uniform(0., 1., (n, n))
            if not self.memmap:
                q, _ = np.linalg.qr(raw)
            del raw
            values = local.uniform(0., 1., rank)
            if not self.memmap:
                self.W[i] = np.sqrt(values)[:, None] * q[:, :rank].T
            self.b[i] = local.uniform(0., 1., n)
            wx = self.W[i] @ self.x0
            self.c[i] = -.5 * (wx @ wx) - self.b[i] @ self.x0 - .02
        h = hashlib.sha256()
        for a in (self.B, self.d, self.b, self.c, self.h):
            h.update(a.tobytes())
        for wi in self.W:
            h.update(memoryview(wi).cast("B"))
        self.instance_hash = h.hexdigest()
        if self.instance_hash != self.manifest["instance_hash"]:
            raise ValueError(f"Instance hash mismatch: {self.instance_hash}")
        self.full_checks = 0
        self.factor_products = 0

    def G(self, x):
        """Factorwise ALL-original-constraint check, with no A_i stacks."""
        self.full_checks += 1
        g = self.b @ x + self.c
        for i in range(self.m):
            wx = self.W[i] @ x
            g[i] += .5 * (wx @ wx)
        self.factor_products += self.m
        return g


class TimedOut(Exception):
    pass


def ball_minimizer(h, linear, radius):
    """Minimize the diagonal convex objective over the Euclidean ball."""
    def at(lam):
        return linear / (2 * h + lam)
    lo, hi = np.finfo(float).tiny, max(1., np.linalg.norm(linear) / radius)
    while np.linalg.norm(at(hi)) > radius:
        hi *= 2
    lam = brentq(lambda v: np.linalg.norm(at(v)) - radius, lo, hi, xtol=1e-13)
    return at(lam)


def solve_gap(p, z, seconds=60., batch=10, ftol=1e-10, feasibility=1e-7):
    started = time.perf_counter()
    deadline = started + seconds
    linear = p.B.T @ z + p.d
    fun = lambda y: float(p.h @ (y * y) - linear @ y)
    jac = lambda y: 2 * p.h * y - linear
    rounds, active = [], set()
    last_x = p.x0.copy()
    last_result = None
    valid = False
    status = "not_started"

    def check():
        if time.perf_counter() >= deadline:
            raise TimedOut()

    def alarm_handler(signum, frame):
        raise TimedOut()

    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        ball_x = ball_minimizer(p.h, linear, p.r)
        all_g = p.G(ball_x)
        check()
        if np.max(all_g) <= feasibility:
            last_x, valid, status = ball_x, True, "ball_minimizer_full_feasible"
        else:
            active.update(np.argsort(all_g)[-min(batch, p.m):].tolist())
            for round_index in range(p.m + 1):
                check()
                ids = sorted(active)
                cache_x = cache_wx = cache_j = None

                def products(x):
                    nonlocal cache_x, cache_wx, cache_j
                    check()
                    if cache_x is None or not np.array_equal(cache_x, x):
                        cache_x, cache_j = x.copy(), None
                        cache_wx = np.array([p.W[i] @ x for i in ids])
                        p.factor_products += len(ids)
                    return cache_wx

                def con(x):
                    wx = products(x)
                    out = np.r_[p.r**2 - x @ x,
                                -(.5 * np.einsum("ij,ij->i", wx, wx)
                                  + p.b[ids] @ x + p.c[ids])]
                    check()
                    return out

                def cjac(x):
                    nonlocal cache_j
                    wx = products(x)
                    if cache_j is None:
                        cache_j = np.vstack([-2*x,
                            *[-(p.W[i].T @ wx[j] + p.b[i]) for j, i in enumerate(ids)]])
                        p.factor_products += len(ids)
                    check()
                    return cache_j

                def callback(x):
                    nonlocal last_x
                    last_x = x.copy()
                    check()

                t = time.perf_counter()
                out = minimize(fun, last_x, jac=jac, method="SLSQP",
                               constraints=[dict(type="ineq", fun=con, jac=cjac)],
                               callback=callback,
                               options=dict(ftol=ftol, maxiter=2000, disp=False))
                last_result, last_x = out, out.x.copy()
                check()
                all_g = p.G(last_x)
                violation = max(0., float(np.max(all_g)), float(last_x @ last_x - p.r**2))
                info = dict(round=round_index, active_count=len(ids), active_ids=ids,
                            status=int(out.status), success=bool(out.success),
                            message=str(out.message), nit=int(out.nit),
                            seconds=time.perf_counter()-t, objective=float(out.fun),
                            all_original_violation=violation)
                if getattr(out, "multipliers", None) is not None:
                    lam = np.asarray(out.multipliers)
                    info["slsqp_kkt_stationarity_inf"] = float(np.max(np.abs(jac(last_x)-cjac(last_x).T @ lam)))
                    info["slsqp_dual_min"] = float(np.min(lam))
                    info["slsqp_complementarity_inf"] = float(np.max(np.abs(lam * con(last_x))))
                rounds.append(info)
                check()
                if not out.success:
                    status = f"restricted_slsqp_status_{out.status}"
                    break
                if violation <= feasibility:
                    valid, status = True, "restricted_slsqp_full_feasible"
                    break
                worst = [int(i) for i in np.argsort(all_g)[::-1]
                         if all_g[i] > feasibility and i not in active][:batch]
                if not worst:
                    status = "restricted_feasibility_failure"
                    break
                active.update(worst)
    except TimedOut:
        status = "recovery_diagnostic_time_limit"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    elapsed = time.perf_counter() - started
    # One separate final verification is recorded even after the optimization timeout.
    check_t = time.perf_counter()
    final_g = p.G(last_x)
    final_violation = max(0., float(np.max(final_g)), float(last_x @ last_x - p.r**2))
    final_objective = fun(last_x)
    raw_gap = -final_objective - float(p.d @ z)
    if final_violation > feasibility:
        valid = False
    return dict(valid=valid, status=status, gap=max(0., raw_gap) if valid else None,
                raw_gap=raw_gap, raw_objective=final_objective,
                all_original_violation=final_violation,
                seconds=elapsed, final_verification_seconds=time.perf_counter()-check_t,
                budget_seconds=seconds, ftol=ftol, feasibility_tolerance=feasibility,
                batch=batch, rounds=rounds, active_count=len(active), active_ids=sorted(active),
                full_checks=p.full_checks, factor_products=p.factor_products,
                solution_hash=vector_digest(last_x),
                interpretation="numerical restricted optimum plus all-original feasibility; no formal upper gap certificate"), last_x


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--method", default="proposed")
    a.add_argument("--checkpoints", type=int, nargs="+", default=[0])
    a.add_argument("--seconds", type=float, default=60.)
    a.add_argument("--batch", type=int, default=10)
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup_t = time.perf_counter()
    with threadpool_limits(limits=1):
        p = ReadOnlyProblem(args.case)
        setup_seconds = time.perf_counter()-setup_t
        point_file = args.case / f"{args.method}_points.npz"
        saved = np.load(point_file)["x"]
        original = json.loads((args.case / f"{args.method}_diagnostics.json").read_text())
        report = dict(case=str(args.case.resolve()), method=args.method,
                      instance_hash=p.instance_hash, instance_hash_verified=True,
                      source_hashes=p.manifest.get("source_hashes"),
                      recovery_source_sha256=digest_file(__file__),
                      points_file_sha256=digest_file(point_file),
                      numpy=np.__version__, scipy=scipy.__version__,
                      python=platform.python_version(), platform=platform.platform(),
                      threads=1, thread_environment={k:os.environ[k] for k in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
                      readonly_existing_factors=p.memmap, factor_bytes=p.factor_bytes,
                      setup_seconds=setup_seconds, results=[])
        for checkpoint in args.checkpoints:
            p.full_checks = p.factor_products = 0
            result, solution = solve_gap(p, saved[checkpoint], args.seconds, args.batch)
            result["checkpoint"] = checkpoint
            result["input_point_hash"] = vector_digest(saved[checkpoint])
            matches = [r for r in original if r["checkpoint"] == checkpoint]
            result["original_diagnostic"] = matches[-1] if matches else None
            np.savez_compressed(args.output / f"checkpoint_{checkpoint}_solution.npz", x=solution)
            report["results"].append(result)
            report["peak_rss_bytes_including_shared_readonly_factor_mapping"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            (args.output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
            print(json.dumps({k:result[k] for k in ("checkpoint", "valid", "status", "gap", "seconds", "active_count", "all_original_violation")}), flush=True)


if __name__ == "__main__":
    main()
