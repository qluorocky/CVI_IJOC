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
import shutil
import signal
import tempfile
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


def completed_case_snapshot(case, require_quiescent=False, stable_single_snapshot=False):
    """Admission evidence only; this function never adopts diagnostic values.

    The controller copies a case into verified only after its worker exits.
    A complete working manifest alone is insufficient because the worker writes
    that state before deleting its factor cache and exiting.
    """
    case = Path(case).resolve()
    if case.parent.name != "verified":
        raise ValueError("Future recovery requires the exited worker's verified snapshot")
    manifest_file, plan_file = case / "manifest.json", case.parent / "plan.json"
    meta, plan = (json.loads(f.read_text()) for f in (manifest_file, plan_file))
    release_scope = plan.get("release_scope", "full_grid")
    if release_scope == "single_only":
        required = {("single",n,1) for n in (10,50,100,200,500,1000)}
        actual = {(s["study"],s["n"],s["m"]) for s in plan["settings"]}
        if actual != required or plan["seeds"] != list(range(10)) or len(plan["methods"]) != 5:
            raise ValueError("Single-only release plan must retain all six dimensions, ten seeds and five methods")
    if stable_single_snapshot and release_scope != "single_only":
        raise ValueError("Stable-target exception is restricted to a single-only release snapshot")
    if meta.get("state") != "complete":
        raise ValueError("Incomplete case refused")
    if meta.get("source_hashes") != plan.get("source_hashes") or meta.get("cap_seconds") != plan.get("seconds"):
        raise ValueError("Manifest/source/budget differs from frozen plan")
    if meta["seed"] not in plan["seeds"] or not any(
            (s["n"], s["m"]) == (meta["n"], meta["m"]) for s in plan["settings"]):
        raise ValueError("Case does not belong to the recorded plan")
    progress_file = case.parent.parent / "progress.json"
    progress = json.loads(progress_file.read_text())
    if case.name in progress.get("active", []) or case.name in progress.get("worker_pids", {}):
        raise ValueError("Active case refused")
    if stable_single_snapshot:
        eligible_names = {f"single_n{n}_m1_seed{seed}" for n in (10,50,100,200,500,1000) for seed in range(10)}
        other_cases = set(progress.get("pending", [])) | set(progress.get("active", [])) | set(progress.get("worker_pids", {}))
        if case.name not in eligible_names or case.name in other_cases or not other_cases <= eligible_names:
            raise ValueError("Stable-target exception refuses pending targets or any nonsingle work")
        if progress.get("state") not in {"waiting_for_single_workers", "complete", "failed"}:
            raise ValueError("Unexpected single-release progress state")
    if require_quiescent and not stable_single_snapshot and (progress.get("state") not in {"complete", "failed"}
                              or progress.get("active") or progress.get("worker_pids")):
        raise ValueError("Heavy offline recovery waits until recorded workers have exited")
    files = {"manifest.json": digest_file(manifest_file)}
    for method in plan["methods"]:
        if meta.get("methods", {}).get(method, {}).get("stage") != "complete":
            raise ValueError(f"Incomplete method refused: {method}")
        run_file = case / f"{method}_run.json"
        point_file = case / f"{method}_points.npz"
        diag_file = case / f"{method}_diagnostics.json"
        run, rows = (json.loads(f.read_text()) for f in (run_file, diag_file))
        expected = {k:meta[k] for k in ("n", "m", "seed", "cap_seconds", "instance_hash", "source_hashes")}
        if any(run.get("metadata", {}).get(k) != v for k, v in expected.items()) or run["metadata"].get("method") != method:
            raise ValueError(f"Run metadata identity mismatch: {method}")
        points_hash = digest_file(point_file)
        if run.get("points_sha256") != points_hash:
            raise ValueError(f"Timed pair checksum mismatch: {method}")
        with np.load(point_file, allow_pickle=False) as saved:
            points = saved["x"]
        checkpoints = run["checkpoints"]
        if points.shape != (len(checkpoints), meta["n"]) or not np.all(np.isfinite(points)) or len(rows) != len(checkpoints) or not len(rows):
            raise ValueError(f"Checkpoint shape/count/finiteness mismatch: {method}")
        previous, previous_k = 0., -1
        for i, (cp, row) in enumerate(zip(checkpoints, rows)):
            elapsed = cp["elapsed"]
            if not np.isfinite(elapsed) or not previous <= elapsed <= plan["seconds"]:
                raise ValueError(f"Invalid checkpoint timing: {method}/{i}")
            if not np.isfinite(cp["k"]) or cp["k"] < previous_k or int(cp["k"]) != cp["k"]:
                raise ValueError(f"Invalid checkpoint iteration index: {method}/{i}")
            if any(row.get(k) != meta[k] for k in ("n", "m", "seed")) or row.get("method") != method or row.get("checkpoint") != i or row.get("elapsed") != elapsed or row.get("k") != cp["k"]:
                raise ValueError(f"Diagnostic/checkpoint identity mismatch: {method}/{i}")
            previous = elapsed
            previous_k = cp["k"]
        files.update({run_file.name:digest_file(run_file), point_file.name:points_hash,
                      diag_file.name:digest_file(diag_file)})
    return dict(case=str(case), manifest_and_data_sha256=files,
                plan_sha256=digest_file(plan_file), plan_file=str(plan_file),
                admitted_methods=list(plan["methods"]),
                release_scope=release_scope, stable_single_snapshot_mode=stable_single_snapshot,
                verified_snapshot=True, recorded_case_worker_absent=True,
                recorded_all_workers_absent=not bool(progress.get("active") or progress.get("worker_pids")),
                adoption_authorized=False)


def assert_snapshot_unchanged(snapshot, require_quiescent=False, stable_single_snapshot=False):
    current = completed_case_snapshot(snapshot["case"], require_quiescent, stable_single_snapshot)
    for key in ("manifest_and_data_sha256", "plan_sha256"):
        if current[key] != snapshot[key]:
            raise ValueError("Original data/plan changed during recovery; sidecar cannot be adopted")


def assert_external_path(path, case):
    path, case = Path(path).resolve(), Path(case).resolve()
    results_root = case.parent.parent
    if path == results_root or path.is_relative_to(results_root):
        raise ValueError("Recovery outputs and scratch storage must stay outside original results")


class ReadOnlyProblem:
    """Independent generator, using existing C-order factors when available."""
    def __init__(self, case, reconstruct_factors=False, scratch_parent=None,
                 reconstruction_seconds=300.):
        self.case = Path(case).resolve()
        self.completion_snapshot = completed_case_snapshot(self.case)
        self.manifest = json.loads((self.case / "manifest.json").read_text())
        self._scratch = None
        self.W = None
        self.factor_reconstruction_seconds = 0.
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
        self.existing_factors = factor_file.exists()
        self.memmap = True
        self.factor_bytes = m * rank * n * 8
        if self.existing_factors:
            if factor_file.stat().st_size != self.factor_bytes:
                raise ValueError("Factor file size does not match manifest dimensions")
            self.W = np.memmap(factor_file, dtype="float64", mode="r", shape=(m, rank, n))
        else:
            if not reconstruct_factors:
                raise ValueError("Deleted cache: explicitly request temporary factor reconstruction")
            scratch_parent = Path(scratch_parent or tempfile.gettempdir()).resolve()
            assert_external_path(scratch_parent, self.case)
            if self.factor_bytes > 128 * 1024**2:
                completed_case_snapshot(self.case, require_quiescent=True)
            reserve = 12 * 1024**3
            if shutil.disk_usage(scratch_parent).free < self.factor_bytes + reserve:
                raise ValueError("Insufficient scratch disk: require factor bytes plus 12 GiB reserve")
            self._scratch = tempfile.TemporaryDirectory(prefix="cvi-recovery-factors-", dir=scratch_parent)
            factor_file = Path(self._scratch.name) / "reconstructed_factors.dat"
            self.W = np.memmap(factor_file, dtype="float64", mode="w+", shape=(m, rank, n))
        self.factor_origin = "existing_readonly" if self.existing_factors else "temporary_reconstructed_readonly"
        self.b, self.c = np.empty((m, n)), np.empty(m)
        construction_started = time.perf_counter()
        for i in range(m):
            if not self.existing_factors and time.perf_counter()-construction_started >= reconstruction_seconds:
                self.close()
                raise TimeoutError("Temporary factor reconstruction deadline; no optimization performed")
            local = np.random.default_rng(np.random.SeedSequence([self.seed, n, i, 202]))
            # Consume exactly the generator's QR input, but do no QR for memmap cases.
            raw = local.uniform(0., 1., (n, n))
            if not self.existing_factors:
                q, _ = np.linalg.qr(raw)
            del raw
            values = local.uniform(0., 1., rank)
            if not self.existing_factors:
                self.W[i] = np.sqrt(values)[:, None] * q[:, :rank].T
                del q
            self.b[i] = local.uniform(0., 1., n)
            wx = self.W[i] @ self.x0
            self.c[i] = -.5 * (wx @ wx) - self.b[i] @ self.x0 - .02
        if not self.existing_factors:
            self.W.flush()
            self.W._mmap.close()
            self.W = np.memmap(factor_file, dtype="float64", mode="r", shape=(m, rank, n))
            self.factor_reconstruction_seconds = time.perf_counter()-construction_started
        self.reconstruction_budget_seconds = reconstruction_seconds
        hash_started = time.perf_counter()
        h = hashlib.sha256()
        for a in (self.B, self.d, self.b, self.c, self.h):
            h.update(a.tobytes())
        for wi in self.W:
            h.update(memoryview(wi).cast("B"))
        self.instance_hash = h.hexdigest()
        self.instance_hash_seconds = time.perf_counter()-hash_started
        if self.instance_hash != self.manifest["instance_hash"]:
            self.close()
            raise ValueError(f"Instance hash mismatch: {self.instance_hash}")
        self.full_checks = 0
        self.factor_products = 0

    def close(self):
        if isinstance(self.W, np.memmap) and not self.W._mmap.closed:
            self.W._mmap.close()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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


def restore_restricted_start(p, x, ids):
    """Move toward the known strict-feasible x0, using exact segment quadratics.

    The previous restricted optimizer need not satisfy newly added constraints.
    A strictly feasible restricted start avoids avoidable SLSQP line-search errors.
    """
    delta = x - p.x0
    if np.linalg.norm(delta) == 0:
        return x.copy(), 1.
    limit = 1.
    triples = [(float(delta @ delta), float(2*p.x0 @ delta),
                float(p.x0 @ p.x0 - p.r**2))]
    for i in ids:
        w0, wd = p.W[i] @ p.x0, p.W[i] @ delta
        triples.append((float(.5*(wd @ wd)), float(w0 @ wd + p.b[i] @ delta),
                        float(.5*(w0 @ w0)+p.b[i] @ p.x0+p.c[i])))
    p.factor_products += 2*len(ids)
    for a, b, c in triples:
        if a+b+c <= -1e-10:
            continue
        if a > 0:
            disc = np.sqrt(max(0., b*b-4*a*c))
            root = (-2*c/(b+disc)) if b+disc > 0 else (-b+disc)/(2*a)
        elif b > 0:
            root = -c/b
        else:
            continue
        limit = min(limit, root)
    weight = .999*max(0., min(1., limit)) if limit < 1 else 1.
    return p.x0 + weight*delta, float(weight)


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
                last_x, start_segment_weight = restore_restricted_start(p, last_x, ids)
                check()
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
                            start_segment_weight=start_segment_weight,
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
    final_objective = fun(last_x)
    raw_gap = -final_objective - float(p.d @ z)
    finite = bool(np.all(np.isfinite(last_x)) and np.all(np.isfinite(final_g))
                  and np.isfinite(final_objective) and np.isfinite(raw_gap))
    final_violation = (max(0., float(np.max(final_g)), float(last_x @ last_x-p.r**2))
                       if finite else float("inf"))
    full_feasible = finite and final_violation <= feasibility
    if not full_feasible:
        valid = False
    if not finite:
        status = "recovery_nonfinite_result"
    final_verification_seconds = time.perf_counter()-check_t
    # Additional numerical optimality evidence; ordinary floats are not intervals.
    # Convexity on the radius-r ball gives p* >= L(y)-(r+||y||)||grad L(y)||.
    quality_t = time.perf_counter()
    quality_budget = 5.
    quality_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, quality_budget)
    try:
        if not finite:
            raise ValueError("Nonfinite candidate cannot receive numerical quality bounds")
        near_ids = [i for i in sorted(active) if abs(final_g[i]) <= 1e-6]
        gradients, constraint_values = [], []
        for i in near_ids:
            gradients.append(p.W[i].T @ (p.W[i] @ last_x) + p.b[i])
            constraint_values.append(final_g[i])
        ball_g = float(last_x @ last_x - p.r**2)
        if abs(ball_g) <= 1e-6:
            gradients.append(2*last_x)
            constraint_values.append(ball_g)
        if gradients:
            amat = np.column_stack(gradients)
            multipliers, residual_norm = nnls(amat, -jac(last_x), maxiter=10*len(gradients))
            lagrangian = final_objective + float(multipliers @ constraint_values)
            objective_lower = float(lagrangian - (p.r+np.linalg.norm(last_x))*residual_norm)
            if not np.isfinite(objective_lower) or not np.isfinite(residual_norm):
                raise ValueError("Nonfinite multiplier-derived objective estimate")
            quality = dict(status="evaluated",
                           multiplier_fit="nonnegative_least_squares_near_active_original_gradients",
                           near_active_constraint_ids=near_ids, multiplier_count=len(multipliers),
                           stationarity_l2=float(residual_norm),
                           complementarity_inf=float(np.max(np.abs(multipliers*constraint_values))),
                           numerical_gap_upper=max(0., -objective_lower-float(p.d @ z)),
                           numerical_gap_lower=max(0., raw_gap) if full_feasible else None,
                           objective_lower=objective_lower,
                           objective_upper=final_objective if full_feasible else None,
                           numerical_objective_bracket_width=float(final_objective-objective_lower) if full_feasible else None,
                           candidate_minus_lower_residual=float(final_objective-objective_lower),
                           full_feasibility_established=full_feasible,
                           disclaimer="floating-point convexity/KKT estimate, not a formal interval certificate")
        else:
            quality = dict(status="no_near_active_gradients", multiplier_fit="none")
    except Exception as error:
        # Preserve solver status/candidate when this separate check fails.
        quality = dict(status="quality_time_limit" if isinstance(error, TimedOut) else "quality_error",
                       error=f"{type(error).__name__}: {error}")
        valid = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, quality_handler)
    quality.update(seconds=time.perf_counter()-quality_t, budget_seconds=quality_budget)
    return dict(valid=valid, status=status, gap=max(0., raw_gap) if valid else None,
                raw_gap=raw_gap, raw_objective=final_objective,
                all_original_violation=final_violation,
                seconds=elapsed, final_verification_seconds=final_verification_seconds,
                budget_seconds=seconds, ftol=ftol, feasibility_tolerance=feasibility,
                batch=batch, rounds=rounds, active_count=len(active), active_ids=sorted(active),
                full_checks=p.full_checks, factor_products=p.factor_products,
                solution_hash=vector_digest(last_x),
                numerical_optimality_check=quality,
                interpretation="numerical restricted optimum plus all-original feasibility; no formal upper gap certificate"), last_x


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--method", default="proposed")
    a.add_argument("--checkpoints", type=int, nargs="+", default=[0])
    a.add_argument("--seconds", type=float, default=60.)
    a.add_argument("--batch", type=int, default=10)
    a.add_argument("--reconstruct-factors", action="store_true")
    a.add_argument("--scratch-parent", type=Path)
    a.add_argument("--reconstruction-seconds", type=float, default=300.)
    a.add_argument("--attempt-label", default="manual_offline_retry")
    a.add_argument("--stable-single-snapshot", action="store_true",
                   help="Read-only completed target in a validated single-only release; other single workers may finish")
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    if not 0 < args.seconds <= 300 or not 0 < args.reconstruction_seconds <= 3600 or args.batch < 1:
        raise ValueError("Invalid separate offline budget or constraint batch")
    if any(i < 0 for i in args.checkpoints) or len(set(args.checkpoints)) != len(args.checkpoints):
        raise ValueError("Checkpoint indices must be nonnegative and unique")
    snapshot = completed_case_snapshot(args.case, require_quiescent=True,
                                       stable_single_snapshot=args.stable_single_snapshot)
    if args.method not in snapshot["admitted_methods"]:
        raise ValueError("Requested method does not belong to the validated plan")
    assert_external_path(args.output, args.case)
    if args.scratch_parent is not None:
        assert_external_path(args.scratch_parent, args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    setup_t = time.perf_counter()
    try:
        with threadpool_limits(limits=1), ReadOnlyProblem(
                args.case, args.reconstruct_factors, args.scratch_parent,
                args.reconstruction_seconds) as p:
            setup_seconds = time.perf_counter()-setup_t
            point_file = args.case / f"{args.method}_points.npz"
            with np.load(point_file, allow_pickle=False) as saved_file:
                saved = saved_file["x"]
            original = json.loads((args.case / f"{args.method}_diagnostics.json").read_text())
            report = dict(case=str(args.case.resolve()), method=args.method,
                          attempt_label=args.attempt_label,
                          completion_guard=snapshot, sidecar_only=True,
                          adoption_authorized=False, automatic_retry=False,
                          separate_offline_budget_not_timed_method=True,
                          instance_hash=p.instance_hash, instance_hash_verified=True,
                          source_hashes=p.manifest.get("source_hashes"),
                          recovery_source_sha256=digest_file(__file__),
                          points_file_sha256=digest_file(point_file),
                          numpy=np.__version__, scipy=scipy.__version__,
                          python=platform.python_version(), platform=platform.platform(),
                          threads=1, thread_environment={k:os.environ[k] for k in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
                          readonly_existing_factors=p.existing_factors,
                          factor_origin=p.factor_origin, factor_bytes=p.factor_bytes,
                          factor_reconstruction_seconds=p.factor_reconstruction_seconds,
                          factor_reconstruction_budget_seconds=p.reconstruction_budget_seconds,
                          instance_hash_seconds=p.instance_hash_seconds,
                          setup_seconds=setup_seconds, results=[])
            for checkpoint in args.checkpoints:
                if not 0 <= checkpoint < len(saved):
                    raise ValueError(f"Invalid checkpoint index {checkpoint}")
                p.full_checks = p.factor_products = 0
                result, solution = solve_gap(p, saved[checkpoint], args.seconds, args.batch)
                result["checkpoint"] = checkpoint
                result["input_point_hash"] = vector_digest(saved[checkpoint])
                result["original_diagnostic"] = original[checkpoint]
                assert_snapshot_unchanged(snapshot, require_quiescent=True,
                                           stable_single_snapshot=args.stable_single_snapshot)
                solution_file = args.output / f"checkpoint_{checkpoint}_solution.npz"
                with solution_file.open("xb") as candidate_file:
                    np.savez_compressed(candidate_file, x=solution)
                report["results"].append(result)
                report["peak_rss_bytes_including_file_backed_factor_mapping"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                (args.output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
                print(json.dumps({k:result[k] for k in ("checkpoint", "valid", "status", "gap", "seconds", "active_count", "all_original_violation")}), flush=True)
    except Exception as error:
        failure = dict(case=str(args.case.resolve()), completion_guard=snapshot,
                       recovery_source_sha256=digest_file(__file__),
                       attempt_label=args.attempt_label, adoption_authorized=False,
                       error=f"{type(error).__name__}: {error}",
                       elapsed_since_setup_start=time.perf_counter()-setup_t,
                       note="No original files edited; any completed sidecar results remain preserved")
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
