#!/usr/bin/env python3
"""Read-only structural audit of saved CVI runs, using independent factors.

This auditor never imports the experiment generator, forms an A_i matrix, or
solves a gap problem.  A structural pass is explicitly not numerical gap
validation.  The default coverage contract is the requested 110-instance,
five-method, 1000-second grid.  Use --plan-scope for smaller validation data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits


METHODS = ("proposed", "oe_true", "oe_low", "oe_high", "golden_ratio")
LEGACY_METHODS = ("proposed", "oe_true", "golden_ratio")
OE_M_FACTORS = {"oe_true": 1., "oe_low": .1, "oe_high": 10.}
FULL_SETTINGS = ({("single", n, 1) for n in (10, 50, 100, 200, 500, 1000)} |
                 {("multi", 100, m) for m in (10, 50, 100, 200, 500)})
SUCCESS_GAP_STATUSES = {"optimal", "slsqp_optimal", 0, "independent_dual_bracket_valid"}
HASH_CHUNK = 8 * 1024 * 1024
RECORD_ATOL = 2e-12
RECORD_RTOL = 2e-12
BALL_ATOL = 1e-10
CERTIFICATE_ATOL = 5e-8
CERTIFICATE_RTOL = 2e-8


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_array_hash(digest, array):
    """Stream a C-contiguous array without a whole-array bytes copy."""
    if not array.flags.c_contiguous:
        raise ValueError("Digest input must already be C-contiguous")
    view = memoryview(array).cast("B")
    for offset in range(0, len(view), HASH_CHUNK):
        digest.update(view[offset:offset + HASH_CHUNK])


def array_hash(array):
    digest = hashlib.sha256()
    update_array_hash(digest, np.ascontiguousarray(array, dtype="<f8"))
    return digest.hexdigest()


def finite(value):
    return (isinstance(value, (int, float, np.number)) and
            not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value)))


def close(left, right, atol=RECORD_ATOL, rtol=RECORD_RTOL):
    return finite(left) and finite(right) and bool(np.isclose(left, right, atol=atol, rtol=rtol))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_json(path):
    # Legacy records use JSON NaN for unavailable gaps/certificates.  They are
    # accepted on input, interpreted explicitly below, and emitted as null.
    return json.loads(Path(path).read_text())


class FactorProblem:
    """Independent deterministic reconstruction with disk-backed PSD factors.

    Largest persistent resident arrays are B (n by n), b (m by n), and a
    checkpoint block.  W (m by ceil(n/2) by n) is always a temporary memmap.
    QR construction has O(n^2) transient storage.  No (m,n,n) array exists.
    """

    def __init__(self, n, m, seed, directory):
        self.n, self.m, self.seed = n, m, seed
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, 101]))
        raw = rng.uniform(0., 1., (n, n))
        self.h = np.r_[rng.uniform(0., 1., n - n // 2), np.zeros(n // 2)]
        self.B = raw - raw.T + np.diag(self.h)
        del raw
        self.d = rng.uniform(0., 1., n)
        self.x0 = -self.d / np.linalg.norm(self.d)
        rank = n - n // 2
        self.factor_path = Path(directory) / "independent_quadratic_factors.dat"
        self.W = np.memmap(self.factor_path, mode="w+", dtype=np.float64,
                           shape=(m, rank, n), order="C")
        self.b = np.empty((m, n))
        self.c = np.empty(m)
        self.eigen_max = np.empty(m)
        self.eigen_min_positive = np.empty(m)
        self.uniform_eigenvalues_valid = True
        for i in range(m):
            local = np.random.default_rng(np.random.SeedSequence([seed, n, i, 202]))
            Q, _ = np.linalg.qr(local.uniform(0., 1., (n, n)))
            values = local.uniform(0., 1., rank)
            # Assign BEFORE the initial product.  self.W[i] is C-contiguous,
            # exactly as the source's preallocated ndarray/memmap slice is.
            self.W[i] = np.sqrt(values)[:, None] * Q[:, :rank].T
            self.b[i] = local.uniform(0., 1., n)
            wx = self.W[i] @ self.x0
            self.c[i] = -.5 * (wx @ wx) - self.b[i] @ self.x0 - .02
            self.eigen_max[i] = max(values)
            self.eigen_min_positive[i] = min(values)
            self.uniform_eigenvalues_valid &= bool(np.all((values > 0) & (values < 1)))
            del Q, values, wx
        self.W.flush()
        digest = hashlib.sha256()
        for array in (self.B, self.d, self.b, self.c, self.h):
            update_array_hash(digest, array)
        for i in range(m):
            update_array_hash(digest, self.W[i])
        self.instance_hash = digest.hexdigest()

    def close(self):
        if hasattr(self, "W"):
            self.W._mmap.close()
            del self.W

    def constraints_block(self, points):
        """G at a bounded block of row points, factor by factor."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.n:
            raise ValueError("Expected a block of row points with shape (q,n)")
        answer = points @ self.b.T + self.c
        for i in range(self.m):
            products = self.W[i] @ points.T
            answer[:, i] += .5 * np.einsum("rq,rq->q", products, products)
        return answer

    def G(self, point):
        return self.constraints_block(np.asarray(point).reshape(1, self.n))[0]

    def J(self, point, factor_block=8):
        """Factorwise Jacobian; only (m,n) output and bounded W slices exist.

        Available for downstream independent checks.  Auditing saved points
        needs G, so it does not compute unnecessary full Jacobians.
        """
        point = np.asarray(point, dtype=np.float64)
        answer = np.empty((self.m, self.n))
        for start in range(0, self.m, factor_block):
            stop = min(self.m, start + factor_block)
            factors = self.W[start:stop]
            wx = np.einsum("mrn,n->mr", factors, point, optimize=False)
            answer[start:stop] = np.einsum("mrn,mr->mn", factors, wx, optimize=False)
        answer += self.b
        return answer


@contextmanager
def independent_problem(n, m, seed, temp_parent=None):
    with tempfile.TemporaryDirectory(prefix="cvi_factor_audit_", dir=temp_parent) as directory:
        problem = None
        try:
            problem = FactorProblem(n, m, seed, directory)
            yield problem
        finally:
            if problem is not None:
                problem.close()


def validate_plan(plan, plan_scope, issues, incomplete,single_only=False):
    if plan_scope and single_only and not plan.get('validation_fixture'):
        issues.append('Plan-scope and single-only release scopes are distinct.')
        return None
    required = ("settings", "seeds", "methods", "seconds", "source_hashes")
    if not isinstance(plan, dict) or any(key not in plan for key in required):
        issues.append("Plan is not an object with settings, seeds, methods, seconds, and source_hashes.")
        return None
    try:
        settings = [(s["study"], s["n"], s["m"]) for s in plan["settings"]]
        seeds, methods = plan["seeds"], plan["methods"]
        if not settings or len(settings) != len(set(settings)):
            raise ValueError("empty or duplicate settings")
        if any(s not in ("single", "multi") or type(n) is not int or type(m) is not int or
               n < 2 or m < 1 or (s == "single" and m != 1) or
               (s == "multi" and m <= 1) for s, n, m in settings):
            raise ValueError("invalid setting identity or dimensions")
        if plan_scope and single_only and any(s!='single' for s,n,m in settings):raise ValueError('single-only layout fixture contains a multi-constraint setting')
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("invalid or duplicate seeds")
        if not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
            raise ValueError("invalid or duplicate method names")
        if plan_scope and set(methods) not in (set(LEGACY_METHODS), set(METHODS)):
            raise ValueError("plan scope requires exactly the declared three legacy methods or all five methods")
        if type(plan["seconds"]) not in (int,float) or not finite(plan["seconds"]) or plan["seconds"] <= 0:
            raise ValueError("nonpositive/nonfinite cap")
        hashes = plan["source_hashes"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("empty source hashes")
        if any(Path(name).name != name or not name.endswith(".py") or
               not isinstance(value, str) or len(value) != 64 or
               any(c not in "0123456789abcdef" for c in value)
               for name, value in hashes.items()):
            raise ValueError("invalid source hash names or SHA256 values")
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"Invalid plan: {exc}.")
        return None
    if not plan_scope:
        expected_settings={("single",n,1) for n in (10,50,100,200,500,1000)} if single_only else FULL_SETTINGS
        if set(settings) != expected_settings:
            incomplete.append("Plan settings differ from the explicitly selected release contract.")
        if single_only and plan.get('release_scope')!='single_only':issues.append('Single-only release requires release_scope=single_only in plan.json.')
        if not single_only and plan.get('release_scope')=='single_only':issues.append('Single-only plan requires explicit --single-only.')
        if list(seeds) != list(range(10)):
            incomplete.append("Plan does not contain exactly seeds 0 through 9.")
        if tuple(methods) != METHODS:
            incomplete.append("Plan does not contain exactly the five requested methods in the declared order.")
        if plan["seconds"] != 1000.:
            incomplete.append("Plan does not declare the requested 1000-second method cap.")
    return {(study, n, m, seed) for study, n, m in settings for seed in seeds}


def validate_saved_pair_identity(saved, manifest, method, actual_point_hash,
                                 required, prefix, issues):
    """Validate new worker pair provenance; legacy plan-scope data may omit it."""
    metadata_present = "metadata" in saved
    checksum_present = "points_sha256" in saved
    metadata_matches = None
    checksum_matches = None
    if not metadata_present:
        if required:
            issues.append(f"{prefix}: full-grid run is missing required pair metadata.")
    elif not isinstance(saved["metadata"], dict):
        metadata_matches = False
        issues.append(f"{prefix}: saved pair metadata is not an object.")
    else:
        mismatches = [key for key in ("n", "m", "seed", "cap_seconds", "instance_hash", "source_hashes")
                      if saved["metadata"].get(key) != manifest.get(key)]
        if saved["metadata"].get("method") != method:
            mismatches.append("method")
        metadata_matches = not mismatches
        if mismatches:
            issues.append(f"{prefix}: saved pair/manifest identity differs for {', '.join(mismatches)}.")
    if not checksum_present:
        if required:
            issues.append(f"{prefix}: full-grid run is missing the required point-file checksum.")
    else:
        checksum_matches = saved["points_sha256"] == actual_point_hash
        if not checksum_matches:
            issues.append(f"{prefix}: recorded point-file checksum differs from the saved archive.")
    return dict(required=required, metadata_present=metadata_present,
                metadata_matches_manifest=metadata_matches,
                point_checksum_present=checksum_present,
                recorded_points_sha256=saved.get("points_sha256"),
                actual_points_sha256=actual_point_hash,
                point_checksum_matches=checksum_matches)


def audit_run(case, ident, meta, method, plan, problem, block_size, issues,
              incomplete, warnings, files, point_hashes, preflight,
              require_pair_identity=True):
    prefix = f"{case.name}/{method}"
    paths = {kind: case / f"{method}_{suffix}"
             for kind, suffix in (("run", "run.json"), ("points", "points.npz"),
                                  ("diagnostics", "diagnostics.json"))}
    absent = [path.name for path in paths.values() if not path.is_file()]
    if absent:
        incomplete.append(f"{prefix}: missing {', '.join(absent)}.")
        return None
    for path in paths.values():
        files[str(path)] = sha_file(path)
    try:
        saved = read_json(paths["run"])
        rows = read_json(paths["diagnostics"])
        samples, result = saved["checkpoints"], saved["result"]
        with np.load(paths["points"], allow_pickle=False) as archive:
            if set(archive.files) != {"x"}:
                issues.append(f"{prefix}: unexpected point archive keys {archive.files}.")
            points = archive["x"]
        if not isinstance(samples, list) or not isinstance(rows, list) or not isinstance(result, dict):
            raise ValueError("run/diagnostic containers have wrong types")
        if any(not isinstance(value, dict) for value in samples + rows):
            raise ValueError("checkpoint/diagnostic entries must be objects")
        if points.dtype != np.dtype("float64") or points.shape != (len(samples), ident[1]):
            raise ValueError(f"point shape/dtype mismatch: {points.shape}, {points.dtype}")
        if len(rows) != len(samples) or not len(samples):
            raise ValueError("empty or unequal checkpoint/point/diagnostic counts")
        if not np.all(np.isfinite(points)):
            raise ValueError("nonfinite saved point")
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        issues.append(f"{prefix}: unreadable or invalid saved records: {exc}.")
        return None
    pair_identity = validate_saved_pair_identity(saved, meta, method,
        files[str(paths["points"])], require_pair_identity, prefix, issues)
    method_meta = meta["methods"].get(method) if isinstance(meta.get("methods"), dict) else None
    if not isinstance(method_meta, dict):
        issues.append(f"{prefix}: manifest lacks method metadata.")
        method_meta = {}
    if method_meta.get("stage") != "complete":
        incomplete.append(f"{prefix}: manifest method stage is not complete.")
    if method_meta.get("checkpoints") != len(samples):
        issues.append(f"{prefix}: manifest checkpoint count differs.")
    if method_meta.get("status") != result.get("status") or method_meta.get("runtime") != result.get("runtime"):
        issues.append(f"{prefix}: manifest/result termination identity differs.")
    runtime = result.get("runtime")
    if not finite(runtime) or runtime < 0:
        issues.append(f"{prefix}: invalid runtime.")
    if result.get("status") == "time_limit" and finite(runtime) and runtime < plan["seconds"] - .01:
        issues.append(f"{prefix}: time-limit runtime is below the cap.")
    if result.get("status") not in ("certificate_stop", "time_limit", "iteration_limit"):
        warnings.append(f"{prefix}: recorded algorithm termination {result.get('status')!r}.")
    expected_fixed_M = OE_M_FACTORS[method] * meta["true_M"] if method in OE_M_FACTORS else None
    if expected_fixed_M is not None and "fixed_M" in result and not close(result["fixed_M"], expected_fixed_M):
        issues.append(f"{prefix}: fixed OE operator estimate differs from its declared factor times true M.")
    if problem is not None:
        initial_matches = (np.array_equal(points[0], problem.x0)
                           if require_pair_identity or pair_identity["metadata_present"] else
                           np.allclose(points[0], problem.x0, atol=1e-14, rtol=0))
        if not initial_matches:
            issues.append(f"{prefix}: checkpoint zero differs from the common initial point.")
    violations = None
    if not preflight:
        violations = np.empty(len(points))
        for start in range(0, len(points), block_size):
            stop = min(len(points), start + block_size)
            values = problem.constraints_block(points[start:stop])
            violations[start:stop] = np.linalg.norm(np.maximum(values, 0), axis=1)
    max_ball = max(0., float(np.max(np.linalg.norm(points, axis=1))) - meta["radius"])
    if max_ball > BALL_ATOL:
        issues.append(f"{prefix}: at least one saved averaged point exceeds the ball.")
    previous_time, previous_k = -1., -1
    valid_count, invalid_count, unavailable_violation_count = 0, 0, 0
    max_violation_error, max_certificate_deficit = 0., 0.
    invalid_details = []
    for index, (sample, row, point) in enumerate(zip(samples, rows, points)):
        tag = f"{prefix}/{index}"
        point_hashes.append(dict(case=case.name, method=method, checkpoint=index,
                                 sha256_float64_C=array_hash(point)))
        if not isinstance(sample, dict) or not isinstance(row, dict):
            issues.append(f"{tag}: checkpoint/diagnostic is not an object.")
            continue
        elapsed, k = sample.get("elapsed"), sample.get("k")
        if expected_fixed_M is not None and "M_estimate" in sample and not close(sample["M_estimate"], expected_fixed_M):
            issues.append(f"{tag}: checkpoint OE operator estimate differs from its declared factor times true M.")
        if (row.get("n"), row.get("m"), row.get("seed"), row.get("method"), row.get("checkpoint")) != (ident[1], ident[2], ident[3], method, index):
            issues.append(f"{tag}: diagnostic identity differs from case/checkpoint.")
        if row.get("elapsed") != elapsed or row.get("k") != k:
            issues.append(f"{tag}: diagnostic/checkpoint elapsed or iteration identity differs.")
        if not finite(elapsed) or not 0 <= elapsed <= plan["seconds"] or (index and elapsed <= previous_time):
            issues.append(f"{tag}: nonfinite, nonmonotone, or out-of-budget elapsed time.")
        elif finite(runtime) and elapsed > runtime + RECORD_ATOL:
            issues.append(f"{tag}: checkpoint elapsed time exceeds method runtime.")
        if type(k) is not int or k < 0 or k < previous_k or (index and k == previous_k):
            issues.append(f"{tag}: invalid or nonincreasing iteration number.")
        if index == 0 and (elapsed != 0 or k != 0):
            issues.append(f"{tag}: checkpoint zero must have elapsed=0 and k=0.")
        previous_time = elapsed if finite(elapsed) else previous_time
        previous_k = k if type(k) is int else previous_k
        valid = row.get("gap_valid")
        violation = row.get("violation")
        diagnostic_error = row.get("diagnostic_error")
        unavailable_violation_explained = (valid is False and not finite(violation) and
            isinstance(diagnostic_error, dict) and diagnostic_error.get("phase") == "violation_check" and
            bool(diagnostic_error.get("message")))
        if unavailable_violation_explained:
            unavailable_violation_count += 1
        elif not finite(violation) or violation < 0:
            issues.append(f"{tag}: violation must be finite and nonnegative.")
        effective_violation = float(violations[index]) if violations is not None else violation
        if violations is not None and finite(violation):
            error = abs(float(violations[index]) - violation)
            max_violation_error = max(max_violation_error, error)
            if not close(float(violations[index]), violation):
                issues.append(f"{tag}: factorwise recomputed violation differs.")
        if method != "proposed" and finite(effective_violation) and effective_violation > np.sqrt(ident[2]) * 1e-7 + RECORD_ATOL:
            issues.append(f"{tag}: saved baseline output exceeds declared projection feasibility tolerance.")
        gap, residual = row.get("gap"), row.get("residual")
        gap_status = row.get("gap_status")
        if type(valid) is not bool:
            issues.append(f"{tag}: gap_valid must be an explicit boolean.")
        if not isinstance(gap_status, (str, int)) or isinstance(gap_status, bool) or gap_status == "":
            issues.append(f"{tag}: gap status/reason is missing or invalid.")
        if valid is True:
            valid_count += 1
            if not finite(gap) or gap < 0 or not finite(residual) or residual < 0:
                issues.append(f"{tag}: valid diagnostic requires finite nonnegative gap/residual.")
            if gap_status not in SUCCESS_GAP_STATUSES:
                issues.append(f"{tag}: valid gap has an unsuccessful/unrecognized solver status {gap_status!r}.")
            if gap_status=='independent_dual_bracket_valid':
                evidence=row.get('diagnostic_adoption',{});quality=evidence.get('numerical_quality',{})
                if evidence.get('kind')!='independent_single_constraint_dual_v2' or evidence.get('quality_passed') is not True or evidence.get('quality_reviewed_by_explicit_cli') is not True or evidence.get('input_point_hash')!=array_hash(points[index]) or evidence.get('instance_hash')!=meta['instance_hash'] or evidence.get('source_hashes')!=meta['source_hashes']:
                    issues.append(f'{tag}: adopted dual-gap provenance/review/point identity mismatch.')
                if any(not finite(quality.get(key)) or not 0<=quality[key]<=1e-8 for key in ('full_feasibility','primal_dual_width','stationarity_l2')):
                    issues.append(f'{tag}: adopted dual-gap quality exceeds the reviewed 1e-8 limits.')
        else:
            invalid_count += 1
            reason = next((row[key] for key in ("gap_failure_reason", "gap_reason", "diagnostic_message", "gap_message", "diagnostic_error")
                           if row.get(key)), None)
            if gap_status in SUCCESS_GAP_STATUSES and not reason:
                issues.append(f"{tag}: invalid diagnostic has a success status with no failure reason.")
            invalid_details.append(dict(checkpoint=index, status=gap_status, reason=reason,
                                        has_finite_unvalidated_gap=finite(gap)))
        if finite(gap) and finite(violation) and (valid is True or finite(residual)):
            if not close(max(gap, violation), residual):
                issues.append(f"{tag}: residual differs from max(gap, violation).")
        elif finite(residual):
            issues.append(f"{tag}: finite residual is reported without a finite gap and violation.")
        diagnostic_seconds = row.get("diagnostic_seconds")
        if not finite(diagnostic_seconds) or diagnostic_seconds < 0:
            issues.append(f"{tag}: invalid untimed diagnostic duration.")
        certificate = sample.get("certificate")
        stored_certificate = row.get("certificate")
        if method == "proposed" and index:
            if not finite(certificate) or certificate < 0 or not close(certificate, stored_certificate, atol=1e-13, rtol=0):
                issues.append(f"{tag}: finite nonnegative paper C certificate is missing or inconsistent.")
            if finite(certificate) and valid is True and finite(gap):
                deficit = max(0., gap - certificate)
                max_certificate_deficit = max(max_certificate_deficit, deficit)
                if deficit > CERTIFICATE_ATOL + CERTIFICATE_RTOL * max(1., abs(gap)):
                    issues.append(f"{tag}: saved valid gap exceeds paper C beyond comparison tolerance.")
            gamma = sample.get("Gamma")
            if not finite(gamma) or gamma <= 0:
                issues.append(f"{tag}: missing/nonpositive Gamma for the geometric paper C bound.")
            elif problem is not None and finite(certificate):
                bound = (1.5 if ident[2] == 1 else 2.) * gamma * (meta["radius"] + np.linalg.norm(problem.x0)) ** 2
                if certificate > bound + 1e-7:
                    issues.append(f"{tag}: paper C exceeds its geometric bound.")
            if "violation" in sample and finite(effective_violation) and not close(sample["violation"], effective_violation):
                issues.append(f"{tag}: timed/saved diagnostic violation differs.")
        elif finite(certificate) != finite(stored_certificate) or (finite(certificate) and not close(certificate, stored_certificate)):
            issues.append(f"{tag}: optional initial/baseline certificate metadata differs.")
    endpoint = samples[-1]
    if result.get("status") == "certificate_stop":
        endpoint_violation = (float(violations[-1]) if violations is not None else
                              rows[-1].get("violation") if finite(rows[-1].get("violation")) else
                              endpoint.get("violation"))
        if (method != "proposed" or not finite(endpoint.get("certificate")) or
                endpoint["certificate"] > meta["epsilon"] or
                endpoint.get("certificate_stop") is not True or endpoint.get("middle_pass") is not True or
                not finite(endpoint_violation) or endpoint_violation > meta["epsilon"]):
            issues.append(f"{prefix}: invalid certificate-stop endpoint.")
    if "final_state_hash" in result and "state_hash" in endpoint and result["final_state_hash"] != endpoint["state_hash"]:
        issues.append(f"{prefix}: result/last-checkpoint state hash differs.")
    return dict(case=case.name, method=method, checkpoints=len(samples), status=result.get("status"),
                runtime=runtime, saved_valid_gaps=valid_count, saved_invalid_gaps=invalid_count,
                invalid_diagnostics=invalid_details, unavailable_violation_records=unavailable_violation_count,
                pair_identity=pair_identity, max_ball_violation=max_ball,
                max_recomputed_violation_difference=max_violation_error if violations is not None else None,
                max_saved_valid_gap_minus_paper_C=max_certificate_deficit,
                independent_gap_checks=0,
                selected_independent_gap_evidence_rows=sum(row.get('gap_status')=='independent_dual_bracket_valid' for row in rows))


def audit(args):
    results, code, out = args.results.resolve(), args.code.resolve(), args.out.resolve()
    issues, incomplete, warnings, instances, runs, point_hashes = [], [], [], [], [], []
    files = {}
    current_hashes = None
    plan_path = results / "plan.json"
    plan, expected = None, None
    if not plan_path.is_file():
        incomplete.append("Missing plan.json: expected coverage and immutable source hashes cannot be established.")
    else:
        files[str(plan_path)] = sha_file(plan_path)
        try:
            plan = read_json(plan_path)
            expected = validate_plan(plan, args.plan_scope, issues, incomplete,args.single_only)
        except (OSError, ValueError) as exc:
            issues.append(f"Unreadable plan.json: {exc}.")
    if expected is not None:
        current_hashes = {path.name: sha_file(path) for path in sorted(code.glob("*.py"))}
        if current_hashes != plan["source_hashes"]:
            issues.append("Current experiment root Python source hashes differ from plan hashes (exact file set required).")
        seen = set()
        manifests = sorted(results.glob("*/manifest.json"))
        for manifest_path in manifests:
            case = manifest_path.parent
            files[str(manifest_path)] = sha_file(manifest_path)
            try:
                meta = read_json(manifest_path)
                n, m, seed = meta["n"], meta["m"], meta["seed"]
                study = "single" if m == 1 else "multi"
                ident = (study, n, m, seed)
                if any(type(value) is not int for value in (n, m, seed)) or n < 2 or m < 1 or seed < 0:
                    raise ValueError("invalid instance identity")
                expected_name = f"{study}_n{n}_m{m}_seed{seed}"
                if case.name != expected_name:
                    issues.append(f"{case.name}: directory name differs from manifest identity {expected_name}.")
                if ident not in expected or ident in seen:
                    issues.append(f"{case.name}: unexpected or duplicate instance identity {ident}.")
                    continue
                seen.add(ident)
                if meta.get("state") != "complete":
                    incomplete.append(f"{case.name}: manifest state is {meta.get('state')!r}, not complete.")
                if meta.get("source_hashes") != plan["source_hashes"]:
                    issues.append(f"{case.name}: manifest/plan source hashes differ.")
                if meta.get("cap_seconds") != plan["seconds"]:
                    issues.append(f"{case.name}: manifest/plan budget differs.")
                lineage_path=case/'lineage.json'
                if lineage_path.exists():
                    lineage=read_json(lineage_path);files[str(lineage_path)]=sha_file(lineage_path)
                    origin=plan.get('inherited_single_provenance',{})
                    if m!=1 or Path(lineage.get('source_case','')).name!=case.name or not lineage.get('source_results') or not lineage.get('source_plan_sha256') or lineage.get('new_plan_sha256')!=files[str(plan_path)] or any(origin.get(key) is not None and lineage.get(key)!=origin[key] for key in ('source_results','source_plan_sha256')):
                        issues.append(f'{case.name}: inherited lineage identity/origin/amended-plan hash mismatch.')
                    copied=lineage.get('copied_artifact_sha256')
                    if not isinstance(copied,dict) or not copied:issues.append(f'{case.name}: missing inherited artifact hash map.')
                    else:
                        for name,value in copied.items():
                            if not isinstance(name,str) or Path(name).name!=name or not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                                issues.append(f'{case.name}: invalid inherited artifact hash identity.');continue
                            if name.endswith(('_run.json','_points.npz')) and (not (case/name).is_file() or sha_file(case/name)!=value):issues.append(f'{case.name}: inherited timed artifact changed: {name}.')
                for key in ("radius", "epsilon", "Lambda", "M0", "L0", "true_M"):
                    if not finite(meta.get(key)) or meta[key] <= 0:
                        raise ValueError(f"missing, nonfinite, or nonpositive {key}")
                if meta.get("scaling") != "none" or meta["radius"] != 2.:
                    issues.append(f"{case.name}: generator radius/scaling differs from this auditor's supported contract.")
                if not isinstance(meta.get("methods"), dict) or set(meta["methods"]) != set(plan["methods"]):
                    issues.append(f"{case.name}: manifest method set differs from plan.")
                recorded_methods = set()
                for suffix in ("_run.json", "_points.npz", "_diagnostics.json"):
                    recorded_methods.update(path.name[:-len(suffix)] for path in case.glob(f"*{suffix}"))
                extras = recorded_methods - set(plan["methods"])
                if extras:
                    issues.append(f"{case.name}: unexpected method run files: {sorted(extras)}.")
                if args.preflight:
                    instance = dict(case=case.name, n=n, m=m, seed=seed, reconstruction="not_performed_preflight")
                    for method in plan["methods"]:
                        run = audit_run(case, ident, meta, method, plan, None, args.point_block,
                                        issues, incomplete, warnings, files, point_hashes, True,
                                        require_pair_identity=not args.plan_scope)
                        if run is not None:
                            runs.append(run)
                else:
                    began = perf_counter()
                    with independent_problem(n, m, seed, args.temp_dir) as problem:
                        if problem.instance_hash != meta.get("instance_hash"):
                            issues.append(f"{case.name}: independent C-layout instance digest differs.")
                        initial = problem.G(problem.x0)
                        if not np.all(initial < -.0199999) or not np.allclose(initial, -.02, atol=2e-12, rtol=0):
                            issues.append(f"{case.name}: strict initial margin differs from -0.02.")
                        if not close(float(min(initial)), meta.get("min_initial_constraint")):
                            issues.append(f"{case.name}: recorded minimum initial constraint differs.")
                        if np.linalg.norm(problem.x0) >= meta["radius"]:
                            issues.append(f"{case.name}: initial point is not strictly inside the ball.")
                        if not problem.uniform_eigenvalues_valid:
                            issues.append(f"{case.name}: generated positive eigenvalues are outside (0,1).")
                        symmetric_error = float(np.max(np.abs((problem.B + problem.B.T) / 2 - np.diag(problem.h))))
                        if symmetric_error > 1e-14 or np.any(problem.h < 0):
                            issues.append(f"{case.name}: monotone operator structure is inconsistent.")
                        # SVD of B is optional: avoid an unneeded expensive norm
                        # solve in the default structural audit of large cases.
                        actual_M = float(np.linalg.norm(problem.B, 2)) if args.verify_operator_norm else None
                        if actual_M is not None and not close(actual_M, meta["true_M"], atol=1e-12, rtol=1e-12):
                            issues.append(f"{case.name}: operator spectral norm differs.")
                        lower, upper = float(max(problem.eigen_max)), float(np.linalg.norm(problem.eigen_max))
                        instance = dict(case=case.name, n=n, m=m, seed=seed,
                            recorded_instance_hash=meta.get("instance_hash"),
                            independent_instance_hash=problem.instance_hash,
                            exact_instance_digest_match=problem.instance_hash == meta.get("instance_hash"),
                            uses_source_generator=False, factor_storage="temporary_C_contiguous_memmap",
                            factor_bytes=int(problem.W.nbytes), dense_A_stack_allocated=False,
                            initial_constraints=initial, initial_point_norm=float(np.linalg.norm(problem.x0)),
                            monotone_symmetric_part_diagonal=True,
                            symmetric_structure_max_error=symmetric_error,
                            PSD_by_sum_of_squares_construction=True,
                            generated_factor_rank=n - n // 2,
                            generated_positive_eigenvalue_min=float(min(problem.eigen_min_positive)),
                            generated_positive_eigenvalue_max=float(max(problem.eigen_max)),
                            eigenvalue_method="known_QR_generator_values_no_dense_A_eigendecomposition",
                            vector_remainder_constant_lower=lower,
                            vector_remainder_constant_upper=upper,
                            vector_remainder_constant_exact=m == 1,
                            initial_L0=meta["L0"], admissible_theorem_L=max(meta["L0"], upper),
                            operator_norm_verification="numerical_SVD" if actual_M is not None else "not_performed",
                            independently_recomputed_M=actual_M, recorded_M=meta["true_M"])
                        for method in plan["methods"]:
                            run = audit_run(case, ident, meta, method, plan, problem, args.point_block,
                                            issues, incomplete, warnings, files, point_hashes, False,
                                            require_pair_identity=not args.plan_scope)
                            if run is not None:
                                runs.append(run)
                        instance["audit_seconds"] = perf_counter() - began
                instances.append(instance)
                print(f"AUDITED {case.name}: {len(instances)}/{len(expected)} planned instances", flush=True)
            except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
                issues.append(f"{case.name}: audit could not complete: {type(exc).__name__}: {exc}.")
        missing = sorted(expected - seen)
        if missing:
            incomplete.append(f"Missing {len(missing)} planned instances: {missing}.")
        expected_runs = len(expected) * len(plan["methods"])
        if len(runs) != expected_runs:
            incomplete.append(f"Audited {len(runs)}/{expected_runs} planned method runs.")
    else:
        expected_runs = None
    adoption_path=results/'diagnostic_adoption_manifest.json';adoption=None
    if adoption_path.exists():
        try:
            adoption=read_json(adoption_path);files[str(adoption_path)]=sha_file(adoption_path)
            if adoption.get('state')!='applied' or adoption.get('quality_reviewed_by_explicit_cli') is not True or adoption.get('timed_pairs_unchanged') is not True:issues.append('Invalid independent gap adoption manifest.')
            for path,value in adoption.get('input_sha256_after',{}).items():
                if not Path(path).is_file() or sha_file(path)!=value:issues.append(f'Adoption input changed: {path}.')
                else:files[path]=value
            snapshot=Path(adoption['helper_snapshot_file'])
            if sha_file(snapshot)!=adoption['helper_snapshot_sha256'] or adoption['helper_snapshot_sha256']!=adoption['helper_sha256']:issues.append('Adoption helper snapshot hash mismatch.')
            files[str(snapshot)]=sha_file(snapshot)
        except (OSError,KeyError,TypeError,ValueError) as error:issues.append(f'Unreadable independent gap adoption manifest: {error}.')
    adopted_count=sum(run.get('selected_independent_gap_evidence_rows',0) for run in runs)
    if adopted_count and adoption is None:issues.append('Adopted independent gap rows lack the explicit adoption manifest.')
    if adoption and adoption.get('rows')!=adopted_count:issues.append('Adoption manifest row count differs from inspected independent gap evidence.')
    status = ("failed" if issues else "incomplete" if incomplete else
              "preflight_only" if args.preflight else "structural_pass")
    limitations = [
        "A structural pass is not independent numerical validation of gap values, optimality, threshold attainment, or the benchmark's scientific conclusions. No gap problem was solved by this auditor.",
        "The audit checks saved gap-valid flags/statuses and residual arithmetic. Invalid diagnostics are unresolved even if an unvalidated finite estimate is present. They cannot establish threshold hits or nonattainment.",
        "Paper C is checked against saved gaps marked valid, not independently validated gaps. The certificate's underlying accumulated affine functional and unsaved inner state cannot be reconstructed from saved averaged points.",
        "PSD and rank follow the deterministic QR/sum-of-squares generator structure and known generated eigenvalues; no A_i matrices or dense eigenvalue decompositions are formed. Numerical reconstruction and hash agreement are not formal finite-precision proofs.",
        "The multi-constraint vector remainder constant is bounded between max_i lambda_max(A_i) and the Euclidean norm of those eigenvalues. The bounds are not claimed exact for m>1. L=max(L0, upper bound) is an admissible theorem bound; the theorem does not require L0 to exceed the minimal smoothness constant.",
        "No optimal multiplier bound is established, so Lambda >= ||lambda*||+1 for the stated worst-case complexity is not verified. The stopping implication permits any positive Lambda.",
        "File and point hashes are created by this audit. They detect later changes but do not prove authenticity or the historical absence of edits. Unsaved algorithm states cannot be independently rehashed.",
        "Factor memmaps use up to 8*m*ceil(n/2)*n bytes of temporary disk and are removed after each case. Operating-system file cache may contribute to measured resident memory; resident storage is not an 8 GB A_i stack.",
    ]
    summary = dict(status=status, structural_validation=status,
        gap_validation="not_independently_performed", independent_gap_solves=0,
        selected_independent_gap_evidence_rows=adoption.get('rows',0) if adoption else 0,
        scope="declared_plan" if args.plan_scope else "single_constraint_release" if args.single_only else "requested_full_grid",
        preflight_only=args.preflight,
        coverage_complete=expected is not None and not incomplete,
        full_requested_grid_records_complete=bool(not args.plan_scope and not args.single_only and expected is not None and not incomplete),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        results=str(results), code=str(code), auditor_sha256=sha_file(__file__),
        plan_snapshot=plan,
        source_identity=dict(recorded_hashes=plan.get("source_hashes") if isinstance(plan, dict) else None,
                             current_hashes=current_hashes,
                             exact_root_Python_file_set_match=(current_hashes == plan["source_hashes"]
                                 if expected is not None else None)),
        expected_instances=len(expected) if expected is not None else None,
        audited_instances=len(instances), expected_method_runs=expected_runs,
        audited_method_runs=len(runs), audited_points=len(point_hashes),
        saved_valid_gap_records=sum(r["saved_valid_gaps"] for r in runs),
        saved_invalid_gap_records=sum(r["saved_invalid_gaps"] for r in runs),
        issues=issues, incomplete_reasons=incomplete, warnings=warnings,
        tolerances=dict(record_atol=RECORD_ATOL, record_rtol=RECORD_RTOL,
                        ball_atol=BALL_ATOL, saved_gap_vs_C_atol=CERTIFICATE_ATOL,
                        saved_gap_vs_C_rtol=CERTIFICATE_RTOL),
        resource_policy=dict(temporary_factor_memmap=True, persistent_dense_A_stack=False,
                             point_block=args.point_block, numerical_threads=1,
                             independent_instance_reconstruction=True),
        limitations=limitations,
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         platform=platform.platform(), byteorder=sys.byteorder),
        instances=instances, runs=runs, file_sha256=files, point_sha256=point_hashes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structural_audit.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + "\n")
    lines = ["# Saved-record structural audit", "", f"**Status: {status}.**", "",
        "**Independent numerical gap validation: not performed.**", "",
        f"Scope: {summary['scope']}. Inspected {len(instances)} instances, {len(runs)} method runs, and {len(point_hashes)} saved points.", "",
        f"Saved diagnostics marked valid: {summary['saved_valid_gap_records']}; invalid/unresolved: {summary['saved_invalid_gap_records']}.", "",
        "Checks cover plan coverage, source/instance/file/point hashes, case/method/checkpoint identities, finite values, budgets, ball membership, factorwise constraint violations, residual arithmetic, diagnostic status honesty, and paper C consistency against saved valid gaps.", "",
        "A structural pass does not establish independent gap agreement or numerical threshold attainment.", ""]
    for title, findings in (("Failures", issues), ("Incomplete coverage", incomplete), ("Warnings", warnings)):
        if findings:
            lines += [f"## {title}", ""] + [f"- {finding}" for finding in findings] + [""]
    lines += ["## Limits", ""] + [f"- {item}" for item in limitations]
    lines += ["", "Detailed identities, statuses, resource declarations, and hashes are in `structural_audit.json`.", ""]
    (out / "structural_audit.md").write_text("\n".join(lines))
    print(f"STRUCTURAL AUDIT {status}: {len(issues)} failures, {len(incomplete)} incomplete reasons; independent gap checks: 0", flush=True)
    return 1 if issues else 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent / "code")
    parser.add_argument("--out", type=Path, help="New audit reports only; defaults to RESULTS/report/structural")
    parser.add_argument("--plan-scope", action="store_true", help="Audit exactly the declared plan (for smoke/legacy data), not the 110-instance contract")
    parser.add_argument("--single-only", action="store_true", help="Require the explicit 60-instance/300-run single-constraint release")
    parser.add_argument("--preflight", action="store_true", help="Inspect record structure without reconstructing factors; never emits structural_pass")
    parser.add_argument("--point-block", type=int, default=8, help="Maximum points per factor-product block")
    parser.add_argument("--temp-dir", type=Path, help="Parent for temporary factor memmaps (up to 4 GB per full-grid case)")
    parser.add_argument("--verify-operator-norm", action="store_true", help="Also independently recompute the spectral norm of B by SVD")
    args = parser.parse_args()
    if args.point_block < 1:
        parser.error("--point-block must be positive")
    args.out = args.out or args.results / "report" / "structural"
    if sys.byteorder != "little":
        parser.error("Exact source instance hashes require the recorded little-endian float64 representation")
    with threadpool_limits(limits=1):
        return audit(args)


if __name__ == "__main__":
    sys.exit(main())
