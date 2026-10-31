"""Paper Algorithms 1--4 with retained-trial certificates from Section 6.

Only ball and affine-halfspace projections are used here. The problem object
provides F, G, J, a stable quadratic remainder, and the radius r.
"""
from dataclasses import dataclass
from time import perf_counter
import hashlib
import numpy as np

from projections import project_ball, project_ball_halfspace


class Deadline(Exception):
    pass


@dataclass
class State:
    x: np.ndarray
    previous_x: np.ndarray
    dual: np.ndarray
    previous_dual: np.ndarray
    fx: np.ndarray
    previous_fx: np.ndarray
    J: np.ndarray
    average: np.ndarray
    eta: float
    M: float
    Gamma: float
    constant: float
    vector: np.ndarray


def state_hash(s):
    h = hashlib.sha256()
    for v in vars(s).values():
        h.update(np.asarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()


def relative_test(lhs, rhs):
    # No unit-sized additive tolerance: tiny steps must pass the same test.
    return lhs <= rhs * (1 + 5e-12)


def scalar_projection(center, r, normal, beta, coefficient):
    """Primal solution and affine-constraint multiplier of the scalar QP."""
    x, info = project_ball_halfspace(center, r, normal, beta)
    aa = float(normal @ normal)
    lam = 0.0
    if info.case == 3:
        lam = coefficient * max(0., (normal @ center - beta) / aa)
    elif info.case == 4:
        tangent = center - normal * ((normal @ center) / aa)
        rho = np.sqrt(max(0., r*r - beta*beta/aa))
        if rho == 0:
            raise ArithmeticError("Degenerate ball/halfspace multiplier")
        radial = np.linalg.norm(tangent) / rho
        lam = coefficient * max(0., (normal @ center - radial*beta) / aa)
    return x, np.array([lam])


def run_proposed(problem, seconds, observe, *, Lambda=1., M0=1., epsilon=1e-5,
                 max_outer=10_000_000):
    """Return only accepted/retained outer outputs. All trial costs are timed.

    observe(k, elapsed, state, diagnostics) is called after accepted output.
    Its small checkpoint-copy cost is included; gap solves are never called.
    """
    if seconds <= 0 or Lambda <= 0 or M0 <= 0:
        raise ValueError("Positive budget and initial parameters required")
    started = perf_counter()
    end = started + seconds
    multi = problem.m > 1
    x0 = problem.x0.copy()
    zero = np.zeros(problem.m)
    initial_F = problem.F(x0)
    initial_J = problem.J(x0)
    initial_G = problem.G(x0)
    counts = dict(F_calls=1, G_calls=1, J_calls=1, inner_steps=0,
                  rejected_operator=0, trial_count=0, projection_calls=0)
    state = State(x0.copy(), x0.copy(), zero.copy(), zero.copy(),
                  initial_F.copy(), initial_F.copy(), initial_J.copy(),
                  x0.copy(), M0, M0, np.nan, 0., np.zeros(problem.n))
    traces = []
    max_balance = 0.

    def check_time():
        if perf_counter() >= end:
            raise Deadline()

    def trial(old, k, s, eta, M_start, upward=False):
        nonlocal max_balance
        check_time()
        gamma = 1. if k == 1 else old.Gamma/(old.Gamma + eta)
        Gamma = eta*gamma
        lower = (1-gamma)*old.average + gamma*old.x
        if k == 1:
            # Algorithms 3--4: the linearization at x0 is cached once.
            G, J = initial_G, initial_J
        else:
            G, J = problem.G(lower), problem.J(lower)
            counts['G_calls'] += 1
            counts['J_calls'] += 1
        M = max(eta, M_start)
        T = int(round(M/eta))
        if T < 1 or abs(T-M/eta) > 1e-8*max(1., M/eta):
            raise ArithmeticError('Nonintegral inner horizon')
        xp, xpp = old.x.copy(), old.previous_x.copy()
        lp, lpp = old.dual.copy(), old.previous_dual.copy()
        fp, fpp = old.fx.copy(), old.previous_fx.copy()
        inv = 0.
        weighted = np.zeros(problem.n)
        affine_c, affine_v = 0., np.zeros(problem.n)
        max_op = max_coupling = 0.
        t = 1
        while t <= T:
            check_time()
            q, previous_M = 0, M
            while True:
                check_time()
                theta = 1. if k == 1 and t == 1 else M/(old.M if t == 1 else previous_M)
                p = (3. if multi else 2.)*M + eta*M*inv
                extrap = fp + theta*(fp-fpp)
                if multi:
                    coupling = (J.T @ lp + theta*(old.J.T @ (lp-lpp))
                                if t == 1 else J.T @ (lp+theta*(lp-lpp)))
                    center = (eta*old.x+p*xp-extrap-coupling)/(eta+p)
                    xc = project_ball(center, problem.r)
                    lc = np.maximum((eta*old.dual+p*lp+G+J@(xc-lower))/(eta+p), 0.)
                else:
                    center = (eta*old.x+p*xp-extrap)/(eta+p)
                    xc, lc = scalar_projection(center, problem.r, J[0],
                                               float(J[0]@lower-G[0]), eta+p)
                counts['projection_calls'] += 1
                fc = problem.F(xc)
                counts['F_calls'] += 1
                step = xc-xp
                # Affine stable difference avoids subtraction cancellation.
                op_left = np.linalg.norm(problem.B @ step)
                op_right = M*np.linalg.norm(step)
                coupling_left = np.linalg.norm(J.T@(lc-lp)) if multi else 0.
                coupling_right = M*np.linalg.norm(lc-lp) if multi else 0.
                ok = relative_test(op_left, op_right) and relative_test(coupling_left, coupling_right)
                if ok or upward:
                    break
                counts['rejected_operator'] += 1
                q += 1
                if q > 60:
                    raise ArithmeticError('Operator search exceeded 60 doublings')
                M *= 2.
            if upward and not ok:
                return None, dict(phase='up', k=k, s=s, eta=eta, M=M,
                                  operator_pass=False, middle_pass=False)
            max_op = max(max_op, op_left/op_right if op_right else 0.)
            max_coupling = max(max_coupling, coupling_left/coupling_right if coupling_right else 0.)
            T = t-1 + 2**q*(T-t+1)
            inv += 1/M
            weighted += xc/M
            affine_c += (fc@xc-lc@(G-J@lower))/M
            affine_v += (fc+J.T@lc)/M
            counts['inner_steps'] += 1
            if t == T:
                break
            xpp, xp = xp, xc
            lpp, lp = lp, lc
            fpp, fp = fp, fc
            t += 1
        balance = abs(eta*inv-1.)
        max_balance = max(max_balance, balance)
        if balance > 1e-8:
            raise ArithmeticError('Inverse-M invariant failed')
        average = (1-gamma)*old.average + gamma*(weighted/inv)
        delta = average-lower
        remainder = problem.remainder(delta)
        middle_left = np.linalg.norm(remainder) if multi else float(remainder[0])
        middle_right = eta/(2*Lambda*gamma)*float(delta@delta)
        middle_pass = bool(relative_test(middle_left, middle_right))
        candidate = State(xc, xp, lc, lp, fc, fp, J, average, eta, M, Gamma,
                          (1-gamma)*old.constant+Gamma*affine_c,
                          (1-gamma)*old.vector+Gamma*affine_v)
        cert = max(0., candidate.constant+problem.r*np.linalg.norm(candidate.vector))
        violation = np.linalg.norm(np.maximum(problem.G(average),0.))
        counts['G_calls'] += 1
        check_time()
        return candidate, dict(k=k, s=s, eta=eta, gamma=gamma, Gamma=Gamma,
            M=M, T=T, balance_error=balance, operator_ratio=max_op,
            coupling_ratio=max_coupling, middle_lhs=middle_left,
            middle_rhs=middle_right, middle_pass=middle_pass,
            certificate=cert, violation=float(violation),
            certificate_stop=bool(middle_pass and cert<=epsilon and violation<=epsilon),
            state_hash=state_hash(candidate))

    observe(0, 0., state, dict(counts, certificate=np.nan,
                              violation=float(np.linalg.norm(np.maximum(initial_G,0.)))))
    completed = 0
    status = 'time_limit'
    try:
        # Joint upward search: each attempt restarts from the complete initial state.
        eta = M0
        for q in range(61):
            counts['trial_count'] += 1
            candidate, info = trial(state, 1, 1, eta, eta, upward=True)
            info.update(phase='up', attempt=q, elapsed=perf_counter()-started)
            traces.append(info)
            if candidate is not None and info['middle_pass']:
                retained, retained_info = candidate, info
                break
            eta *= 2.
        else:
            raise ArithmeticError('Joint upward search failed')
        if not retained_info['certificate_stop']:
            for s in range(2, 63):
                eta /= 2.
                counts['trial_count'] += 1
                candidate, info = trial(state, 1, s, eta, retained.M)
                info.update(phase='down', elapsed=perf_counter()-started,
                            retained_before_hash=state_hash(retained))
                traces.append(info)
                if not info['middle_pass']:
                    info['rollback_to_hash'] = state_hash(retained)
                    break
                retained, retained_info = candidate, info
                if info['certificate_stop']:
                    break
            else:
                raise ArithmeticError('First downward search failed to terminate')
        state = retained
        completed = 1
        retained_info['retained'] = True
        observe(1, perf_counter()-started, state, dict(counts, **retained_info))
        if retained_info['certificate_stop']:
            status = 'certificate_stop'
        else:
            for k in range(2, max_outer+1):
                before = state_hash(state)
                eta, M_start = state.eta/2., state.M
                for s in range(1, 63):
                    counts['trial_count'] += 1
                    candidate, info = trial(state, k, s, eta, M_start)
                    if info['middle_pass']:
                        break
                    eta *= 2.
                    M_start = candidate.M
                else:
                    raise ArithmeticError('Middle search failed')
                state = candidate
                completed = k
                info.update(phase='ordinary', start_state_hash=before, retained=True)
                traces.append(info)
                observe(k, perf_counter()-started, state, dict(counts, **info))
                if info['certificate_stop']:
                    status = 'certificate_stop'
                    break
            else:
                status = 'iteration_limit'
    except Deadline:
        pass
    # The last completed retained point is not relabeled with the deadline time.
    return dict(status=status, runtime=perf_counter()-started, completed=completed,
                counts=counts, trace=traces, max_balance_error=max_balance,
                final_state_hash=state_hash(state))
