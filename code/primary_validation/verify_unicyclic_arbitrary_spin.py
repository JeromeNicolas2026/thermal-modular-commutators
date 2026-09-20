#!/usr/bin/env python3
"""Independent certificates for the arbitrary-spin unicyclic onset law.

This file deliberately does not import any code from the v5 bundle.  It builds
irreducible spin matrices for arbitrary half-integer spin, constructs a finite
axial XXZ--DM Hamiltonian, expands the *normalized* Gibbs state as a formal
power series, takes normalized partial traces, expands both matrix logarithms,
and finally forms

    M_beta(AB,BC) = i Tr rho_beta [log rho_AB, log rho_BC].

All computations in ``exact_certificate`` use SymPy algebraic arithmetic; no
floating-point value enters.  The graph may have extra tree vertices, while
the measured sites are always A=0, B=1, C=2 and the oriented unique cycle is
(0,1,...,n-1,0).

Conjectured coefficient (our orientation conventions):

    [beta^(n+1)] M = 2 (-1)^n h_B prod_{x in cycle} kappa_x Im W,
    kappa_x = s_x(s_x+1)/3,
    W = t_01 t_12 ... t_(n-1,0).

The script prints the exact coefficient, target, and residual for a collection
of heterogeneous spins, generic longitudinal exchanges/fields, and a graph
with an attached tree.  A nonzero exit status means that a hard assertion
failed.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import reduce
from operator import mul

import sympy as sp
from sympy import I, Matrix, Rational as R, eye, zeros


def kron_all(factors):
    out = Matrix([[1]])
    for factor in factors:
        out = Matrix(sp.kronecker_product(out, factor))
    return out


def spin_matrices(two_s: int):
    """Return (I,S+,S-,Sz) in the descending-m magnetic basis."""
    assert isinstance(two_s, int) and two_s >= 1
    d = two_s + 1
    s = R(two_s, 2)
    ms = [s - j for j in range(d)]
    plus = zeros(d)
    for col in range(1, d):
        m = ms[col]
        plus[col - 1, col] = sp.sqrt(s * (s + 1) - m * (m + 1))
    minus = plus.T
    z = sp.diag(*ms)
    return eye(d), plus, minus, z


def site_operator(local, site, identities):
    return kron_all([local if x == site else identities[x]
                     for x in range(len(identities))])


def build_hamiltonian(two_spins, edges, fields):
    """Build H. Each edge is (x,y,t_xy,Jz_xy), with its shown orientation."""
    locals_ = [spin_matrices(q) for q in two_spins]
    identities = [ops[0] for ops in locals_]
    plus = [site_operator(locals_[x][1], x, identities)
            for x in range(len(two_spins))]
    minus = [site_operator(locals_[x][2], x, identities)
             for x in range(len(two_spins))]
    zops = [site_operator(locals_[x][3], x, identities)
            for x in range(len(two_spins))]
    dim = math.prod(q + 1 for q in two_spins)
    H = zeros(dim)
    for x, y, t, jz in edges:
        H += R(1, 2) * (t * plus[x] * minus[y]
                        + sp.conjugate(t) * minus[x] * plus[y])
        H += jz * zops[x] * zops[y]
    for x, h in enumerate(fields):
        H -= h * zops[x]
    return sp.expand(H)


def strides(dims):
    return [math.prod(dims[j + 1:]) for j in range(len(dims))]


def tuples_for_dims(dims):
    if not dims:
        return [()]
    import itertools
    return list(itertools.product(*[range(d) for d in dims]))


def flatten_tuple(digits, dims):
    return sum(a * st for a, st in zip(digits, strides(dims)))


def normalized_partial_trace(M, dims, keep):
    """Return Tr_keep^c(M)/dim(keep^c), in the tensor order given by keep."""
    keep = tuple(keep)
    traced = tuple(x for x in range(len(dims)) if x not in keep)
    kdims = [dims[x] for x in keep]
    tdims = [dims[x] for x in traced]
    ktuples = tuples_for_dims(kdims)
    ttuples = tuples_for_dims(tdims)
    out = zeros(math.prod(kdims))

    # Precompute global basis indices for every (kept, traced) configuration.
    indices = {}
    for ka, kvals in enumerate(ktuples):
        for ta, tvals in enumerate(ttuples):
            digits = [0] * len(dims)
            for x, v in zip(keep, kvals):
                digits[x] = v
            for x, v in zip(traced, tvals):
                digits[x] = v
            indices[(ka, ta)] = flatten_tuple(digits, dims)

    dt = math.prod(tdims) if tdims else 1
    for a in range(len(ktuples)):
        for b in range(len(ktuples)):
            out[a, b] = sp.expand(sum(
                M[indices[(a, t)], indices[(b, t)]]
                for t in range(len(ttuples))) / dt)
    return out


class MatrixSeries:
    def __init__(self, order, dim):
        self.order = order
        self.dim = dim

    def mul(self, a, b):
        out = [zeros(self.dim) for _ in range(self.order + 1)]
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                if i + j <= self.order:
                    out[i + j] += ai * bj
        return [sp.expand(x) for x in out]

    def log_at_identity(self, a):
        assert a[0] == eye(self.dim)
        x = [m.copy() for m in a]
        x[0] = zeros(self.dim)
        out = [zeros(self.dim) for _ in range(self.order + 1)]
        power = [eye(self.dim)] + [zeros(self.dim)
                                  for _ in range(self.order)]
        for k in range(1, self.order + 1):
            power = self.mul(power, x)
            c = R((-1) ** (k + 1), k)
            out = [sp.expand(o + c * p) for o, p in zip(out, power)]
        return out


def scalar_inverse(series):
    assert series[0] == 1
    out = [sp.Integer(0)] * len(series)
    out[0] = sp.Integer(1)
    for n in range(1, len(series)):
        out[n] = sp.expand(-sum(series[k] * out[n - k]
                                for k in range(1, n + 1)))
    return out


def scalar_times_matrix_series(scalar, matrices, order):
    dim = matrices[0].rows
    out = [zeros(dim) for _ in range(order + 1)]
    for i, si in enumerate(scalar):
        for j, mj in enumerate(matrices):
            if i + j <= order:
                out[i + j] += si * mj
    return [sp.expand(x) for x in out]


def lift_ab(M, dC):
    return Matrix(sp.kronecker_product(M, eye(dC)))


def lift_bc(M, dA):
    return Matrix(sp.kronecker_product(eye(dA), M))


def exact_coefficients(two_spins, edges, fields, order):
    """Exact beta-series coefficients of M_beta(AB,BC), through ``order``."""
    dims = [q + 1 for q in two_spins]
    H = build_hamiltonian(two_spins, edges, fields)
    full_dim = math.prod(dims)

    # Coefficients of exp(-beta H), their normalized traces, and their ABC
    # partial traces.  Normalization is applied only after tracing, avoiding
    # needless dense full-space series arithmetic.
    e = []
    e_abc = []
    power = eye(full_dim)
    for k in range(order + 1):
        ek = sp.expand(R((-1) ** k, math.factorial(k)) * power)
        e.append(ek)
        e_abc.append(normalized_partial_trace(ek, dims, (0, 1, 2)))
        power = sp.expand(power * H)
    tau_e = [sp.expand(sp.trace(x) / full_dim) for x in e]
    r_abc = scalar_times_matrix_series(
        scalar_inverse(tau_e), e_abc, order)

    dims3 = dims[:3]
    dA, dB, dC = dims3
    dABC = dA * dB * dC
    r_ab = [normalized_partial_trace(x, dims3, (0, 1)) for x in r_abc]
    r_bc = [normalized_partial_trace(x, dims3, (1, 2)) for x in r_abc]

    lab_small = MatrixSeries(order, dA * dB).log_at_identity(r_ab)
    lbc_small = MatrixSeries(order, dB * dC).log_at_identity(r_bc)
    lab = [lift_ab(x, dC) for x in lab_small]
    lbc = [lift_bc(x, dA) for x in lbc_small]

    S = MatrixSeries(order, dABC)
    ab = S.mul(lab, lbc)
    ba = S.mul(lbc, lab)
    comm = [sp.expand(x - y) for x, y in zip(ab, ba)]
    weighted = S.mul(r_abc, comm)
    return [sp.simplify(sp.expand(I * sp.trace(x) / dABC))
            for x in weighted]


def kappa(two_s):
    return R(two_s * (two_s + 2), 12)


def cycle_edges(n, t_cycle, jz_cycle):
    assert len(t_cycle) == len(jz_cycle) == n
    return [(j, (j + 1) % n, t_cycle[j], jz_cycle[j])
            for j in range(n)]


def predicted_coefficient(n, two_spins, t_cycle, hB):
    W = sp.prod(t_cycle)
    kap = sp.prod(kappa(two_spins[x]) for x in range(n))
    return sp.expand(2 * (-1) ** n * hB * kap * sp.im(sp.expand(W)))


@dataclass
class Case:
    name: str
    ncycle: int
    two_spins: list[int]
    t_cycle: list
    jz_cycle: list
    fields: list
    tree_edges: list[tuple] = None


FAILURES = []


def check_case(case):
    n = case.ncycle
    edges = cycle_edges(n, case.t_cycle, case.jz_cycle)
    edges += case.tree_edges or []
    order = n + 1
    started = time.time()
    coeffs = exact_coefficients(case.two_spins, edges, case.fields, order)
    target = predicted_coefficient(n, case.two_spins, case.t_cycle,
                                   case.fields[1])
    residual = sp.simplify(sp.expand(coeffs[order] - target))
    low = [sp.simplify(coeffs[k]) for k in range(1, order)]
    ok_low = all(x == 0 for x in low)
    ok_target = residual == 0
    print("=" * 76)
    print(case.name)
    print(f"  dims       = {[q + 1 for q in case.two_spins]}")
    print(f"  cycle n    = {n}; W = {sp.expand(sp.prod(case.t_cycle))}")
    print(f"  kappas     = {[kappa(case.two_spins[x]) for x in range(n)]}")
    print(f"  m_1..m_{n} = {low}")
    print(f"  m_{n+1}     = {coeffs[order]}")
    print(f"  target     = {target}")
    print(f"  residual   = {residual}")
    print(f"  status     = {'PASS' if ok_low and ok_target else 'FAIL'}"
          f" ({time.time() - started:.1f}s)")
    if not ok_low:
        FAILURES.append(case.name + ": nonzero lower coefficient")
    if not ok_target:
        FAILURES.append(case.name + ": wrong leading coefficient")


CASES = [
    Case(
        name="E1: triangle, spins (1/2,1,3/2), generic XXZ and fields",
        ncycle=3,
        two_spins=[1, 2, 3],
        t_cycle=[1 + 2 * I, 2 - I, -1 + 3 * I],
        jz_cycle=[R(2, 3), R(-5, 7), R(7, 4)],
        fields=[R(-3, 5), R(11, 6), R(13, 9)],
    ),
    Case(
        name="E2: triangle plus a generic complex tree edge",
        ncycle=3,
        two_spins=[1, 2, 1, 3],
        t_cycle=[2 + I, 1 - 2 * I, 3 + I],
        jz_cycle=[R(-4, 5), R(6, 7), R(5, 3)],
        fields=[R(2, 9), R(-7, 5), R(4, 3), R(-8, 11)],
        tree_edges=[(2, 3, -2 + 3 * I, R(9, 8))],
    ),
    Case(
        name="E3: four-cycle, spins (1/2,1,1/2,1/2), generic XXZ/fields",
        ncycle=4,
        two_spins=[1, 2, 1, 1],
        t_cycle=[1 + I, 2 - I, -1 + 2 * I, 3 + I],
        jz_cycle=[R(2, 5), R(-3, 4), R(5, 6), R(7, 9)],
        fields=[R(1, 3), R(-5, 4), R(7, 8), R(-2, 7)],
    ),
    Case(
        name="E4: four-cycle plus attached spin-1 tree site",
        ncycle=4,
        two_spins=[1, 1, 1, 1, 2],
        t_cycle=[2 + I, 1 - I, 3 + 2 * I, -1 + I],
        jz_cycle=[R(-1, 3), R(2, 7), R(4, 5), R(-6, 11)],
        fields=[R(-2, 5), R(3, 2), R(5, 9), R(-7, 8), R(11, 10)],
        tree_edges=[(3, 4, 2 - 3 * I, R(13, 12))],
    ),
    Case(
        name="E5: five-cycle, all spins 1/2, generic XXZ/fields",
        ncycle=5,
        two_spins=[1, 1, 1, 1, 1],
        t_cycle=[1 + I, 2 - I, 1 - 2 * I, -1 + I, 2 + I],
        jz_cycle=[R(1, 2), R(-2, 3), R(3, 5), R(-4, 7), R(5, 8)],
        fields=[R(2, 3), R(-4, 5), R(6, 7), R(-8, 9), R(10, 11)],
    ),
    Case(
        name="E6: four-cycle, spins (3/2,1,1/2,1/2)",
        ncycle=4,
        two_spins=[3, 2, 1, 1],
        t_cycle=[2 - I, -1 + 2 * I, 1 + 3 * I, 2 + I],
        jz_cycle=[R(5, 7), R(-7, 9), R(9, 11), R(-11, 13)],
        fields=[R(-3, 8), R(7, 6), R(-9, 10), R(11, 12)],
    ),
    Case(
        name="E7: five-cycle with one spin 1, generic XXZ/fields",
        ncycle=5,
        two_spins=[1, 2, 1, 1, 1],
        t_cycle=[1 + 2 * I, -2 + I, 3 - I, 1 + I, -1 + 3 * I],
        jz_cycle=[R(-2, 5), R(3, 7), R(-5, 9), R(7, 11), R(-11, 13)],
        fields=[R(-1, 4), R(9, 7), R(5, 8), R(-7, 10), R(13, 12)],
    ),
    Case(
        name="E8: six-cycle, all spins 1/2, generic XXZ/fields",
        ncycle=6,
        two_spins=[1, 1, 1, 1, 1, 1],
        t_cycle=[1 + I, 2 - I, -1 + 2 * I, 3 + I, 1 - 2 * I, 2 + I],
        jz_cycle=[R(1, 3), R(-2, 5), R(3, 7), R(-4, 9), R(5, 11), R(-6, 13)],
        fields=[R(1, 5), R(-8, 7), R(3, 8), R(-4, 9), R(5, 12), R(-6, 13)],
    ),
    Case(
        name="E9: four-cycle with remote cycle spin 3/2",
        ncycle=4,
        two_spins=[1, 1, 1, 3],
        t_cycle=[-1 + 2 * I, 2 + I, 1 - I, 3 + 2 * I],
        jz_cycle=[R(-3, 7), R(5, 8), R(-7, 9), R(9, 10)],
        fields=[R(2, 5), R(-6, 7), R(8, 9), R(-10, 11)],
    ),
    Case(
        name="E10: triangle with overlap spin 3/2",
        ncycle=3,
        two_spins=[1, 3, 2],
        t_cycle=[1 - 3 * I, -2 + I, 2 + 2 * I],
        jz_cycle=[R(4, 9), R(-5, 11), R(7, 13)],
        fields=[R(-5, 6), R(13, 10), R(17, 14)],
    ),
    Case(
        name="E11: triangle with a two-edge tree grafted at overlap B",
        ncycle=3,
        two_spins=[1, 1, 1, 2, 1],
        t_cycle=[1 + I, 2 - I, -1 + 3 * I],
        jz_cycle=[R(3, 5), R(-5, 7), R(7, 9)],
        fields=[R(-2, 3), R(4, 5), R(-6, 7), R(8, 9), R(-10, 11)],
        tree_edges=[
            (1, 3, 2 + 3 * I, R(-11, 13)),
            (3, 4, -3 + 2 * I, R(13, 15)),
        ],
    ),
]


def main(argv):
    selected = CASES
    if len(argv) > 1:
        wanted = set(argv[1:])
        selected = [c for c in CASES if c.name.split(":", 1)[0] in wanted]
        unknown = wanted - {c.name.split(":", 1)[0] for c in selected}
        if unknown:
            raise SystemExit(f"unknown case label(s): {sorted(unknown)}")
    for case in selected:
        check_case(case)
    print("=" * 76)
    if FAILURES:
        print("FAILED:")
        for item in FAILURES:
            print("  -", item)
        return 1
    print(f"All {len(selected)} exact arbitrary-spin/unicyclic certificates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
