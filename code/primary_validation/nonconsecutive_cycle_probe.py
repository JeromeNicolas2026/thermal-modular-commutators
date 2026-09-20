#!/usr/bin/env python3
"""Algebraic beta-series probe for modular commutators on small spin rings.

This is a standalone certificate. It does not edit or import the manuscript.
It expands exp(-beta H), the normalized reduced states, and their logarithms
as truncated matrix power series.  Matrix entries are evaluated in double
precision, but no finite-beta differencing or matrix logarithm is used.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import factorial

import numpy as np


def spin_matrices(two_s: int):
    s = two_s / 2
    m = np.arange(s, -s - 1, -1, dtype=float)
    d = two_s + 1
    sz = np.diag(m).astype(complex)
    sp = np.zeros((d, d), dtype=complex)
    # column j has m[j]; S+ sends it to m[j]+1, namely row j-1.
    for j in range(1, d):
        mj = m[j]
        sp[j - 1, j] = np.sqrt(s * (s + 1) - mj * (mj + 1))
    return sp, sp.conj().T, sz


def kron_all(fs):
    out = np.array([[1.0 + 0j]])
    for f in fs:
        out = np.kron(out, f)
    return out


def local_op(m, site, dims):
    return kron_all([m if i == site else np.eye(d) for i, d in enumerate(dims)])


def ring_hamiltonian(two_spins, t, h, jz=None):
    n = len(two_spins)
    dims = [u + 1 for u in two_spins]
    one = [spin_matrices(u) for u in two_spins]
    sp = [local_op(one[i][0], i, dims) for i in range(n)]
    sm = [local_op(one[i][1], i, dims) for i in range(n)]
    sz = [local_op(one[i][2], i, dims) for i in range(n)]
    D = int(np.prod(dims))
    H = np.zeros((D, D), dtype=complex)
    for i in range(n):
        j = (i + 1) % n
        H += .5 * (t[i] * (sp[i] @ sm[j]) + np.conj(t[i]) * (sm[i] @ sp[j]))
        if jz is not None:
            H += jz[i] * (sz[i] @ sz[j])
    for i in range(n):
        H -= h[i] * sz[i]
    return H


def smul(a, b, order):
    """Scalar-series times matrix-series."""
    out = [np.zeros_like(b[0]) for _ in range(order + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i + j <= order:
                out[i + j] += x * y
    return out


def mmul(a, b, order):
    out = [np.zeros((a[0].shape[0], b[0].shape[1]), dtype=complex)
           for _ in range(order + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i + j <= order:
                out[i + j] += x @ y
    return out


def sinv(a, order):
    out = [0j] * (order + 1)
    out[0] = 1 / a[0]
    for n in range(1, order + 1):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1)) / a[0]
    return out


def slog(a, order):
    d = a[0].shape[0]
    ident = np.eye(d, dtype=complex)
    # All uses below have a[0]=I.  Keep the general constant check explicit.
    assert np.max(np.abs(a[0] - ident)) < 1e-12
    x = [u.copy() for u in a]
    x[0] = np.zeros_like(ident)
    out = [np.zeros_like(ident) for _ in range(order + 1)]
    cur = [ident] + [np.zeros_like(ident) for _ in range(order)]
    for k in range(1, order + 1):
        cur = mmul(cur, x, order)
        c = (-1) ** (k + 1) / k
        for j in range(order + 1):
            out[j] += c * cur[j]
    return out


def partial_trace_normalized(M, dims, keep):
    keep = tuple(sorted(keep))
    rest = tuple(i for i in range(len(dims)) if i not in keep)
    perm = keep + rest
    p = len(keep)
    arr = M.reshape(tuple(dims) * 2).transpose(perm + tuple(i + len(dims) for i in perm))
    dk = int(np.prod([dims[i] for i in keep]))
    dr = int(np.prod([dims[i] for i in rest]))
    arr = arr.reshape(dk, dr, dk, dr)
    return np.trace(arr, axis1=1, axis2=3) / dr


def embed(op, op_sites, union_sites, dims):
    """Embed op (whose axes follow sorted(op_sites)) on sorted(union_sites)."""
    op_sites = tuple(sorted(op_sites))
    union_sites = tuple(sorted(union_sites))
    du = [dims[i] for i in union_sites]
    dop = [dims[i] for i in op_sites]
    pos = {s: k for k, s in enumerate(union_sites)}
    out = np.zeros((int(np.prod(du)), int(np.prod(du))), dtype=complex)
    for ii in np.ndindex(*du):
        ai = tuple(ii[pos[s]] for s in op_sites)
        ai_flat = np.ravel_multi_index(ai, dop)
        for jj in np.ndindex(*du):
            if any(ii[pos[s]] != jj[pos[s]] for s in union_sites if s not in op_sites):
                continue
            aj = tuple(jj[pos[s]] for s in op_sites)
            aj_flat = np.ravel_multi_index(aj, dop)
            out[np.ravel_multi_index(ii, du), np.ravel_multi_index(jj, du)] = op[ai_flat, aj_flat]
    return out


def relative_density_series(H, order):
    D = H.shape[0]
    e = []
    P = np.eye(D, dtype=complex)
    for k in range(order + 1):
        e.append(((-1) ** k / factorial(k)) * P)
        P = P @ H
    tau = [np.trace(u) / D for u in e]
    return smul(sinv(tau, order), e, order)


def modular_coefficients(r, dims, A, B, C, order):
    X, Y = (A, B), (B, C)
    U = tuple(sorted((A, B, C)))
    rU = [partial_trace_normalized(u, dims, U) for u in r]
    rX = [partial_trace_normalized(u, dims, X) for u in r]
    rY = [partial_trace_normalized(u, dims, Y) for u in r]
    LX = slog(rX, order)
    LY = slog(rY, order)
    LXU = [embed(u, X, U, dims) for u in LX]
    LYU = [embed(u, Y, U, dims) for u in LY]
    comm = [u - v for u, v in zip(mmul(LXU, LYU, order), mmul(LYU, LXU, order))]
    prod = mmul(rU, comm, order)
    dU = prod[0].shape[0]
    return np.array([(1j * np.trace(u) / dU).real for u in prod])


def cyclic_sign(A, B, C, n):
    """+1 iff A,B,C occur in that order in the fixed forward orientation."""
    b = (B - A) % n
    c = (C - A) % n
    return 1 if 0 < b < c else -1


def kappa(two_s):
    s = Fraction(two_s, 2)
    return s * (s + 1) / 3


def predicted(two_spins, A, B, C):
    n = len(two_spins)
    p = Fraction(2 * ((-1) ** n) * cyclic_sign(A, B, C, n), 1)
    for u in two_spins:
        p *= kappa(u)
    return p


def rational(x, cap=10**9):
    return Fraction(float(x)).limit_denominator(cap)


def run_case(two_spins, full_triples=True, generic_jz=False):
    n = len(two_spins)
    dims = [u + 1 for u in two_spins]
    order = n + 1
    t = [1 + 0j] * n
    t[-1] = 1j  # Im product = +1 in the fixed forward orientation.
    jz = None
    if generic_jz:
        jz = [(3 * i - 2) / (2 * i + 3) for i in range(n)]
    triples = list(permutations(range(n), 3)) if full_triples else [(0, n // 3, 2 * n // 3)]
    # The leading term is linear in fields.  Probe every one-hot field.
    observed = {}
    for field in range(n):
        h = [0.] * n
        h[field] = 1.
        H = ring_hamiltonian(two_spins, t, h, jz)
        r = relative_density_series(H, order)
        for A, B, C in triples:
            cc = modular_coefficients(r, dims, A, B, C, order)
            observed[(A, B, C, field)] = cc
    worst_low = 0.
    worst_main = 0.
    unexpected = []
    samples = []
    for A, B, C in triples:
        target = float(predicted(two_spins, A, B, C))
        vec = []
        for field in range(n):
            cc = observed[(A, B, C, field)]
            worst_low = max(worst_low, np.max(np.abs(cc[:order])))
            got = cc[order]
            exp = target if field == B else 0.
            worst_main = max(worst_main, abs(got - exp))
            vec.append(rational(got))
            if abs(got - exp) > 2e-9:
                unexpected.append(((A, B, C), field, got, exp))
        if cyclic_sign(A, B, C, n) == 1 and not (B == (A + 1) % n and C == (B + 1) % n):
            samples.append(((A, B, C), vec, rational(target)))
    print(f"n={n}, 2s={two_spins}, D={np.prod(dims)}, Jz={'generic' if generic_jz else '0'}")
    print(f"  all {len(triples)} ordered triples x {n} fields: low<{worst_low:.2e}, main err={worst_main:.2e}")
    for row in samples[:min(4, len(samples))]:
        print(f"  nonconsecutive {row[0]}: field-vector={row[1]}, predicted B={row[2]}")
    if unexpected:
        print("  UNEXPECTED", unexpected[:10])
    return not unexpected and worst_low < 2e-9


if __name__ == "__main__":
    ok = True
    for n in range(3, 7):
        ok &= run_case([1] * n, full_triples=True)
    # Mixed irreducible spins.  Use representative nonconsecutive triples for
    # n=5,6 to keep the scratch probe quick, and exhaustive triples for n=3,4.
    for ss, exhaustive in [([1, 2, 3], True), ([1, 2, 1, 3], True),
                           ([1, 2, 1, 1, 1], False), ([1, 2, 1, 1, 1, 1], False)]:
        ok &= run_case(ss, full_triples=exhaustive)
    raise SystemExit(0 if ok else 1)
