#!/usr/bin/env python3
"""Small exact SymPy certificate for a nonconsecutive mixed-spin 4-cycle."""

import itertools
from math import factorial

import sympy as sp


def kron_all(fs):
    out = sp.Matrix([[1]])
    for f in fs:
        out = sp.kronecker_product(out, f)
    return sp.Matrix(out)


def spin(two_s):
    s = sp.Rational(two_s, 2)
    d = two_s + 1
    ms = [s - j for j in range(d)]
    z = sp.diag(*ms)
    p = sp.zeros(d)
    for j in range(1, d):
        m = ms[j]
        p[j - 1, j] = sp.sqrt(s * (s + 1) - m * (m + 1))
    return p, p.T, z


def flat(idx, dims):
    ans = 0
    for i, d in zip(idx, dims):
        ans = ans * d + i
    return ans


def reduce_norm(M, dims, keep):
    keep = tuple(sorted(keep))
    rest = tuple(i for i in range(len(dims)) if i not in keep)
    kd = tuple(dims[i] for i in keep)
    rd = tuple(dims[i] for i in rest)
    out = sp.zeros(sp.prod(kd))
    for a in itertools.product(*[range(d) for d in kd]):
        for b in itertools.product(*[range(d) for d in kd]):
            acc = 0
            for r in itertools.product(*[range(d) for d in rd]):
                ii = [0] * len(dims)
                jj = [0] * len(dims)
                for q, site in enumerate(keep):
                    ii[site], jj[site] = a[q], b[q]
                for q, site in enumerate(rest):
                    ii[site] = jj[site] = r[q]
                acc += M[flat(ii, dims), flat(jj, dims)]
            out[flat(a, kd), flat(b, kd)] = acc / sp.prod(rd)
    return out


def embed(M, sites, union, dims):
    sites = tuple(sorted(sites))
    union = tuple(sorted(union))
    sd = tuple(dims[i] for i in sites)
    ud = tuple(dims[i] for i in union)
    pos = {s: i for i, s in enumerate(union)}
    out = sp.zeros(sp.prod(ud))
    for a in itertools.product(*[range(d) for d in ud]):
        for b in itertools.product(*[range(d) for d in ud]):
            if any(a[pos[s]] != b[pos[s]] for s in union if s not in sites):
                continue
            aa = tuple(a[pos[s]] for s in sites)
            bb = tuple(b[pos[s]] for s in sites)
            out[flat(a, ud), flat(b, ud)] = M[flat(aa, sd), flat(bb, sd)]
    return out


def mmul(a, b, N):
    out = [sp.zeros(a[0].rows, b[0].cols) for _ in range(N + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i + j <= N:
                out[i + j] += x * y
    return out


def smul(a, b, N):
    out = [sp.zeros(b[0].rows, b[0].cols) for _ in range(N + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i + j <= N:
                out[i + j] += x * y
    return out


def sinv(a, N):
    out = [sp.S.Zero] * (N + 1)
    out[0] = 1 / a[0]
    for n in range(1, N + 1):
        out[n] = -sum(a[k] * out[n-k] for k in range(1, n+1)) / a[0]
    return out


def slog(a, N):
    d = a[0].rows
    x = list(a)
    x[0] = sp.zeros(d)
    out = [sp.zeros(d) for _ in range(N + 1)]
    cur = [sp.eye(d)] + [sp.zeros(d) for _ in range(N)]
    for k in range(1, N + 1):
        cur = mmul(cur, x, N)
        c = sp.Rational((-1) ** (k + 1), k)
        out = [u + c * v for u, v in zip(out, cur)]
    return out


def coefficient(two_s=(1, 2, 1, 1), A=0, B=2, C=3):
    n, N = 4, 5
    dims = tuple(u + 1 for u in two_s)
    mats = [spin(u) for u in two_s]
    P, M, Z = [], [], []
    for i in range(n):
        P.append(kron_all([mats[j][0] if i == j else sp.eye(dims[j]) for j in range(n)]))
        M.append(kron_all([mats[j][1] if i == j else sp.eye(dims[j]) for j in range(n)]))
        Z.append(kron_all([mats[j][2] if i == j else sp.eye(dims[j]) for j in range(n)]))
    t = [1, 1, 1, sp.I]
    jz = [sp.Rational(3, 5), sp.Rational(-7, 4), sp.Rational(11, 9), sp.Rational(-2, 7)]
    H = sp.zeros(sp.prod(dims))
    for i in range(n):
        j = (i + 1) % n
        H += sp.Rational(1, 2) * (t[i] * P[i] * M[j] + sp.conjugate(t[i]) * M[i] * P[j])
        H += jz[i] * Z[i] * Z[j]
    H -= Z[B]
    e, power = [], sp.eye(H.rows)
    for k in range(N + 1):
        e.append(sp.Rational((-1) ** k, factorial(k)) * power)
        power = power * H
    tau = [sp.trace(u) / H.rows for u in e]
    r = smul(sinv(tau, N), e, N)
    X, Y, U = (A, B), (B, C), tuple(sorted((A, B, C)))
    rU = [reduce_norm(u, dims, U) for u in r]
    LX = slog([reduce_norm(u, dims, X) for u in r], N)
    LY = slog([reduce_norm(u, dims, Y) for u in r], N)
    LX = [embed(u, X, U, dims) for u in LX]
    LY = [embed(u, Y, U, dims) for u in LY]
    comm = [u-v for u, v in zip(mmul(LX, LY, N), mmul(LY, LX, N))]
    ans = [sp.simplify(sp.I * sp.trace(u) / rU[0].rows) for u in mmul(rU, comm, N)]
    return ans


if __name__ == '__main__':
    coefficients = coefficient()
    target = sp.Rational(1, 48)
    assert coefficients[:5] == [0] * 5
    assert coefficients[5] == target
    print("[ok] mixed-spin nonconsecutive 4-cycle:")
    print(f"     m_1=...=m_4=0 and m_5={target}")
