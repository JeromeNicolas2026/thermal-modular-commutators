#!/usr/bin/env python3
"""Independent re-derivation of the arbitrary-spin unicyclic onset coefficient.

Deliberately shares no code and no library with either v06 engine.  The
overall method is the same one (formal Gibbs series, normalized partial traces,
expansion of log(1+Y), contraction of the commutator): this is an independent
implementation, not an independent method.  What differs is the arithmetic:
  * no SymPy, no NumPy, no floating point anywhere;
  * spin matrices are put in a *diagonal integer gauge* S^+ -> superdiagonal of
    1, S^- -> subdiagonal of (s-m)(s+m+1), so every matrix entry is a Gaussian
    rational.  The gauge is a local similarity D = tensor_x D_x,
    under which rho -> D rho D^{-1}, rho_X -> D_X rho_X D_X^{-1} (partial trace
    over the complement is unaffected by cyclicity inside the traced factor),
    hence log rho_X -> D_X log rho_X D_X^{-1} and M is invariant.  This removes
    every radical from the computation.
  * the Gibbs series is propagated with a *sparse* H acting on dense series
    coefficients; the reduced logarithms are expanded by the noncommutative
    log(1+Y) series.

Conventions (identical to the v5/v06 manuscript):
    H = sum_{edges} [ (1/2)(t_xy S_x^+ S_y^- + conj(t_xy) S_x^- S_y^+)
                      + Jz_xy S_x^z S_y^z ]  -  sum_x h_x S_x^z
    M_beta(AB,BC) = i Tr_ABC rho_ABC,beta
                       [log rho_AB tensor I_C, I_A tensor log rho_BC]
    W_gamma       = product of t along the cycle oriented A -> ... -> B -> ... -> C -> ... -> A
Claim under test:
    [beta^{n+1}] M = 2 (-1)^n h_B (prod_{x in cycle} kappa_x) Im W_gamma,
    kappa_x = s_x(s_x+1)/3.
"""

from fractions import Fraction as F
from itertools import product

ZERO = (F(0), F(0))


# ---------------------------------------------------------------- scalars ---
def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cscale(a, s):
    return (a[0] * s, a[1] * s)


def cnum(re, im=0):
    return (F(re), F(im))


# ---------------------------------------------------------------- matrices --
def zeros(d):
    return [[ZERO] * d for _ in range(d)]


def eye(d):
    M = zeros(d)
    for i in range(d):
        M[i][i] = cnum(1)
    return M


def madd(A, B):
    return [[cadd(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mscale(A, s):
    return [[cscale(a, s) for a in ra] for ra in A]


def mmul(A, B):
    d = len(A)
    out = [[ZERO] * d for _ in range(d)]
    Bt = list(zip(*B))
    for i in range(d):
        Ai = A[i]
        oi = out[i]
        nz = [(k, Ai[k]) for k in range(d) if Ai[k] != ZERO]
        for j in range(d):
            Bj = Bt[j]
            sr = F(0)
            si = F(0)
            for k, (ar, ai) in nz:
                br, bi = Bj[k]
                sr += ar * br - ai * bi
                si += ar * bi + ai * br
            oi[j] = (sr, si)
    return out


def sparse_of(A):
    """(row, col, value) list of the nonzero entries."""
    return [(i, j, A[i][j]) for i in range(len(A)) for j in range(len(A))
            if A[i][j] != ZERO]


def sparse_dense(Hs, M, d):
    """H @ M with H given as a nonzero list."""
    out = [[ZERO] * d for _ in range(d)]
    for i, k, (vr, vi) in Hs:
        oi = out[i]
        Mk = M[k]
        for j in range(d):
            br, bi = Mk[j]
            if br or bi:
                cr, ci = oi[j]
                oi[j] = (cr + vr * br - vi * bi, ci + vr * bi + vi * br)
    return out


def mtrace(A):
    r = F(0)
    i = F(0)
    for k in range(len(A)):
        r += A[k][k][0]
        i += A[k][k][1]
    return (r, i)


# ------------------------------------------------------------ spin algebra --
def spin_ops_integer_gauge(two_s):
    """(S+, S-, Sz) for spin s = two_s/2 in the gauge that clears all radicals.

    Basis ordered by descending m.  S^+ has 1 on the superdiagonal;
    S^- has (s-m)(s+m+1) on the subdiagonal; S^z = diag(m).
    """
    d = two_s + 1
    ms = [F(two_s, 2) - k for k in range(d)]
    sp = zeros(d)
    sm = zeros(d)
    sz = zeros(d)
    s = F(two_s, 2)
    for col in range(1, d):
        m = ms[col]                      # S^+ |m> = c_m |m+1>
        c2 = (s - m) * (s + m + 1)       # = c_m^2, a positive integer
        sp[col - 1][col] = cnum(1)
        sm[col][col - 1] = (F(c2), F(0))
    for k in range(d):
        sz[k][k] = (ms[k], F(0))
    return sp, sm, sz


def check_su2(two_s):
    """[S^z,S^+]=S^+, [S^+,S^-]=2S^z, tau(S^+S^-)=2 kappa in this gauge."""
    sp, sm, sz = spin_ops_integer_gauge(two_s)
    d = two_s + 1
    c1 = madd(mmul(sz, sp), mscale(mmul(sp, sz), F(-1)))
    c2 = madd(mmul(sp, sm), mscale(mmul(sm, sp), F(-1)))
    assert c1 == sp, two_s
    assert c2 == mscale(sz, F(2)), two_s
    s = F(two_s, 2)
    kappa = s * (s + 1) / 3
    tr = mtrace(mmul(sp, sm))
    assert (tr[0] / d, tr[1]) == (2 * kappa, F(0)), two_s
    return kappa


def kappa_of(two_s):
    s = F(two_s, 2)
    return s * (s + 1) / 3


# ------------------------------------------------------------- many bodies --
class Lattice:
    def __init__(self, two_spins):
        self.two_spins = list(two_spins)
        self.dims = [q + 1 for q in two_spins]
        self.N = len(self.dims)
        self.D = 1
        for d in self.dims:
            self.D *= d
        self.strides = []
        acc = 1
        for d in reversed(self.dims):
            self.strides.append(acc)
            acc *= d
        self.strides.reverse()

    def index(self, digits):
        return sum(x * s for x, s in zip(digits, self.strides))

    def digits(self, idx):
        out = []
        for s, d in zip(self.strides, self.dims):
            out.append((idx // s) % d)
        return out

    def embed(self, local, site):
        """Site operator as a nonzero list on the full space."""
        d = self.dims[site]
        st = self.strides[site]
        nz = []
        for a in range(d):
            for b in range(d):
                v = local[a][b]
                if v == ZERO:
                    continue
                nz.append((a, b, v))
        out = []
        for idx in range(self.D):
            db = (idx // st) % d
            for a, b, v in nz:
                if b == db:
                    out.append((idx + (a - b) * st, idx, v))
        return out

    def ptrace_keep(self, M, keep):
        """Partial trace of a dense full-space matrix onto the ordered `keep`."""
        keep = list(keep)
        rest = [x for x in range(self.N) if x not in keep]
        dk = 1
        for x in keep:
            dk *= self.dims[x]
        out = [[ZERO] * dk for _ in range(dk)]
        ranges_rest = [range(self.dims[x]) for x in rest]
        ranges_keep = [range(self.dims[x]) for x in keep]
        for kr in product(*ranges_keep):
            row_k = 0
            for x, v in zip(keep, kr):
                row_k = row_k * self.dims[x] + v
            for kc in product(*ranges_keep):
                col_k = 0
                for x, v in zip(keep, kc):
                    col_k = col_k * self.dims[x] + v
                sr = F(0)
                si = F(0)
                for rr in product(*ranges_rest):
                    dig_r = [0] * self.N
                    dig_c = [0] * self.N
                    for x, v in zip(keep, kr):
                        dig_r[x] = v
                    for x, v in zip(keep, kc):
                        dig_c[x] = v
                    for x, v in zip(rest, rr):
                        dig_r[x] = v
                        dig_c[x] = v
                    vr, vi = M[self.index(dig_r)][self.index(dig_c)]
                    sr += vr
                    si += vi
                out[row_k][col_k] = (sr, si)
        return out


def build_H(lat, edges, fields, fields_x=None, gx=None):
    """edges: list of (x, y, t_xy as (re,im), Jz).  fields: list of h_x.

    Transverse fields, OUTSIDE the scope of the theorem, are added by either
    of two equivalent arguments; they are used only to certify that the axial
    restriction is necessary (see t4_transverse_boundary.py):
        fields_x : real h^x_x, term  -h^x_x S^x_x
        gx       : complex g_x,   term  -(1/2)(g_x S^+_x + conj(g_x) S^-_x)
    """
    if fields_x is not None and gx is not None:
        raise ValueError("use either fields_x or gx, not both")

    D = lat.D
    H = zeros(D)
    ops = [spin_ops_integer_gauge(q) for q in lat.two_spins]
    emb = {}
    for x in range(lat.N):
        sp, sm, sz = ops[x]
        emb[(x, '+')] = lat.embed(sp, x)
        emb[(x, '-')] = lat.embed(sm, x)
        emb[(x, 'z')] = lat.embed(sz, x)

    def add_product(nz1, nz2, coeff):
        d1 = {}
        for i, j, v in nz1:
            d1.setdefault(j, []).append((i, v))
        for k, j2, v2 in nz2:
            for i, v1 in d1.get(k, ()):
                val = cmul(cmul(v1, v2), coeff)
                H[i][j2] = cadd(H[i][j2], val)

    for (x, y, t, jz) in edges:
        add_product(emb[(x, '+')], emb[(y, '-')], cscale(t, F(1, 2)))
        add_product(emb[(x, '-')], emb[(y, '+')],
                    cscale((t[0], -t[1]), F(1, 2)))
        if jz:
            add_product(emb[(x, 'z')], emb[(y, 'z')], cnum(jz))
    for x, h in enumerate(fields):
        if h:
            for i, j, v in emb[(x, 'z')]:
                H[i][j] = cadd(H[i][j], cscale(v, -F(h)))
    if fields_x:
        for x, h in enumerate(fields_x):
            if h:
                for tag in ('+', '-'):
                    for i, j, v in emb[(x, tag)]:
                        H[i][j] = cadd(H[i][j], cscale(v, -F(h) / 2))
    if gx:
        for x, g in enumerate(gx):
            if g == ZERO:
                continue
            for tag, coeff in (('+', cscale(g, F(-1, 2))),
                               ('-', cscale((g[0], -g[1]), F(-1, 2)))):
                for i, j, v in emb[(x, tag)]:
                    H[i][j] = cadd(H[i][j], cmul(v, coeff))
    return H


# --------------------------------------------------------------- series -----
def gibbs_series(lat, H, K):
    """[beta^k] exp(-beta H) for k = 0..K."""
    Hs = sparse_of(H)
    E = [eye(lat.D)]
    cur = eye(lat.D)
    for k in range(1, K + 1):
        cur = sparse_dense(Hs, cur, lat.D)          # H^k
        E.append(mscale(cur, F((-1) ** k, factorial(k))))
    return E


def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def series_inverse_scalar(Z, K):
    """1/Z(beta) as a series, Z[0] != 0.  Z entries are (re, im) tuples."""
    inv = [None] * (K + 1)
    z0 = Z[0]
    assert z0 != ZERO
    den = z0[0] ** 2 + z0[1] ** 2
    inv0 = (z0[0] / den, -z0[1] / den)
    inv[0] = inv0
    for k in range(1, K + 1):
        acc = ZERO
        for j in range(1, k + 1):
            acc = cadd(acc, cmul(Z[j], inv[k - j]))
        inv[k] = cmul(cscale(acc, F(-1)), inv0)
    return inv


def series_mat_scalar(Mser, sser, K):
    out = []
    for k in range(K + 1):
        acc = zeros(len(Mser[0]))
        for j in range(k + 1):
            acc = madd(acc, [[cmul(v, sser[k - j]) for v in row]
                             for row in Mser[j]])
        out.append(acc)
    return out


def series_matmul(A, B, K):
    d = len(A[0])
    out = []
    for k in range(K + 1):
        acc = zeros(d)
        for j in range(k + 1):
            acc = madd(acc, mmul(A[j], B[k - j]))
        out.append(acc)
    return out


def series_log_at_identity(R, K, dS):
    """log(dS * R) up to an additive multiple of the identity.

    R is the series of the reduced state; dS*R[0] must be the identity.
    Returns the series of log(1+Y), Y = dS*R - 1 = O(beta).
    """
    d = len(R[0])
    assert mscale(R[0], F(dS)) == eye(d)
    Y = [zeros(d)] + [mscale(R[k], F(dS)) for k in range(1, K + 1)]
    L = [zeros(d) for _ in range(K + 1)]
    P = [eye(d)] + [zeros(d) for _ in range(K)]      # Y^0
    for m in range(1, K + 1):
        P = series_matmul(P, Y, K)                   # Y^m
        sgn = F((-1) ** (m + 1), m)
        for k in range(K + 1):
            L[k] = madd(L[k], mscale(P[k], sgn))
    return L


def extend(op, pos, dims):
    """Embed an operator acting on the sites `pos` (ordered) of a 3-site space."""
    dA, dB, dC = dims
    D = dA * dB * dC
    out = zeros(D)
    idx = [(a, b, c) for a in range(dA) for b in range(dB) for c in range(dC)]
    lin = {t: n for n, t in enumerate(idx)}
    sub = [tuple(t[p] for p in pos) for t in idx]
    subdims = [dims[p] for p in pos]
    sublin = {}
    n = 0
    for t in product(*[range(x) for x in subdims]):
        sublin[t] = n
        n += 1
    for r, tr in enumerate(idx):
        for c, tc in enumerate(idx):
            if all(tr[p] == tc[p] for p in range(3) if p not in pos):
                v = op[sublin[sub[r]]][sublin[sub[c]]]
                if v != ZERO:
                    out[r][c] = v
    return out


def modular_series_from_H(lat, H, marked, K):
    """Series of M_beta(AB,BC) for an already assembled Hamiltonian H.

    This entry point is useful for coefficient-extraction certificates whose
    tagged Hamiltonian terms need not be paired into Hermitian couplings.
    """
    E = gibbs_series(lat, H, K)
    Z = [mtrace(Ek) for Ek in E]
    invZ = series_inverse_scalar(Z, K)
    A, B, C = marked
    RABC = [lat.ptrace_keep(Ek, [A, B, C]) for Ek in E]
    RABC = series_mat_scalar(RABC, invZ, K)
    dA, dB, dC = (lat.dims[A], lat.dims[B], lat.dims[C])

    tri = Lattice([lat.two_spins[A], lat.two_spins[B], lat.two_spins[C]])
    RAB = [tri.ptrace_keep(M, [0, 1]) for M in RABC]
    RBC = [tri.ptrace_keep(M, [1, 2]) for M in RABC]

    LAB = series_log_at_identity(RAB, K, dA * dB)
    LBC = series_log_at_identity(RBC, K, dB * dC)
    LABx = [extend(M, (0, 1), (dA, dB, dC)) for M in LAB]
    LBCx = [extend(M, (1, 2), (dA, dB, dC)) for M in LBC]

    comm = []
    for k in range(K + 1):
        acc = zeros(dA * dB * dC)
        for j in range(k + 1):
            acc = madd(acc, mmul(LABx[j], LBCx[k - j]))
            acc = madd(acc, mscale(mmul(LBCx[j], LABx[k - j]), F(-1)))
        comm.append(acc)
    out = []
    for k in range(K + 1):
        acc = ZERO
        for j in range(k + 1):
            acc = cadd(acc, mtrace(mmul(RABC[j], comm[k - j])))
        out.append(cmul(cnum(0, 1), acc))            # multiply by i
    return out


def modular_series(lat, edges, fields, marked, K, fields_x=None, gx=None):
    """Series of M_beta(AB,BC) up to beta^K.  marked = (A,B,C) site labels."""
    H = build_H(lat, edges, fields, fields_x, gx)
    return modular_series_from_H(lat, H, marked, K)
