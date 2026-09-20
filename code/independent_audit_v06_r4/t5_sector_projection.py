#!/usr/bin/env python3
"""Exact symbolic certificate for the local transverse order-seven sector.

This file is deliberately self-contained.  It imports neither the numerical
engine in this archive nor SymPy/NumPy, and it performs no floating-point
arithmetic.  All scalar coefficients are Gaussian rationals represented by
pairs of :class:`fractions.Fraction` objects.

The Hamiltonian convention is

    H_AB = (t_AB S_A^+ S_B^- + tb_AB S_A^- S_B^+)/2
           + J_AB^z S_A^z S_B^z,
    H_BC = (t_BC S_B^+ S_C^- + tb_BC S_B^- S_C^+)/2,
    H_z  = -h_B^z S_B^z,
    H_perp = -(g_x S_x^+ + gb_x S_x^-)/2.

Here ``tb`` and ``gb`` are *formal independent variables*.  The physical
specialization is tb=conj(t), gb=conj(g).  For

    M_beta = i Tr rho_beta [log rho_AB, log rho_BC],

the script extracts, rather than samples, the following two multidegrees of
the coefficient m_7=[beta^7]M_beta:

    [J h tBC tbBC tAB gbA gB] m_7 = -i/46080,
    [J h tBC tbBC tbAB gA gbB] m_7 = +i/46080.

Consequently, after the physical specialization,

    Pi_loc m_7
      = J h tBC tbBC (tAB gbA gB - tbAB gA gbB)/(46080 i)
      = J h |tBC|^2 Im(tAB conj(gA) gB)/23040.

WHY THIS IS AN IDENTITY, NOT INTERPOLATION
-------------------------------------------
For each of the two target monomials the calculation takes place in

    Q(i)[x_1,...,x_7]/(x_1^2,...,x_7^2).

A polynomial is stored exactly as ``{squarefree monomial mask: coefficient}``.
Terms containing a squared target variable are discarded because they cannot
contribute to the squarefree target coefficient.  Parameters absent from the
target may be set to zero for the same reason: Hamiltonian perturbation theory
contains no negative powers.  Thus the coefficient of mask 2^7-1 returned
below is literally the requested multivariate coefficient; no parameter is
ever assigned a numerical test value.

The complete finite-dimensional construction is repeated in this coefficient
ring: exp(-beta H), its trace normalization, both normalized partial traces,
the noncommutative log(1+Y) series, the commutator, and its contraction with
rho_ABC.  Hard assertions check the maximally mixed zeroth order, the absence
of the target multidegree below order seven, and both displayed coefficients.

AMBIENT-INDEPENDENCE AND LIMITS
-------------------------------
The computation is on the open three-qubit path A-B-C.  This also proves the
same projected coefficient in any finite ambient graph: coefficient extraction
sets every Hamiltonian parameter not occurring in the target monomial to zero,
after which all other sites are maximally mixed spectators and their dimension
factors cancel between the partial traces and the partition function.

The certificate proves only the displayed projected sector for spin 1/2.  It
does not claim that the full m_7 is local, classify every transverse sector, or
extend the axial onset theorem to transverse fields.

On an ordinary laptop the two symbolic orientations normally finish in well
under a minute.  The script reports an integer millisecond wall-clock duration;
that measurement is not used in any mathematical operation.
"""

from fractions import Fraction as F
from itertools import product
from time import monotonic_ns


# A Gaussian rational is a pair (real part, imaginary part).
CZ = (F(0), F(0))
CO = (F(1), F(0))


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1],
            a[0] * b[1] + a[1] * b[0])


def cscale(a, q):
    return (a[0] * q, a[1] * q)


# A polynomial is a dictionary mask -> Gaussian rational.  Multiplication is
# in Q(i)[x_1,...,x_7]/(x_1^2,...,x_7^2).
def pconst(re=0, im=0):
    z = (F(re), F(im))
    return {} if z == CZ else {0: z}


def pvar(bit, q=F(1)):
    return {} if q == 0 else {1 << bit: (F(q), F(0))}


def padd(a, b):
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for m, z in b.items():
        w = cadd(out.get(m, CZ), z)
        if w == CZ:
            out.pop(m, None)
        else:
            out[m] = w
    return out


def pscale(a, q):
    """Multiply a polynomial by a rational or Gaussian rational."""
    z = q if isinstance(q, tuple) else (F(q), F(0))
    if not a or z == CZ:
        return {}
    return {m: w for m, c in a.items() if (w := cmul(c, z)) != CZ}


def pmul(a, b):
    if not a or not b:
        return {}
    out = {}
    for ma, za in a.items():
        for mb, zb in b.items():
            if ma & mb:
                continue
            m = ma | mb
            w = cadd(out.get(m, CZ), cmul(za, zb))
            if w == CZ:
                out.pop(m, None)
            else:
                out[m] = w
    return out


def pinv_constant(a):
    """Inverse of a polynomial known to be a nonzero scalar constant."""
    assert set(a) == {0}
    z = a[0]
    den = z[0] * z[0] + z[1] * z[1]
    assert den != 0
    return {0: (z[0] / den, -z[1] / den)}


# Matrices with polynomial entries.
def mzero(rows, cols=None):
    cols = rows if cols is None else cols
    return [[{} for _ in range(cols)] for _ in range(rows)]


def meye(d):
    out = mzero(d)
    for k in range(d):
        out[k][k] = pconst(1)
    return out


def madd(a, b):
    return [[padd(x, y) for x, y in zip(ra, rb)]
            for ra, rb in zip(a, b)]


def mscale(a, q):
    return [[pscale(x, q) for x in row] for row in a]


def mmul(a, b):
    nr, nk, nc = len(a), len(b), len(b[0])
    assert len(a[0]) == nk
    out = mzero(nr, nc)
    # The loop order avoids calling pmul for structural zeroes.
    for i in range(nr):
        for k in range(nk):
            aik = a[i][k]
            if not aik:
                continue
            for j in range(nc):
                if b[k][j]:
                    out[i][j] = padd(out[i][j], pmul(aik, b[k][j]))
    return out


def mtrace(a):
    out = {}
    for k in range(len(a)):
        out = padd(out, a[k][k])
    return out


def kron_numeric(*factors):
    """Kronecker product of small matrices with rational entries."""
    out = [[F(1)]]
    for a in factors:
        old = out
        out = [[F(0) for _ in range(len(old[0]) * len(a[0]))]
               for _ in range(len(old) * len(a))]
        for i, row in enumerate(old):
            for j, x in enumerate(row):
                if x == 0:
                    continue
                for r, ar in enumerate(a):
                    for c, y in enumerate(ar):
                        out[i * len(a) + r][j * len(a[0]) + c] = x * y
    return out


I2 = [[F(1), F(0)], [F(0), F(1)]]
SP = [[F(0), F(1)], [F(0), F(0)]]
SM = [[F(0), F(0)], [F(1), F(0)]]
SZ = [[F(1, 2), F(0)], [F(0), F(-1, 2)]]


def three_site(op_a=I2, op_b=I2, op_c=I2):
    return kron_numeric(op_a, op_b, op_c)


def add_letter(hmat, numeric, bit, prefactor=F(1)):
    """Add prefactor*x_bit*numeric to a polynomial Hamiltonian."""
    v = pvar(bit, prefactor)
    for i, row in enumerate(numeric):
        for j, q in enumerate(row):
            if q:
                hmat[i][j] = padd(hmat[i][j], pscale(v, q))


def build_oriented_hamiltonian(orientation):
    """Hamiltonian restricted to one of the two target multidegrees."""
    # Common bit order: J, h, tBC, tbBC, followed by the three oriented
    # A-B/transverse letters.  Each bit occurs once in the target mask.
    hmat = mzero(8)
    add_letter(hmat, three_site(SZ, SZ, I2), 0)
    add_letter(hmat, three_site(I2, SZ, I2), 1, F(-1))
    add_letter(hmat, three_site(I2, SP, SM), 2, F(1, 2))
    add_letter(hmat, three_site(I2, SM, SP), 3, F(1, 2))
    if orientation == "forward":
        # tAB, gbA, gB
        add_letter(hmat, three_site(SP, SM, I2), 4, F(1, 2))
        add_letter(hmat, three_site(SM, I2, I2), 5, F(-1, 2))
        add_letter(hmat, three_site(I2, SP, I2), 6, F(-1, 2))
    elif orientation == "conjugate":
        # tbAB, gA, gbB
        add_letter(hmat, three_site(SM, SP, I2), 4, F(1, 2))
        add_letter(hmat, three_site(SP, I2, I2), 5, F(-1, 2))
        add_letter(hmat, three_site(I2, SM, I2), 6, F(-1, 2))
    else:
        raise ValueError(orientation)
    return hmat


def factorial(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def gibbs_series(hmat, order):
    """Coefficients [beta^k] exp(-beta H), k=0,...,order."""
    dim = len(hmat)
    powers = [meye(dim)]
    for _ in range(order):
        powers.append(mmul(hmat, powers[-1]))
    return [mscale(powers[k], F((-1) ** k, factorial(k)))
            for k in range(order + 1)]


def scalar_series_inverse(z, order):
    out = [{} for _ in range(order + 1)]
    out[0] = pinv_constant(z[0])
    for k in range(1, order + 1):
        acc = {}
        for j in range(1, k + 1):
            acc = padd(acc, pmul(z[j], out[k - j]))
        out[k] = pscale(pmul(out[0], acc), F(-1))
    return out


def matrix_scalar_series(a, z, order):
    out = []
    for k in range(order + 1):
        acc = mzero(len(a[0]))
        for j in range(k + 1):
            term = [[pmul(x, z[k - j]) for x in row] for row in a[j]]
            acc = madd(acc, term)
        out.append(acc)
    return out


def matrix_series_product(a, b, order):
    out = []
    for k in range(order + 1):
        acc = mzero(len(a[0]))
        for j in range(k + 1):
            acc = madd(acc, mmul(a[j], b[k - j]))
        out.append(acc)
    return out


def partial_trace_three(a, keep):
    """Partial trace of an 8x8 three-qubit matrix, preserving keep order."""
    keep = tuple(keep)
    rest = tuple(x for x in range(3) if x not in keep)
    kept_tuples = list(product((0, 1), repeat=len(keep)))
    rest_tuples = list(product((0, 1), repeat=len(rest)))

    def lin(d):
        return 4 * d[0] + 2 * d[1] + d[2]

    out = mzero(1 << len(keep))
    for ir, kr in enumerate(kept_tuples):
        for ic, kc in enumerate(kept_tuples):
            val = {}
            for rr in rest_tuples:
                dr = [0, 0, 0]
                dc = [0, 0, 0]
                for x, q in zip(keep, kr):
                    dr[x] = q
                for x, q in zip(keep, kc):
                    dc[x] = q
                for x, q in zip(rest, rr):
                    dr[x] = dc[x] = q
                val = padd(val, a[lin(dr)][lin(dc)])
            out[ir][ic] = val
    return out


def series_log_normalized(reduced, order):
    """log(d*reduced), omitting the irrelevant scalar -log(d) I."""
    dim = len(reduced[0])
    assert mscale(reduced[0], F(dim)) == meye(dim)
    y = [mzero(dim)] + [mscale(reduced[k], F(dim))
                        for k in range(1, order + 1)]
    out = [mzero(dim) for _ in range(order + 1)]
    power = [meye(dim)] + [mzero(dim) for _ in range(order)]
    for n in range(1, order + 1):
        power = matrix_series_product(power, y, order)
        q = F((-1) ** (n + 1), n)
        for k in range(order + 1):
            out[k] = madd(out[k], mscale(power[k], q))
    return out


def extend_ab(a):
    """Embed a 4x4 AB matrix as AB tensor I_C."""
    out = mzero(8)
    for ar, br, ac, bc, c in product((0, 1), repeat=5):
        r = 4 * ar + 2 * br + c
        col = 4 * ac + 2 * bc + c
        out[r][col] = dict(a[2 * ar + br][2 * ac + bc])
    return out


def extend_bc(a):
    """Embed a 4x4 BC matrix as I_A tensor BC."""
    out = mzero(8)
    for a_site, br, cr, bc, cc in product((0, 1), repeat=5):
        r = 4 * a_site + 2 * br + cr
        col = 4 * a_site + 2 * bc + cc
        out[r][col] = dict(a[2 * br + cr][2 * bc + cc])
    return out


def modular_series(hmat, order=7):
    e = gibbs_series(hmat, order)
    z = [mtrace(x) for x in e]
    inv_z = scalar_series_inverse(z, order)
    rho = matrix_scalar_series(e, inv_z, order)
    assert rho[0] == mscale(meye(8), F(1, 8))
    assert mtrace(rho[0]) == pconst(1)
    assert all(not mtrace(rho[k]) for k in range(1, order + 1))

    rho_ab = [partial_trace_three(x, (0, 1)) for x in rho]
    rho_bc = [partial_trace_three(x, (1, 2)) for x in rho]
    log_ab = [extend_ab(x) for x in series_log_normalized(rho_ab, order)]
    log_bc = [extend_bc(x) for x in series_log_normalized(rho_bc, order)]

    comm = []
    for k in range(order + 1):
        acc = mzero(8)
        for j in range(k + 1):
            acc = madd(acc, mmul(log_ab[j], log_bc[k - j]))
            acc = madd(acc, mscale(mmul(log_bc[j], log_ab[k - j]), F(-1)))
        comm.append(acc)

    result = []
    for k in range(order + 1):
        value = {}
        for j in range(k + 1):
            value = padd(value, mtrace(mmul(rho[j], comm[k - j])))
        result.append(pscale(value, (F(0), F(1))))
    return result


def extract_orientation(name, expected):
    target = (1 << 7) - 1
    series = modular_series(build_oriented_hamiltonian(name), 7)
    below = [series[k].get(target, CZ) for k in range(7)]
    found = series[7].get(target, CZ)
    assert below == [CZ] * 7, (name, below)
    assert found == expected, (name, found, expected)
    return found


def show(z):
    re, im = z
    if im == 0:
        return str(re)
    if re == 0:
        return f"{im} i"
    sign = "+" if im > 0 else "-"
    return f"{re} {sign} {abs(im)} i"


def main():
    started = monotonic_ns()
    forward = extract_orientation("forward", (F(0), F(-1, 46080)))
    print("[PASS] [J h tBC tbBC tAB gbA gB] m7 =", show(forward))
    conjugate = extract_orientation("conjugate", (F(0), F(1, 46080)))
    print("[PASS] [J h tBC tbBC tbAB gA gbB] m7 =", show(conjugate))
    assert conjugate == (forward[0], -forward[1])
    print("[PASS] physical specialization gives")
    print("       Pi_loc m7 = J h tBC tbBC"
          " (tAB gbA gB - tbAB gA gbB)/(46080 i)")
    elapsed_ms = (monotonic_ns() - started) // 1_000_000
    print("[PASS] exact symbolic extraction completed in", elapsed_ms, "ms")


if __name__ == "__main__":
    main()
