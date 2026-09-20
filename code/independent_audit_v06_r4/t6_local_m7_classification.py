#!/usr/bin/env python3
"""Exact classification of m_7 in a stated eight-parameter local submodel.

The graph is the open qubit path A-B-C.  Its only nonzero Hamiltonian
parameters are

    t_AB, t_BC, g_A, g_B, g_C, J_AB^z, J_BC^z, h_B^z.

To make this a polynomial identity rather than a collection of evaluations,
the ten oriented complex letters for ``t``, ``conj(t)``, ``g`` and
``conj(g)``, together with ``J_AB``, ``J_BC`` and ``h_B``, are thirteen
independent indeterminates.  All coefficients lie in Q(i).  No numerical
library and no project engine is imported.

Within precisely this submodel, the output certifies both that m_0,...,m_6
vanish and that m_7 has exactly ten complex monomials, i.e. the five real
sectors in ``closed_form`` below. It makes no completeness claim after any
additional on-site field or coupling is enabled.
"""

from fractions import Fraction as F


# The first ten variables are formally independent.  Hermiticity is imposed
# only when the ten monomials are paired into imaginary parts at the end.
NAMES = ("tAB", "tbAB", "tBC", "tbBC", "gA", "gbA", "gB", "gbB",
         "gC", "gbC", "JAB", "JBC", "hB")
NV = len(NAMES)
RADIX = 8                         # every exponent is at most the beta order 7
WEIGHT = [RADIX ** k for k in range(NV)]
ZERO = (F(0), F(0))


# ------------------------------------------------------- Gaussian rationals
def cnum(re, im=0):
    return (F(re), F(im))


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1],
            a[0] * b[1] + a[1] * b[0])


def cscale(a, q):
    return (a[0] * q, a[1] * q)


# ------------------------------------------ Q(i)[the thirteen parameters]
# A monomial x_0^e_0 ... x_12^e_12 is encoded by sum e_k 8^k.  Because all
# total degrees are <=7, integer addition is exactly monomial multiplication.
def pconst(z):
    return {} if z == ZERO else {0: z}


def pvar(k, z=cnum(1)):
    return {} if z == ZERO else {WEIGHT[k]: z}


def padd(a, b):
    out = dict(a)
    for key, value in b.items():
        value = cadd(out.get(key, ZERO), value)
        if value == ZERO:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def pscale(a, q):
    if q == 0:
        return {}
    return {key: cscale(value, q) for key, value in a.items()}


def pmul(a, b):
    if not a or not b:
        return {}
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            key = ka + kb
            value = cadd(out.get(key, ZERO), cmul(va, vb))
            if value == ZERO:
                out.pop(key, None)
            else:
                out[key] = value
    return out


def monomial(**powers):
    return sum(powers.get(name, 0) * WEIGHT[k]
               for k, name in enumerate(NAMES))


def decode(key):
    exponents = []
    for _ in range(NV):
        exponents.append(key % RADIX)
        key //= RADIX
    assert key == 0
    return tuple(exponents)


# ----------------------------------------------------------- matrices/series
def zeros(d):
    return [[{} for _ in range(d)] for _ in range(d)]


def eye(d):
    out = zeros(d)
    for k in range(d):
        out[k][k] = pconst(cnum(1))
    return out


def madd(A, B):
    return [[padd(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mscale(A, q):
    return [[pscale(a, q) for a in row] for row in A]


def mmul(A, B):
    d = len(A)
    out = zeros(d)
    for i in range(d):
        for k in range(d):
            if not A[i][k]:
                continue
            for j in range(d):
                if B[k][j]:
                    out[i][j] = padd(out[i][j], pmul(A[i][k], B[k][j]))
    return out


def mtrace(A):
    out = {}
    for k in range(len(A)):
        out = padd(out, A[k][k])
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


def factorial(n):
    value = 1
    for k in range(2, n + 1):
        value *= k
    return value


# ---------------------------------------------------- the formal Hamiltonian
def index(a, b, c):
    return 4 * a + 2 * b + c


def add_entry(H, row, col, variable, coefficient):
    H[row][col] = padd(H[row][col], pvar(variable, cnum(coefficient)))


def formal_H():
    """Qubit Hamiltonian in the conventions of indep_engine.py.

    Basis bits 0,1 mean m=+1/2,-1/2.  The off-diagonal entries implement
    (t S_x^+ S_y^- + tb S_x^- S_y^+)/2 and
    -(g S_x^+ + gb S_x^-)/2 directly.
    """
    H = zeros(8)
    tAB, tbAB, tBC, tbBC = 0, 1, 2, 3
    gA, gbA, gB, gbB, gC, gbC = 4, 5, 6, 7, 8, 9
    JAB, JBC, hB = 10, 11, 12

    for c in range(2):
        add_entry(H, index(0, 1, c), index(1, 0, c), tAB, F(1, 2))
        add_entry(H, index(1, 0, c), index(0, 1, c), tbAB, F(1, 2))
    for a in range(2):
        add_entry(H, index(a, 0, 1), index(a, 1, 0), tBC, F(1, 2))
        add_entry(H, index(a, 1, 0), index(a, 0, 1), tbBC, F(1, 2))

    for a, b, c in ((a, b, c) for a in range(2)
                    for b in range(2) for c in range(2)):
        bits = [a, b, c]
        for site, gv, gbv in ((0, gA, gbA), (1, gB, gbB), (2, gC, gbC)):
            target = list(bits)
            if bits[site] == 1:                  # S^+ |down> = |up>
                target[site] = 0
                add_entry(H, index(*target), index(*bits), gv, F(-1, 2))
            else:                                # S^- |up> = |down>
                target[site] = 1
                add_entry(H, index(*target), index(*bits), gbv, F(-1, 2))

        mA, mB, mC = (F(1, 2) if bit == 0 else F(-1, 2)
                       for bit in bits)
        row = index(*bits)
        add_entry(H, row, row, JAB, mA * mB)
        add_entry(H, row, row, JBC, mB * mC)
        add_entry(H, row, row, hB, -mB)
    return H


def ptrace(M, keep):
    """Partial trace of an 8 by 8 matrix over the omitted qubit(s)."""
    keep = tuple(keep)
    rest = tuple(x for x in range(3) if x not in keep)
    out = zeros(2 ** len(keep))
    bits = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]

    def subindex(bit):
        value = 0
        for site in keep:
            value = 2 * value + bit[site]
        return value

    for br in bits:
        for bc in bits:
            if all(br[site] == bc[site] for site in rest):
                r, c = subindex(br), subindex(bc)
                out[r][c] = padd(out[r][c], M[index(*br)][index(*bc)])
    return out


def extend(op, sites):
    """Embed a two-qubit operator on AB or BC into ABC."""
    sites = tuple(sites)
    omitted = tuple(x for x in range(3) if x not in sites)
    out = zeros(8)
    bits = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]

    def subindex(bit):
        value = 0
        for site in sites:
            value = 2 * value + bit[site]
        return value

    for br in bits:
        for bc in bits:
            if all(br[site] == bc[site] for site in omitted):
                out[index(*br)][index(*bc)] = dict(op[subindex(br)][subindex(bc)])
    return out


def series_log(Q, K):
    """log Q for Q=I+O(beta), as a noncommutative formal series."""
    assert Q[0] == eye(len(Q[0]))
    d = len(Q[0])
    Y = [zeros(d)] + Q[1:]
    out = [zeros(d) for _ in range(K + 1)]
    power = [eye(d)] + [zeros(d) for _ in range(K)]
    for q in range(1, K + 1):
        power = series_matmul(power, Y, K)
        for k in range(K + 1):
            out[k] = madd(out[k], mscale(power[k], F((-1) ** (q + 1), q)))
    return out


def scalar_inverse(Z, K):
    """Inverse of a scalar polynomial-valued beta series; Z_0=8."""
    out = [pconst(cnum(F(1, 8)))]
    for k in range(1, K + 1):
        acc = {}
        for j in range(1, k + 1):
            acc = padd(acc, pmul(Z[j], out[k - j]))
        out.append(pscale(acc, F(-1, 8)))
    return out


def modular_polynomial(K=7):
    """Return the exact coefficients m_0,...,m_K as formal polynomials."""
    H = formal_H()
    E = [eye(8)]
    Hpower = eye(8)
    for k in range(1, K + 1):
        Hpower = mmul(H, Hpower)
        E.append(mscale(Hpower, F((-1) ** k, factorial(k))))

    # Scalar factors d_spectator and Z commute with everything.  Hence
    # log rho_AB and log rho_BC may be replaced inside the commutator by the
    # logs of (Tr_C exp(-beta H))/2 and (Tr_A exp(-beta H))/2.
    QAB = [mscale(ptrace(M, (0, 1)), F(1, 2)) for M in E]
    QBC = [mscale(ptrace(M, (1, 2)), F(1, 2)) for M in E]
    LAB = [extend(M, (0, 1)) for M in series_log(QAB, K)]
    LBC = [extend(M, (1, 2)) for M in series_log(QBC, K)]

    commutator = []
    for k in range(K + 1):
        acc = zeros(8)
        for j in range(k + 1):
            acc = madd(acc, mmul(LAB[j], LBC[k - j]))
            acc = madd(acc, mscale(mmul(LBC[j], LAB[k - j]), F(-1)))
        commutator.append(acc)

    numerator = []
    for k in range(K + 1):
        acc = {}
        for j in range(k + 1):
            acc = padd(acc, mtrace(mmul(E[j], commutator[k - j])))
        numerator.append(acc)

    invZ = scalar_inverse([mtrace(M) for M in E], K)
    result = []
    for k in range(K + 1):
        value = {}
        for j in range(k + 1):
            value = padd(value, pmul(numerator[j], invZ[k - j]))
        # M_beta = i Tr rho [log rho_AB,log rho_BC].
        result.append({key: (-z[1], z[0]) for key, z in value.items()})
    return result


# ---------------------------------------------------------- expected answer
def add_imaginary_part(out, z, zbar, coefficient):
    """Add coefficient*Im(z), with zbar the formal conjugate monomial."""
    half = coefficient / 2
    out[z] = cadd(out.get(z, ZERO), (F(0), -half))
    out[zbar] = cadd(out.get(zbar, ZERO), (F(0), half))


def closed_form():
    """Formal polynomial for

      hB/23040 * [
        JAB (tBC*tbBC-JBC^2) Im(tAB*gbA*gB)
       +JBC (tAB*tbAB-JAB^2) Im(tBC*gbB*gC)
       +(JAB*JBC/2) Im(tAB*tBC*gbA*gC) ].
    """
    out = {}
    q = F(1, 23040)

    left = monomial(tAB=1, gbA=1, gB=1)
    leftbar = monomial(tbAB=1, gA=1, gbB=1)
    right = monomial(tBC=1, gbB=1, gC=1)
    rightbar = monomial(tbBC=1, gB=1, gbC=1)
    across = monomial(tAB=1, tBC=1, gbA=1, gC=1)
    acrossbar = monomial(tbAB=1, tbBC=1, gA=1, gbC=1)

    def times(key, **powers):
        return key + monomial(**powers)

    add_imaginary_part(out,
                       times(left, tBC=1, tbBC=1, JAB=1, hB=1),
                       times(leftbar, tBC=1, tbBC=1, JAB=1, hB=1), q)
    add_imaginary_part(out,
                       times(left, JAB=1, JBC=2, hB=1),
                       times(leftbar, JAB=1, JBC=2, hB=1), -q)
    add_imaginary_part(out,
                       times(right, tAB=1, tbAB=1, JBC=1, hB=1),
                       times(rightbar, tAB=1, tbAB=1, JBC=1, hB=1), q)
    add_imaginary_part(out,
                       times(right, JAB=2, JBC=1, hB=1),
                       times(rightbar, JAB=2, JBC=1, hB=1), -q)
    add_imaginary_part(out,
                       times(across, JAB=1, JBC=1, hB=1),
                       times(acrossbar, JAB=1, JBC=1, hB=1), q / 2)
    return {key: value for key, value in out.items() if value != ZERO}


def main():
    series = modular_polynomial(7)
    assert all(value == {} for value in series[:7])
    expected = closed_form()
    assert len(expected) == 10
    assert series[7] == expected

    # Structural assertions: degree seven, five conjugate pairs, and no hidden
    # sector beyond the displayed formula.
    assert all(sum(decode(key)) == 7 for key in series[7])
    assert len(series[7]) == 10
    print("[PASS] P1  m_0,...,m_6 vanish as formal polynomial identities")
    print("[PASS] P2  m_7 contains exactly 10 monomials = 5 real sectors")
    print("[PASS] P3  complete closed form in the stated eight-parameter submodel")
    print("\nm_7 = h_B/23040 [")
    print("  J_AB (|t_BC|^2-J_BC^2) Im(t_AB conj(g_A) g_B)")
    print(" +J_BC (|t_AB|^2-J_AB^2) Im(t_BC conj(g_B) g_C)")
    print(" +(J_AB J_BC/2) Im(t_AB t_BC conj(g_A) g_C) ].")


if __name__ == "__main__":
    main()
