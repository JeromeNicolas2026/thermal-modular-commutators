#!/usr/bin/env python3
"""
verify_modular_flux_triangle.py
===============================

Independent verification of the analytic results for the signed modular
commutator

    M_rho(X,Y) = i Tr( rho [ log rho_X , log rho_Y ] ),      X = AB,  Y = BC,

on the axial XXZ-DM triangle

    H = sum_{<xy>} [ (1/2)( t_xy S^+_x S^-_y + conj(t_xy) S^-_x S^+_y )
                     + J^z_xy S^z_x S^z_y ]  -  sum_x h_x S^z_x ,

with  t_xy = J^perp_xy + i D^z_xy,  Phi = Im W = Im(t_AB t_BC t_CA).

PART 1 (exact, sympy).  The chain rho_beta -> partial traces -> matrix
logarithms -> trace is expanded as a truncated power series in beta in exact
Gaussian-rational arithmetic, with no floating point anywhere.  Certifies:

  THEOREM 1 (quartic)   m_4 = -(1/32) h_B Phi, for all couplings,
                        independent of h_A, h_C and of every J^z.

  THEOREM 2 (quintic)   m_5 = (Phi/384) *
                          [ ( 3 h_A -   h_B - 2 h_C ) J^z_AB
                          + (-2 h_A -   h_B + 3 h_C ) J^z_BC
                          + (         - 3 h_B        ) J^z_CA ],
                        for all couplings.  In particular m_5 vanishes
                        identically whenever all J^z = 0.

  THEOREM 3 (sextic, purely transverse)  for J^z = 0,
                        m_6 = (Phi/1536) *
                          [ 4 h_B (h_A^2 + h_A h_C + h_B^2 + h_C^2)
                          + h_A |t_AB|^2 + h_C |t_BC|^2
                          + 4 h_B (|t_AB|^2 + |t_BC|^2) + 6 h_B |t_CA|^2 ].
                        The endpoint fields h_A, h_C therefore reappear at
                        sixth order with no longitudinal exchange at all.

  Theorem 3 is identified from three calibration couplings whose squared
  amplitudes (|t_AB|^2, |t_BC|^2, |t_CA|^2) form an invertible 3x3 matrix
  (determinant -9, asserted below); the remaining couplings tested are
  independent out-of-sample checks, not part of the identification.

PART 2 (exact, symbolic real couplings).  Leading coefficients of the two
competing diagnostics:

  BOND COHERENCE        <S^+_x S^-_y>_beta = -(beta/8) conj(t_xy) + O(beta^2),
  WILSON LOOP           G_beta = Im( product of the three oriented bond
                        coherences ) = (beta^3/512) Phi + O(beta^4),
  SCALAR CHIRALITY      <chi>_beta = -(beta^2/32)
                        ( h_A D^z_BC + h_B D^z_CA + h_C D^z_AB ) + O(beta^3).

  G_beta is gauge invariant and needs no field; <chi>_beta is a sum of
  single-bond DM amplitudes, hence gauge dependent and nonzero on a tree.

PART 3 (non-perturbative, numpy/scipy).  Full matrix exponential and log.
Exact vanishing theorems at all temperatures, gauge behaviour, and the
flux-free tree comparison.

Every check is a hard assertion; the script exits nonzero on any failure.

Tested with Python 3.12, sympy 1.14.0, numpy 2.4.4, scipy 1.17.1.
Runtime a few minutes, dominated by the order-6 sympy expansions
(about 4 s per sextic coefficient on the reference machine).
"""

import sys
import numpy as np
import sympy as sp
from scipy.linalg import expm, logm
from sympy import Rational as R, I, eye, zeros, Matrix

EDGES = [(0, 1), (1, 2), (2, 0)]          # oriented A->B->C->A
EDGE_NAME = {(0, 1): "J^z_AB", (1, 2): "J^z_BC", (2, 0): "J^z_CA"}
hA, hB, hC = sp.symbols('h_A h_B h_C', real=True)


def _spin_ops(backend):
    exact = backend == "exact"
    if exact:
        sx = Matrix([[0, R(1, 2)], [R(1, 2), 0]])
        sy = Matrix([[0, -I*R(1, 2)], [I*R(1, 2), 0]])
        sz = Matrix([[R(1, 2), 0], [0, -R(1, 2)]])
        idm, unit = eye(2), I
        kr = lambda a, b: Matrix(sp.kronecker_product(a, b))
    else:
        sx = np.array([[0, 1], [1, 0]], dtype=complex)/2
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)/2
        sz = np.array([[1, 0], [0, -1]], dtype=complex)/2
        idm, unit = np.eye(2, dtype=complex), 1j
        kr = np.kron
    ops = {"id": idm}
    for name, m in (("x", sx), ("y", sy), ("z", sz),
                    ("+", sx + unit*sy), ("-", sx - unit*sy)):
        ops[name] = []
        for site in range(3):
            fac = [idm, idm, idm]
            fac[site] = m
            ops[name].append(kr(kr(fac[0], fac[1]), fac[2]))
    return ops


def hamiltonian(t, Jz, h, ops, backend):
    exact = backend == "exact"
    half = R(1, 2) if exact else 0.5
    conj = sp.conjugate if exact else np.conj
    mul = (lambda a, b: a*b) if exact else (lambda a, b: a @ b)
    H = zeros(8, 8) if exact else np.zeros((8, 8), dtype=complex)
    for e in EDGES:
        x, y = e
        H = H + half*(t[e]*mul(ops["+"][x], ops["-"][y])
                      + conj(t[e])*mul(ops["-"][x], ops["+"][y]))
        H = H + Jz[e]*mul(ops["z"][x], ops["z"][y])
    for k in range(3):
        H = H - h[k]*ops["z"][k]
    return sp.expand(H) if exact else H


def wilson(t):
    return sp.expand(t[(0, 1)]*t[(1, 2)]*t[(2, 0)])


def symbolic_hamiltonian():
    """Hamiltonian with fully symbolic real couplings J^perp, D^z, J^z, h.

    Returns (H, D^z dict, t dict)."""
    ops = _spin_ops("exact")
    Jp = {e: sp.Symbol(f'Jperp_{EDGE_NAME[e][4:]}', real=True) for e in EDGES}
    Dz = {e: sp.Symbol(f'D_{EDGE_NAME[e][4:]}', real=True) for e in EDGES}
    Jz = {e: sp.Symbol(f'Jz_{EDGE_NAME[e][4:]}', real=True) for e in EDGES}
    t = {e: Jp[e] + I*Dz[e] for e in EDGES}
    H = hamiltonian(t, Jz, [hA, hB, hC], ops, "exact")
    return H, Dz, t


class ExactSeries:
    """Truncated matrix power series in beta, exact rational arithmetic."""

    def __init__(self, order):
        self.order = order
        self.ops = _spin_ops("exact")

    def mul(self, a, b):
        out = [zeros(8, 8) for _ in range(self.order+1)]
        for i, Ai in enumerate(a):
            for j, Bj in enumerate(b):
                if i + j <= self.order:
                    out[i+j] += Ai*Bj
        return [sp.expand(m) for m in out]

    def scal_mul(self, c, a):
        out = [zeros(8, 8) for _ in range(self.order+1)]
        for i, ci in enumerate(c):
            for j, Aj in enumerate(a):
                if i + j <= self.order:
                    out[i+j] += ci*Aj
        return [sp.expand(m) for m in out]

    @staticmethod
    def tau(M):
        return sp.expand(sp.trace(M)/8)

    def scal_inv(self, c):
        inv = [sp.Integer(0)]*(self.order+1)
        inv[0] = sp.Integer(1)
        for n in range(1, self.order+1):
            inv[n] = sp.expand(-sum(c[k]*inv[n-k] for k in range(1, n+1)))
        return inv

    def log(self, a):
        """log of a matrix series with a[0] = I."""
        x = [m.copy() for m in a]
        x[0] = zeros(8, 8)
        out = [zeros(8, 8) for _ in range(self.order+1)]
        cur = [eye(8)] + [zeros(8, 8)]*self.order
        for n in range(1, self.order+1):
            cur = self.mul(cur, x)
            out = [o + R((-1)**(n+1), n)*c for o, c in zip(out, cur)]
        return [sp.expand(m) for m in out]

    @staticmethod
    def _E_AB(M):
        out = zeros(4, 4)
        for i in range(4):
            for j in range(4):
                out[i, j] = M[2*i, 2*j] + M[2*i+1, 2*j+1]
        return sp.expand(Matrix(sp.kronecker_product(out, eye(2)))/2)

    @staticmethod
    def _E_BC(M):
        out = zeros(4, 4)
        for i in range(4):
            for j in range(4):
                out[i, j] = M[i, j] + M[4+i, 4+j]
        return sp.expand(Matrix(sp.kronecker_product(eye(2), out))/2)

    def coefficients(self, t, Jz, h):
        """Returns [m_0, ..., m_order] of M_beta(AB, BC)."""
        H = hamiltonian(t, Jz, h, self.ops, "exact")
        e, P = [], eye(8)
        for k in range(self.order+1):
            e.append(sp.expand(R((-1)**k, sp.factorial(k))*P))
            P = sp.expand(P*H)
        r = self.scal_mul(self.scal_inv([self.tau(m) for m in e]), e)
        LAB = self.log([self._E_AB(m) for m in r])
        LBC = self.log([self._E_BC(m) for m in r])
        comm = [sp.expand(u - v) for u, v in
                zip(self.mul(LAB, LBC), self.mul(LBC, LAB))]
        return [sp.simplify(sp.expand(I*self.tau(m)))
                for m in self.mul(r, comm)]


def theorem_1(t):
    return sp.expand(-hB*sp.im(wilson(t))/32)


def theorem_2(t, Jz):
    br = ((3*hA - hB - 2*hC)*Jz[(0, 1)]
          + (-2*hA - hB + 3*hC)*Jz[(1, 2)]
          + (-3*hB)*Jz[(2, 0)])
    return sp.expand(sp.im(wilson(t))*br/384)


def theorem_3(t):
    """Sextic coefficient in the purely transverse model (all J^z = 0)."""
    u, v, w = (sp.Abs(t[e])**2 for e in EDGES)
    br = (4*hB*(hA**2 + hA*hC + hB**2 + hC**2)
          + hA*u + hC*v + 4*hB*(u + v) + 6*hB*w)
    return sp.expand(sp.im(wilson(t))*br/1536)


# --- calibration used to IDENTIFY Theorem 3 ---------------------------------
# The classification (gauge invariance + K-oddness + Theta-oddness) leaves
#     m_6 / Phi  =  sum_{|alpha|=3} a_alpha h^alpha  +  sum_{x,e} b_{x,e} h_x |t_e|^2 .
# The cubic part carries no |t|^2 and is read off directly; the nine b_{x,e}
# require three couplings whose squared amplitudes form an invertible matrix.
CALIBRATION = [
    {(0, 1): sp.Integer(1), (1, 2): sp.Integer(1), (2, 0): I},      # |t|^2 = (1,1,1)
    {(0, 1): sp.Integer(2), (1, 2): sp.Integer(1), (2, 0): I},      # |t|^2 = (4,1,1)
    {(0, 1): sp.Integer(1), (1, 2): sp.Integer(1), (2, 0): 2*I},    # |t|^2 = (1,1,4)
]


def calibration_matrix():
    return Matrix([[sp.Abs(t[e])**2 for e in EDGES] for t in CALIBRATION])


def identify_theorem_3(series6):
    """Solve for the cubic coefficients and the nine b_{x,e} from CALIBRATION.

    Returns (cubic_part, {(field, edge): b}, determinant)."""
    M = calibration_matrix()
    det = sp.simplify(M.det())
    reduced, cubics = [], []
    for t in CALIBRATION:
        m6 = series6.coefficients(t, JZ_ZERO, [hA, hB, hC])[6]
        red = sp.expand(m6/sp.im(wilson(t)))
        poly = sp.Poly(red, hA, hB, hC)
        cubics.append(sp.expand(sum(c*hA**m[0]*hB**m[1]*hC**m[2]
                                    for m, c in zip(poly.monoms(), poly.coeffs())
                                    if sum(m) == 3)))
        reduced.append(red)
    b = {}
    for j, hx in enumerate((hA, hB, hC)):
        rhs = Matrix([sp.expand(r).coeff(hx, 1).subs(
            {hA: 0, hB: 0, hC: 0}, simultaneous=True) for r in reduced])
        sol = M.solve(rhs)
        for k, e in enumerate(EDGES):
            b[(j, e)] = sp.nsimplify(sol[k])
    return cubics, b, det


_N = _spin_ops("numeric")


def gibbs(t, Jz, h, beta):
    r = expm(-beta*hamiltonian(t, Jz, h, _N, "numeric"))
    return r/np.trace(r)


def modular_commutator(t, Jz, h, beta):
    r = gibbs(t, Jz, h, beta)
    rAB = np.trace(r.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3), axis1=2, axis2=3)
    rBC = np.trace(r.reshape(2, 4, 2, 4).transpose(1, 3, 0, 2), axis1=2, axis2=3)
    LAB = np.kron(logm(rAB), np.eye(2))
    LBC = np.kron(np.eye(2), logm(rBC))
    return (1j*np.trace(r @ (LAB @ LBC - LBC @ LAB))).real


def scalar_chirality(t, Jz, h, beta):
    r = gibbs(t, Jz, h, beta)
    S = [_N["x"], _N["y"], _N["z"]]
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    C = sum(eps[a, b, c]*S[a][0] @ S[b][1] @ S[c][2]
            for a in range(3) for b in range(3) for c in range(3) if eps[a, b, c])
    return np.trace(r @ C).real


def coherence_loop(t, Jz, h, beta):
    r = gibbs(t, Jz, h, beta)
    g = [np.trace(r @ (_N["+"][x] @ _N["-"][y])) for (x, y) in EDGES]
    return (g[0]*g[1]*g[2]).imag


TRANSVERSE = [
    ("t=(2+3i, 1-i, 5+7i)", {(0, 1): 2+3*I, (1, 2): 1-I,           (2, 0): 5+7*I}),
    ("t=(1+i, 3, -2+5i)  ", {(0, 1): 1+I,   (1, 2): sp.Integer(3), (2, 0): -2+5*I}),
    ("t=(i, 2-7i, 4+i)   ", {(0, 1): I,     (1, 2): 2-7*I,         (2, 0): 4+I}),
    ("t=(1, 1, i)        ", {(0, 1): sp.Integer(1), (1, 2): sp.Integer(1), (2, 0): I}),
]
JZ_ZERO = {e: sp.Integer(0) for e in EDGES}
JZ_GEN = {(0, 1): R(3, 5), (1, 2): R(-7, 4), (2, 0): R(11, 9)}

failures = []


def check(label, condition):
    print(f"    [{'ok ' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


if __name__ == "__main__":
    S5 = ExactSeries(5)

    print("=" * 72)
    print("PART 1a -- Theorem 1 (quartic), exact")
    print("=" * 72)
    for lab, t in TRANSVERSE[:2]:
        for jlab, Jz in (("J^z = 0    ", JZ_ZERO), ("J^z generic", JZ_GEN)):
            m = S5.coefficients(t, Jz, [hA, hB, hC])
            print(f"  {lab}  {jlab}   Im W = {sp.im(wilson(t))}")
            print(f"      m_2 = {m[2]}    m_3 = {m[3]}    m_4 = {m[4]}")
            check("m_2 = m_3 = 0", m[2] == 0 and m[3] == 0)
            check("m_4 = -h_B Im(W)/32", sp.simplify(m[4] - theorem_1(t)) == 0)

    print()
    print("=" * 72)
    print("PART 1b -- Theorem 2 (quintic), exact")
    print("=" * 72)
    print("  (i) purely transverse model: m_5 must vanish for every t")
    for lab, t in TRANSVERSE:
        m5 = S5.coefficients(t, JZ_ZERO, [hA, hB, hC])[5]
        print(f"      {lab}  Im W = {str(sp.im(wilson(t))):>4}   m_5 = {m5}")
        check(f"m_5 = 0 at J^z = 0 for {lab.strip()}", sp.simplify(m5) == 0)

    print("  (ii) one J^z at a time, minimal gauge t = (1,1,i) so that Im W = 1")
    tmin = TRANSVERSE[3][1]
    for e in EDGES:
        Jz = dict(JZ_ZERO)
        Jz[e] = sp.Integer(1)
        m5 = S5.coefficients(tmin, Jz, [hA, hB, hC])[5]
        print(f"      {EDGE_NAME[e]} = 1 :  m_5 = {m5}")
        check(f"{EDGE_NAME[e]} row matches Theorem 2",
              sp.simplify(m5 - theorem_2(tmin, Jz)) == 0)

    print("  (iii) generic t and generic J^z simultaneously")
    for lab, t in TRANSVERSE[:3]:
        m5 = S5.coefficients(t, JZ_GEN, [hA, hB, hC])[5]
        print(f"      {lab}  m_5 = {m5}")
        check(f"Theorem 2 holds for {lab.strip()}",
              sp.simplify(m5 - theorem_2(t, JZ_GEN)) == 0)

    print("  (iv) reflection A <-> C together with J^z_AB <-> J^z_BC")
    swap = {hA: hC, hC: hA}
    Jsw = {(0, 1): JZ_GEN[(1, 2)], (1, 2): JZ_GEN[(0, 1)], (2, 0): JZ_GEN[(2, 0)]}
    lhs = theorem_2(tmin, JZ_GEN).subs(swap, simultaneous=True)
    check("reflection symmetry of the quintic coefficients",
          sp.simplify(lhs - theorem_2(tmin, Jsw)) == 0)

    print("  (v) uniform field: only the edge opposite the overlap survives")
    hs = sp.Symbol('h')
    unif = {hA: hs, hB: hs, hC: hs}
    got = sp.simplify(theorem_2(tmin, JZ_GEN).subs(unif, simultaneous=True))
    want = sp.simplify(-hs*JZ_GEN[(2, 0)]*sp.im(wilson(tmin))/128)
    print(f"      m_5(uniform h) = {got}   expected -h J^z_CA Im(W)/128 = {want}")
    check("uniform-field reduction", sp.simplify(got - want) == 0)

    print()
    print("=" * 72)
    print("PART 1c -- Theorem 3 (sextic, purely transverse), exact")
    print("=" * 72)
    S6 = ExactSeries(6)

    print("  (i) identification: calibration matrix of squared amplitudes")
    M = calibration_matrix()
    cubics, bcoef, det = identify_theorem_3(S6)
    print(f"      rows (|t_AB|^2, |t_BC|^2, |t_CA|^2) = {M.tolist()}")
    print(f"      determinant = {det}")
    check("calibration matrix is invertible", det != 0)
    print(f"      cubic part, read off from each calibration coupling:")
    for cub in cubics:
        print(f"        {cub}")
    check("cubic part agrees across all calibration couplings",
          all(sp.simplify(c - cubics[0]) == 0 for c in cubics))
    check("cubic part equals h_B(h_A^2 + h_A h_C + h_B^2 + h_C^2)/384",
          sp.simplify(cubics[0] - hB*(hA**2 + hA*hC + hB**2 + hC**2)/384) == 0)
    print("      solved coefficients b_{x,e} of h_x |t_e|^2 :")
    want = {(0, (0, 1)): R(1, 1536), (0, (1, 2)): 0, (0, (2, 0)): 0,
            (1, (0, 1)): R(1, 384), (1, (1, 2)): R(1, 384), (1, (2, 0)): R(1, 256),
            (2, (0, 1)): 0, (2, (1, 2)): R(1, 1536), (2, (2, 0)): 0}
    for j, hx in enumerate(("h_A", "h_B", "h_C")):
        row = "  ".join(f"{EDGE_NAME[e][4:]}: {bcoef[(j, e)]}" for e in EDGES)
        print(f"        {hx} :  {row}")
    check("solved b_{x,e} match the closed form",
          all(sp.simplify(bcoef[k] - want[k]) == 0 for k in want))
    check("reflection A<->C, AB<->BC on the solved coefficients",
          sp.simplify(bcoef[(0, (0, 1))] - bcoef[(2, (1, 2))]) == 0
          and sp.simplify(bcoef[(0, (1, 2))] - bcoef[(2, (0, 1))]) == 0
          and sp.simplify(bcoef[(0, (2, 0))] - bcoef[(2, (2, 0))]) == 0
          and sp.simplify(bcoef[(1, (0, 1))] - bcoef[(1, (1, 2))]) == 0)

    print("  (ii) out-of-sample checks (not used in the identification)")
    extra = [("t=(3, 2i, 1+i)     ",
              {(0, 1): sp.Integer(3), (1, 2): 2*I, (2, 0): 1+I})]
    for lab, t in TRANSVERSE[:3] + extra:
        m6 = S6.coefficients(t, JZ_ZERO, [hA, hB, hC])[6]
        print(f"      {lab}  m_6 = {sp.expand(m6)}")
        check(f"Theorem 3 holds for {lab.strip()}",
              sp.simplify(m6 - theorem_3(t)) == 0)
    print("      endpoint fields reappear at order six with no longitudinal exchange:")
    tref = TRANSVERSE[0][1]
    cA = sp.expand(theorem_3(tref)).coeff(hA, 1).coeff(hB, 0).coeff(hC, 0)
    cC = sp.expand(theorem_3(tref)).coeff(hC, 1).coeff(hA, 0).coeff(hB, 0)
    print(f"        coefficient of h_A = {cA} = Phi |t_AB|^2 / 1536")
    print(f"        coefficient of h_C = {cC} = Phi |t_BC|^2 / 1536")
    check("h_A enters at order six at J^z = 0", cA != 0)
    check("h_C enters at order six at J^z = 0", cC != 0)

    print()
    print("=" * 72)
    print("PART 1d -- competing diagnostics, exact, symbolic real couplings")
    print("=" * 72)
    Hs, Dz, Ts = symbolic_hamiltonian()
    Hs2 = sp.expand(Hs*Hs)
    ops = _spin_ops("exact")
    tau = lambda X: sp.expand(sp.trace(X)/8)

    print("  (i) bond coherences to first order")
    lead = {}
    for e in EDGES:
        x, y = e
        O = sp.expand(ops["+"][x]*ops["-"][y])
        c1 = sp.expand(-tau(Hs*O))
        lead[e] = c1
        print(f"      <S^+_{'ABC'[x]} S^-_{'ABC'[y]}> = {c1} * beta + O(beta^2)")
        check(f"coherence on {EDGE_NAME[e][4:]} equals -conj(t)/8",
              tau(O) == 0 and sp.simplify(c1 + sp.conjugate(Ts[e])/8) == 0)

    print("  (ii) coherence Wilson loop")
    Gc = sp.simplify(sp.im(sp.expand(lead[EDGES[0]]*lead[EDGES[1]]*lead[EDGES[2]])))
    Phi_sym = sp.simplify(sp.im(sp.expand(Ts[(0, 1)]*Ts[(1, 2)]*Ts[(2, 0)])))
    print(f"      [beta^3] G = {sp.factor(Gc)}      Phi = {Phi_sym}")
    check("G = (beta^3/512) Phi + O(beta^4)",
          sp.simplify(sp.expand(Gc - Phi_sym/512)) == 0)

    print("  (iii) thermal scalar chirality")
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
           (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    Sx = [ops["x"], ops["y"], ops["z"]]
    CHI = zeros(8, 8)
    for (a, b_, c), sgn in eps.items():
        CHI += sgn*Sx[a][0]*Sx[b_][1]*Sx[c][2]
    CHI = sp.expand(CHI)
    k0, k1 = tau(CHI), sp.expand(-tau(Hs*CHI))
    k2 = sp.expand(R(1, 2)*(tau(Hs2*CHI) - tau(Hs2)*tau(CHI)))
    claim = sp.expand(-(hA*Dz[(1, 2)] + hB*Dz[(2, 0)] + hC*Dz[(0, 1)])/32)
    print(f"      [beta^0] = {k0}   [beta^1] = {sp.simplify(k1)}")
    print(f"      [beta^2] = {k2}")
    check("<chi> starts at order beta^2", k0 == 0 and sp.simplify(k1) == 0)
    check("<chi> = -(beta^2/32)(h_A D_BC + h_B D_CA + h_C D_AB) + O(beta^3)",
          sp.simplify(sp.expand(k2 - claim)) == 0)
    check("<chi> leading term is a sum of single-bond DM amplitudes, "
          "not a loop product", sp.Poly(k2, *Dz.values()).total_degree() == 1)

    print()
    print("=" * 72)
    print("PART 3 -- non-perturbative checks (all temperatures)")
    print("=" * 72)
    tn = {e: complex(sp.N(v)) for e, v in TRANSVERSE[0][1].items()}
    Jn = {e: float(v) for e, v in JZ_GEN.items()}
    h = [0.3, 0.7, -0.4]
    TOL = 1e-12

    cases = [
        ("zero field, flux present  ", tn, Jn, [0.0, 0.0, 0.0]),
        ("real Wilson product       ", {(0, 1): 1+1j, (1, 2): 1+1j, (2, 0): 1j}, Jn, h),
        ("tree A-B-C, complex bonds ", {(0, 1): 1+1.3j, (1, 2): 0.8-0.5j, (2, 0): 0.0},
         {(0, 1): 0.6, (1, 2): -1.75, (2, 0): 0.0}, h),
    ]
    for lab, tt, JJ, hh in cases:
        vals = [modular_commutator(tt, JJ, hh, b) for b in (0.5, 1.0, 3.0)]
        print(f"  {lab}  M = " + "  ".join(f"{v: .2e}" for v in vals))
        check(f"M vanishes: {lab.strip()}", max(abs(v) for v in vals) < TOL)

    print("  beta^4 asymptotics of Theorem 1")
    pred = -h[1]*40/32
    for b in (0.05, 0.025, 0.0125):
        ratio = modular_commutator(tn, Jn, h, b)/b**4
        print(f"      beta={b:7.4f}   M/beta^4 = {ratio: .5f}   (theorem: {pred: .5f})")
    check("M/beta^4 converges to the theorem value",
          abs(modular_commutator(tn, Jn, h, 0.0125)/0.0125**4 - pred) < 0.01)

    print("  competing diagnostics on the flux-free tree")
    tt, JJ = cases[2][1], cases[2][2]
    for b in (0.5, 1.0, 2.0):
        print(f"      beta={b:4.1f}   M={modular_commutator(tt,JJ,h,b): .2e}"
              f"   <chi>={scalar_chirality(tt,JJ,h,b): .4e}"
              f"   G={coherence_loop(tt,JJ,h,b): .2e}")
    check("scalar chirality is nonzero on a flux-free tree",
          abs(scalar_chirality(tt, JJ, h, 1.0)) > 1e-3)
    check("coherence loop vanishes on a flux-free tree",
          abs(coherence_loop(tt, JJ, h, 1.0)) < TOL)

    print("  axial gauge transformation t_xy -> t_xy exp(i(phi_x - phi_y))")
    phi = [0.0, 0.9, -0.35]
    ph = {(0, 1): np.exp(1j*(phi[0]-phi[1])), (1, 2): np.exp(1j*(phi[1]-phi[2])),
          (2, 0): np.exp(1j*(phi[2]-phi[0]))}
    tg = {e: tn[e]*ph[e] for e in EDGES}
    for name, f in (("M    ", modular_commutator), ("<chi>", scalar_chirality),
                    ("G    ", coherence_loop)):
        a, bb = f(tn, Jn, h, 0.4), f(tg, Jn, h, 0.4)
        print(f"      {name} before {a: .8e}   after {bb: .8e}")
    check("M is gauge invariant",
          abs(modular_commutator(tn, Jn, h, 0.4) -
              modular_commutator(tg, Jn, h, 0.4)) < 1e-10)
    check("G is gauge invariant",
          abs(coherence_loop(tn, Jn, h, 0.4) - coherence_loop(tg, Jn, h, 0.4)) < 1e-10)
    check("scalar chirality is NOT gauge invariant",
          abs(scalar_chirality(tn, Jn, h, 0.4) -
              scalar_chirality(tg, Jn, h, 0.4)) > 1e-4)

    print()
    print("=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All checks passed.")
