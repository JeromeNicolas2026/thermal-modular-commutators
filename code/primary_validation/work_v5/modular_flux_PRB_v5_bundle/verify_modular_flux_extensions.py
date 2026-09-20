#!/usr/bin/env python3
"""
verify_modular_flux_extensions.py
=================================

EXACT (SymPy, Gaussian-rational arithmetic, no floating point in PARTS E1-E3)
certificates for the higher-order results of the modular-flux paper, at the
same reproducibility standard as verify_modular_flux_triangle.py:

  E1  Ring n=4, transverse, regions AB/BC, symbolic fields:
        m1..m4 = 0,  m5 = h_B/128  for t=(1,1,1,i)      (Im W_4 = 1)
        m5 = h_B/64  for t=(2,1,1,i)                     (Im W_4 = 2)
        m5 unchanged with generic rational J^z.
      Together with the classification (degree n+1 = oriented loop + one field;
      no slot for longitudinal exchange), this certifies
        M_beta(AB,BC) = (+1/128) beta^5 h_B Im W_4 + O(beta^6)   (n=4).

  E2  Transverse septic on the triangle (J^z=0), symbolic fields:
        m7 = -Im(W^2) (2 h_A + 41 h_B + 2 h_C) / 92160
      certified exactly at two generic couplings:
        t=(2+3i, 1-i, 5+7i)  (W=18+40i,   Im W^2=1440  -> m7 = -(2h_A+41h_B+2h_C)/64)
        t=(1+i, 3, -2+5i)    (W=-21+9i,   Im W^2=-378  -> m7 = 21(2h_A+41h_B+2h_C)/5120)

  E3  Complete sextic with arbitrary J^z (triangle), symbolic fields.
      Calibration in exact arithmetic at t=(1,1,i) from six J^z configs,
      then closed form
        m6 = Phi/1536 * [ 4h_B(h_A^2+h_Ah_C+h_B^2+h_C^2)
                          + h_A|t_AB|^2 + h_C|t_BC|^2
                          + 4h_B(|t_AB|^2+|t_BC|^2) + 6h_B|t_CA|^2
                          + h_A(J_AB^2 + 3 J_AB J_CA - 4 J_BC J_CA)
                          + h_B(J_AB^2 + J_BC^2 + 2 J_CA^2
                                + J_AB J_BC - J_BC J_CA - J_AB J_CA)
                          + h_C(J_BC^2 + 3 J_BC J_CA - 4 J_AB J_CA) ]
      is asserted (i) to reproduce the six calibration runs and (ii) at two
      OUT-OF-SAMPLE generic (t, J^z) points, all exact.

  PART N (numerical evidence only, clearly labeled as such):
        n=5 ring: m6 = -1/512 (mpmath series, 60 digits);
        n=6 ring: m7 = +1/2048 (Richardson extrapolation, 2.4e-3 relative);
        4-ring low-T plateau ~ 0.314621219... at beta=200 (scipy).

Every check is a hard assertion; the script exits nonzero on any failure.
Runtime: ~5 minutes (dominated by the order-7 and 16x16 exact series).
"""

import sys
import time
import sympy as sp
from sympy import Rational as R, I, eye, zeros, Matrix

failures = []

def check(label, condition):
    print(f"    [{'ok ' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)

# ---------------------------------------------------------- exact engine ---
def spin_ops_N(N):
    sx = Matrix([[0, R(1, 2)], [R(1, 2), 0]])
    sy = Matrix([[0, -I*R(1, 2)], [I*R(1, 2), 0]])
    sz = Matrix([[R(1, 2), 0], [0, -R(1, 2)]])
    def kronN(factors):
        out = Matrix([[1]])
        for f in factors:
            out = Matrix(sp.kronecker_product(out, f))
        return out
    ops = {}
    for nm, m in (('x', sx), ('y', sy), ('z', sz), ('+', sx+I*sy), ('-', sx-I*sy)):
        ops[nm] = [kronN([m if s == site else eye(2) for s in range(N)])
                   for site in range(N)]
    return ops

def H_ring(N, t, Jz, h, ops):
    d = 2**N
    H = zeros(d)
    for i in range(N):
        j = (i+1) % N
        H = H + R(1, 2)*(t[i]*(ops['+'][i]*ops['-'][j])
                         + sp.conjugate(t[i])*(ops['-'][i]*ops['+'][j]))
        if Jz is not None:
            H = H + Jz[i]*(ops['z'][i]*ops['z'][j])
    for i in range(N):
        H = H - h[i]*ops['z'][i]
    return sp.expand(H)

def E_region(O, N, keep):
    """(1/d_rest) Tr_rest(O) tensor I_rest ; site s <-> bit (N-1-s)."""
    d = 2**N
    keep = sorted(keep)
    traced = [s for s in range(N) if s not in keep]
    dt = 2**len(traced)
    bitpos = {s: N-1-s for s in range(N)}
    dk = 2**len(keep)
    Rm = zeros(dk)
    for a in range(dk):
        for b in range(dk):
            acc = sp.Integer(0)
            for k in range(dt):
                ii = jj = 0
                for s in range(N):
                    if s in keep:
                        q = keep.index(s)
                        bi = (a >> (len(keep)-1-q)) & 1
                        bj = (b >> (len(keep)-1-q)) & 1
                    else:
                        q = traced.index(s)
                        bi = (k >> (len(traced)-1-q)) & 1
                        bj = bi
                    ii = (ii << 1) | bi
                    jj = (jj << 1) | bj
                acc += O[ii, jj]
            Rm[a, b] = sp.expand(acc/dt)
    E = zeros(d)
    for i in range(d):
        for j in range(d):
            ok = True
            for s in traced:
                p = bitpos[s]
                if ((i >> p) & 1) != ((j >> p) & 1):
                    ok = False
                    break
            if not ok:
                continue
            a = b = 0
            for s in keep:
                p = bitpos[s]
                a = (a << 1) | ((i >> p) & 1)
                b = (b << 1) | ((j >> p) & 1)
            E[i, j] = Rm[a, b]
    return E

class Series:
    def __init__(s, order, d):
        s.N = order
        s.d = d
    def mul(s, a, b):
        out = [zeros(s.d) for _ in range(s.N+1)]
        for i in range(len(a)):
            for j in range(len(b)):
                if i+j <= s.N:
                    out[i+j] += a[i]*b[j]
        return [sp.expand(m) for m in out]
    def smul(s, c, a):
        out = [zeros(s.d) for _ in range(s.N+1)]
        for i, ci in enumerate(c):
            for j in range(len(a)):
                if i+j <= s.N:
                    out[i+j] += ci*a[j]
        return [sp.expand(m) for m in out]
    def sinv(s, c):
        inv = [sp.Integer(0)]*(s.N+1)
        inv[0] = sp.Integer(1)
        for n in range(1, s.N+1):
            inv[n] = sp.expand(-sum(c[k]*inv[n-k] for k in range(1, n+1)))
        return inv
    def log(s, a):
        N, d = s.N, s.d
        x = [m.copy() for m in a]
        x[0] = zeros(d)
        out = [zeros(d) for _ in range(N+1)]
        cur = [eye(d)] + [zeros(d) for _ in range(N)]
        for n in range(1, N+1):
            cur = s.mul(cur, x)
            coef = R((-1)**(n+1), n)
            out = [sp.expand(o+coef*c) for o, c in zip(out, cur)]
        return out

def M_series(N, t, Jz, h, order, keepA=(0, 1), keepB=(1, 2), ops=None):
    if ops is None:
        ops = spin_ops_N(N)
    H = H_ring(N, t, Jz, h, ops)
    d = 2**N
    e = [None]*(order+1)
    P = eye(d)
    for k in range(order+1):
        e[k] = sp.expand(R((-1)**k, sp.factorial(k))*P)
        P = sp.expand(P*H)
    tau = [sp.expand(sp.trace(e[k])/d) for k in range(order+1)]
    S = Series(order, d)
    r = S.smul(S.sinv(tau), e)
    EA = [E_region(r[k], N, keepA) for k in range(order+1)]
    EB = [E_region(r[k], N, keepB) for k in range(order+1)]
    LA = S.log(EA)
    LB = S.log(EB)
    comm = [sp.expand(u-v) for u, v in zip(S.mul(LA, LB), S.mul(LB, LA))]
    Mser = S.mul(r, comm)
    return [sp.simplify(sp.expand(I*sp.trace(m)/d)) for m in Mser]

hA, hB, hC, hD = sp.symbols('h_A h_B h_C h_D', real=True)
TMIN3 = [sp.Integer(1), sp.Integer(1), I]

def m6_full(t, Jz, h):
    hA_, hB_, hC_ = h
    W = t[0]*t[1]*t[2]
    Phi = sp.im(W)
    u = [sp.expand(sp.conjugate(t[i])*t[i]) for i in range(3)]
    J0, J1, J2 = Jz
    br = (4*hB_*(hA_**2+hA_*hC_+hB_**2+hC_**2)
          + hA_*u[0] + hC_*u[1] + 4*hB_*(u[0]+u[1]) + 6*hB_*u[2]
          + hA_*(J0**2 + 3*J0*J2 - 4*J1*J2)
          + hB_*(J0**2 + J1**2 + 2*J2**2 + J0*J1 - J1*J2 - J0*J2)
          + hC_*(J1**2 + 3*J1*J2 - 4*J0*J2))
    return sp.expand(Phi*br/1536)

# =================================================================== E1 ====
def part_E1():
    print("=" * 72)
    print("E1 -- anneau n=4, certificat exact du coefficient dominant")
    print("=" * 72)
    t0 = time.time()
    t4 = [sp.Integer(1), sp.Integer(1), sp.Integer(1), I]
    mc = M_series(4, t4, [0, 0, 0, 0], [hA, hB, hC, hD], 5)
    check("m1..m4 = 0 (exact)",
          all(sp.simplify(mc[k]) == 0 for k in range(1, 5)))
    check("m5 = h_B/128 (exact, champs symboliques)",
          sp.simplify(mc[5] - hB/128) == 0)
    print(f"    [{round(time.time()-t0, 1)}s] m5 = {sp.simplify(mc[5])}")

    t0 = time.time()
    t4b = [sp.Integer(2), sp.Integer(1), sp.Integer(1), I]
    mc = M_series(4, t4b, [0, 0, 0, 0], [hA, hB, hC, hD], 5)
    check("t=(2,1,1,i) : m5 = h_B/64 = h_B Im W/128 (exact)",
          sp.simplify(mc[5] - hB/64) == 0)
    print(f"    [{round(time.time()-t0, 1)}s]")

    t0 = time.time()
    Jg = [R(3, 5), R(-7, 4), R(11, 9), R(-2, 7)]
    mc = M_series(4, t4, Jg, [hA, hB, hC, hD], 5)
    check("J^z generiques : m5 inchange = h_B/128 (exact)",
          sp.simplify(mc[5] - hB/128) == 0)
    print(f"    [{round(time.time()-t0, 1)}s]")

# =================================================================== E2 ====
def part_E2():
    print("=" * 72)
    print("E2 -- m7 transverse : -Im(W^2)(2h_A+41h_B+2h_C)/92160, exact")
    print("=" * 72)
    for t7 in ([2+3*I, 1-I, 5+7*I], [1+I, sp.Integer(3), -2+5*I]):
        t0 = time.time()
        W = sp.expand(t7[0]*t7[1]*t7[2])
        imW2 = sp.im(sp.expand(W**2))
        mc = M_series(3, t7, [0, 0, 0], [hA, hB, hC], 7)
        target = sp.expand(-imW2*(2*hA+41*hB+2*hC)/92160)
        check(f"m7 exact pour t={t7} (Im W^2 = {imW2})",
              sp.simplify(mc[7] - target) == 0)
        check("ordres impairs < 7 nuls a J^z=0 (exact)",
              all(sp.simplify(mc[k]) == 0 for k in (1, 3, 5)))
        print(f"    [{round(time.time()-t0, 1)}s] m7 = {sp.simplify(sp.expand(mc[7]))}")
    # annulation quand Im(W^2)=0
    t0 = time.time()
    mc = M_series(3, TMIN3, [0, 0, 0], [hA, hB, hC], 7)
    check("m7 = 0 quand Im(W^2)=0 (jauge (1,1,i)), exact",
          sp.simplify(mc[7]) == 0)
    print(f"    [{round(time.time()-t0, 1)}s]")

# =================================================================== E3 ====
def part_E3():
    print("=" * 72)
    print("E3 -- m6 complet avec J^z : calibration exacte + hors-echantillon")
    print("=" * 72)
    p3 = hB*(hA**2 + hA*hC + hB**2 + hC**2)/384
    bpart = hA/1536 + hB*R(7, 768) + hC/1536
    configs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    t0 = time.time()
    for cfg in configs:
        Jz = [sp.Integer(v) for v in cfg]
        m6 = M_series(3, TMIN3, Jz, [hA, hB, hC], 6)[6]
        check(f"calibration J^z={cfg} reproduite par la formule fermee",
              sp.simplify(sp.expand(m6) - m6_full(TMIN3, Jz, [hA, hB, hC])) == 0)
        print(f"    [{round(time.time()-t0, 1)}s] J^z={cfg} : residu = "
              f"{sp.expand(m6 - p3 - bpart)}")
    for t_e, Jz_e in (
        ([1+I, sp.Integer(3), -2+5*I], [R(3, 5), R(-7, 4), R(11, 9)]),
        ([2+3*I, 1-I, 5+7*I], [R(-1, 2), R(4, 5), R(2, 3)]),
    ):
        t0 = time.time()
        m6 = M_series(3, t_e, Jz_e, [hA, hB, hC], 6)[6]
        check(f"hors-echantillon exact t={t_e}",
              sp.simplify(sp.expand(m6 - m6_full(t_e, Jz_e, [hA, hB, hC]))) == 0)
        print(f"    [{round(time.time()-t0, 1)}s]")

# ==================================================== N (numerique) ========
def part_N():
    print("=" * 72)
    print("N -- indications numeriques (PAS des certificats exacts)")
    print("=" * 72)
    import numpy as np
    import mpmath as mp
    from scipy.linalg import expm, logm
    mp.mp.dps = 60

    def mpkron(A, B):
        ra, ca, rb, cb = A.rows, A.cols, B.rows, B.cols
        out = mp.zeros(ra*rb, ca*cb)
        for i in range(ra):
            for j in range(ca):
                aij = A[i, j]
                if aij == 0:
                    continue
                for k in range(rb):
                    for l in range(cb):
                        out[i*rb+k, j*cb+l] = aij*B[k, l]
        return out

    def opsN(N):
        sx = mp.matrix([[0, mp.mpf('0.5')], [mp.mpf('0.5'), 0]])
        sy = mp.matrix([[0, -0.5j], [0.5j, 0]])
        sz = mp.matrix([[mp.mpf('0.5'), 0], [0, -mp.mpf('0.5')]])
        def kronN(fs):
            out = mp.matrix([[mp.mpf(1)]])
            for f in fs:
                out = mpkron(out, f)
            return out
        return {nm: [kronN([m if s == site else mp.eye(2) for s in range(N)])
                     for site in range(N)]
                for nm, m in (('+', sx+1j*sy), ('-', sx-1j*sy), ('z', sz))}

    def Hm(N, t, h, ops):
        d = 2**N
        H = mp.zeros(d)
        for i in range(N):
            j = (i+1) % N
            H += mp.mpf('0.5')*(t[i]*ops['+'][i]*ops['-'][j]
                                + mp.conj(t[i])*ops['-'][i]*ops['+'][j])
        for i in range(N):
            H -= h[i]*ops['z'][i]
        return H

    def Ereg(O, N, keep):
        d = 2**N
        keep = sorted(keep)
        traced = [s for s in range(N) if s not in keep]
        dt = 2**len(traced)
        bitpos = {s: N-1-s for s in range(N)}
        dk = 2**len(keep)
        Rm = mp.zeros(dk)
        for a in range(dk):
            for b in range(dk):
                acc = mp.mpf(0)
                for k in range(dt):
                    ii = jj = 0
                    for s in range(N):
                        if s in keep:
                            q = keep.index(s)
                            bi = (a >> (len(keep)-1-q)) & 1
                            bj = (b >> (len(keep)-1-q)) & 1
                        else:
                            q = traced.index(s)
                            bi = (k >> (len(traced)-1-q)) & 1
                            bj = bi
                        ii = (ii << 1) | bi
                        jj = (jj << 1) | bj
                    acc += O[ii, jj]
                Rm[a, b] = acc/dt
        E = mp.zeros(d)
        for i in range(d):
            for j in range(d):
                if any(((i >> bitpos[s]) & 1) != ((j >> bitpos[s]) & 1)
                       for s in traced):
                    continue
                a = b = 0
                for s in keep:
                    a = (a << 1) | ((i >> bitpos[s]) & 1)
                    b = (b << 1) | ((j >> bitpos[s]) & 1)
                E[i, j] = Rm[a, b]
        return E

    def mptr(A):
        return sum(A[i, i] for i in range(A.rows))

    class MS:
        def __init__(s, order, d):
            s.N, s.d = order, d
        def mul(s, a, b):
            out = [mp.zeros(s.d) for _ in range(s.N+1)]
            for i in range(len(a)):
                for j in range(len(b)):
                    if i+j <= s.N:
                        out[i+j] += a[i]*b[j]
            return out
        def smul(s, c, a):
            out = [mp.zeros(s.d) for _ in range(s.N+1)]
            for i, ci in enumerate(c):
                for j in range(len(a)):
                    if i+j <= s.N:
                        out[i+j] += ci*a[j]
            return out
        def sinv(s, c):
            inv = [mp.mpf(0)]*(s.N+1)
            inv[0] = mp.mpf(1)
            for n in range(1, s.N+1):
                inv[n] = -sum(c[k]*inv[n-k] for k in range(1, n+1))
            return inv
        def log(s, a):
            N, d = s.N, s.d
            x = [m.copy() for m in a]
            x[0] = mp.zeros(d)
            out = [mp.zeros(d) for _ in range(N+1)]
            cur = [mp.eye(d)] + [mp.zeros(d) for _ in range(N)]
            for n in range(1, N+1):
                cur = s.mul(cur, x)
                cf = mp.mpf((-1)**(n+1))/n
                out = [o+cf*c for o, c in zip(out, cur)]
            return out

    def M_ser(N, t, h, order):
        ops = opsN(N)
        H = Hm(N, t, h, ops)
        d = 2**N
        e = [None]*(order+1)
        P = mp.eye(d)
        for k in range(order+1):
            e[k] = mp.mpf((-1)**k)/mp.factorial(k)*P
            P = P*H
        tau = [mptr(e[k])/d for k in range(order+1)]
        S = MS(order, d)
        r = S.smul(S.sinv(tau), e)
        EA = [Ereg(r[k], N, (0, 1)) for k in range(order+1)]
        EB = [Ereg(r[k], N, (1, 2)) for k in range(order+1)]
        LA = S.log(EA)
        LB = S.log(EB)
        comm = [u-v for u, v in zip(S.mul(LA, LB), S.mul(LB, LA))]
        return [1j*mptr(m)/d for m in S.mul(r, comm)]

    mc5 = M_ser(5, [mp.mpf(1)]*4+[mp.mpc(0, 1)], [0, mp.mpf(1), 0, 0, 0], 6)
    check("N1 (numerique) : n=5, m6 = -1/512",
          abs(mp.re(mc5[6]) + mp.mpf(1)/512) < mp.mpf('1e-40'))
    check("N1 (numerique) : n=5, m1..m5 nuls",
          all(abs(mp.re(mc5[k])) < mp.mpf('1e-50') for k in range(1, 6)))

    # n=6 par Richardson double precision
    def M_np(N, t, h, beta):
        sx = np.array([[0, 1], [1, 0]], complex)/2
        sy = np.array([[0, -1j], [1j, 0]], complex)/2
        idm = np.eye(2)
        def loc(m, s):
            out = np.array([[1.0+0j]])
            for k in range(N):
                out = np.kron(out, m if k == s else idm)
            return out
        Sp = [loc(sx+1j*sy, s) for s in range(N)]
        Sm = [loc(sx-1j*sy, s) for s in range(N)]
        Sz = [loc(np.array([[1, 0], [0, -1]], complex)/2, s) for s in range(N)]
        d = 2**N
        H = np.zeros((d, d), complex)
        for i in range(N):
            j = (i+1) % N
            H += 0.5*(t[i]*Sp[i]@Sm[j]+np.conj(t[i])*Sm[i]@Sp[j])
        for i in range(N):
            H -= h[i]*Sz[i]
        r = expm(-beta*H)
        r /= np.trace(r)
        def redm(rho, keep):
            perm = list(keep)+[s for s in range(N) if s not in keep]
            rr = rho.reshape([2]*(2*N)).transpose(perm+[x+N for x in perm])
            dk, dn = 2**len(keep), 2**(N-len(keep))
            return np.trace(rr.reshape(dk, dn, dk, dn), axis1=1, axis2=3)
        LA = np.kron(logm(redm(r, (0, 1))), np.eye(2**(N-2)))
        LB = np.kron(np.eye(2), np.kron(logm(redm(r, (1, 2))), np.eye(2**(N-3))))
        return (1j*np.trace(r@(LA@LB-LB@LA))).real

    bs = [0.10, 0.071, 0.050]
    vals = [M_np(6, [1, 1, 1, 1, 1, 1j], [0, 1, 0, 0, 0, 0], b)/b**7 for b in bs]
    cf = np.polynomial.polynomial.polyfit(bs, vals, 1)
    target_n6 = 1/2048
    rel_n6 = abs(cf[0]-target_n6)/abs(target_n6)
    print(f"    n=6 : m7 extrapole = {cf[0]:.8f} (1/2048 = {target_n6:.8f})")
    print(f"          erreur relative = {rel_n6:.3e}")
    check("N2 (numerique) : n=6, m7 = +1/2048 a 2.5e-3 relatif",
          rel_n6 < 2.5e-3)

    m200 = M_np(4, [1, 1, 1, 1j], [0, 0.7, 0, 0], 200.0)
    target_plateau = 0.3146212191
    print(f"    plateau basse-T de l'anneau n=4 : M(200) = {m200:.10f}")
    check("N3 (numerique) : plateau = 0.3146212191 a 5e-9 absolu",
          abs(m200-target_plateau) < 5e-9)

if __name__ == "__main__":
    part_E1()
    part_E2()
    part_E3()
    part_N()
    print()
    print("=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All exact extension certificates passed (E1-E3) + numerical evidence (N).")
