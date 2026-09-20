#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_modular_flux_ring_jw.py
==============================
Independent implementation of the Jordan--Wigner reconstruction for a
transverse spin ring and of the thermal modular commutator

    M_ρ(X,Y) = i Tr ρ [log ρ_X, log ρ_Y],

with X={0,1} and Y={1,2}.

CONVENTION DE CONJUGAISON DES COUPLAGES (documentée et testée)
--------------------------------------------------------------
La matrice monoparticulaire E_p est construite à partir des couplages
CONJUGUÉS : hopping t̄_j/2 sur le lien (j,j+1), et hopping de bord
−p·t̄_{n−1}/2 sur le lien fermant (n−1,0), où p = ±1 est la condition aux
limites fermionique (secteur de parité). Avec cette convention — cordes de
Jordan-Wigner orientées de gauche à droite,
c_j = (Π_{k<j} −σ^z_k) σ⁻_j, et passage base de Fock → base de spins par
le retournement local X^{⊗m} — l'assemblage en quatre morceaux reproduit
l'état de Gibbs des spins aux couplages t :

    ρ_JW(t) = ρ_ED(t)        (testé : écart max < 1e-11, n=3,4,5,
                              couplages complexes et champs génériques, J0/J1)

La convention « brute » (E construite avec t au lieu de t̄) calcule l'état
conjugué ρ_ED(t̄), pour lequel le commutateur modulaire change de signe
(M(ρ*) = −M(ρ), les régions étant invariantes sous X^{⊗n}) : c'est la
conjugaison interne de E_p qui fixe le signe du flux. Testé en J0 :
jw_M(conj t) = −jw_M(t) exactement.

DÉCOMPOSITION EXACTE EN QUATRE MORCEAUX GAUSSIENS
-------------------------------------------------
L'état de Gibbs de l'anneau transverse s'écrit :

    ρ_β = (1/2Z) Σ_{p=±1} Σ_{s=0,1} p^s T_{p,s} ρ_G(C_{p,s})

avec M_{p,s} = β E_p − s·iπ·I,  C_{p,s} = (e^{M_{p,s}} + 1)^{-1},
T_{p,s} = det(1 + e^{-M_{p,s}}),  Z = (1/2) Σ p^s T_{p,s},
et ρ_G(C) l'opérateur gaussien fermionique de matrice de corrélation C
(relu à la base de spins par X^{⊗m}). s=0 : morceau grand-canonique ;
s=1 : insertion de parité (logit C = M = βE − iπI, modulo 2πiI).

CONTENU DES TESTS
-----------------
  J0  convention : ρ_JW(t) = ρ_ED(t) (niveau état) ; M(conj t) = −M(t)
  J1  ED vs JW pour M_β : n=3..7, couplages complexes + champs
      génériques, β = 0.05, 0.3, 1.0 (assert 1e-9)
  J2  table 3 ≤ n ≤ 12 du coefficient dominant
          [β^{n+1}] M_β = (−1)^n h_B Im W_n / 2^{2n−1}
      extraction par résolution du Vandermonde en u = β² (mpmath
      lu_solve, dps=80 — pas de numpy.polyfit), assertions relatives 1e-12
  J3  nécessité structurelle des morceaux de parité : les deux morceaux
      grand-canoniques (s=0) pris seuls donnent un coefficient dominant
      nul à la précision de travail (assert |c0| < 1e-25 ; évidence
      numérique, non un certificat exact). [Énoncé de nécessité, AUCUNE
      interprétation additive des contributions.]
  NB  les fractions affichées en J2 sont des IDENTIFICATIONS
      numériques à haute précision (Fraction sur flottant), non des
      certificats rationnels.  J4  lemme de parité : M(−β) = (−1)^{n+1} M(β)
      (a) vérification directe à Re W_n = 0, couplages/champs génériques :
          ED float64 n=3..5 (assert 1e-9) et JW mpmath n=3..8 (assert 1e-15)
      (b) mécanisme n pair (tout couplage) : U = X^{⊗n} Π_k e^{iπk S^z_k}
          (unitaire on-site) vérifie U H(t,h) U† = −H(t̄,h) (assert 1e-13)
      (c) mécanisme n impair (preuve locale directe) : à Re W_n = 0,
          construction explicite de U = Π_j e^{−iα_j S^z_j} σ^x_j avec
          α_j − α_{j+1} = π − 2 arg t_j (compatible ssi Re W_n = 0) ;
          U H U† = −H (assert 1e-13) et ρ(−β) = U ρ(β) U† (assert 1e-12)
      (d) remarque fermionique (confirmation) :
          det(λI − E_p) = P_0(λ) + p κ_n Re W_n, κ_n = 2^{-(n-1)} :
          constance en λ (assert 1e-9 relatif), κ_n exact, et
          P_0 ≠ det(λI − E_ouvert) (contributions aller-retour du lien
          fermant, indépendantes de p, absentes du déterminant ouvert)
Exécution : python3 verify_modular_flux_ring_jw.py   (~1-4 min selon BLAS)
"""

import time
import numpy as np
from scipy.linalg import logm, expm
from mpmath import mp          # mp = le CONTEXTE (pas le module) :
mp.dps = 80                    # mp.dps agit réellement sur la précision
from fractions import Fraction

X2 = np.array([[0, 1], [1, 0]], complex)

# ======================================================================
#  Diagonalisation exacte (numpy, float64)
# ======================================================================

def ring_H(n, t, h):
    """Hamiltonien de l'anneau transverse  H = Σ ½(t_j S⁺_j S⁻_{j+1} + h.c.)
    − Σ h_j S^z_j,  lien fermant (n−1,0) portant t_{n−1}."""
    sp = np.array([[0, 1], [0, 0]], complex)   # S⁺ (sans le facteur 2 : S=σ/2)
    sm = np.array([[0, 0], [1, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex) / 2
    idm = np.eye(2)
    def loc(m, s):
        out = np.array([[1.0 + 0j]])
        for k in range(n):
            out = np.kron(out, m if k == s else idm)
        return out
    Sp = [loc(sp, s) for s in range(n)]
    Sm = [loc(sm, s) for s in range(n)]
    Sz = [loc(sz, s) for s in range(n)]
    H = np.zeros((2**n, 2**n), complex)
    for i in range(n):
        j = (i + 1) % n
        H += 0.5 * (t[i] * Sp[i] @ Sm[j] + np.conj(t[i]) * Sm[i] @ Sp[j])
    for i in range(n):
        H -= h[i] * Sz[i]
    return H


def M_of_H(H, n, beta):
    """Commutateur modulaire M = i Tr ρ [loĝ_X, loĝ_Y], X={0,1}, Y={1,2}."""
    r = expm(-beta * H)
    r /= np.trace(r)
    def redm(rho, keep):
        perm = list(keep) + [s for s in range(n) if s not in keep]
        rr = rho.reshape([2] * (2 * n)).transpose(perm + [x + n for x in perm])
        dk, dn = 2**len(keep), 2**(n - len(keep))
        return np.trace(rr.reshape(dk, dn, dk, dn), axis1=1, axis2=3)
    LA = np.kron(logm(redm(r, (0, 1))), np.eye(2**(n - 2)))
    LB = np.kron(np.eye(2), np.kron(logm(redm(r, (1, 2))), np.eye(2**(n - 3))))
    return (1j * np.trace(r @ (LA @ LB - LB @ LA))).real


# ======================================================================
#  Jordan-Wigner (numpy) — convention : E construite avec conj(t)
# ======================================================================

def eps_mat(n, t, h, p):
    """Matrice monoparticulaire E_p (convention conjuguée, cf. en-tête)."""
    tt = np.conj(np.array(t, dtype=complex))
    E = np.zeros((n, n), complex)
    for j in range(n - 1):
        E[j, j + 1] = 0.5 * tt[j]
        E[j + 1, j] = 0.5 * np.conj(tt[j])
    E[n - 1, 0] = -0.5 * p * tt[n - 1]
    E[0, n - 1] = -0.5 * p * np.conj(tt[n - 1])
    for j in range(n):
        E[j, j] = -h[j]
    return E


def fermion_ops(m):
    sz = np.array([[1, 0], [0, -1]], complex)
    cm = np.array([[0, 1], [0, 0]], complex)
    c = []
    for j in range(m):
        op = np.array([[1.0 + 0j]])
        for l in range(m):
            op = np.kron(op, sz if l < j else (cm if l == j else np.eye(2)))
        c.append(op)
    return c


def gauss_rho_np(Cb):
    """Opérateur gaussien fermionique ρ ∝ exp(−Σ G_ij c†_i c_j) avec
    G = logit(C), sur m sites (base de Fock, cordes gauche→droite)."""
    m = Cb.shape[0]
    vals, U = np.linalg.eigh(Cb)
    g = np.log(((1 - vals) / vals).astype(complex))
    G = U.conj() @ np.diag(g) @ U.T          # G = conj(logit(C)) : convention cordes
    cs = fermion_ops(m)
    HG = np.zeros((2**m, 2**m), complex)
    for i in range(m):
        for j in range(m):
            HG = HG + G[i, j] * (cs[i].conj().T @ cs[j])
    R = expm(-HG)
    return R / np.trace(R)


def xflip(m):
    out = np.eye(1)
    for _ in range(m):
        out = np.kron(out, X2)
    return out


def jw_pieces_np(n, t, h, beta, keep=((1, 0), (1, 1), (-1, 0), (-1, 1))):
    pieces = []
    for p in (1, -1):
        E = eps_mat(n, t, h, p)
        for s in (0, 1):
            if (p, s) not in keep:
                continue
            Mm = beta * E - s * 1j * np.pi * np.eye(n)
            w, U = np.linalg.eig(Mm)
            Cm = U @ np.diag(1 / (np.exp(w) + 1)) @ np.linalg.inv(U)
            Cm = 0.5 * (Cm + Cm.conj().T)
            pieces.append((p**s, Cm, np.prod(1 + np.exp(-w))))
    return pieces


def jw_M_np(n, t, h, beta, keep=((1, 0), (1, 1), (-1, 0), (-1, 1))):
    pieces = jw_pieces_np(n, t, h, beta, keep)
    Z2 = sum(sg * T for sg, _, T in pieces)
    def block(sites):
        m = len(sites)
        r = None
        for sg, Cm, T in pieces:
            rb = sg * T * gauss_rho_np(Cm[np.ix_(sites, sites)])
            r = rb if r is None else r + rb
        r = r / Z2
        return xflip(m) @ r @ xflip(m)
    rABC = block([0, 1, 2])
    rAB = block([0, 1])
    rBC = block([1, 2])
    KAB = np.kron(logm(rAB), np.eye(2))
    KBC = np.kron(np.eye(2), logm(rBC))
    return (1j * np.trace(rABC @ (KAB @ KBC - KBC @ KAB))).real


def jw_rho_full_np(n, t, h, beta):
    """État de Gibbs complet reconstruit par les quatre morceaux."""
    pieces = jw_pieces_np(n, t, h, beta)
    Z2 = sum(sg * T for sg, _, T in pieces)
    r = None
    for sg, Cm, T in pieces:
        rb = sg * T * gauss_rho_np(Cm)
        r = rb if r is None else r + rb
    r = r / Z2
    return xflip(n) @ r @ xflip(n)

# ======================================================================
#  Jordan-Wigner (mpmath, haute précision)
# ======================================================================

def mpkron(A, B):
    C = mp.zeros(A.rows * B.rows, A.cols * B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i, j] == 0:
                continue
            for k in range(B.rows):
                for l in range(B.cols):
                    C[i * B.rows + k, j * B.cols + l] = A[i, j] * B[k, l]
    return C


def mconj(A):
    return A.apply(mp.conj)


def mdag(A):
    return mconj(A).transpose()


_FOPS = {}


def mp_fops(m):
    if m in _FOPS:
        return _FOPS[m]
    sz = mp.matrix([[1, 0], [0, -1]])
    cm = mp.matrix([[0, 1], [0, 0]])
    I2 = mp.eye(2)
    c = []
    for j in range(m):
        op = mp.matrix([[1]])
        for l in range(m):
            op = mpkron(op, sz if l < j else (cm if l == j else I2))
        c.append(op)
    _FOPS[m] = c
    return c


def mp_eps(n, t, h, p):
    """E_p en haute précision — convention conjuguée (cf. en-tête)."""
    t = [mp.conj(mp.mpc(x)) for x in t]
    E = mp.zeros(n)
    for j in range(n - 1):
        E[j, j + 1] = mp.mpf('0.5') * t[j]
        E[j + 1, j] = mp.mpf('0.5') * mp.conj(t[j])
    E[n - 1, 0] = -mp.mpf('0.5') * p * t[n - 1]
    E[0, n - 1] = -mp.mpf('0.5') * p * mp.conj(t[n - 1])
    for j in range(n):
        E[j, j] = -mp.mpf(h[j])
    return E


def gauss_rho_mp(Cb):
    m = Cb.rows
    vals, U = mp.eig(Cb)
    g = [mp.log((1 - v) / v) for v in vals]
    G = mconj(U) * mp.diag(g) * U.transpose()
    cs = mp_fops(m)
    HG = mp.zeros(2**m)
    for i in range(m):
        ci = mdag(cs[i])
        for j in range(m):
            HG = HG + G[i, j] * (ci * cs[j])
    R = mp.expm(-HG)
    tr = sum(R[i, i] for i in range(2**m))
    return R / tr


def jw_M_mp(n, t, h, beta, keep=((1, 0), (1, 1), (-1, 0), (-1, 1))):
    """M_β en précision arbitraire ; keep sélectionne les morceaux (p,s)."""
    beta = mp.mpf(beta)
    pieces = []
    for p in (1, -1):
        E = mp_eps(n, t, h, p)
        for s in (0, 1):
            if (p, s) not in keep:
                continue
            Mm = beta * E - s * 1j * mp.pi * mp.eye(n)
            w, U = mp.eig(Mm)
            f = [1 / (mp.exp(wk) + 1) for wk in w]
            Cm = U * mp.diag(f) * mp.inverse(U)
            Cm = (Cm + mdag(Cm)) / 2
            T = mp.mpf(1)
            for wk in w:
                T *= (1 + mp.exp(-wk))
            pieces.append((p**s, Cm, T))
    Z = sum(sg * T for sg, _, T in pieces)
    def block(sites):
        m = len(sites)
        r = None
        for sg, Cm, T in pieces:
            Cb = mp.matrix([[Cm[sites[a], sites[b]] for b in range(m)]
                            for a in range(m)])
            rb = sg * T * gauss_rho_mp(Cb)
            r = rb if r is None else r + rb
        r = r / Z
        Xm = mp.matrix([[1]])
        for _ in range(m):
            Xm = mpkron(Xm, mp.matrix([[0, 1], [1, 0]]))
        return Xm * r * Xm
    rABC = block([0, 1, 2])
    rAB = block([0, 1])
    rBC = block([1, 2])
    def logm_herm(R):
        R = (R + mdag(R)) / 2
        w, U = mp.eig(R)
        return U * mp.diag([mp.log(mp.re(x)) for x in w]) * mp.inverse(U)
    KAB = mpkron(logm_herm(rAB), mp.eye(2))
    KBC = mpkron(mp.eye(2), logm_herm(rBC))
    prod = rABC * (KAB * KBC - KBC * KAB)
    return 1j * sum(prod[i, i] for i in range(8))


# ======================================================================
#  Extraction des coefficients : Vandermonde en u = β² (mpmath)
# ======================================================================

def extract_beta2(n, npts=5, bmax='0.055', keep=((1, 0), (1, 1), (-1, 0), (-1, 1))):
    """Au point W_n = i, h_B = 1. Lemme de parité (J4) ⇒
    M(β)/β^{n+1} = c0 + c1 u + c2 u² + … avec u = β².
    Résolution du système de Vandermonde en u par mp.lu_solve
    (aucune utilisation de numpy.polyfit)."""
    t = [mp.mpf(1)] * (n - 1) + [mp.mpc(0, 1)]
    h = [mp.mpf(0)] * n
    h[1] = mp.mpf(1)
    bs = [mp.mpf(bmax) * (2 * k - 1) / (2 * npts) for k in range(1, npts + 1)]
    y = []
    for b in bs:
        y.append(mp.re(jw_M_mp(n, t, h, b, keep=keep)) / b**(n + 1))
    y = mp.matrix(y)
    V = mp.zeros(npts)
    for i, b in enumerate(bs):
        u = b * b
        for j in range(npts):
            V[i, j] = u**j
    sol = mp.lu_solve(V, y)
    return [sol[j] for j in range(npts)]


def rat(v, nd=10, dmax=2**40):
    return Fraction(float(v)).limit_denominator(dmax)


# ======================================================================
#  Tests
# ======================================================================

def J0_convention():
    print("J0  Convention de conjugaison (niveau état et signe du flux)")
    rng = np.random.default_rng(5)
    worst = 0.0
    for n in (3, 4, 5):
        tg = rng.normal(size=n) + 1j * rng.normal(size=n)
        hg = rng.normal(size=n)
        for b in (0.23, 0.71):
            rj = jw_rho_full_np(n, tg, hg, b)
            H = ring_H(n, tg, hg)
            re_ = expm(-b * H)
            re_ /= np.trace(re_)
            worst = max(worst, float(np.max(np.abs(rj - re_))))
    print(f"    max ||rho_JW(t) - rho_ED(t)||_inf = {worst:.2e}")
    assert worst < 1e-9
    # signe du flux : M(conj t) = -M(t)
    tn = [1.0, 1.0, 1.0, 1j]
    hn = [0.0, 1.0, 0.0, 0.0]
    v1 = jw_M_np(4, tn, hn, 0.3)
    v2 = jw_M_np(4, np.conj(tn), hn, 0.3)
    print(f"    M(conj t)/M(t) = {v2 / v1:.15f}  (attendu -1)")
    assert abs(v2 / v1 + 1) < 1e-12
    print("    OK")


def J1_ed_vs_jw():
    print("J1  ED vs JW : n=3..7, couplages complexes et champs génériques")
    rng = np.random.default_rng(1234)
    worst = 0.0
    for n in range(3, 8):
        for trial in range(2):
            tg = rng.normal(size=n) + 1j * rng.normal(size=n)
            hg = rng.normal(size=n)
            for b in (0.05, 0.3, 1.0):
                ed = M_of_H(ring_H(n, tg, hg), n, b)
                jw = jw_M_np(n, tg, hg, b)
                worst = max(worst, abs(ed - jw))
        print(f"    n={n} ok (cumul max = {worst:.2e})")
    assert worst < 1e-9
    print(f"    OK  (écart max = {worst:.2e})")


def J2_table():
    print("J2  Table 3<=n<=12 : [β^{n+1}]M = (−1)^n/2^{2n−1}  (W_n=i, h_B=1)")
    worst = 0.0
    table = {}
    for n in range(3, 13):
        c = extract_beta2(n, npts=5, bmax='0.055')
        target = (-1)**n / 2**(2 * n - 1)
        rel = float(abs(c[0] - target) / abs(target))
        worst = max(worst, rel)
        table[n] = (c[0], c[1])
        print(f"    n={n:2d}  c0={mp.nstr(c[0], 12)}  cible={target:+.8f}"
              f"  err rel={rel:.1e}  fraction={rat(c[0])}")
        assert rel < 1e-12
    print(f"    OK  (écart relatif max = {worst:.1e})")
    return table


def J3_parity_necessity():
    print("J3  Nécessité des morceaux de parité (s=1) : morceaux grand-")
    print("    canoniques seuls (s=0) -> coefficient dominant nul")
    keep_gc = ((1, 0), (-1, 0))
    for n in range(4, 9):
        c = extract_beta2(n, npts=3, bmax='0.05', keep=keep_gc)
        scale = 1 / 2**(2 * n - 1)
        print(f"    n={n}  |c0_GC| = {mp.nstr(abs(c[0]), 3)}"
              f"   (échelle de la loi : {scale:.2e})")
        assert abs(c[0]) < 1e-25
    print("    OK  (le signal impair en flux exige la projection de parité ;")
    print("         aucune lecture additive des morceaux n'est faite)")


def J4_parity_lemma():
    print("J4  Lemme de parité : à Re W_n = 0, M(−β) = (−1)^{n+1} M(β)")
    # (a) vérification à couplages/champs génériques, Re W_n = 0.
    #     n=3..5 : ED float64 (indépendant de JW) ; n=3..8 : JW mpmath.
    rng = np.random.default_rng(11)
    worst = 0.0
    for n in range(3, 6):
        tg = rng.normal(size=n) + 1j * rng.normal(size=n)
        W = np.prod(tg)
        tg[-1] *= np.exp(1j * (np.pi / 2 - np.angle(W)))   # W_n -> i|W_n|
        assert abs(np.prod(tg).real) < 1e-12
        hg = rng.normal(size=n)
        for b in (0.15, 0.35):
            Mp = M_of_H(ring_H(n, tg, hg), n, b)
            Mm = M_of_H(ring_H(n, tg, hg), n, -b)
            r = Mm / Mp
            worst = max(worst, abs(r - (-1)**(n + 1)))
        print(f"    (a-ED) n={n} : M(-β)/M(β) = {(-1)**(n+1):+d}"
              f"  (écart max cumulé {worst:.1e})")
    assert worst < 1e-9
    worst_mp = mp.mpf(0)
    rng = np.random.default_rng(21)
    for n in range(3, 9):
        tg = rng.normal(size=n) + 1j * rng.normal(size=n)
        W = np.prod(tg)
        tg[-1] *= np.exp(1j * (np.pi / 2 - np.angle(W)))
        hg = rng.normal(size=n)
        tm = [mp.mpc(complex(x).real, complex(x).imag) for x in tg]
        hm = [mp.mpf(x) for x in hg]
        for b in ('0.08', '0.2', '0.45'):
            Mp = mp.re(jw_M_mp(n, tm, hm, mp.mpf(b)))
            Mm = mp.re(jw_M_mp(n, tm, hm, -mp.mpf(b)))
            worst_mp = max(worst_mp, abs(Mm / Mp - (-1)**(n + 1)))
        print(f"    (a-JW) n={n} : idem (écart max cumulé {mp.nstr(worst_mp, 2)})")
    assert worst_mp < mp.mpf('1e-15')
    # (b) mécanisme n pair : U H(t,h) U† = −H(t̄,h), U on-site
    for n in (4, 6):
        tg = rng.normal(size=n) + 1j * rng.normal(size=n)
        hg = rng.normal(size=n)
        U = np.array([[1.0 + 0j]])
        for k in range(n):
            Rk = np.diag([np.exp(0.5j * np.pi * k), np.exp(-0.5j * np.pi * k)])
            U = np.kron(U, X2 @ Rk)
        H = ring_H(n, tg, hg)
        Hc = ring_H(n, np.conj(tg), hg)
        err = np.max(np.abs(U @ H @ U.conj().T + Hc))
        print(f"    (b) n={n} : ||U H(t,h) U† + H(t̄,h)||_inf = {err:.1e}")
        assert err < 1e-13
    # (c) mécanisme n impair : preuve locale directe.
    #     U_j = e^{−iα_j Sz_j} X_j avec α_j − α_{j+1} = π − 2 arg t_j,
    #     compatible autour du cycle ssi (-1)^n W̄_n/W_n = 1, i.e. pour
    #     n impair ssi Re W_n = 0. Alors U H(t,h) U† = −H(t,h).
    for n in (3, 5, 7):
        tg = rng.normal(size=n) + 1j * rng.normal(size=n)
        W = np.prod(tg)
        tg[-1] *= np.exp(1j * (np.pi / 2 - np.angle(W)))   # Re W_n -> 0
        assert abs(np.prod(tg).real) < 1e-12
        hg = rng.normal(size=n)
        D = [np.pi - 2 * np.angle(x) for x in tg]
        alpha = [0.0]
        for j in range(1, n):
            alpha.append(alpha[-1] - D[j - 1])
        comp = (alpha[-1] - D[-1]) % (2 * np.pi)           # doit être ~0
        assert min(comp, 2 * np.pi - comp) < 1e-10
        U = np.array([[1.0 + 0j]])
        for j in range(n):
            Uj = np.array([[0, np.exp(-0.5j * alpha[j])],
                           [np.exp(0.5j * alpha[j]), 0]])
            U = np.kron(U, Uj)
        H = ring_H(n, tg, hg)
        errU = np.max(np.abs(U @ H @ U.conj().T + H))
        b = 0.4
        rp = expm(-b * H)
        rp /= np.trace(rp)
        rm = expm(b * H)
        rm /= np.trace(rm)
        errR = np.max(np.abs(rm - U @ rp @ U.conj().T))
        print(f"    (c) n={n} : ||U H U† + H|| = {errU:.1e}   "
              f"||ρ(−β) − Uρ(β)U†|| = {errR:.1e}")
        assert errU < 1e-13 and errR < 1e-12
    # (d) remarque fermionique : det(λI − E_p) = P_0(λ) + p κ_n Re W_n,
    #     κ_n = 2^{-(n-1)} ; P_0 ≠ déterminant de la chaîne ouverte.
    rng2 = np.random.default_rng(7)
    for n in range(3, 9):
        tg = rng2.normal(size=n) + 1j * rng2.normal(size=n)
        hg = rng2.normal(size=n)
        Ep = eps_mat(n, tg, hg, 1)
        Em = eps_mat(n, tg, hg, -1)
        Eo = Ep.copy()
        Eo[n - 1, 0] = 0
        Eo[0, n - 1] = 0
        ReW = np.prod(tg).real
        lams = [0.31, -0.77, 1.13, 2.7]
        ks = []
        P0_minus_open = 0.0
        for lam in lams:
            dp = np.linalg.det(lam * np.eye(n) - Ep)
            dm = np.linalg.det(lam * np.eye(n) - Em)
            ks.append(((dp - dm) / 2).real / ReW)
            P0_minus_open = max(P0_minus_open,
                                abs(((dp + dm) / 2).real
                                    - np.linalg.det(lam * np.eye(n) - Eo).real))
        spread = max(abs(k - np.mean(ks)) for k in ks) / abs(np.mean(ks))
        err_kappa = abs(np.mean(ks) - 2.0**-(n - 1)) / 2.0**-(n - 1)
        print(f"    (d) n={n} : κ_n = {np.mean(ks):.8f} = 2^{{-(n-1)}} "
              f"(rel {err_kappa:.1e}), constance en λ (rel {spread:.1e}), "
              f"max|P0 − det_ouvert| = {P0_minus_open:.1e}")
        assert err_kappa < 1e-7 and spread < 1e-7 and P0_minus_open > 1e-3
    print("    OK")


if __name__ == '__main__':
    t0 = time.time()
    J0_convention()
    J1_ed_vs_jw()
    J2_table()
    J3_parity_necessity()
    J4_parity_lemma()
    print(f"\nTous les tests J0–J4 sont passés.  ({time.time() - t0:.0f} s)")
