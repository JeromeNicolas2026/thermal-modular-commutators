#!/usr/bin/env python3
"""Direct-Gibbs cross-check of the arbitrary-spin unicyclic onset law.

This is intentionally independent of ``verify_unicyclic_arbitrary_spin.py``:
it uses no formal series and imports no project code.  At each positive and
negative real beta it diagonalizes the full Hamiltonian, forms the normalized
Gibbs matrix, takes ordinary partial traces, diagonalizes the reduced density
matrices to evaluate their logarithms, and computes the modular commutator.

For p=n+1 we evenize M(beta)/beta**p using +/- beta, then extrapolate it as a
polynomial in beta**2 to beta=0.  This removes the p+1,p+3,... corrections and
gives an independent sign and normalization check of

  c_p = 2*(-1)**n*h_B*prod(s_x*(s_x+1)/3)*Im(prod(t_cycle)).

The direct calculation is necessarily floating point.  The companion script
provides the exact algebraic certificates and verifies all lower coefficients.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def spin_matrices(s):
    d = int(round(2 * s + 1))
    m = np.array([s - j for j in range(d)], dtype=float)
    plus = np.zeros((d, d), dtype=complex)
    for col in range(1, d):
        mm = m[col]
        plus[col - 1, col] = math.sqrt(s * (s + 1) - mm * (mm + 1))
    return np.eye(d), plus, plus.conj().T, np.diag(m)


def kron_all(factors):
    out = np.array([[1.0 + 0.0j]])
    for factor in factors:
        out = np.kron(out, factor)
    return out


def build_hamiltonian(spins, edges, fields):
    loc = [spin_matrices(s) for s in spins]
    ids = [q[0] for q in loc]
    plus = [kron_all([loc[x][1] if y == x else ids[y]
                      for y in range(len(spins))]) for x in range(len(spins))]
    minus = [kron_all([loc[x][2] if y == x else ids[y]
                       for y in range(len(spins))]) for x in range(len(spins))]
    zops = [kron_all([loc[x][3] if y == x else ids[y]
                      for y in range(len(spins))]) for x in range(len(spins))]
    dim = math.prod(int(2 * s + 1) for s in spins)
    H = np.zeros((dim, dim), dtype=complex)
    for x, y, t, jz in edges:
        H += 0.5 * (t * plus[x] @ minus[y]
                    + np.conjugate(t) * minus[x] @ plus[y])
        H += jz * zops[x] @ zops[y]
    for x, h in enumerate(fields):
        H -= h * zops[x]
    assert np.linalg.norm(H - H.conj().T) < 2e-13
    return H


def partial_trace(rho, dims, keep):
    """Ordinary partial trace, preserving the order specified in keep."""
    keep = list(keep)
    traced = [x for x in range(len(dims)) if x not in keep]
    N = len(dims)
    perm = keep + traced + [x + N for x in keep] + [x + N for x in traced]
    dk = math.prod(dims[x] for x in keep)
    dt = math.prod(dims[x] for x in traced)
    tensor = rho.reshape(tuple(dims + dims)).transpose(perm)
    tensor = tensor.reshape(dk, dt, dk, dt)
    return np.trace(tensor, axis1=1, axis2=3)


def hermitian_log(rho):
    vals, vecs = eigh(rho)
    if vals.min() <= 0:
        raise ArithmeticError(f"reduced state not positive: lambda_min={vals.min()}")
    return (vecs * np.log(vals)) @ vecs.conj().T


class DirectFunctional:
    def __init__(self, spins, edges, fields):
        self.spins = spins
        self.dims = [int(2 * s + 1) for s in spins]
        self.H = build_hamiltonian(spins, edges, fields)
        self.evals, self.evecs = eigh(self.H)

    def __call__(self, beta):
        exponents = -beta * self.evals
        weights = np.exp(exponents - exponents.max())
        weights /= weights.sum()
        rho = (self.evecs * weights) @ self.evecs.conj().T
        rho_abc = partial_trace(rho, self.dims, (0, 1, 2))
        rho_ab = partial_trace(rho, self.dims, (0, 1))
        rho_bc = partial_trace(rho, self.dims, (1, 2))
        lab = np.kron(hermitian_log(rho_ab), np.eye(self.dims[2]))
        lbc = np.kron(np.eye(self.dims[0]), hermitian_log(rho_bc))
        ans = 1j * np.trace(rho_abc @ (lab @ lbc - lbc @ lab))
        if abs(ans.imag) > 2e-11:
            raise ArithmeticError(f"M has spurious imaginary part {ans.imag}")
        return float(ans.real)


def cycle_edges(n, t, jz):
    return [(x, (x + 1) % n, t[x], jz[x]) for x in range(n)]


def kappa(s):
    return s * (s + 1) / 3.0


@dataclass
class Case:
    name: str
    n: int
    spins: list[float]
    t: list[complex]
    jz: list[float]
    fields: list[float]
    tree_edges: list[tuple] | None = None


CASES = [
    Case(
        "D1 triangle: spins (1/2,1,3/2)", 3,
        [0.5, 1.0, 1.5],
        [1 + 2j, 2 - 1j, -1 + 3j],
        [2 / 3, -5 / 7, 7 / 4],
        [-3 / 5, 11 / 6, 13 / 9],
    ),
    Case(
        "D2 four-cycle plus attached spin-1 tree site", 4,
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [2 + 1j, 1 - 1j, 3 + 2j, -1 + 1j],
        [-1 / 3, 2 / 7, 4 / 5, -6 / 11],
        [-2 / 5, 3 / 2, 5 / 9, -7 / 8, 11 / 10],
        [(3, 4, 2 - 3j, 13 / 12)],
    ),
    Case(
        "D3 five-cycle: a remote cycle site has spin 1", 5,
        [0.5, 0.5, 0.5, 1.0, 0.5],
        [1 + 1j, -2 + 1j, 1 - 3j, 2 + 1j, -1 + 2j],
        [2 / 5, -3 / 7, 5 / 9, -7 / 11, 11 / 13],
        [-1 / 4, 8 / 7, 3 / 5, -5 / 8, 7 / 10],
    ),
    Case(
        "D4 six-cycle: all spins 1/2", 6,
        [0.5] * 6,
        [1 + 1j, 2 - 1j, -1 + 2j, 3 + 1j, 1 - 2j, 2 + 1j],
        [1 / 3, -2 / 5, 3 / 7, -4 / 9, 5 / 11, -6 / 13],
        [1 / 5, -8 / 7, 3 / 8, -4 / 9, 5 / 12, -6 / 13],
    ),
]


def target(case):
    W = np.prod(case.t)
    product_kappa = np.prod([kappa(case.spins[x]) for x in range(case.n)])
    return (2 * (-1) ** case.n * case.fields[1]
            * product_kappa * W.imag)


def extrapolate(case):
    p = case.n + 1
    edges = cycle_edges(case.n, case.t, case.jz) + (case.tree_edges or [])
    functional = DirectFunctional(case.spins, edges, case.fields)
    if p <= 5:
        betas = np.array([0.025, 0.032, 0.040, 0.050, 0.063])
    else:
        betas = np.array([0.045, 0.055, 0.067, 0.080, 0.095])
    evenized = []
    for beta in betas:
        yp = functional(beta) / beta ** p
        ym = functional(-beta) / (-beta) ** p
        evenized.append(0.5 * (yp + ym))
    x = betas ** 2
    # Cubic fit is the reported estimate; quadratic and quartic fits provide a
    # conservative internal estimate of extrapolation/numerical stability.
    estimates = [np.polynomial.polynomial.polyfit(x, evenized, degree)[0]
                 for degree in (2, 3, 4)]
    return betas, np.array(evenized), estimates


def main():
    failures = []
    for case in CASES:
        betas, evenized, estimates = extrapolate(case)
        predicted = target(case)
        estimate = estimates[1]
        scale = max(abs(predicted), 1e-14)
        relerr = abs(estimate - predicted) / scale
        spread = max(estimates) - min(estimates)
        product_kappa = np.prod([kappa(case.spins[x])
                                 for x in range(case.n)])
        denominator = case.fields[1] * np.prod(case.t).imag * product_kappa
        normalized = estimate / denominator
        wanted_normalized = 2 * (-1) ** case.n
        ok = relerr < 2e-5 and abs(spread) / scale < 5e-4
        print("=" * 76)
        print(case.name)
        print(f"  dimensions              = {[int(2*s+1) for s in case.spins]}")
        print(f"  p=n+1                   = {case.n + 1}")
        print(f"  Im W                    = {np.prod(case.t).imag:.12g}")
        print(f"  evenized samples        = {evenized.tolist()}")
        print(f"  extrapolants deg 2/3/4  = {estimates}")
        print(f"  extrapolated c_p        = {estimate:.12g}")
        print(f"  predicted c_p           = {predicted:.12g}")
        print(f"  relative error          = {relerr:.3e}")
        print(f"  c_p/(h_B ImW prod kap)  = {normalized:.12g}"
              f"  [target {wanted_normalized:+d}]")
        print(f"  status                  = {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(case.name)
    print("=" * 76)
    if failures:
        print("Direct-Gibbs failures:")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"All {len(CASES)} direct-Gibbs cross-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
