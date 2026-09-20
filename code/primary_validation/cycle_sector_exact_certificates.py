#!/usr/bin/env python3
"""Exact certificates for the limits of a cycle-by-cycle expansion.

The script uses Gaussian-rational SymPy arithmetic throughout.  Its default
three checks take roughly two minutes.  ``--full`` also extracts an exact
mixed Fourier mode for two triangles sharing an edge.
"""

from pathlib import Path
import argparse
import sys

import sympy as sp


BUNDLE = Path(__file__).resolve().parent / "work_v5" / "modular_flux_PRB_v5_bundle"
sys.path.insert(0, str(BUNDLE))
import verify_modular_flux_extensions as engine  # noqa: E402


R = sp.Rational
I = sp.I


def modular_series(number_of_sites, edges, fields, order):
    """Return exact coefficients for M_beta(AB,BC), with A=0,B=1,C=2.

    Each edge is ``(x,y,t_xy,Jz_xy)``.  All spins are one half.
    """
    dimension = 2**number_of_sites
    ops = engine.spin_ops_N(number_of_sites)
    hamiltonian = sp.zeros(dimension)
    for x, y, t_xy, jz_xy in edges:
        hamiltonian += R(1, 2) * (
            t_xy * ops["+"][x] * ops["-"][y]
            + sp.conjugate(t_xy) * ops["-"][x] * ops["+"][y]
        )
        hamiltonian += jz_xy * ops["z"][x] * ops["z"][y]
    for x, h_x in enumerate(fields):
        hamiltonian -= h_x * ops["z"][x]

    series = engine.Series(order, dimension)
    exponential = []
    power = sp.eye(dimension)
    for k in range(order + 1):
        exponential.append(sp.expand(R((-1) ** k, sp.factorial(k)) * power))
        power = sp.expand(power * hamiltonian)
    trace_series = [sp.expand(sp.trace(term) / dimension) for term in exponential]
    density = series.smul(series.sinv(trace_series), exponential)
    log_ab = series.log(
        [engine.E_region(term, number_of_sites, (0, 1)) for term in density]
    )
    log_bc = series.log(
        [engine.E_region(term, number_of_sites, (1, 2)) for term in density]
    )
    commutator = [
        sp.expand(left - right)
        for left, right in zip(series.mul(log_ab, log_bc), series.mul(log_bc, log_ab))
    ]
    product = series.mul(density, commutator)
    return [
        sp.simplify(sp.expand(I * sp.trace(term) / dimension)) for term in product
    ]


def theta_additivity():
    # Two girth-four cycles A-B-C-D-A and A-B-C-E-A share AB and BC.
    # W_1=W_2=i, h_B=1.  Each primitive square contributes 1/128.
    edges = [
        (0, 1, 1, 0),
        (1, 2, 1, 0),
        (2, 3, 1, 0),
        (3, 0, I, 0),
        (2, 4, 1, 0),
        (4, 0, I, 0),
    ]
    coefficients = modular_series(5, edges, [0, 1, 0, 0, 0], 5)
    assert coefficients[:5] == [0] * 5
    assert coefficients[5] == R(1, 64)
    print("[ok] theta: m_1=...=m_4=0 and m_5=1/64=2(1/128)")


def off_target_cycle():
    # Unique cycle A-B-D-A has W=i, while C is a leaf joined to B.
    # The primitive order-four cycle sector is zero, but the flux is detected
    # after neutral backtracks: the first term is at order eight.
    edges = [
        (0, 1, 1, 0),
        (1, 3, 1, 0),
        (3, 0, I, 0),
        (1, 2, 1, 0),
    ]
    coefficients = modular_series(4, edges, [0, 1, 0, 0], 8)
    assert coefficients[:8] == [0] * 8
    assert coefficients[8] == -R(7, 368640)
    print("[ok] off-target cycle: m_1=...=m_7=0 and m_8=-7/368640")


def cactus_mixing():
    # Bow tie (a cactus): marked triangle A-B-C-A with W_1=i and an
    # unmarked triangle B-D-E-B with W_2=1.  At degree seven, grading rules
    # out a one-cycle term (Im W_1^2=0); the nonzero value is therefore a
    # two-cycle circulation.  Changing W_2 from +1 to -1 flips it.
    edges = [
        (0, 1, 1, 0),
        (1, 2, 1, 0),
        (2, 0, I, 0),
        (1, 3, 1, 0),
        (3, 4, 1, 0),
        (4, 1, 1, 0),
    ]
    coefficients = modular_series(5, edges, [0, 1, 0, 0, 0], 7)
    assert coefficients == [0, 0, 0, 0, -R(1, 32), 0, R(13, 768), -R(1, 768)]
    flipped_edges = edges[:-1] + [(4, 1, -1, 0)]
    flipped = modular_series(5, flipped_edges, [0, 1, 0, 0, 0], 7)
    assert flipped[7] == R(1, 768)
    print("[ok] cactus mixing: m_7=-1/768 and W_2 -> -W_2 reverses its sign")


def shared_edge_fourier_mode():
    # Triangles A-B-C-A and A-B-D-A share AB.  Put W_1=i^p, W_2=i^q.
    # The (1,1) discrete Fourier mode of m_7 is the coefficient of the
    # genuinely mixed Im(W_1 W_2) harmonic; it is nonzero.
    representatives = {}
    for p, q in [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)]:
        edges = [
            (0, 1, 1, 0),
            (1, 2, 1, 0),
            (2, 0, I**p, 0),
            (1, 3, 1, 0),
            (3, 0, I**q, 0),
        ]
        representatives[p, q] = modular_series(4, edges, [0, 1, 0, 0], 7)[7]

    values = {}
    for p in range(4):
        for q in range(4):
            if (p, q) in representatives:
                values[p, q] = representatives[p, q]
            elif ((-p) % 4, (-q) % 4) in representatives:
                values[p, q] = -representatives[(-p) % 4, (-q) % 4]
            else:
                # These four points are fixed by conjugation and hence vanish.
                assert p % 2 == 0 and q % 2 == 0
                values[p, q] = 0
    mode_11 = sp.simplify(
        sum(values[p, q] * I ** (-(p + q)) for p in range(4) for q in range(4))
        / 16
    )
    sine_coefficient = sp.simplify(2 * I * mode_11)
    assert mode_11 == 29 * I / 92160
    assert sine_coefficient == -R(29, 46080)
    print("[ok] shared-edge mixing: coefficient of sin(phi_1+phi_2)=-29/46080")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run the shared-edge DFT")
    arguments = parser.parse_args()
    theta_additivity()
    off_target_cycle()
    cactus_mixing()
    if arguments.full:
        shared_edge_fourier_mode()
    print("All requested exact certificates passed.")
