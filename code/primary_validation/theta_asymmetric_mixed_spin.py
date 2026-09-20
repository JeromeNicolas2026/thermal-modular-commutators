#!/usr/bin/env python3
"""Exact asymmetric theta-graph certificate for the girth corollary.

The two four-cycles A-B-C-D-A and A-B-C-E-A share the path A-B-C.
All hopping magnitudes are distinct.  The graph has one spin-1 site, generic
longitudinal exchanges, and nonzero longitudinal fields.  Exact arithmetic
checks unequal fluxes and the predicted sum of the two primitive sectors.
"""

import sympy as sp

from verify_unicyclic_arbitrary_spin import I, R, exact_coefficients, kappa


def main():
    two_spins = [1, 1, 1, 2, 1]
    edges = [
        (0, 1, 1, R(1, 3)),
        (1, 2, 2 + I, R(-2, 5)),
        (2, 3, 1 + I, R(3, 7)),
        (3, 0, -14 - I, R(-4, 9)),
        (2, 4, 3 + I, R(5, 11)),
        (4, 0, 1 + R(4, 5) * I, R(-6, 13)),
    ]
    fields = [R(2, 5), R(-4, 3), R(3, 7), R(-5, 6), R(7, 9)]

    w1 = sp.expand(sp.prod(edge[2] for edge in edges[:4]))
    w2 = sp.expand(edges[0][2] * edges[1][2] * edges[4][2] * edges[5][2])
    squared_magnitudes = [
        sp.expand(sp.re(edge[2]) ** 2 + sp.im(edge[2]) ** 2)
        for edge in edges
    ]
    coefficients = exact_coefficients(two_spins, edges, fields, 5)

    cycle_1 = (0, 1, 2, 3)
    cycle_2 = (0, 1, 2, 4)
    target = sp.expand(2 * fields[1] * (
        sp.prod(kappa(two_spins[x]) for x in cycle_1) * sp.im(w1)
        + sp.prod(kappa(two_spins[x]) for x in cycle_2) * sp.im(w2)
    ))

    assert len(set(squared_magnitudes)) == 6
    assert sp.im(w1) == -43
    assert sp.im(w2) == 9
    assert coefficients[:5] == [0] * 5
    assert target == R(317, 288)
    assert coefficients[5] == target

    print("[ok] asymmetric theta: distinct hopping magnitudes")
    print("[ok] Im W_1=-43, Im W_2=9")
    print("[ok] m_0=...=m_4=0 and m_5=317/288")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
