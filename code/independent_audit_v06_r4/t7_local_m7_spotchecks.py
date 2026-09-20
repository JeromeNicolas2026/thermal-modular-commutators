#!/usr/bin/env python3
"""Physical spot checks of the stated eight-parameter open-path formula.

Unlike the self-contained polynomial certificate t6, this script deliberately
uses indep_engine.py.  The two files therefore test different failure modes:
t6 proves the formal identity in a standalone qubit implementation, while the
checks here verify its Hermitian specialization in the audit engine.
"""

from fractions import Fraction as F

from indep_engine import Lattice, ZERO, cnum, modular_series


def mul(a, b):
    return (a[0] * b[0] - a[1] * b[1],
            a[0] * b[1] + a[1] * b[0])


def conj(a):
    return (a[0], -a[1])


def norm2(a):
    return a[0] ** 2 + a[1] ** 2


def im3(a, b, c):
    return mul(mul(a, b), c)[1]


def complete_formula(tAB, tBC, JAB, JBC, hB, gA, gB, gC):
    left = im3(tAB, conj(gA), gB)
    right = im3(tBC, conj(gB), gC)
    across = mul(mul(mul(tAB, tBC), conj(gA)), gC)[1]
    return hB * (
        JAB * (norm2(tBC) - JBC ** 2) * left
        + JBC * (norm2(tAB) - JAB ** 2) * right
        + JAB * JBC * across / 2
    ) / 23040


def measured(point):
    tAB, tBC, JAB, JBC, hB, gA, gB, gC = point
    series = modular_series(
        Lattice([1, 1, 1]),
        [(0, 1, tAB, JAB), (1, 2, tBC, JBC)],
        [F(0), hB, F(0)], (0, 1, 2), 7, gx=[gA, gB, gC])
    return series


# Five deterministic nondegenerate exact points. They are intentionally unrelated
# to the four points in t4_transverse_boundary.py.
POINTS = [
    (cnum(2, 3), cnum(-1, 2), F(4, 5), F(-3, 7), F(5, 6),
     cnum(1, -2), cnum(3, 1), cnum(-2, 5)),
    (cnum(-3, 1), cnum(2, -5), F(-7, 3), F(4, 9), F(-11, 6),
     cnum(2, 7), cnum(-1, 3), cnum(5, -4)),
    (cnum(1, -4), cnum(-3, -2), F(2), F(5, 2), F(7, 4),
     cnum(-2, 1), cnum(4, -3), cnum(1, 6)),
    (cnum(5, 2), cnum(-4, 3), F(3, 8), F(-9, 5), F(-2, 7),
     cnum(3, -5), cnum(2, 1), cnum(-6, -1)),
    (cnum(-2, -7), cnum(1, 4), F(-5, 11), F(7, 6), F(13, 9),
     cnum(4, 3), cnum(-5, 2), cnum(2, -3)),
]


def main():
    for number, point in enumerate(POINTS, 1):
        series = measured(point)
        target = (complete_formula(*point), F(0))
        assert all(value == ZERO for value in series[:7])
        assert series[7] == target
        print(f"[PASS] Q{number}  nondegenerate point:"
              f" m_0,...,m_6=0 and m_7={target[0]}")

    # Isolate the two one-sided sectors.  The second assertion fixes the sign
    # of the reflected B-C contribution in the orientation A -> B -> C.
    p = POINTS[0]
    left = (p[0], p[1], p[2], F(0), p[4], p[5], p[6], p[7])
    right = (p[0], p[1], F(0), p[3], p[4], p[5], p[6], p[7])
    assert measured(left)[7] == (complete_formula(*left), F(0))
    assert measured(right)[7] == (complete_formula(*right), F(0))
    assert complete_formula(*left) != 0 and complete_formula(*right) != 0
    print("[PASS] Q6  original and reflected one-sided sectors isolated; sign fixed")

    # Setting g_C=0 does not suppress the mixed correction proportional to
    # J_AB J_BC^2.  This guards against calling the old t4 expression the
    # complete coefficient of this submodel when J_BC is enabled.
    no_gC = (*p[:7], ZERO)
    assert measured(no_gC)[7] == (complete_formula(*no_gC), F(0))
    old_projection = (p[4] * p[2] * norm2(p[1])
                      * im3(p[0], conj(p[5]), p[6]) / 23040)
    assert complete_formula(*no_gC) != old_projection
    print("[PASS] Q7  g_C=0 still leaves the J_AB J_BC^2 mixed correction")


if __name__ == "__main__":
    main()
