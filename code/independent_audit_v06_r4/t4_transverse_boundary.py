#!/usr/bin/env python3
"""Exact spot checks and counterexamples at the transverse-field boundary.

This script disproves an extension with the same universal leading asymptotic
under arbitrary transverse fields. Exact rational arithmetic throughout.

Transverse term, with the gauge-invariant phase combination:

    H_perp = -(1/2) sum_x ( g_x S_x^+ + conj(g_x) S_x^- ),      g_x in Q(i).

For the minimal open path A-B-C of qubits used below, with J^z_BC=0, the
full-series checks agree with

    m_7^loc = J^z_AB h^z_B |t_BC|^2 Im( t_AB conj(g_A) g_B ) / 23040.

The universal statement is the projected multidegree proved symbolically in
``t5_sector_projection.py``. N1-N5 below are exact specialization checks, not
a proof of a polynomial identity or of ambient independence. N6-N7 are full
counterexamples.
"""

from fractions import Fraction as F
from indep_engine import Lattice, modular_series, cnum, ZERO


def m_series(sp, ed, hz, gx, K=7, marked=(0, 1, 2)):
    return modular_series(Lattice(sp), ed, hz, marked, K, gx=gx)


def chain(tAB, tBC, jz, hz, gA, gB, gC=ZERO, K=7):
    """Open path A - B - C.  No cycle anywhere."""
    return m_series([1, 1, 1], [(0, 1, tAB, jz), (1, 2, tBC, F(0))],
                    [F(0), hz, F(0)], [gA, gB, gC], K)


def cycle(n, tAB=cnum(1, 1), tBC=cnum(1), other=F(1), jz=F(1), hz=F(1),
          gA=cnum(1), gB=cnum(1), K=7):
    ed = [(i, (i + 1) % n,
           tAB if i == 0 else (tBC if i == 1 else (other, F(0))),
           jz if i == 0 else F(0)) for i in range(n)]
    hzs = [F(0)] * n
    hzs[1] = hz
    gs = [ZERO] * n
    gs[0], gs[1] = gA, gB
    return m_series([1] * n, ed, hzs, gs, K)


def closed_form(tAB, tBC, jz, hz, gA, gB):
    """Projected formula, and full m_7 in the checked J^z_BC=0 submodel."""
    def mul(u, v):
        return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
    z = mul(mul(tAB, (gA[0], -gA[1])), gB)
    return jz * hz * (tBC[0] ** 2 + tBC[1] ** 2) * z[1] / 23040


def report(tag, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{extra}")
    assert cond, tag


POINTS = [(cnum(1, 1), cnum(1), F(1), F(1), cnum(1), cnum(1)),
          (cnum(2, -3), cnum(1, 2), F(5, 3), F(-7, 4), cnum(1, 2), cnum(-3, 1)),
          (cnum(-1, 4), cnum(3, -1), F(-2, 9), F(11, 6), cnum(2, -5), cnum(4, 3)),
          (cnum(3, 2), cnum(0, 2), F(4, 7), F(1, 3), cnum(-1, -1), cnum(2, 7))]

# ------------------------------------------------------------------- N1 -----
ok = True
for p in POINTS:
    s = chain(*p)
    ok &= all(v == ZERO for v in s[:7]) and s[7] == (closed_form(*p), F(0))
report("N1  minimal open-path formula: four exact full-series checks", ok,
       "   four nondegenerate complex points, m_0..m_6 = 0, m_7 matches")

report("N2  with J^z_BC=0, changing g_C at one point leaves m_7 unchanged",
       chain(*POINTS[1], gC=cnum(5, -2))[7] == chain(*POINTS[1])[7],
       f"   m_7 = {chain(*POINTS[1])[7][0]} either way")

# ------------------------------------------------------------------- N3 -----
p = POINTS[1]
s7 = cycle(7, tAB=p[0], tBC=p[1], jz=p[2], hz=p[3], gA=p[4], gB=p[5])
s8 = cycle(8, tAB=p[0], tBC=p[1], jz=p[2], hz=p[3], gA=p[4], gB=p[5])
report("N3  one ambient-graph spot check at order 7",
       s7[7] == s8[7] == (closed_form(*p), F(0))
       and all(v == ZERO for v in s7[:7]) and all(v == ZERO for v in s8[:7]),
       f"   open path, C_7 and C_8 all give m_7 = {s7[7][0]}")

# ------------------------------------------------------------------- N4 -----
base = cycle(7)[7]
singles = {
    "t_AB": cycle(7, tAB=ZERO),
    "g_A": cycle(7, gA=ZERO), "g_B": cycle(7, gB=ZERO),
    "J^z_AB": cycle(7, jz=F(0)), "h^z_B": cycle(7, hz=F(0)),
    "t_BC": cycle(7, tBC=cnum(0)),
}
report("N4  six deletion checks at the base point",
       base == (F(1, 23040), F(0)) and all(s[7] == ZERO for s in singles.values()),
       "   removed one at a time: " + ", ".join(singles))

report("N5  two scaling checks at the base point",
       cycle(7, tBC=cnum(2))[7][0] / base[0] == 4
       and cycle(7, other=F(2))[7][0] / base[0] == 1,
       "   doubling t_BC alone gives x4; doubling all cycle amplitudes"
       " outside AB and BC together gives x1")

# ------------------------------------------------------------------- N6 -----
axial6 = modular_series(
    Lattice([1] * 6),
    [(i, (i + 1) % 6, cnum(1, 1) if i == 0 else cnum(1),
      F(1) if i == 0 else F(0)) for i in range(6)],
    [F(0), F(1), F(0), F(0), F(0), F(0)], (0, 1, 2), 7)
report("N6  C_6: the total beta^7 onset coefficient is contaminated",
       axial6[7] == (F(1, 2048), F(0)) and cycle(6)[7] == (F(49, 92160), F(0)),
       f"   {axial6[7][0]} -> 49/92160 = 1/2048 + 1/23040")

report("N7  exact C_7 point: the total series starts before the Wilson onset",
       all(v == ZERO for v in cycle(7)[:7])
       and cycle(7)[7] == (F(1, 23040), F(0)),
       "   m_7 = 1/23040 while the cycle onset sits at order 8")

print("\nN6-N7 show that the axial hypothesis cannot be dropped from the\n"
      "universal theorem. The symbolic t5\n"
      "certificate, not these finite spot checks, proves the projected\n"
      "open-path contribution and its spectator independence.")
