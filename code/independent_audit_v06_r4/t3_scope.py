#!/usr/bin/env python3
"""Scope probes: how far the hypotheses of the unicyclic theorem can be relaxed.

Three narrowly stated questions, all answered at exact specializations:
  (a) in THIS C_4 specialization, where only the winding transverse channel
      is activated, do transverse fields move the onset?
      (they do in general: see t4_transverse_boundary.py)
  (b) do three K4 points agree with the marked-triangle girth prediction?
  (c) does one witness rule out a uniform O(beta^{n+3}) remainder?
"""

from fractions import Fraction as F
from indep_engine import (Lattice, kappa_of, cnum, modular_series, ZERO)


def modular_general(sp, ed, h, marked, K, gx=None):
    """Thin convenience wrapper around the single engine pipeline."""
    return modular_series(Lattice(sp), ed, h, marked, K, gx=gx)


def report(tag, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{extra}")
    assert cond, tag


# ---------------------------------------------------- (a) transverse fields --
sp = [1, 2, 1, 3]
t = [cnum(3, -1), cnum(-2, 5), cnum(1, 1), cnum(2, 3)]
ed = [(0, 1, t[0], F(5, 9)), (1, 2, t[1], F(-7, 11)),
      (2, 3, t[2], F(4, 13)), (3, 0, t[3], F(-9, 8))]
h = [F(7, 5), F(-4, 3), F(11, 7), F(-6, 13)]
pk = F(1)
for q in sp:
    pk *= kappa_of(q)
pred = 2 * h[1] * pk * F(-22)                      # Im W = -22, n = 4

axial = modular_general(sp, ed, h, (0, 1, 3), 6)
real_g = modular_general(sp, ed, h, (0, 1, 3), 6,
                         gx=[cnum(2), cnum(-3), cnum(1), cnum(-2)])
real_h = modular_series(Lattice(sp), ed, h, (0, 1, 3), 6,
                        fields_x=[F(2), F(-3), F(1), F(-2)])
try:
    modular_series(Lattice(sp), ed, h, (0, 1, 3), 1,
                   fields_x=[F(2), F(-3), F(1), F(-2)],
                   gx=[cnum(2), cnum(-3), cnum(1), cnum(-2)])
    interfaces_exclusive = False
except ValueError:
    interfaces_exclusive = True
cplx_x = modular_general(sp, ed, h, (0, 1, 3), 6,
                         gx=[cnum(2, -1), cnum(-3, 2), cnum(1, 4), cnum(-2, -3)])
report("A1  C_4 specialization: onset unchanged by real transverse fields",
       axial[:7] == real_g[:7] == real_h[:7] and interfaces_exclusive
       and axial[5] == (pred, F(0)),
       "   fields_x and real gx agree exactly and are mutually exclusive")
report("A2  C_4 specialization: onset unchanged by complex transverse fields",
       axial[:7] == cplx_x[:7],
       f"   m_5 = {axial[5][0]} = predicted {pred};  m_6 also unchanged")

qb = [1, 1, 1, 1]
tq = [cnum(1, 1), cnum(2, -1), cnum(-1, 3), cnum(1, 2)]
edq = [(i, (i + 1) % 4, tq[i], F(0)) for i in range(4)]
hq = [F(0), F(1), F(0), F(0)]
a = modular_general(qb, edq, hq, (0, 1, 2), 7)
b = modular_general(qb, edq, hq, (0, 1, 2), 7, gx=[cnum(1, 1), ZERO, ZERO, ZERO])
first = next(k for k in range(8) if a[k] != b[k])
report("A3  C_4 specialization: only the winding channel is activated,"
       " first change at order 7",
       first == 7,
       f"   first differing order = {first}; generic winding and local orders"
       " are tied on C_4,\n        but this specialization has no adjacent"
       " transverse-field pair -- see t4")

# --------------------------------------------------- (b) girth on K4 --------
def k4(tAB, tBC, tCA, tAD, tBD, tCD, jz=F(0)):
    e = [(0, 1, tAB, jz), (1, 2, tBC, jz), (2, 0, tCA, jz),
         (0, 3, tAD, jz), (1, 3, tBD, jz), (2, 3, tCD, jz)]
    return modular_series(Lattice([1, 1, 1, 1]), e,
                          [F(0), F(1), F(0), F(0)], (0, 1, 2), 4)

runs = [k4(cnum(1), cnum(1), cnum(0, 1), cnum(1), cnum(1), cnum(1)),
        k4(cnum(1), cnum(1), cnum(0, 1), cnum(0, 1), cnum(1, 2), cnum(-1, 1)),
        k4(cnum(1), cnum(1), cnum(0, 1), cnum(1), cnum(1), cnum(1), jz=F(3, 7))]
report("B1  K4: three exact points agree with the marked-triangle prediction",
       all(all(v == ZERO for v in r[1:4]) and r[4] == (F(-1, 32), F(0))
           for r in runs),
       "   m_4 = -1/32 for two off-target edge assignments;"
       " baseline repeated with uniform Jz=3/7")

# --------------------------------------------------- (c) sharpness ----------
report("C1  one C_4 witness rules out a uniform O(beta^{n+3}) remainder",
       axial[6] != ZERO, f"   m_6 = {axial[6][0]} != 0")

print("\nAll scope probes passed.")
