#!/usr/bin/env python3
"""Smoke test: the gauge trick, the su(2) identities, and the v5 qubit triangle."""

import time
from fractions import Fraction as F
from indep_engine import (Lattice, check_su2, kappa_of, modular_series,
                          cnum, cmul)


def show(series):
    return [f"{v[0]}" + ("" if v[1] == 0 else f" + {v[1]}i") for v in series]


for two_s in (1, 2, 3, 4, 5):
    k = check_su2(two_s)
    print(f"  spin {F(two_s,2)}: su(2) ok in integer gauge, kappa = {k}")

# --- v5 qubit triangle: M = -(beta^4 h_B/32) Im(t_AB t_BC t_CA) + O(beta^5)
t0 = time.time()
lat = Lattice([1, 1, 1])
edges = [(0, 1, cnum(1, 2), 0), (1, 2, cnum(2, -1), 0), (2, 0, cnum(-1, 3), 0)]
fields = [F(0), F(1), F(0)]
ser = modular_series(lat, edges, fields, (0, 1, 2), 4)
# W = (1+2i)(2-i)(-1+3i), evaluated in exact Gaussian rationals.
W = cmul(cmul(cnum(1, 2), cnum(2, -1)), cnum(-1, 3))
print(f"\n  qubit triangle, h_B=1, Jz=0:  W = {W[0]} + {W[1]}i")
print("  series m_0..m_4 =", show(ser))
print("  v5 prediction m_4 = -Im(W)/32 =", -W[1] / 32)
assert all(v == (F(0), F(0)) for v in ser[:4]), ser[:4]
assert ser[4] == (-W[1] / 32, F(0)), ser[4]
print(f"  PASS  ({time.time()-t0:.1f}s)")
