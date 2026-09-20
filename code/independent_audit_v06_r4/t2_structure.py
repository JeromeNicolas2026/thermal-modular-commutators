#!/usr/bin/env python3
"""Ten exact structural probes, with extensions explicitly labelled."""

import time
from fractions import Fraction as F
from indep_engine import Lattice, kappa_of, modular_series, cnum

Z = (F(0), F(0))


def series(two_spins, edges, fields, marked, K, fields_x=None):
    return modular_series(Lattice(two_spins), edges, fields, marked, K,
                          fields_x=fields_x)


def exact_prod(ts):
    """Exact Gaussian-rational product; no floating point anywhere."""
    r, i = F(1), F(0)
    for (a, b) in ts:
        r, i = r * a - i * b, r * b + i * a
    return (r, i)


def report(tag, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{extra}")
    assert cond, tag


# ---------------------------------------------------------------- Y1 spin 5/2
t0 = time.time()
ts = [cnum(1, 2), cnum(3, -1), cnum(-2, 1)]
ed = [(0, 1, ts[0], F(3, 5)), (1, 2, ts[1], F(-4, 7)), (2, 0, ts[2], F(9, 8))]
h = [F(5, 3), F(-7, 4), F(2, 9)]
s = series([1, 5, 2], ed, h, (0, 1, 2), 4)
W = exact_prod(ts)
pred = 2 * F(-1) * h[1] * kappa_of(1) * kappa_of(5) * kappa_of(2) * W[1]
report("Y1  spin-5/2 site on the triangle (kappa=35/12)",
       all(v == Z for v in s[:4]) and s[4] == (pred, F(0)),
       f"   m_4 = {s[4][0]} = predicted {pred}   ({time.time()-t0:.1f}s)")

# ------------------------------------------------- Y2 six-cycle nonconsecutive
t0 = time.time()
tsix = [cnum(2, 1), cnum(-1, 2), cnum(3, 1), cnum(1, -1), cnum(-2, 3), cnum(1, 2)]
ed = [(i, (i + 1) % 6, tsix[i], jz) for i, jz in
      enumerate([F(1, 3), F(-2, 5), F(3, 7), F(-4, 9), F(5, 11), F(-6, 13)])]
h = [F(2, 7), F(-5, 3), F(4, 9), F(-1, 8), F(7, 6), F(-3, 11)]
s = series([1, 2, 1, 1, 1, 1], ed, h, (0, 2, 4), 7)
W = exact_prod(tsix)
pk = F(1)
for q in [1, 2, 1, 1, 1, 1]:
    pk *= kappa_of(q)
pred = 2 * h[2] * pk * W[1]
report("Y2  six-cycle, spin-1 site, nonconsecutive (A,B,C)=(0,2,4)",
       all(v == Z for v in s[:7]) and s[7] == (pred, F(0)),
       f"   m_7 = {s[7][0]} = predicted {pred}   ({time.time()-t0:.1f}s)")

# ------------------------------------------- Y3 field only at an interior site
t0 = time.time()
t4 = [cnum(1, 1), cnum(2, -1), cnum(-1, 3), cnum(1, 2)]
ed = [(i, (i + 1) % 4, t4[i], jz) for i, jz in
      enumerate([F(2, 3), F(-3, 5), F(4, 7), F(-5, 9)])]
s = series([1, 1, 1, 1], ed, [F(0), F(0), F(11, 6), F(0)], (0, 1, 3), 5)
report("Y3  only h at an interior arc vertex (h_A=h_B=h_C=0): onset must vanish",
       all(v == Z for v in s[:6]), f"   ({time.time()-t0:.1f}s)")

# ------------------------------------------------------- Y4 h_A, h_C but no h_B
t0 = time.time()
s = series([1, 1, 1, 1], ed, [F(7, 5), F(0), F(0), F(-4, 3)], (0, 1, 3), 5)
report("Y4  h_A and h_C nonzero, h_B = 0: onset must vanish",
       all(v == Z for v in s[:6]), f"   ({time.time()-t0:.1f}s)")

# ------------------------------------------ Y5 exchange of A and C flips sign
t0 = time.time()
hh = [F(7, 5), F(-4, 3), F(11, 7), F(-6, 13)]
sABC = series([1, 1, 1, 1], ed, hh, (0, 1, 3), 5)
sCBA = series([1, 1, 1, 1], ed, hh, (3, 1, 0), 5)
report("Y5  exchanging A and C reverses the sign",
       sABC[5] == (-sCBA[5][0], -sCBA[5][1]) and sABC[5] != Z,
       f"   m_5 = {sABC[5][0]} vs {sCBA[5][0]}   ({time.time()-t0:.1f}s)")

# ------------------------------------- Y6 real Wilson loop: onset zero, not all
t0 = time.time()
treal = [cnum(1, 1), cnum(1, -1), cnum(0, 3), cnum(0, 2)]   # W = -12, real
edr = [(i, (i + 1) % 4, treal[i], jz) for i, jz in
       enumerate([F(2, 3), F(-3, 5), F(4, 7), F(-5, 9)])]
s = series([1, 1, 1, 1], edr, hh, (0, 1, 3), 7)
assert exact_prod(treal)[1] == 0 and exact_prod(treal)[0] != 0
report("Y6  complex amplitudes with a REAL Wilson loop: onset vanishes",
       s[5] == Z, f"   W = {exact_prod(treal)[0]} real; m_5 = 0"
       f"   ({time.time()-t0:.1f}s)")

# ---------------------------- Y7 a big spin on a grafted tree must not show up
t0 = time.time()
tri = [cnum(1, 2), cnum(2, -1), cnum(-1, 3)]
base = [(0, 1, tri[0], F(2, 5)), (1, 2, tri[1], F(-3, 7)), (2, 0, tri[2], F(5, 8))]
hb = [F(3, 4), F(-9, 5), F(7, 11)]
s_no = series([1, 1, 1], base, hb, (0, 1, 2), 4)
leaf = base + [(1, 3, cnum(-2, 5), F(13, 6))]
s_leaf5 = series([1, 1, 1, 5], leaf, hb + [F(-8, 3)], (0, 1, 2), 4)
s_leaf1 = series([1, 1, 1, 1], leaf, hb + [F(-8, 3)], (0, 1, 2), 4)
report("Y7  spin-5/2 leaf grafted at B leaves the onset untouched",
       s_no[4] == s_leaf5[4] == s_leaf1[4] and s_no[4] != Z,
       f"   m_4 = {s_no[4][0]}   ({time.time()-t0:.1f}s)")

# ---------------------------------------------------- Y8 cactus counterexample
t0 = time.time()
# two triangles ABC and BDE sharing only B; h_B = 1 only; Jz = 0;
# unit-modulus amplitudes with W_ABC = i and W_BDE = 1.
cact = [(0, 1, cnum(1, 0), 0), (1, 2, cnum(1, 0), 0), (2, 0, cnum(0, 1), 0),
        (1, 3, cnum(1, 0), 0), (3, 4, cnum(1, 0), 0), (4, 1, cnum(1, 0), 0)]
s = series([1, 1, 1, 1, 1], cact, [F(0), F(1), F(0), F(0), F(0)], (0, 1, 2), 7)
report("Y8  cactus of two triangles sharing B: reproduces the v06 value at order 7\n        (this reproduces the number only; the sector itself is isolated by the\n         phase-variation certificate of the v06 bundle, which is stronger)",
       s[7] == (F(-1, 768), F(0)),
       f"   m_7 = {s[7][0]} (claimed -1/768)   ({time.time()-t0:.1f}s)")

# ------------------------------------------------------- Y9 theta graph, girth
t0 = time.time()
# theta graph: the two 4-cycles 0-1-2-3-0 and 0-1-2-4-0 both carry A,B,C.
# They are NOT internally disjoint: they share the path 0-1-2.  Marked (0,1,2).
th = [(0, 1, cnum(1, 0), 0), (1, 2, cnum(1, 0), 0),
      (2, 3, cnum(1, 0), 0), (3, 0, cnum(0, 1), 0),
      (2, 4, cnum(1, 0), 0), (4, 0, cnum(0, 1), 0)]
s = series([1, 1, 1, 1, 1], th, [F(0), F(1), F(0), F(0), F(0)], (0, 1, 2), 5)
report("Y9  theta specialization: the exact value agrees with 2 x 1/128",
       s[5] == (F(1, 64), F(0)),
       f"   m_5 = {s[5][0]} = 2 x 1/128   ({time.time()-t0:.1f}s)")

# ------------------------------- Y10 transverse field does NOT move the onset
# (this probe was written expecting the opposite; the result is the finding.)
t0 = time.time()
s_ax = series([1, 1, 1], base, [F(0), F(1), F(0)], (0, 1, 2), 5)
s_tx = series([1, 1, 1], base, [F(0), F(1), F(0)], (0, 1, 2), 5,
              fields_x=[F(1, 2), F(-1, 3), F(2, 5)])
report("Y10 triangle specialization: chosen real transverse fields leave"
       " m_0..m_5 unchanged",
       s_ax[:6] == s_tx[:6] and s_ax[4] != Z,
       f"   m_4 = {s_ax[4][0]} with and without transverse fields"
       f"   ({time.time()-t0:.1f}s)")
print("\nAll structural checks passed.")
