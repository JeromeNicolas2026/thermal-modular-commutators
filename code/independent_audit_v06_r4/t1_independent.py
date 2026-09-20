#!/usr/bin/env python3
"""Independent audit of the v06 unicyclic onset law."""

import time
from fractions import Fraction as F
from indep_engine import Lattice, kappa_of, modular_series, cnum


def exact_W(ts):
    r, i = F(1), F(0)
    for (a, b) in ts:
        r, i = r * a - i * b, r * b + i * a
    return (r, i)


def run(name, two_spins, edges, fields, marked, cycle_sites, cycle_ts, K,
        expect_zero_lead=False):
    t0 = time.time()
    lat = Lattice(two_spins)
    ser = modular_series(lat, edges, fields, marked, K)
    W = exact_W(cycle_ts)
    n = len(cycle_sites)
    prod_k = F(1)
    for x in cycle_sites:
        prod_k *= kappa_of(two_spins[x])
    hB = F(fields[marked[1]])
    target = 2 * F((-1) ** n) * hB * prod_k * W[1]
    low = ser[:K]
    lead = ser[K]
    ok_low = all(v == (F(0), F(0)) for v in low)
    ok_lead = (lead == (target, F(0)))
    print(f"--- {name}")
    print(f"    dims={lat.dims}  n={n}  W={W[0]}{'+' if W[1]>=0 else '-'}{abs(W[1])}i"
          f"  prod kappa={prod_k}  h_B={hB}")
    print(f"    m_1..m_{K-1} all zero : {ok_low}")
    print(f"    m_{K} exact           : {lead[0]}"
          + (f" + {lead[1]}i" if lead[1] != 0 else ""))
    print(f"    predicted             : {target}")
    print(f"    status                : {'PASS' if (ok_low and ok_lead) else 'FAIL'}"
          f"  ({time.time()-t0:.1f}s)")
    assert ok_low and ok_lead, name
    return ser


# =========================================================== X1: replicate E1
run("X1  cross-engine replication of certificate E1 (triangle, spins 1/2,1,3/2)",
    two_spins=[1, 2, 3],
    edges=[(0, 1, cnum(1, 2), F(2, 3)),
           (1, 2, cnum(2, -1), F(-5, 7)),
           (2, 0, cnum(-1, 3), F(7, 4))],
    fields=[F(-3, 5), F(11, 6), F(13, 9)],
    marked=(0, 1, 2), cycle_sites=[0, 1, 2],
    cycle_ts=[cnum(1, 2), cnum(2, -1), cnum(-1, 3)], K=4)

# ============================================== X2: new nonconsecutive 4-cycle
# 4-cycle 0-1-2-3-0, spins (1/2, 1, 1/2, 3/2); marked A=0, B=1, C=3 (not
# consecutive: the arc B->C has length 2).  Generic Jz and fields everywhere,
# amplitudes of pairwise different moduli.
run("X2  NEW: nonconsecutive 4-cycle, spins (1/2,1,1/2,3/2), generic Jz+fields",
    two_spins=[1, 2, 1, 3],
    edges=[(0, 1, cnum(3, -1), F(5, 9)),
           (1, 2, cnum(-2, 5), F(-7, 11)),
           (2, 3, cnum(1, 1), F(4, 13)),
           (3, 0, cnum(2, 3), F(-9, 8))],
    fields=[F(7, 5), F(-4, 3), F(11, 7), F(-6, 13)],
    marked=(0, 1, 3), cycle_sites=[0, 1, 2, 3],
    cycle_ts=[cnum(3, -1), cnum(-2, 5), cnum(1, 1), cnum(2, 3)], K=5)

# ==================== X3: new nonconsecutive 5-cycle + tree at an interior arc
# 5-cycle 0-1-2-3-4-0 with a leaf 5 grafted on site 2, which is *interior* to
# the arc B->C.  Marked A=0, B=1, C=3.
run("X3  NEW: nonconsecutive 5-cycle + leaf grafted inside the arc B->C",
    two_spins=[1, 1, 1, 1, 1, 1],
    edges=[(0, 1, cnum(2, 1), F(3, 7)),
           (1, 2, cnum(-1, 3), F(-5, 6)),
           (2, 3, cnum(4, -1), F(2, 9)),
           (3, 4, cnum(1, -2), F(-8, 15)),
           (4, 0, cnum(3, 2), F(7, 12)),
           (2, 5, cnum(-3, 4), F(11, 10))],
    fields=[F(2, 3), F(-9, 5), F(4, 7), F(-1, 6), F(8, 11), F(-5, 13)],
    marked=(0, 1, 3), cycle_sites=[0, 1, 2, 3, 4],
    cycle_ts=[cnum(2, 1), cnum(-1, 3), cnum(4, -1), cnum(1, -2), cnum(3, 2)],
    K=6)

# ================================ X4: spin 2 on the cycle (beyond their range)
run("X4  NEW: triangle with a spin-2 site (kappa=2), generic Jz+fields",
    two_spins=[1, 2, 4],
    edges=[(0, 1, cnum(1, -3), F(-2, 5)),
           (1, 2, cnum(2, 2), F(6, 7)),
           (2, 0, cnum(-1, 1), F(3, 4))],
    fields=[F(-7, 6), F(5, 4), F(9, 11)],
    marked=(0, 1, 2), cycle_sites=[0, 1, 2],
    cycle_ts=[cnum(1, -3), cnum(2, 2), cnum(-1, 1)], K=4)
