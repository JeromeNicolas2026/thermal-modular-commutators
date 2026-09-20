#!/usr/bin/env python3
"""Exact nonconsecutive-site certificates using the v5 SymPy series engine.

Default: exact n=4 and n=5 checks.  Pass --include-n6 for the exact n=6
check (about 7.5 minutes on the current runner).  No manuscript file is
modified.
"""

import argparse
import importlib.util
from pathlib import Path
import time

import sympy as sp


ENGINE = (Path(__file__).parent / "work_v5" / "modular_flux_PRB_v5_bundle"
          / "verify_modular_flux_extensions.py")


def load_engine():
    spec = importlib.util.spec_from_file_location("v5_extensions", ENGINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_numeric_parameters(v, n, triple, expected):
    A, B, C = triple
    t = [sp.Integer(1)] * (n - 1) + [sp.I]
    h = [sp.Integer(0)] * n
    h[B] = sp.Integer(1)
    before = time.time()
    got = v.M_series(n, t, [sp.Integer(0)] * n, h, n + 1,
                     keepA=(A, B), keepB=(B, C))
    elapsed = time.time() - before
    target = [sp.Integer(0)] * (n + 1) + [expected]
    assert got == target, (n, triple, got, target)
    print(f"n={n}, (A,B,C)={triple}, elapsed={elapsed:.1f}s: {got}")


def run_symbolic_n4(v):
    h = sp.symbols("h0:4", real=True)
    t = [sp.Integer(1), sp.Integer(1), sp.Integer(1), sp.I]
    jz = [sp.Rational(3, 5), sp.Rational(-7, 4),
          sp.Rational(11, 9), sp.Rational(-2, 7)]
    cases = [((0, 1, 3), h[1] / 128),
             ((0, 2, 3), h[2] / 128),
             ((0, 3, 1), -h[3] / 128)]
    for (A, B, C), expected in cases:
        before = time.time()
        got = v.M_series(4, t, jz, h, 5,
                         keepA=(A, B), keepB=(B, C))
        target = [sp.Integer(0)] * 5 + [expected]
        assert got == target, ((A, B, C), got, target)
        print(f"n=4 symbolic, (A,B,C)={(A,B,C)}, elapsed={time.time()-before:.1f}s: {got}")

    # Independent Gaussian-rational coupling point: Im(prod t)=15/2.
    tg = [1 + sp.I, 2 - sp.I, sp.Rational(3, 2), -1 + 2 * sp.I]
    A, B, C = 0, 2, 3
    before = time.time()
    got = v.M_series(4, tg, jz, h, 5,
                     keepA=(A, B), keepB=(B, C))
    target = [sp.Integer(0)] * 5 + [sp.Rational(15, 256) * h[B]]
    assert got == target, (got, target)
    print(f"n=4 generic t, ImW=15/2, elapsed={time.time()-before:.1f}s: {got}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-n6", action="store_true")
    args = ap.parse_args()
    v = load_engine()
    run_symbolic_n4(v)
    run_numeric_parameters(v, 5, (0, 2, 4), -sp.Rational(1, 512))
    if args.include_n6:
        run_numeric_parameters(v, 6, (0, 2, 4), sp.Rational(1, 2048))
