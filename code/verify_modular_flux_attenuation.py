#!/usr/bin/env python3
"""Verify the thermal ring-length attenuation reported in the manuscript.

The exact quantity ``r_P(a)`` is evaluated from its one-dimensional integral.
The modular commutator is evaluated independently with the Jordan--Wigner
reconstruction in ``verify_modular_flux_ring_jw.py``.  The longest calculation
uses n=70 only where the signal remains well resolved in double precision.
Every numerical claim printed by this script is protected by an assertion.
"""

from math import log, pi

import numpy as np
from scipy.integrate import quad

from verify_modular_flux_ring_jw import jw_M_np


def parity_factor(a):
    """Bulk fermion-parity attenuation factor for a=beta*J_perp."""
    if a <= 0:
        raise ValueError("a must be positive")
    integral, error = quad(
        lambda k: np.log(abs(np.tanh(0.5 * a * np.cos(k)))),
        0.0,
        0.5 * np.pi,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
        limit=400,
    )
    if error > 2.0e-11:
        raise ArithmeticError("quadrature did not reach the required accuracy")
    return float(np.exp((2.0 / np.pi) * integral))


def parity_length(a):
    return 1.0 / log(1.0 / parity_factor(a))


def calibration_response(n, a):
    """M_{a,n} for t=(1,...,1,i), h_B=1, with J_perp=1."""
    transverse = [1.0] * (n - 1) + [1.0j]
    fields = [0.0] * n
    fields[1] = 1.0
    return jw_M_np(n, transverse, fields, a)


def last_size_ratio(a, nmax):
    previous = calibration_response(nmax - 1, a)
    current = calibration_response(nmax, a)
    if previous == 0:
        raise ArithmeticError("vanishing denominator in consecutive-size ratio")
    if min(abs(previous), abs(current)) < 1.0e-13:
        raise ArithmeticError("modular signal is too close to the roundoff floor")
    return abs(current / previous)


def figure_ratio_data(a_values=(1.0, 2.0, 4.0, 8.0), nmax=20):
    """Return R_n(a), n=4,...,nmax, for the manuscript figure."""
    ns = np.arange(4, nmax + 1)
    data = {}
    for a in a_values:
        responses = [calibration_response(n, a) for n in range(3, nmax + 1)]
        ratios = np.abs(np.asarray(responses[1:]) / np.asarray(responses[:-1]))
        if not np.isfinite(ratios).all():
            raise FloatingPointError("nonfinite modular size ratio")
        data[a] = ratios
    return ns, data


def run_assertions():
    reference = {
        0.5: 0.12371856660497445,
        1.0: 0.24021084850791066,
        2.0: 0.43341676993303113,
        4.0: 0.6593866624946259,
        8.0: 0.8192273444426504,
        16.0: 0.9061934094220743,
    }
    for a, target in reference.items():
        value = parity_factor(a)
        if abs(value - target) > 2.0e-13:
            raise AssertionError(f"incorrect parity integral at a={a}")

    # log r_P(a) = log(a/4) - a^2/24 + 7 a^4/3840 + O(a^6).
    for a in (0.05, 0.10):
        residual = log(parity_factor(a)) - log(a / 4.0) + a * a / 24.0
        scaled = residual / a**4
        if abs(scaled - 7.0 / 3840.0) > 2.0e-6:
            raise AssertionError("high-temperature expansion check failed")

    if abs(parity_length(128.0) / 128.0 - 2.0 / pi) > 4.0e-5:
        raise AssertionError("low-temperature length check failed")

    # Adaptive sizes keep every modular signal above the roundoff floor.
    checks = ((0.5, 13, 2.0e-10), (1.0, 18, 2.0e-10),
              (2.0, 25, 2.0e-9), (4.0, 35, 2.0e-8),
              (8.0, 50, 2.0e-8), (16.0, 70, 8.0e-8))
    rows = []
    for a, nmax, tolerance in checks:
        ratio = last_size_ratio(a, nmax)
        target = parity_factor(a)
        relative = abs(ratio - target) / target
        if relative > tolerance:
            raise AssertionError(f"large-n modular ratio check failed at a={a}")
        rows.append((a, nmax, ratio, target, relative))
    return rows


def main():
    rows = run_assertions()
    print(" a       n_max       R_n(a)          r_P(a)       relative discrepancy")
    for a, nmax, ratio, target, relative in rows:
        print(f"{a:4.1f}      {nmax:3d}    {ratio:.12f}   {target:.12f}   {relative:.2e}")
    print(f"low-temperature limit: 2/pi = {2.0 / pi:.12f}")
    print("all attenuation assertions passed")


if __name__ == "__main__":
    main()
