#!/usr/bin/env python3
"""Generate both figures for the modular-flux manuscript.

The two purely transverse models are

  triangle: t=(1,1,i),     h=(0,0.7,0),
  4-ring:   t=(1,1,1,i),   h=(0,0.7,0,0).

The reduced modular Hamiltonians act on AB=(0,1) and BC=(1,2).  The site D
of the four-site ring is outside AB union BC and is traced out.

The second figure uses the Jordan--Wigner reconstruction for homogeneous
rings and compares consecutive-size ratios with the exact bulk
fermion-parity-string attenuation factor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verify_modular_flux_attenuation import (
    figure_ratio_data,
    parity_factor,
    parity_length,
)


OUT = Path(__file__).resolve().parent.parent


def local_operator(matrix, site, nsites):
    identity = np.eye(2, dtype=complex)
    result = np.array([[1.0 + 0.0j]])
    for current in range(nsites):
        result = np.kron(result, matrix if current == site else identity)
    return result


def ring_hamiltonian(transverse, fields):
    nsites = len(transverse)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    raising = [local_operator(sx + 1j * sy, site, nsites)
               for site in range(nsites)]
    lowering = [local_operator(sx - 1j * sy, site, nsites)
                for site in range(nsites)]
    zops = [local_operator(sz, site, nsites) for site in range(nsites)]
    matrix = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site, coupling in enumerate(transverse):
        neighbor = (site + 1) % nsites
        matrix += 0.5 * (
            coupling * raising[site] @ lowering[neighbor]
            + np.conjugate(coupling) * lowering[site] @ raising[neighbor]
        )
    for site, field in enumerate(fields):
        matrix -= field * zops[site]
    return matrix


def gibbs_from_spectrum(eigenvalues, eigenvectors, beta):
    weights = np.exp(-beta * (eigenvalues - eigenvalues.min()))
    return (eigenvectors * weights) @ eigenvectors.conj().T / weights.sum()


def reduced_density(rho, keep, nsites):
    keep = list(keep)
    drop = [site for site in range(nsites) if site not in keep]
    permutation = keep + drop
    tensor = rho.reshape([2] * (2 * nsites)).transpose(
        permutation + [site + nsites for site in permutation]
    )
    dim_keep = 2**len(keep)
    dim_drop = 2**len(drop)
    tensor = tensor.reshape(dim_keep, dim_drop, dim_keep, dim_drop)
    return np.trace(tensor, axis1=1, axis2=3)


def hermitian_log(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    if values.min() <= 0:
        raise FloatingPointError("reduced state is not positive definite")
    return (vectors * np.log(values)) @ vectors.conj().T


def modular_commutator(rho, nsites):
    log_ab = hermitian_log(reduced_density(rho, (0, 1), nsites))
    log_bc = hermitian_log(reduced_density(rho, (1, 2), nsites))
    full_ab = np.kron(log_ab, np.eye(2 ** (nsites - 2), dtype=complex))
    full_bc = np.kron(
        np.eye(2, dtype=complex),
        np.kron(log_bc, np.eye(2 ** (nsites - 3), dtype=complex)),
    )
    commutator = full_ab @ full_bc - full_bc @ full_ab
    return float(np.real(1j * np.trace(rho @ commutator)))


def thermal_curve(hamiltonian, betas):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return np.array([
        modular_commutator(
            gibbs_from_spectrum(eigenvalues, eigenvectors, beta),
            int(np.log2(hamiltonian.shape[0])),
        )
        for beta in betas
    ])


def thermal_network_figure():
    triangle = ring_hamiltonian([1.0, 1.0, 1.0j], [0.0, 0.7, 0.0])
    square = ring_hamiltonian([1.0, 1.0, 1.0, 1.0j], [0.0, 0.7, 0.0, 0.0])

    beta_full = np.linspace(0.0, 40.0, 401)
    triangle_full = thermal_curve(triangle, beta_full)
    square_full = thermal_curve(square, beta_full)

    beta_triangle = np.linspace(0.025, 0.8, 100)
    triangle_short = thermal_curve(triangle, beta_triangle)
    beta_square = np.linspace(0.05, 0.9, 100)
    square_short = thermal_curve(square, beta_square)

    m4_triangle = -0.7 / 32
    m6_triangle = (4 * 0.7**3 + 8 * 0.7 + 6 * 0.7) / 1536
    m5_square = 0.7 / 128

    if not all(np.isfinite(values).all() for values in
               (triangle_full, square_full, triangle_short, square_short)):
        raise RuntimeError("nonfinite value in a thermal curve")
    triangle_expected = m4_triangle + beta_triangle[0]**2 * m6_triangle
    if abs(triangle_short[0] / beta_triangle[0]**4
           - triangle_expected) > 5.0e-9:
        raise AssertionError("triangle curve does not approach m4+beta^2*m6")
    if abs(square_short[0] / beta_square[0]**5 - m5_square) > 1.0e-5:
        raise AssertionError("four-site curve does not approach m5")
    if abs(square_full[-1] - 0.3145980200) > 2.0e-8:
        raise AssertionError("unexpected four-site low-temperature value")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))

    axes[0].plot(beta_full, triangle_full, color="#1f5a94", linewidth=1.7,
                 label="triangle")
    axes[0].plot(beta_full, square_full, color="#a64b32", linewidth=1.7,
                 label="four-site ring")
    axes[0].axhline(0.0, color="0.75", linewidth=0.7)
    axes[0].axhline(0.3146212191, color="#a64b32", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel(r"$\beta |t_{AB}|$")
    axes[0].set_ylabel(r"$\mathfrak{M}_\beta(AB,BC)$")
    axes[0].set_title("(a) Thermal crossover", loc="left")
    axes[0].legend(frameon=False, loc="lower right")

    triangle_ratio = triangle_short / beta_triangle**4
    axes[1].plot(beta_triangle, triangle_ratio, color="#1f5a94", linewidth=1.7,
                 label="exact")
    axes[1].axhline(m4_triangle, color="#a3312e", linestyle="--",
                    linewidth=1.1, label=r"$m_4$")
    axes[1].plot(beta_triangle, m4_triangle + m6_triangle * beta_triangle**2,
                 color="#d18b20", linestyle=":", linewidth=1.7,
                 label=r"$m_4+\beta^2m_6$")
    axes[1].set_xlabel(r"$\beta |t_{AB}|$")
    axes[1].set_ylabel(r"$\mathfrak{M}_\beta/\beta^4$")
    axes[1].set_title("(b) Triangle", loc="left")
    axes[1].legend(frameon=False)

    square_ratio = square_short / beta_square**5
    axes[2].plot(beta_square, square_ratio, color="#a64b32", linewidth=1.7,
                 label="exact")
    axes[2].axhline(m5_square, color="#2d7f5e", linestyle="--",
                    linewidth=1.2, label=r"$m_5^{\mathrm{ring}}$")
    axes[2].set_xlabel(r"$\beta |t_{AB}|$")
    axes[2].set_ylabel(r"$\mathfrak{M}_\beta/\beta^5$")
    axes[2].set_title("(c) Four-site ring", loc="left")
    axes[2].legend(frameon=False)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "modular_flux_thermal_networks.pdf", bbox_inches="tight")
    fig.savefig(OUT / "modular_flux_thermal_networks.png", dpi=260,
                bbox_inches="tight")
    plt.close(fig)


def ring_attenuation_figure():
    # Long-size checks through n=70 belong to the dedicated verification
    # script.  Figure generation evaluates only the displayed n <= 20 data.
    a_values = (1.0, 2.0, 4.0, 8.0)
    reference = {
        1.0: 0.24021084850791066,
        2.0: 0.43341676993303113,
        4.0: 0.6593866624946259,
        8.0: 0.8192273444426504,
    }
    for a, target in reference.items():
        if abs(parity_factor(a) - target) > 2.0e-13:
            raise AssertionError(f"incorrect parity integral at a={a}")
    ns, ratio_data = figure_ratio_data(a_values=a_values, nmax=20)
    tolerances = {1.0: 1.0e-10, 2.0: 1.0e-9,
                  4.0: 2.0e-7, 8.0: 2.0e-4}
    for a in a_values:
        target = parity_factor(a)
        relative = abs(ratio_data[a][-1] - target) / target
        if relative > tolerances[a]:
            raise AssertionError(f"plotted modular ratios do not approach r_P at a={a}")

    colors = ("#315f9b", "#2d8064", "#bc6c25", "#8f3d56")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for a, color in zip(a_values, colors):
        axes[0].plot(ns, ratio_data[a], marker="o", markersize=2.6,
                     linewidth=1.15, color=color, label=rf"$a={a:g}$")
        axes[0].axhline(parity_factor(a), color=color, linestyle=":",
                        linewidth=0.9)
    axes[0].set_xlabel(r"ring size $n$")
    axes[0].set_ylabel(r"$R_n(a)$")
    axes[0].set_xlim(3.7, 20.3)
    axes[0].set_xticks((4, 8, 12, 16, 20))
    axes[0].set_title("(a) Modular size ratios", loc="left")
    axes[0].legend(frameon=False, ncol=2)

    grid = np.geomspace(0.5, 128.0, 180)
    scaled_lengths = np.array([parity_length(a) / a for a in grid])
    if not np.isfinite(scaled_lengths).all():
        raise FloatingPointError("nonfinite parity-string length")
    if abs(scaled_lengths[-1] - 2.0 / np.pi) > 4.0e-5:
        raise AssertionError("low-temperature asymptote is not resolved")
    axes[1].semilogx(grid, scaled_lengths, color="#315f9b", linewidth=1.7)
    axes[1].axhline(2.0 / np.pi, color="#a3312e", linestyle="--",
                    linewidth=1.1, label=r"$2/\pi$")
    axes[1].set_xlabel(r"$a=\beta J_\perp$")
    axes[1].set_ylabel(r"$\xi_{\rm P}(a)/a$")
    axes[1].set_title("(b) Exact parity-string length", loc="left")
    axes[1].legend(frameon=False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "modular_flux_ring_attenuation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "modular_flux_ring_attenuation.png", dpi=260,
                bbox_inches="tight")
    plt.close(fig)


def main():
    thermal_network_figure()
    ring_attenuation_figure()


if __name__ == "__main__":
    main()
