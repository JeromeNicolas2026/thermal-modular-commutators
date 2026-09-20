# Thermal modular commutators

This repository contains the source, preprint, figures, analytic proof
modules, and validation code for

> Jérôme Nicolas, "Thermal modular commutators as joint probes of gauge flux
> and broken time-reversal symmetry in axial spin networks," version 0.6.0.

Version 0.6.0 is archived on Zenodo under the version-specific DOI
[10.5281/zenodo.22860690](https://doi.org/10.5281/zenodo.22860690).

## Main exact result

For a finite, simple, connected, unicyclic axial spin network whose unique
cycle has length `n`, with arbitrary irreducible spins, longitudinal
exchanges, longitudinal fields, and grafted trees, the manuscript proves

```text
M_beta(AB,BC)
  = 2 (-1)^n beta^(n+1) h_B
      product_(x in gamma) [s_x(s_x+1)/3]
      Im(W_gamma)
    + O(beta^(n+2)).
```

The three marked cycle sites need not be consecutive. The displayed term is
the leading admissible coefficient. No nonzero-coupling assumption is made.

The manuscript also proves an ambient-independent primitive coefficient for
each simple cycle and a shortest-cycle sum at the first admissible order. It
does not assert an all-orders cycle decomposition.

## Manuscript

- `modular_flux_PRB_v06.tex`  requires REVTeX 4.2.
- `modular_flux_preprint_v06.tex` is the portable article-class source.
- `modular_flux_preprint_v06.pdf` is the rendered portable preprint.
- `unicyclic_proof_section.tex` contains the complete proof of the
  arbitrary-spin unicyclic onset theorem.
- `cycle_axial_sections.tex` contains the primitive-cycle results, the exact
  transverse-field counterexample, and the certified boundary of the axial
  theorem.

Build the portable preprint with

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error modular_flux_preprint_v06.tex
```

## Reproducibility

The pinned environment for the final release pass is recorded in
`requirements.txt`. Create a clean Python 3.12 environment and install it with

```bash
python3 -m pip install -r requirements.txt
```

Run the standard validation suite from the repository root with

```bash
python3 run_all.py
```

Run the longer certificate variants with

```bash
python3 run_all.py --full
```

The standard-library audit can also be run independently with

```bash
python3 code/independent_audit_v06_r4/run_all.py
```

The audit is an independent implementation of the formal-series calculation.
It is not an independent method. Its full revision history, including the
retracted transverse-field conjecture, is retained in
`code/independent_audit_v06_r4/README_independent_audit.md`.

The primary suite and its individual commands are documented in
`code/primary_validation/README.md`. The two detailed validation reports are
in `notes`.

To regenerate the two figures in the repository root, run

```bash
python3 code/generate_modular_flux_figures.py
```

## Axial boundary

The accepted exact audit proves a projected local order-seven
transverse-field sector and classifies one explicit eight-parameter
three-qubit path. The seven-cycle counterexample has a local non-Wilson
coefficient at order seven before its primitive cycle term at order eight.

The shared-path clause in the shortest-cycle corollary has a dedicated exact
check in `code/primary_validation/theta_asymmetric_mixed_spin.py`. The test
uses unequal cycle fluxes, six distinct hopping magnitudes, mixed spins,
generic longitudinal exchanges, and nonzero fields.

## File integrity

`SHA256SUMS_v06.txt` verifies every file in the archived release except the
manifest itself. Run

```bash
sha256sum -c SHA256SUMS_v06.txt
```

The release ZIP is attached to the GitHub release and deposited unchanged on
Zenodo. Its external SHA-256 digest is reported in both release records.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite version 0.6.0 through
its version-specific Zenodo DOI,
[10.5281/zenodo.22860690](https://doi.org/10.5281/zenodo.22860690), which
identifies the exact archived files.

## Licenses

The repository uses scoped licenses.

- Python source files under `code/` and the root `run_all.py` are licensed
  under the MIT License in `LICENSE-CODE`.
- The manuscript, figures, notes, and repository documentation are licensed
  under CC BY 4.0 as stated in `LICENSE-CONTENT.md`.

See `LICENSE.md` for the scope summary.

## Scope

This release does not claim a proof of `R_n -> r_P`, a general cactus theorem,
all-orders cycle additivity, an arbitrary-temperature XXZ attenuation law, or
an arbitrary-spin transverse-field-background formula.
