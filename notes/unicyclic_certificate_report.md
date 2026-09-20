# Independent computational audit of the arbitrary-spin unicyclic onset

## Result audited

With the orientation
`gamma=(A=0,B=1,C=2,...,n-1,A)` and the Hamiltonian convention

```text
H_xy = 1/2 (t_xy S_x^+ S_y^- + conjugate(t_xy) S_x^- S_y^+)
       + Jz_xy S_x^z S_y^z,
H_field = - sum_x h_x S_x^z,
```

the proposed coefficient is

```text
[beta^(n+1)] M_beta(AB,BC)
  = 2 (-1)^n h_B (product_{x in gamma} kappa_x) Im W_gamma,
kappa_x = s_x(s_x+1)/3.
```

No normalization or sign discrepancy was found.

## Certificate 1: exact formal-series engine

`verify_unicyclic_arbitrary_spin.py` is a fresh implementation and imports no
code from the v5 bundle.  It constructs the standard irreducible spin
matrices, expands the normalized Gibbs state, performs normalized partial
traces, expands the two matrix logarithms at the identity, and contracts the
commutator.  Gaussian-rational couplings and the radicals in the spin matrices
are kept exact by SymPy.

Every listed case has nonzero longitudinal exchanges and nonzero fields at
all sites.  Tree couplings are complex and have nonzero longitudinal parts.

| Case | Graph / spin dimensions | `W_gamma` | Exact leading coefficient | Target | Lower orders |
|---|---|---:|---:|---:|---:|
| E1 | triangle, spins `(1/2,1,3/2)` | `-13+9i` | `m4=-55/8` | `-55/8` | `m1...m3=0` |
| E2 | triangle `(1/2,1,1/2)` + spin-`3/2` leaf at C | `15-5i` | `m4=-7/12` | `-7/12` | `m1...m3=0` |
| E3 | 4-cycle, spins `(1/2,1,1/2,1/2)` | `-20+10i` | `m5=-25/96` | `-25/96` | `m1...m4=0` |
| E4 | spin-`1/2` 4-cycle + spin-1 leaf at remote cycle site | `-14+8i` | `m5=3/32` | `3/32` | `m1...m4=0` |
| E5 | spin-`1/2` 5-cycle | `-10+20i` | `m6=1/32` | `1/32` | `m1...m5=0` |
| E6 | 4-cycle, spins `(3/2,1,1/2,1/2)` | `-35-5i` | `m5=-175/288` | `-175/288` | `m1...m4=0` |
| E7 | 5-cycle, spins `(1/2,1,1/2,1/2,1/2)` | `70-10i` | `m6=15/224` | `15/224` | `m1...m5=0` |
| E8 | spin-`1/2` 6-cycle | `-50+100i` | `m7=-25/448` | `-25/448` | `m1...m6=0` |
| E9 | 4-cycle with remote cycle spin `3/2` | `-17+19i` | `m5=-285/448` | `-285/448` | `m1...m4=0` |
| E10 | triangle with overlap spin `3/2` | `-12+16i` | `m4=-26/3` | `-26/3` | `m1...m3=0` |
| E11 | spin-`1/2` triangle + two-edge tree grafted at B, intermediate spin 1 | `-6+8i` | `m4=-1/5` | `-1/5` | `m1...m3=0` |

Thus all 11 exact specializations pass.  They test both parities of `n`, both
signs of `Im W_gamma` and `h_B`, all placements relevant to the measured
triplet, higher spin on an unmeasured cycle site, and trees grafted at C, B,
and an unmeasured cycle site.

These are exact specialization certificates, not a replacement for a symbolic
proof with arbitrary parameters.

## Certificate 2: direct Gibbs-state engine

`verify_unicyclic_direct_gibbs.py` shares no project or formal-series code with
the first engine.  It diagonalizes the full Hamiltonian at positive and
negative real beta, forms the normalized Gibbs density matrix, takes ordinary
partial traces, evaluates the two reduced logarithms by Hermitian spectral
calculus, and computes the defining trace directly.  For `p=n+1`, it evenizes
`M(beta)/beta^p` using `+/- beta` and extrapolates in `beta^2`.

| Case | `p` | Extrapolated `c_p` | Predicted `c_p` | Relative error | Normalized ratio |
|---|---:|---:|---:|---:|---:|
| D1, mixed-spin triangle | 4 | `-6.87499974683` | `-6.875` | `3.68e-8` | `-1.99999992635` |
| D2, 4-cycle + spin-1 tree leaf | 5 | `0.0937500000162` | `0.09375` | `1.73e-10` | `2.00000000034` |
| D3, 5-cycle with remote spin 1 | 6 | `0.297619043981` | `0.297619047619` | `1.22e-8` | `-1.99999997555` |
| D4, spin-`1/2` 6-cycle | 7 | `-0.0558035725832` | `-0.0558035714286` | `2.07e-8` | `2.00000004138` |

The last column is
`c_p/[h_B Im(W_gamma) product(kappa_x)]`; it directly confirms `-2` for odd
cycles and `+2` for even cycles.

## Certificate 3: nonconsecutive marked sites

Three complementary checks cover the stronger statement in which `A`, `B`,
and `C` are any distinct vertices of the cycle, oriented in that cyclic
order.  Here `AB` and `BC` denote the overlapping two-site subsystems, not
necessarily graph edges.

- `exact_qubit_nonconsecutive_probe.py` uses exact SymPy arithmetic.  Its
  default run computes the complete beta series through the onset order for
  symbolic fields and generic rational `Jz` on a four-cycle, including the
  orientation-reversed value `-h_B/128`; it also verifies a generic complex
  hopping point with `Im W=15/2` and the five-cycle coefficient `-1/512`.
- `exact_mixed_spin_probe.py` independently checks a nonconsecutive
  four-cycle with spins `(1/2,1,1/2,1/2)` and generic rational `Jz`, obtaining
  `m_1=...=m_4=0` and `m_5=1/48` exactly.
- `nonconsecutive_cycle_probe.py` is a separate algebraic-series engine in
  NumPy.  It exhausts every ordered marked triple and every one-hot field for
  qubit cycles of lengths 3 through 6, then checks mixed-spin cases through
  length 6.  The largest coefficient error is `5.55e-17`; every lower-order
  coefficient is zero to at most `5.78e-18`.

These checks show that the coefficient is independent of the three arc
lengths.  With a cycle orientation fixed externally, reversing the cyclic
order of `A,B,C` reverses the sign.

## Certificate 4: cycle-sector boundary tests

`cycle_sector_exact_certificates.py` uses exact SymPy series to separate the
valid primitive-cycle statement from invalid all-orders additivity claims.
Its default checks give:

| Graph | Exact result | Meaning |
|---|---:|---|
| theta graph with two marked 4-cycles | `m5=1/64=2(1/128)` | shortest primitive cycles add at first admissible order |
| triangle omitting `C`, with `C` attached as a leaf | `m1...m7=0`, `m8=-7/368640` | an off-target cycle can reappear at later decorated order |
| two triangles sharing only `B` | `m7=-1/768` | distinct cycle circulations mix at later order |

Thus the ambient-independent result is the coefficient of the primitive
multidegree `h_x(W_eta-conjugate(W_eta))`.  A first-girth sum is valid, but a
global sum of independent functions of Wilson loops is not.

## Reproduction

Tested with Python 3.12.13, SymPy 1.14.0, NumPy 2.3.5, and SciPy 1.17.0.

```bash
python3 verify_unicyclic_arbitrary_spin.py
python3 verify_unicyclic_direct_gibbs.py
python3 nonconsecutive_cycle_probe.py
python3 exact_mixed_spin_probe.py
python3 exact_qubit_nonconsecutive_probe.py
python3 cycle_sector_exact_certificates.py
```

The two exact scripts that import the v5 series engine expect the unchanged
`verify_modular_flux_extensions.py` under
`work_v5/modular_flux_PRB_v5_bundle/`, as arranged in the validation bundle.

At the audited revision, the SHA-256 hashes of the two fully independent main
engines are

```text
8ba8dfc3ad1fff59a09f00a53ffc109438f88492ecb001ab988d38405f3f2d23  verify_unicyclic_arbitrary_spin.py
e0ba2e453f0705c752010b1bc46522069a0e487f566011aa8ef8fbc2f9eb4ea1  verify_unicyclic_direct_gibbs.py
```
