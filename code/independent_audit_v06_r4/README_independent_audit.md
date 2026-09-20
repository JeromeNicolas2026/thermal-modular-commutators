# Independent audit of the v06 unicyclic onset law — revision 4

This directory contains a second exact implementation, written from scratch.
It shares no code or external library with either engine in the primary
validation suite. It uses the same overall method —
formal Gibbs series, normalized partial traces, the noncommutative expansion
of `log(1+Y)`, and contraction of the modular commutator — so it is an
**independent implementation, not an independent method**.

Revision history:

* **r1** proposed dropping the axial-field hypothesis. That was wrong.
* **r2** retracted the proposal and gave exact counterexamples.
* **r3** identified the gauge-invariant open-path expression and checked it at
  several exact points.
* **r4** separates proof from specialization checks. Self-contained symbolic
  extractors prove the projected order-seven sector and, separately, classify
  one explicitly delimited open-path submodel. Every statement about a full
  coefficient or a finite graph is labelled as such.

## Conventions and claim under audit

Let the interaction graph be finite, simple, connected, and unicyclic, with
arbitrary trees grafted onto its unique cycle.  Every site carries an
irreducible spin `s_x` in `{1/2,1,3/2,...}`.  With complex hoppings and real
`J^z_xy,h_x`, all fixed as `beta -> 0`, consider the axial Hamiltonian

    H = sum_(xy) [ (t_xy S_x^+ S_y^- + tb_xy S_x^- S_y^+)/2
                    + J^z_xy S_x^z S_y^z ] - sum_x h_x S_x^z,

where `tb_xy=conj(t_xy)` in the physical model and no transverse field is
present. For three distinct marked sites `A,B,C`, not necessarily consecutive
on the cycle,

    M_beta(AB,BC) = i Tr_ABC rho_ABC,beta
                       [ log(rho_AB,beta) tensor I_C,
                         I_A tensor log(rho_BC,beta) ],
    M_beta = sum_(q>=0) beta^q m_q.

Let `gamma` be the unique cycle, of length `n`, oriented through the marked
sites in the order `A -> B -> C -> A`, and let `W_gamma` be the corresponding
product of oriented hopping amplitudes. The axial theorem tested here is

    m_0 = ... = m_n = 0,
    m_(n+1) = 2 (-1)^n h_B (product_(x in gamma) kappa_x) Im(W_gamma),
    kappa_x = s_x(s_x+1)/3.

The phrase *primitive cycle coefficient* below means the coefficient of the
two conjugate monomials `h_B W_gamma` and `h_B conj(W_gamma)` in `m_(n+1)`.

## Exact arithmetic and independence

* The numerical engine uses no SymPy, NumPy, or floating-point arithmetic.
  Every mathematical target in every test is a Gaussian rational. The
  floating-point clocks used only to print timings do not enter any target.
* The radicals in the spin matrices are removed by a site-factorized diagonal
  similarity `D = tensor_x D_x`. In this representation `S^+` has ones on its
  superdiagonal and `S^-` has the integers `(s-m)(s+m+1)` on its
  subdiagonal. Because the similarity factorizes by site, partial traces,
  reduced logarithms, and the final trace transform covariantly and `M_beta`
  is unchanged. The su(2) relations and `tau(S^+S^-)=2 kappa` are checked at
  run time. This algebraic similarity is distinct from the physical local
  U(1) gauge transformation of the couplings.
* The engine stores `H` sparsely and propagates exact dense series
  coefficients. The ordinary tests all call one common modular-series
  pipeline.
* `t5_sector_projection.py` and `t6_local_m7_classification.py` are deliberately
  self-contained. They repeat the construction over exact polynomial rings
  and therefore supply implementation-independent symbolic extractions.

## Files

| file | content |
|---|---|
| `indep_engine.py` | integer-similarity spin algebra, exact Gibbs series, normalized partial traces, noncommutative logarithms, and the modular commutator; optional real (`fields_x`) or complex (`gx`) transverse fields |
| `t0_smoke.py` | su(2) identities and the v5 qubit-triangle law |
| `t1_independent.py` | replication of certificate E1 and three further generic axial points |
| `t2_structure.py` | ten exact structural probes, including separately labelled multicyclic and transverse specializations |
| `t3_scope.py` | a girth specialization on K4, a sharpness counterexample, and one explicitly labelled `C_4` transverse specialization |
| `t4_transverse_boundary.py` | exact full-series spot checks and two complete transverse counterexamples |
| `t5_sector_projection.py` | symbolic extraction of the two local order-seven multidegrees, without parameter sampling |
| `t6_local_m7_classification.py` | exhaustive symbolic `m_7` in one stated eight-parameter open-path submodel |
| `t7_local_m7_spotchecks.py` | five physical exact checks of the `t6` formula through `indep_engine.py`, plus isolation and scope guards |
| `run_all.py` | runs `t0` through `t7` in order and stops at the first failed assertion |

In normal, non-optimized Python mode, every script exits nonzero on a failed
hard assertion. The archive uses only the Python 3.12 standard library. Do not
invoke the audit with Python's `-O` option; `run_all.py` rejects optimized mode
because assertions are certificates.

From the unpacked archive, run the full audit with

    python3 run_all.py

Then verify the distributed sources with

    sha256sum -c SHA256SUMS_independent_audit.txt

## Further axial checks

The following are additional exact specializations, not substitutes for the
all-parameter proof in the v06 proof memo.

* `X2`: a nonconsecutive four-cycle with spins `(1/2,1,1/2,3/2)`, generic
  `J^z`, and nonzero longitudinal fields at every site gives `m_5=55/18`.
* `X3`: a nonconsecutive five-cycle with a leaf grafted inside the arc
  `B -> C` gives `m_6=423/512`.
* `X4` and `Y1` extend the tested values of
  `kappa=s(s+1)/3` to spins `2` and `5/2`.
* `Y3` and `Y4` check that the onset selects `h_B`, rather than a field at an
  interior arc vertex or at `A,C`.
* `Y6` uses individually complex amplitudes with a real Wilson product and
  obtains a vanishing onset.
* `Y7` checks that a spin-`5/2` leaf at `B` leaves the onset unchanged.
* `B1` consists of three exact K4 specializations. They are compatible with
  the marked-triangle girth formula, but are not presented as a proof for all
  K4 couplings.
* `C1` is one `C_4` specialization with `J^z != 0` and `m_6 != 0`. It rules
  out a uniform improvement of the theorem's `O(beta^(n+2))` remainder to
  `O(beta^(n+3))`; it does not assert nonvanishing for every `n` and every
  parameter choice.

Also reproduced are E1 (`m_4=-55/8`), the total cactus value at order seven
(`-1/768`), and the theta-graph girth value (`m_5=1/64`). The cactus sector is
isolated only by the phase-variation certificate in the v06 bundle. The two
four-cycles in the theta specialization share the path `0-1-2`.

## The transverse-field boundary

Write

    H_perp = -(1/2) sum_x (g_x S_x^+ + gb_x S_x^-).

Here `gb_x=conj(g_x)` physically. With `S^+=S^x+iS^y` and
`H_perp=-h^x S^x-h^y S^y`, the convention is `g=h^x-i h^y`.
Under the physical local U(1) transformation,

    t_xy -> exp(i(theta_x-theta_y)) t_xy,
    g_x  -> exp(i theta_x) g_x,

so `t_AB conj(g_A) g_B` is gauge invariant.

Let `A-B-C` now denote an open path of three qubits with consecutive bonds
`AB` and `BC`. Treat barred and unbarred parameters as formally independent
and let `Pi_loc` project `m_7` onto the following two multidegrees:

    J^z_AB h^z_B t_BC tb_BC t_AB gb_A g_B,
    J^z_AB h^z_B t_BC tb_BC tb_AB g_A gb_B.

Then `t5_sector_projection.py` proves, by exact symbolic extraction,

    Pi_loc m_7
      = J^z_AB h^z_B t_BC tb_BC
          (t_AB gb_A g_B - tb_AB g_A gb_B)/(46080 i).

After physical specialization this is

    Pi_loc m_7
      = J^z_AB h^z_B |t_BC|^2
          Im(t_AB conj(g_A) g_B)/23040.                 (T)

The extractor works in
`Q(i)[x_1,...,x_7]/(x_1^2,...,x_7^2)`. Discarding a square is exact for this
coefficient because perturbation theory has only nonnegative parameter
degrees: a term already divisible by `x_j^2` can never contribute later to
the squarefree target. No interpolation or numerical specialization is used.

### Spectator-factorization lemma

Each `m_q` is a polynomial in the Hamiltonian parameters. Extracting either
multidegree in `(T)` sets every absent parameter to zero. In any finite
ambient graph this leaves

    H = H_ABC tensor I_spectators.

Thus the Gibbs operator factorizes, and the spectator dimension cancels
between the partition function and all normalized partial traces. The two
projected coefficients are therefore the same as on the isolated open path.
This proves ambient independence of `Pi_loc m_7`. It does **not** say that the
full `m_7` of a general transverse model is local or contains no other sectors.

### Additional classification in a fixed local submodel

This result is not needed for the ambient projection theorem above. It records
what happens when the graph is exactly the open qubit path, `h_A=h_C=0`, and
the only active parameters are

    t_AB, t_BC, g_A, g_B, g_C, J^z_AB, J^z_BC, h^z_B.

Define

    I_AB = Im(t_AB conj(g_A) g_B),
    I_BC = Im(t_BC conj(g_B) g_C),
    I_AC = Im(t_AB t_BC conj(g_A) g_C).

The exhaustive polynomial calculation in `t6_local_m7_classification.py`
proves `m_0=...=m_6=0` and

    m_7 = h^z_B/23040 [
        J^z_AB (|t_BC|^2-(J^z_BC)^2) I_AB
      + J^z_BC (|t_AB|^2-(J^z_AB)^2) I_BC
      + (J^z_AB J^z_BC/2) I_AC ].                         (P)

It finds exactly ten formal complex monomials, paired into the five real
sectors displayed in `(P)`, and no others. `t7_local_m7_spotchecks.py`
independently checks the physical specialization at five further exact
complex points. Formula `(P)` is complete only in the eight-parameter
submodel just listed. For example, enabling `h_A` or `h_C` creates additional
order-seven sectors. The general article-level statement remains the projected
coefficient `(T)`.

### Independent full-series checks (`t4_transverse_boundary.py`)

| | exact statement checked |
|---|---|
| `N1` | at four Gaussian-rational complex points of the minimal open-path family, `m_0,...,m_6=0` and the full `m_7` equals `(T)` |
| `N2` | at one such point, changing `g_C` leaves the full `m_7` unchanged |
| `N3` | at one point, the open path, `C_7`, and `C_8` give the same full `m_7` |
| `N4` | at the base point, removing `t_AB`, `g_A`, `g_B`, `J^z_AB`, `h^z_B`, or `t_BC` separately kills `m_7` |
| `N5` | at the base point, doubling `t_BC` gives `x4`, while doubling every cycle amplitude outside `AB` and `BC` gives `x1` |
| `N6` | on `C_6`, the axial onset coefficient changes from `1/2048` to `49/92160 = 1/2048 + 1/23040` |
| `N7` | on `C_7`, `m_7=1/23040`, while the primitive cycle onset is at order eight |

N1-N5 are exact specialization checks. The symbolic extraction in `t5`,
not the finite comparison in N3, proves `(T)` and its spectator independence.
The stronger equality for the entire `J^z_BC=0` open-path submodel also follows
from the independently implemented classification `(P)` in `t6`.
N6 and N7 are complete counterexamples and by themselves show that the axial
hypothesis cannot be dropped from the universal v06 theorem.

## Consequences and limits

For an ambient model containing the marked consecutive path `A-B-C`, the
nonzero coefficient `(T)` shows that the full `m_7` polynomial is not
identically zero. Thus generic fixed transverse data in this setting contain
a non-Wilson-loop open-link contribution by order seven. Special parameter
values can still produce cancellations between sectors. In the isolated
eight-parameter path submodel of `(P)`, after setting `J^z_BC=0`, the simple
nonvanishing condition

    J^z_AB h^z_B |t_BC|^2 Im(t_AB conj(g_A) g_B) != 0

does guarantee a nonzero total `m_7`. In the consecutive-marked six-cycle of
N6 this order coincides with the Wilson onset, giving an exact contamination.
Whenever the marked
sites form the consecutive path `A-B-C`, this certified local multidegree is
at order seven and is therefore earlier than the primitive `beta^(n+1)` cycle
term for every `n>=7`. For real transverse fields in the chosen spin frame,
its phase factor is `h_A^x h_B^x Im(t_AB)`.

These statements do not classify the complete transverse expansion, exclude
lower-order transverse sectors, or identify the first transverse correction
for `n=3,4,5`. No such order table is claimed in this revision. Other local
multidegrees may occur when further transverse fields or Ising couplings are
enabled.

One statement survives without additional assumptions: the primitive
coefficient of `h_B W_gamma` is unchanged by transverse fields, because its
coefficient extraction sets every transverse-field variable to zero. It need
not be the leading behavior of the full modular commutator.
