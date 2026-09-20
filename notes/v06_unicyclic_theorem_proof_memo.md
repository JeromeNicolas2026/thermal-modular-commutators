# Proof memo for the v06 unicyclic theorem

This memo fixes the proposed major result before any rewrite of the
manuscript. It uses the conventions of **modular_flux_PRB_v06.tex** and records
the proof repair that must replace the informal effective-hopping substitution
in v5.

## 1. Final theorem and exact scope

Let \(G=(V,E)\) be a finite, simple, connected, unicyclic graph. Its unique
cycle has length \(n\geq 3\). At every vertex \(x\), let \(\mathcal H_x\)
carry the irreducible spin-\(s_x\) representation, where

\[
s_x\in\left\{\frac12,1,\frac32,\ldots\right\},
\qquad
\kappa_x=\frac{s_x(s_x+1)}{3}.
\]

Consider the axial Hamiltonian

\[
H=\sum_{\{x,y\}\in E}
\left[
\frac12\left(t_{xy}S_x^+S_y^-+\overline{t_{xy}}S_x^-S_y^+\right)
+J^z_{xy}S_x^zS_y^z
\right]
-\sum_{x\in V}h_xS_x^z,
\]

where \(J^z_{xy},h_x\in\mathbb R\), \(t_{yx}=\overline{t_{xy}}\), and all
parameters are fixed as \(\beta\to0\). Here “arbitrary fields” means the
longitudinal fields displayed above; transverse fields are not covered.

Choose any three distinct vertices \(A,B,C\) on the unique cycle and orient
the cycle so that they occur in the cyclic order

\[
\gamma=(A,\ldots,B,\ldots,C,\ldots,A).
\]

With \(X=\{A,B\}\), \(Y=\{B,C\}\), and

\[
W_\gamma=\prod_{(x,y)\in\gamma}t_{xy},
\]

one has

\[
\boxed{
\mathfrak M_\beta(AB,BC)
=2(-1)^n\beta^{n+1}h_B
\left(\prod_{x\in V(\gamma)}\kappa_x\right)
\operatorname{Im}W_\gamma
+O(\beta^{n+2}).}
\]

Every Taylor coefficient through order \(n\) vanishes. No nonzero-coupling
hypothesis is needed: if \(W_\gamma=0\), the statement follows by
polynomiality. When the marked vertices are consecutive this is exactly the
conjecture proposed for v06. When every \(s_x=1/2\), it reduces to

\[
2(-1)^n4^{-n}=\frac{(-1)^n}{2^{2n-1}},
\]

the coefficient in v5.

## 2. Gauge grading and pruning

Write

\[
\mathfrak M_\beta(AB,BC)=\sum_{q\geq0}\beta^q m_q.
\]

Finite dimensionality makes this series analytic near \(\beta=0\), and
scaling \(H\mapsto\lambda H\) shows that \(m_q\) is a real homogeneous
polynomial of total degree \(q\) in \(t,\bar t,J^z,h\).

For a transverse monomial, let \(d_e=a_e-b_e\), where \(a_e,b_e\) are the
exponents of \(t_e,\bar t_e\). Local \(U(1)\) gauge invariance imposes
\(\operatorname{div}d=0\). On a unicyclic graph, every such integral flow is
zero on the bridges and equals \(k\) times the oriented cycle flow on
\(\gamma\). Complex-conjugation oddness excludes \(k=0\). Hence, if
\(D_\perp\) is the transverse degree,

\[
D_\perp\geq\sum_e|d_e|\geq n.
\]

Physical time-reversal oddness imposes an odd, hence nonzero, total field
degree. Therefore \(m_q=0\) for \(q\leq n\). Equality at \(q=n+1\) forces:

- one occurrence of every transverse cycle edge, all with one coherent
  orientation;
- exactly one longitudinal field;
- no \(J^z\), transverse bridge, or neutral backtrack.

Consequently,

\[
m_{n+1}=\operatorname{Im}W_\gamma\sum_{x\in V}c_xh_x.
\]

The coefficient of \(W_\gamma h_x\) contains no tree coupling. Setting every
parameter absent from this monomial to zero makes all attached trees decoupled
tensor factors. If \(x\notin\gamma\), the reduced state on \(ABC\) is
independent of \(h_x\), so \(c_x=0\). If \(x\in\gamma\), all off-cycle tensor
factors have normalized trace one and disappear. Thus the leading coefficient
is determined by the isolated marked cycle. This is the required pruning
argument.

## 3. Local spin identities and the arbitrary-spin triangle

Let \(\tau_x=(2s_x+1)^{-1}\operatorname{Tr}_x\). Rotational invariance and the
\(\mathfrak{su}(2)\) commutation relations give

\[
\tau_x(S_x^a)=0,\qquad
\tau_x(S_x^aS_x^b)=\kappa_x\delta_{ab},\qquad
\tau_x(S_x^aS_x^bS_x^c)
=\frac{i\kappa_x}{2}\varepsilon_{abc}.
\]

Equivalently,

\[
\tau_x(S^+S^-)=\tau_x(S^-S^+)=2\kappa_x,
\]

and the six permutations of \(S^+,S^-,S^z\) have trace
\(+\kappa_x\) or \(-\kappa_x\), three of each.

For the triangle, gauge grading first reduces \(m_4\) to

\[
m_4=(c_Ah_A+c_Bh_B+c_Ch_C)
\operatorname{Im}(t_{AB}t_{BC}t_{CA}),
\]

with no possible \(J^z\). In the gauge
\(t_{AB}=t_{BC}=1,t_{CA}=i\), use

\[
T_0=\frac{i}{2}\tau(H^2[P,Q]),\qquad
T_1=i\tau(H[P,B_2]),\qquad
T_2=i\tau(H[A_2,Q]).
\]

At \(h=0\), \(J^z=0\), and in this minimal transverse gauge, the local
identities give, with
\(K_\triangle=\kappa_A\kappa_B\kappa_C\),

| field | \(\partial T_0\) | \(\partial T_1\) | \(\partial T_2\) | \(\partial m_4\) |
|---|---:|---:|---:|---:|
| \(h_A\) | 0 | 0 | 0 | 0 |
| \(h_B\) | \(2K_\triangle\) | \(-2K_\triangle\) | \(-2K_\triangle\) | \(-2K_\triangle\) |
| \(h_C\) | 0 | 0 | 0 | 0 |

A concise verification is the termwise lifting argument. In every nonzero
square-free contraction of \(h_rW_\triangle\), each vertex other than \(r\)
carries one \(S^+\) and one \(S^-\), while \(r\) carries
\(S^+,S^-,S^z\). Any splitting among two or more normalized local traces
leaves a singleton traceless generator and vanishes. Thus every surviving local
contraction is \(4\kappa_x\) times its spin-\(1/2\) value. Multiplying the
exact qubit table by \(\prod_{x=A,B,C}4\kappa_x\) gives the table above.

Hence:

\[
\boxed{
\mathfrak M_\beta(AB,BC)
=-2\beta^4h_B\kappa_A\kappa_B\kappa_C
\operatorname{Im}(t_{AB}t_{BC}t_{CA})+O(\beta^5).}
\]

This does not extend the Pauli anticommutator identities themselves; it
replaces them by second- and third-moment identities valid in every
irreducible spin representation.

## 4. Square-free arc cumulants

This is the step missing from the v5 decimation proof.

By the pruning result of Section 2, the coefficient now under study is
unchanged when every attached tree is removed. Throughout Sections 4 and 5,
therefore, \(H\) denotes its restriction \(H_\gamma\) to the isolated cycle.

Split the oriented cycle into its three internally disjoint marked arcs

\[
P_1:A\longrightarrow B,\qquad
P_2:B\longrightarrow C,\qquad
P_3:C\longrightarrow A,
\]

of lengths \(\ell_1,\ell_2,\ell_3\), with
\(\ell_1+\ell_2+\ell_3=n\). Put \(U=\{A,B,C\}\), let
\(I=V(\gamma)\setminus U\), and take the normalized partial trace over all
vertices in \(I\). This defines the operator on
\(\mathcal H_U\)

\[
G(\beta)=\tau_I(e^{-\beta H})
=\frac{1}{d_I}\operatorname{Tr}_I(e^{-\beta H}),\qquad
d_I=\prod_{x\in I}(2s_x+1),
\]
\[
Q(\beta)=-\log G(\beta),\qquad
\mathcal F(Q)=\mathfrak M_{e^{-Q}/\operatorname{Tr}_Ue^{-Q}}(AB,BC).
\]

It is important to use the dimensionless \(Q=-\log G\). If one instead sets
\(K=-\beta^{-1}\log G\), the composed functional is \(\mathcal F(\beta K)\),
not \(\mathcal F(K)\).

For an oriented arc

\[
P=(v_0=u,v_1,\ldots,v_\ell=v),\qquad
T_P=\prod_{j=0}^{\ell-1}t_{v_jv_{j+1}},
\]

denote by \([P]\) projection onto the **primitive multiaffine sector** of
multidegree exactly \(P\): every oriented edge of \(P\) occurs once and no
other field, bond, or longitudinal parameter occurs. Thus \([P]G\) still
contains the monomial \(T_P\). Equivalently, polarize in the edge variables,
set every other parameter to zero, and retain that multidegree. Then

\[
[P]G
=(-\beta)^\ell
\left(\prod_{x\in P^\circ}\kappa_x\right)
\frac{T_P}{2}S_u^+S_v^-.
\]

The bond factors contribute \(2^{-\ell}\), each internal spin gives
\(2\kappa_x\), and all \(\ell!\) orderings give the same contraction,
canceling the exponential factorial.

### Proper-subpath lemma

In any block of the primitive sector under consideration, if the selected
edges of \(P\) form a nonempty proper subset, that block has zero partial
trace, even if it contains the unique diagonal field. A selected component
has an internal endpoint at which the total axial charge is \(+1\) or \(-1\);
its local trace is therefore zero. Adding \(S^z\) does not change that charge.
The same statement holds when the block contains pieces of other internally
disjoint arcs.

This prevents a full arc from being split among different blocks in the set
partitions generated by \(\log G\).

### Pairwise shuffle cancellation

For two complete, internally disjoint arcs sharing at most one retained
endpoint, still in the primitive coefficient with no additional insertion,

\[
[P_iP_j]G
=\frac12\bigl([P_i]G[P_j]G+[P_j]G[P_i]G\bigr).
\]

The same identity holds for a complete arc and a field on a retained vertex.
Among all shuffles, exactly half place the possibly noncommuting endpoint
letter of the first ingredient before that of the second, and half reverse
the order. Since proper-subpath jets vanish,

\[
[P_iP_j]\log G=0,\qquad [P_iF]\log G=0.
\]

If the field lies at an internal vertex of its arc, the raw arc-field jet is
also zero because

\[
\tau_x\!\left(\operatorname{Sym}(S^+,S^-,S^z)\right)=0.
\]

All partitions containing a proper subarc again vanish, so the corresponding
jet of \(\log G\) is zero. Pairing that internal field with either of the two
other arcs instead factorizes through \(\tau_x(S_x^z)=0\). Hence every
quadratic jet involving an internal field is zero.

## 5. Why generated multispin terms do not change the answer

The modular functional at the maximally mixed state satisfies

\[
D\mathcal F(0)=D^2\mathcal F(0)=0.
\]

Indeed, the commutator of the reduced logarithms begins quadratically, and
the only quadratic contribution is the trace of a commutator.

First suppose the unique field \(F\) is on the retained set \(U\). The
underlying variables are the individual edge variables and the field
variable. The proper-subpath lemma proves that no surviving block can split
an arc; only then may the three full arcs be treated as atomic ingredients.
Thus the multivariate chain rule for the primitive coefficient is organized
by set partitions of \(\{P_1,P_2,P_3,F\}\):

- four singletons give the polarized fourth derivative of the triangle
  functional;
- a pair and two singletons would give a third derivative, but every relevant
  pair jet of \(Q=-\log G\) vanishes by the shuffle lemma;
- a triple and a singleton, or two pairs, can meet only
  \(D^2\mathcal F(0)=0\);
- a four-element block can meet only \(D\mathcal F(0)=0\).

Thus every effective multipolar operator in this primitive four-ingredient
sector is canceled at the dangerous pair level or killed by the first two
vanishing derivatives. Only the three singleton arc hoppings and the
singleton retained field survive.

If \(F\) is at an internal vertex of an arc, its singleton jet vanishes after
the partial trace. The proper-subpath lemma forces any potentially surviving
block containing \(F\) to contain that complete arc, but the pair jet
\([P_iF]Q\) vanishes by the symmetrized three-letter trace. Any larger block
meets only \(D^2\mathcal F(0)\) or \(D\mathcal F(0)\). Hence every
internal-cycle field coefficient is zero.

For each arc, the effective hopping amplitude is

\[
t_{uv}^{\mathrm{eff}}
=(-\beta)^{\ell-1}
\left(\prod_{x\in P^\circ}\kappa_x\right)T_P.
\]

Indeed, at the primitive linear level,

\[
[P]Q=-[P]G
=\frac{\beta}{2}\left(
t_{uv}^{\mathrm{eff}}S_u^+S_v^-+\mathrm{H.c.}\right),
\]

where the displayed Hermitian combination combines the two conjugate arc
orientations. This is the precise bridge from the dimensionless cumulant to
the triangle lemma.

Substitution in the arbitrary-spin triangle is now justified. The three arc
signs give

\[
(-1)^{(\ell_1-1)+(\ell_2-1)+(\ell_3-1)}=(-1)^{n-3}.
\]

Together with the triangular coefficient,

\[
-2(-1)^{n-3}=2(-1)^n.
\]

The arc interiors and marked vertices supply exactly
\(\prod_{x\in\gamma}\kappa_x\), completing the proof.

## 6. Nonconsecutive marked sites

The proof already covers nonconsecutive \(A,B,C\). The notation \(AB\) then
means the two-site subsystem \(\{A,B\}\), not a physical bond. The coefficient
is independent of the three arc lengths.

For a fixed external orientation of \(\gamma\), define

\[
\varepsilon_\gamma(A,B,C)=
\begin{cases}
+1,&A,B,C\text{ occur in that cyclic order},\\
-1,&A,C,B\text{ occur in that cyclic order}.
\end{cases}
\]

The formula with that fixed orientation acquires the factor
\(\varepsilon_\gamma(A,B,C)\). Equivalently, orient \(\gamma\) from the outset
in the order \(A,B,C\). Exchanging \(A\) and \(C\) reverses
\(\mathfrak M\), as it should.

This must not be confused with two disjoint bond regions. If \(X\) and \(Y\)
are disjoint, their extended reduced logarithms commute and
\(\mathfrak M_\rho(X,Y)=0\) identically at every temperature.

## 7. What survives cycle by cycle

There is an exact ambient-independent statement, but not an all-orders sum of
independent cycle responses.

After formal complexification, treat \(t_e,\bar t_e,h_x,J_e^z\) as
algebraically independent. For a simple cycle \(\eta\) of length \(r\), let
\(\operatorname{Prim}_\eta m_{r+1}\) be the real, conjugation-odd sector
generated by \(h_x(W_\eta-\overline W_\eta)\), with no other factor. This
sector contains the two conjugate orientations and is zero unless \(\eta\)
contains \(A,B,C\). If it does, orient it in the order
\(A\to\cdots\to B\to\cdots\to C\to\cdots\to A\); then

\[
\boxed{
\operatorname{Prim}_\eta m_{r+1}
=2(-1)^r h_B
\left(\prod_{x\in V(\eta)}\kappa_x\right)
\operatorname{Im}W_\eta.}
\]

The proof is coefficient extraction: set every variable outside this
multidegree to zero and apply the isolated marked-cycle theorem. If the cycle
omits one marked site, the reduced state factorizes. The modular commutator is
then either a commutator of disjoint logarithms or a nested-region trace, both
zero.

As an optional first-order corollary, suppose the nonzero transverse support
graph contains a cycle and has finite girth \(g\). Then \(m_q=0\) for
\(q\leq g\), and

\[
m_{g+1}=2(-1)^g h_B
\sum_{\substack{\eta:\ |\eta|=g\\A,B,C\in V(\eta)}}
\left(\prod_{x\in V(\eta)}\kappa_x\right)
\operatorname{Im}W_\eta,
\]

where each unoriented cycle is counted once and then oriented in the marked
order. Indeed, every nonzero integral circulation decomposes into oriented
simple cycles, so its \(\ell^1\)-norm is at least \(g\); equality forces one
unit circulation on a simple \(g\)-cycle. Field oddness adds one degree.
This remains true when shortest cycles overlap. If the transverse support is a forest, the local
real-gauge criterion instead gives \(\mathfrak M_\beta=0\) exactly at every
temperature. This is a first-admissible-order statement, not a cactus theorem.

No broader additivity should be asserted. At later orders there are neutral
returns, \(J^z\) and field decorations, higher harmonics, and products of
distinct circulations. Exact counterexamples include:

- a spin-\(1/2\) cactus with \(J^z=0\), only \(h_B=1\), and two triangles
  \(ABC\) and \(BDE\) sharing only \(B\), all remaining amplitudes of unit modulus,
  \(W_{ABC}=i\), and \(W_{BDE}=1\). It has \(m_7=-1/768\). On the
  unit-modulus phase torus the mixed sector is
  \[
  -\frac1{1536}\left[
  \operatorname{Im}(W_1W_2)+
  \operatorname{Im}(W_1\overline W_2)\right]
  =-\frac1{768}\operatorname{Im}W_1\operatorname{Re}W_2;
  \]
- a spin-\(1/2\), \(J^z=0\) triangle \(A-B-D-A\) with a leaf \(B-C\), only
  \(h_B=1\), and \(t_{AB}=t_{BD}=t_{BC}=1,t_{DA}=i\). Although the flux
  cycle omits \(C\), neutral returns make it visible later:
  \(m_1=\cdots=m_7=0\) and \(m_8=-7/368640\).

Thus the canonical decomposition is by edge multidegree, not by a sum of
independent functions of fundamental Wilson loops.

## 8. Independent certificates

Two independent engines audit the unicyclic theorem.

1. **verify_unicyclic_arbitrary_spin.py** performs the complete formal
   high-temperature series with exact SymPy arithmetic. Eleven generic
   specializations pass for \(n=3,4,5,6\), spins \(1/2,1,3/2\), nonzero
   longitudinal couplings and fields, and trees grafted at several locations,
   including a depth-two tree grafted directly at \(B\). In every case
   \(m_1=\cdots=m_n=0\) and the predicted \(m_{n+1}\) is exact.
2. **verify_unicyclic_direct_gibbs.py** uses no formal series. It diagonalizes
   the full Hamiltonian, builds ordinary Gibbs density matrices and partial
   traces, evaluates reduced logarithms spectrally, and extrapolates
   \(\beta\to0\). Four tests give the normalized ratio \(-2\) for odd \(n\)
   and \(+2\) for even \(n\), with relative errors between
   \(1.7\times10^{-10}\) and \(3.7\times10^{-8}\).

Additional scripts audit every ordered marked triple through \(n=6\), exact
mixed-spin nonconsecutive cases, the shortest-cycle sum on a theta graph, and
the two failures of all-orders cycle additivity above.

These computations are exact specialization certificates except for the
explicitly numerical direct-Gibbs and broad-scan engines. They support but do
not replace the all-parameter proof.

## 9. Editorial consequence for v06

The strongest clean organization is:

1. state the marked unicyclic theorem for arbitrary irreducible spins and
   arbitrary longitudinal axial data, allowing nonconsecutive marked sites;
2. prove the invariant triangle lemma;
3. prove the proper-subpath and pairwise-shuffle lemmas using \(Q=-\log G\);
4. give the consecutive spin-\(1/2\) ring law as an immediate specialization;
5. state at most the primitive single-cycle coefficient as the controlled
   multicyclic extension.

The v5 sentences that merely substitute the first induced hopping into a
\(\beta\)-dependent effective Hamiltonian are not sufficient and should not
be retained as the proof. The even-\(n\) improvement to
\(O(\beta^{n+3})\) proved in v5 for the transverse qubit ring is not claimed
by this arbitrary-axial theorem. The conjectured limit \(R_n\to r_{\rm P}\),
general cactus laws, and all-temperature XXZ attenuation remain outside the
agreed v06 scope.
