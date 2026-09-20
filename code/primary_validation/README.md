# Primary validation suite

The scripts in this directory certify exact specializations and numerical
cross-checks of the v06 theorem. They supplement the analytic proof. They do
not replace it.

The pinned environment for the final release pass is Python 3.12 with SymPy
1.14.0, NumPy 2.3.5, SciPy 1.17.0, mpmath 1.3.0, and Matplotlib 3.10.8. Exact
pins are recorded in `../../requirements.txt`.

Run the standard checks from this directory with

```bash
python3 verify_unicyclic_arbitrary_spin.py
python3 verify_unicyclic_direct_gibbs.py
python3 nonconsecutive_cycle_probe.py
python3 exact_mixed_spin_probe.py
python3 exact_qubit_nonconsecutive_probe.py
python3 cycle_sector_exact_certificates.py
python3 theta_asymmetric_mixed_spin.py
```

`exact_qubit_nonconsecutive_probe.py` and
`cycle_sector_exact_certificates.py` load the included v5 exact series engine
from `work_v5/modular_flux_PRB_v5_bundle`.

`theta_asymmetric_mixed_spin.py` is the hard exact check of the shortest-cycle
sum when two four-cycles share the path A-B-C. It uses unequal fluxes, six
distinct hopping magnitudes, one spin-1 site, generic longitudinal exchanges,
and nonzero fields.

See `../../notes/unicyclic_certificate_report.md` for the certified values,
method distinctions, and numerical tolerances.
