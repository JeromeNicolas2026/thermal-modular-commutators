#!/usr/bin/env python3
"""Run the v06 validation suites from the repository root.

The default run checks the release manifest, the independent exact audit,
the primary validation suite, and the inherited v5 checks. Use ``--full`` to
enable the longer n=6 and complete cycle-sector variants. Figure generation
and LaTeX compilation are explicit because they modify tracked artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(f"\n[{cwd.relative_to(ROOT) if cwd != ROOT else '.'}] "
          + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS_v06.txt"
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"invalid manifest line {line_number}: {raw_line!r}"
            ) from exc
        relative = relative.removeprefix("*").removeprefix("./")
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: {digest} != {expected}"
            )
        print(f"{relative}: OK")


def independent_audit() -> None:
    audit = ROOT / "code" / "independent_audit_v06_r4"
    run([sys.executable, "run_all.py"], cwd=audit)


def primary_validation(full: bool) -> None:
    primary = ROOT / "code" / "primary_validation"
    commands = [
        [sys.executable, "verify_unicyclic_arbitrary_spin.py"],
        [sys.executable, "verify_unicyclic_direct_gibbs.py"],
        [sys.executable, "nonconsecutive_cycle_probe.py"],
        [sys.executable, "exact_mixed_spin_probe.py"],
        [sys.executable, "exact_qubit_nonconsecutive_probe.py"],
        [sys.executable, "cycle_sector_exact_certificates.py"],
        [sys.executable, "theta_asymmetric_mixed_spin.py"],
    ]
    if full:
        commands[4].append("--include-n6")
        commands[5].append("--full")
    for command in commands:
        run(command, cwd=primary)


def inherited_validation() -> None:
    code = ROOT / "code"
    for name in (
        "verify_modular_flux_triangle.py",
        "verify_modular_flux_extensions.py",
        "verify_modular_flux_ring_jw.py",
        "verify_modular_flux_attenuation.py",
    ):
        run([sys.executable, name], cwd=code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the longer primary certificate variants",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="regenerate the tracked PDF and PNG figures",
    )
    parser.add_argument(
        "--tex",
        action="store_true",
        help="compile the portable preprint with latexmk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_manifest()
    independent_audit()
    primary_validation(args.full)
    inherited_validation()
    if args.figures:
        run([sys.executable, "code/generate_modular_flux_figures.py"])
    if args.tex:
        run([
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "modular_flux_preprint_v06.tex",
        ])
    if args.figures or args.tex:
        print(
            "\nTracked artifacts were rebuilt. Regenerate "
            "SHA256SUMS_v06.txt before archival."
        )
    print("\nAll requested checks passed.")


if __name__ == "__main__":
    main()
