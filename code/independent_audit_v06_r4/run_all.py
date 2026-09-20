#!/usr/bin/env python3
"""Run the independent audit certificates in their documented order."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [ROOT / f"t{number}_{name}.py" for number, name in (
    (0, "smoke"),
    (1, "independent"),
    (2, "structure"),
    (3, "scope"),
    (4, "transverse_boundary"),
    (5, "sector_projection"),
    (6, "local_m7_classification"),
    (7, "local_m7_spotchecks"),
)]


def main():
    if not __debug__:
        raise SystemExit("Do not run this audit with Python -O; assertions are certificates.")
    for script in SCRIPTS:
        print(f"\n=== {script.name} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nAll revision-4 audit certificates passed.")


if __name__ == "__main__":
    main()
