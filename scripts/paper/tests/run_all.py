"""Exécute toutes les suites de tests unitaires du dossier tests/.

Retour : 0 si tout passe, ≠0 sinon.
Usage : python scripts/paper/tests/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = [
    "test_hp_hash.py",
    "test_registry.py",
    "test_log_parser.py",
    "test_experiments.py",
    "test_multi_exp_group.py",
    "test_mutable_defaults.py",
    "test_seed_reproducibility.py",
]


def _pick_python() -> str:
    # Prefer the active virtualenv's Python so suites pick up project deps.
    cand = Path(sys.prefix) / "bin" / "python"
    if cand.exists():
        return str(cand)
    return sys.executable


def main() -> int:
    py = _pick_python()
    fails = []
    for s in SUITES:
        p = HERE / s
        if not p.exists():
            print(f"[SKIP] {s} (missing)")
            continue
        proc = subprocess.run([py, str(p)], capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"[PASS] {s}")
        else:
            print(f"[FAIL] {s}")
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            fails.append(s)
    print()
    if fails:
        print(f"OVERALL: FAIL ({len(fails)}/{len(SUITES)} suites failed)")
        return 1
    print(f"OVERALL: PASS ({len(SUITES)} suites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
