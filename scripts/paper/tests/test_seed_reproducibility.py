"""Régression : curtis et ours doivent produire le MÊME état initial pour
le même seed (reproductibilité atomique de la comparaison).

Historique du bug : ``curtis_baseline/main.py:105`` utilisait
``env.reset(seed=np.random.randint(0, 10000))`` → grille d'éval aléatoire à
chaque run → ours vs curtis ne comparaient pas la même instance physique.

Ce test instancie un env minigrid avec le même seed dans les 2 codebases et
compare les grilles initiales. S'il échoue, la comparaison paper est biaisée.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


_SNIPPET = r"""
import sys, hashlib, numpy as np
sys.path.insert(0, __ROOT__)
bad = [m for m in list(sys.modules) if m.startswith("uncertain_worms")]
for m in bad:
    del sys.modules[m]
from uncertain_worms.environments.minigrid.minigrid_env import MinigridEnvironment
env = MinigridEnvironment(env_name="MyMiniGrid-LavaWall-v0", max_steps=40, fully_obs=False)
s = env.reset(seed=0)
print("agent_pos=" + str(tuple(s.agent_pos)))
print("agent_dir=" + str(int(s.agent_dir)))
h = hashlib.sha256(np.ascontiguousarray(s.grid).tobytes()).hexdigest()[:16]
print("grid_hash=" + h)
"""


def _run_side(side: str) -> dict:
    root = str(REPO / "curtis_baseline") if side == "curtis" else str(REPO)
    code = _SNIPPET.replace("__ROOT__", repr(root))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    out = {}
    for ln in proc.stdout.splitlines():
        if "=" in ln and not ln.startswith(("pybullet", "All joint")):
            k, _, v = ln.partition("=")
            out[k.strip()] = v.strip()
    return out


def test_curtis_ours_same_grid_for_same_seed() -> None:
    ours = _run_side("ours")
    curtis = _run_side("curtis")
    assert ours == curtis, (
        f"ours vs curtis initial state mismatch on seed=0\n"
        f"  ours  : {ours}\n  curtis: {curtis}"
    )


if __name__ == "__main__":
    test_curtis_ours_same_grid_for_same_seed()
    print("OK — seed reproducibility test passed")
