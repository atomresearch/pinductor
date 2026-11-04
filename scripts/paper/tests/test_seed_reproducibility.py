"""The two codebases must build the same environment from the same seed.

Every comparison in the paper is paired: for a given seed and episode,
method and baseline are meant to face the identical instance. That holds
only if both reset their environment from the seed the runner passes. A
codebase that draws its own random seed instead evaluates on a different
grid, and the resulting difference in reward measures the instances rather
than the methods.

The test resets one MiniGrid environment on each side with the same seed
and compares the initial states. A failure here invalidates the paired
comparison itself, not only this file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


# Source run in a subprocess per side: only the import root differs, and
# the grid is compared through a hash so the two states never have to be
# unpickled in the same interpreter.
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
    """Return the initial state one codebase produces for seed 0."""
    root = str(REPO / "curtis_baseline") if side == "curtis" else str(REPO)
    code = _SNIPPET.replace("__ROOT__", repr(root))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    out = {}
    for ln in proc.stdout.splitlines():
        # Imported simulation libraries print banners containing "=";
        # those prefixes exclude them from the parsed output.
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
