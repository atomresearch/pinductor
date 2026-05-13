"""Régression : mutable default args shared entre instances (Python gotcha).

Fix historique : ``def __init__(self, episodes=[])`` partage la MÊME liste
entre toutes les instances. Bug très subtil — dans ``regenerate_datasets.py``
ça a créé des pkl cumulatifs (N=10 avait 19 trajs au lieu de 10).

Ce test crée 2 instances indépendantes sans argument, modifie l'une, et
vérifie que l'autre reste intacte.

Le test tourne dans un seul contexte (ours) ; on a fixé les deux codebases
(curtis + ours) avec le même pattern ``Optional[T] = None; self.x = x or default()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root


def test_replay_buffer_no_shared_episodes() -> None:
    from uncertain_worms.structs import ReplayBuffer, Episode
    a = ReplayBuffer()
    b = ReplayBuffer()
    assert a.episodes is not b.episodes, (
        "ReplayBuffer() shares mutable default episodes list!")
    a.episodes.append(Episode())
    assert len(a.episodes) == 1 and len(b.episodes) == 0, (
        f"a={len(a.episodes)} b={len(b.episodes)} — shared!")


def test_particle_belief_no_shared_particles() -> None:
    from uncertain_worms.structs import ParticleBelief
    a = ParticleBelief()
    b = ParticleBelief()
    assert a.particles is not b.particles, "shared default particles dict!"
    a.particles[42] = 1
    assert 42 not in b.particles, "shared!"


def test_replay_buffer_save_load_roundtrip(tmp_path=None) -> None:
    """Sanity : save/load ne fuit pas entre instances via default list."""
    import tempfile
    import os
    from uncertain_worms.structs import ReplayBuffer, Episode
    tmp = tempfile.mkdtemp()
    a = ReplayBuffer()
    a.episodes.append(Episode())
    a.save_to_file(os.path.join(tmp, "a.pkl"))
    # 2e buffer avant save-load : doit toujours être vide
    c = ReplayBuffer()
    assert len(c.episodes) == 0


if __name__ == "__main__":
    test_replay_buffer_no_shared_episodes()
    test_particle_belief_no_shared_particles()
    test_replay_buffer_save_load_roundtrip()
    print("OK — all mutable defaults tests passed")
