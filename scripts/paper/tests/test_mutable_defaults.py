"""Regression tests: containers must not be shared between instances.

A mutable default argument (``def __init__(self, episodes=[])``) is
evaluated once, so every instance built without that argument shares one
list. The consequence here was silent and specific: successive demonstration
slices accumulated into each other, and a file meant to hold N
trajectories held the ones from every previous slice as well.

Both codebases were changed to the ``Optional[T] = None`` pattern; the
test exercises one of them, since a single process cannot import both.
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
    """Writing one buffer to disk must not populate the next one."""
    import tempfile
    import os
    from uncertain_worms.structs import ReplayBuffer, Episode
    tmp = tempfile.mkdtemp()
    a = ReplayBuffer()
    a.episodes.append(Episode())
    a.save_to_file(os.path.join(tmp, "a.pkl"))
    # A buffer created afterwards must still be empty.
    c = ReplayBuffer()
    assert len(c.episodes) == 0


if __name__ == "__main__":
    test_replay_buffer_no_shared_episodes()
    test_particle_belief_no_shared_particles()
    test_replay_buffer_save_load_roundtrip()
    print("OK — all mutable defaults tests passed")
