"""Regression tests for grouping atoms that several experiments share.

Experiments that resolve to identical overrides produce the same hash and
are merged into one group by ``group_atoms``, so one subprocess serves all
of them. The failure this pins down is that the group's atoms are then
keyed only by episode index, which collides across experiments: one atom
gets its result and the others stay pending forever, with no error
anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import Atom
from paper_runner import group_atoms
from hyperparams import DEFAULT_LLM


def test_group_atoms_preserves_multi_exp() -> None:
    """Experiments with identical overrides share a single group."""
    atoms = [
        Atom(exp_id="EXP_A", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM, extra={}),
        Atom(exp_id="EXP_B", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM, extra={}),
        Atom(exp_id="EXP_C", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM, extra={}),
    ]
    groups = group_atoms(atoms)
    assert len(groups) == 1, f"expected 1 group (same hp_hash), got {len(groups)}"
    assert len(groups[0].atoms) == 3, (
        f"expected 3 atoms in group, got {len(groups[0].atoms)}")
    # Each experiment must still be represented separately.
    exp_ids = {a.exp_id for a in groups[0].atoms}
    assert exp_ids == {"EXP_A", "EXP_B", "EXP_C"}


def test_group_atoms_separates_different_hp_hash() -> None:
    """A difference in `extra` separates the groups."""
    atoms = [
        Atom(exp_id="EXP_A", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM, extra={}),
        Atom(exp_id="EXP_B", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM,
             extra={"agent.num_model_attempts": 7}),  # diff (default=5)
    ]
    groups = group_atoms(atoms)
    assert len(groups) == 2


def test_launch_group_multi_exp_run_ids_unique() -> None:
    """Every experiment of a group gets its own registry row.

    Reproduces what launch_group does, without the subprocess.
    """
    import tempfile
    from registry import Registry
    from paper_runner import group_atoms
    from hyperparams import resolve_overrides, hp_hash

    tmp = Path(tempfile.mkdtemp()) / "test_registry.db"
    registry = Registry(tmp)

    atoms = [
        Atom(exp_id=f"EXP_{i}", env="lava", condition="ours", seed=0,
             episode_idx=0, llm_model=DEFAULT_LLM, extra={})
        for i in range(5)
    ]
    groups = group_atoms(atoms)
    assert len(groups) == 1
    g = groups[0]
    # One pending row per atom of the group.
    rids = []
    for atom in g.atoms:
        rid = registry.upsert_pending(
            exp_id=atom.exp_id, env=g.env, condition=g.condition,
            seed=g.seed, episode_idx=atom.episode_idx, hp_hash=g.hp_hash,
            llm_model=g.llm_model, overrides=g.overrides,
        )
        rids.append(rid)
    # Distinct ids: no atom overwrote another.
    assert len(set(rids)) == 5, f"expected 5 distinct rids, got {set(rids)}"


if __name__ == "__main__":
    test_group_atoms_preserves_multi_exp()
    test_group_atoms_separates_different_hp_hash()
    test_launch_group_multi_exp_run_ids_unique()
    print("OK — all multi_exp_group tests passed")
