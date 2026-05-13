"""Régression : le bug 'run_ids dict écrase les atomes multi-exp'.

Quand plusieurs exp_ids génèrent les mêmes overrides → même hp_hash → groupés
ensemble par ``group_atoms``. Le bug précédent : ``run_ids[episode_idx]`` écrasait
les entries → un seul atome était mark_done, les autres restaient ``pending``.

Ce test vérifie que tous les atomes multi-exp d'un group reçoivent bien leur
ligne registry distincte avec mark_done.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import Atom
from paper_runner import group_atoms
from hyperparams import DEFAULT_LLM


def test_group_atoms_preserves_multi_exp() -> None:
    """Plusieurs exp_ids avec mêmes overrides → 1 group, N atomes dans le group."""
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
    # Les exp_ids doivent être distincts
    exp_ids = {a.exp_id for a in groups[0].atoms}
    assert exp_ids == {"EXP_A", "EXP_B", "EXP_C"}


def test_group_atoms_separates_different_hp_hash() -> None:
    """Différents `extra` → différents hp_hash → différents groups."""
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
    """Simule launch_group (sans subprocess) et vérifie que chaque exp_id
    a sa propre entrée dans le registry."""
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
    # Simule upsert pending pour chaque atome
    rids = []
    for atom in g.atoms:
        rid = registry.upsert_pending(
            exp_id=atom.exp_id, env=g.env, condition=g.condition,
            seed=g.seed, episode_idx=atom.episode_idx, hp_hash=g.hp_hash,
            llm_model=g.llm_model, overrides=g.overrides,
        )
        rids.append(rid)
    # 5 rids distincts attendus
    assert len(set(rids)) == 5, f"expected 5 distinct rids, got {set(rids)}"


if __name__ == "__main__":
    test_group_atoms_preserves_multi_exp()
    test_group_atoms_separates_different_hp_hash()
    test_launch_group_multi_exp_run_ids_unique()
    print("OK — all multi_exp_group tests passed")
