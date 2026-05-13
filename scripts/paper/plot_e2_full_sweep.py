"""E2 full sweep figure : 4 envs × 6 NDs × 3 conditions.

Layout : single x-axis with sequential per-env blocks. Within each
env block, ND ∈ {2, 4, 6, 8, 10, 12} groups, each group holding 3
bars (ours / curtis_their / tabular). Vertical dotted line between
env blocks.

y-axis : mean episode reward (mean ± SEM). WR annotated above each
bar.

Disk-scan fallback : when an (exp_id, env, cond) bucket has < 30
rows in the registry, augment from the on-disk ``result.json``
files using best-per-(seed, ep) anti-regression dedup.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.neurips2024(usetex=False, family="serif"))
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[2]
DB = _REPO / "outputs" / "paper_runs" / "registry.db"
RUNS = _REPO / "outputs" / "paper_runs" / "runs"
OUT = _REPO / "outputs" / "paper_runs" / "plots" / "pretty"
OUT.mkdir(parents=True, exist_ok=True)
WR_THR = 0.05

ENVS = ["lava", "unlock", "four_rooms", "corners"]
NDS = [2, 4, 6, 8, 10, 12]
CONDS = [
    ("ours",         "#56B4E9"),
    ("curtis_their", "#D55E00"),
    ("tabular",      "#999999"),
]


def fetch_rewards(conn, exp_id: str, env: str, condition: str) -> list:
    cur = conn.cursor()
    cur.execute(
        "SELECT reward FROM runs WHERE exp_id=? AND env=? AND condition=? "
        "AND status IN ('done','skipped_dedup') AND reward IS NOT NULL",
        (exp_id, env, condition),
    )
    return [r[0] for r in cur.fetchall()]


def fetch_rewards_disk(exp_id: str, env: str, condition: str) -> list:
    """Per (seed, ep) keeps the value from the oldest result.json
    (smallest mtime) — anchors plots on the earliest surviving run."""
    pattern = re.compile(
        rf"^{re.escape(exp_id)}__{re.escape(env)}__{re.escape(condition)}__s(\d+)__"
    )
    rows: dict = {}  # (seed, ep) -> (mtime, reward)
    for run_dir in RUNS.glob(f"{exp_id}__{env}__{condition}__s*"):
        m = pattern.match(run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        rj = run_dir / "result.json"
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text())
            mtime = rj.stat().st_mtime
        except Exception:
            continue
        rewards = d.get("rewards") or []
        for ep, r in enumerate(rewards):
            key = (seed, ep)
            r_f = float(r)
            cur = rows.get(key)
            if cur is None or mtime < cur[0]:
                rows[key] = (mtime, r_f)
    return [v[1] for v in rows.values()]


def main() -> int:
    conn = sqlite3.connect(DB)

    # Compute (mean, sem, wr, n) per (env, ND, cond)
    data = {}
    for env in ENVS:
        for nd in NDS:
            exp_id = f"E2_offline_ND{nd}"
            for cond, _ in CONDS:
                rewards = fetch_rewards(conn, exp_id, env, cond)
                disk_rewards = fetch_rewards_disk(exp_id, env, cond)
                if len(disk_rewards) > len(rewards):
                    rewards = disk_rewards
                if not rewards:
                    data[(env, nd, cond)] = (0.0, 0.0, 0.0, 0)
                else:
                    arr = np.array(rewards)
                    n = len(arr)
                    mean = float(arr.mean())
                    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    wr = float(np.sum(arr > WR_THR)) / n
                    data[(env, nd, cond)] = (mean, sem, wr, n)
    conn.close()

    # Layout : x positions for each (env, nd, cond) bar
    # Within an env block : 6 NDs × 3 conds = 18 bars + 3 spacing
    # Between env blocks : separator gap
    bar_w = 0.25
    nd_spacing = bar_w * len(CONDS) + 0.4   # space per ND group (3 bars + small pad)
    env_gap = nd_spacing * 0.6              # gap between envs

    fig, ax = plt.subplots(figsize=(12.0, 3.6))

    x_cursor = 0.0
    env_block_centers = []
    nd_xticks = []
    nd_xlabels = []
    env_separators = []

    for env_idx, env in enumerate(ENVS):
        block_start = x_cursor
        for nd in NDS:
            nd_center = x_cursor
            for i, (cond, color) in enumerate(CONDS):
                m, sem, wr, n = data[(env, nd, cond)]
                x_pos = nd_center + (i - 1) * bar_w
                ax.bar(
                    x_pos, m, bar_w, yerr=sem, color=color,
                    label=cond if (env_idx == 0 and nd == NDS[0]) else None,
                    edgecolor="black", linewidth=0.4,
                    error_kw={"elinewidth": 0.6, "capsize": 1.5},
                )
                if n > 0:
                    ax.text(
                        x_pos, m + sem + 0.015, f"{int(wr*100)}",
                        ha="center", va="bottom", fontsize=5.5,
                    )
            nd_xticks.append(nd_center)
            nd_xlabels.append(str(nd))
            x_cursor += nd_spacing
        env_block_centers.append((block_start + x_cursor - nd_spacing) / 2.0)
        if env_idx < len(ENVS) - 1:
            sep_x = x_cursor - nd_spacing / 2.0
            env_separators.append(sep_x)
            x_cursor += env_gap

    # vertical separators between env blocks
    for sep_x in env_separators:
        ax.axvline(sep_x, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    # x ticks : NDs at the bottom
    ax.set_xticks(nd_xticks)
    ax.set_xticklabels(nd_xlabels, fontsize=7)

    # env labels above each block
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(env_block_centers)
    ax_top.set_xticklabels(
        [e.replace("_", " ").title() for e in ENVS],
        fontsize=9, fontweight="bold",
    )
    ax_top.tick_params(length=0, pad=2)

    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel(r"$N_D$ (number of demos)")
    ax.set_title(
        "E2 — full sweep (mean ± SEM ; WR% above bar, n=30)",
        pad=18,
    )
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()

    pdf = OUT / "E2_full_sweep_4envs_6NDs_3conds.pdf"
    png = OUT / "E2_full_sweep_4envs_6NDs_3conds.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    print()
    print(f"  {'env':<12} {'ND':>3} {'cond':<14} {'mean':>6} {'sem':>6} {'WR':>6} {'n':>4}")
    for env in ENVS:
        for nd in NDS:
            for cond, _ in CONDS:
                m, s, wr, n = data[(env, nd, cond)]
                print(f"  {env:<12} {nd:>3d} {cond:<14} {m:>6.3f} {s:>6.3f} {wr:>5.1%} {n:>4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
