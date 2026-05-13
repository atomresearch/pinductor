"""E4 figure : 2 envs × 3 LLMs grouped bars.

Data sources :
  - qwen3-14b      : E4_qwen__qwen3_14b
  - qwen3.6-plus   : E2_offline_ND10 condition='ours' (default LLM)
  - claude-opus-4.7: E4_anthropic__claude_opus_4_7

For qwen3.6-plus we scan the disk (``outputs/paper_runs/runs/E2_offline_ND10__<env>__ours__*``)
because the registry rows for the regression batches were purged but the
on-disk ``result.json`` files survive. We deduplicate per (seed, ep) by
keeping the row with the latest mtime.

y-axis : mean episode reward (avec error bars = SEM).
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


def fetch_rewards(conn: sqlite3.Connection, exp_filter: str, env: str) -> list:
    cur = conn.cursor()
    cur.execute(
        "SELECT reward FROM runs WHERE " + exp_filter + " AND env=? "
        "AND status IN ('done','skipped_dedup') AND reward IS NOT NULL",
        (env,),
    )
    return [r[0] for r in cur.fetchall()]


def fetch_rewards_disk(exp_id: str, env: str, condition: str) -> list:
    """Disk-scan: parses ``result.json`` from every run dir matching
    ``<exp_id>__<env>__<condition>__s<seed>__<hash>``.

    Per (seed, ep) keeps the value from the **oldest** result.json
    (smallest mtime). Anchors plots on the earliest surviving run per
    cell rather than the latest re-write or the best score, to avoid
    post-hoc cherry-picking after config drifts re-tagged cells with
    new hp_hashes.
    """
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

    llms = [
        ("qwen3-14b",      "exp_id='E4_qwen__qwen3_14b'", None),
        ("qwen3.6-plus",   "exp_id='E2_offline_ND10' AND condition='ours'",
         ("E2_offline_ND10", "ours")),
        ("opus 4.7",       "exp_id='E4_anthropic__claude_opus_4_7'", None),
    ]
    envs = ["lava", "unlock"]

    # Color per LLM (colorblind-safe)
    colors = ["#E69F00", "#56B4E9", "#009E73"]

    # Mean reward + SEM ; WR = fraction of episodes with reward > 0.05
    # (annotated above each bar).
    WR_THR = 0.05
    data = {}  # data[(env, llm_name)] = (mean, sem, wr, n)
    for env in envs:
        for (llm_name, llm_filter, disk_fallback) in llms:
            rewards = fetch_rewards(conn, llm_filter, env)
            if disk_fallback is not None:
                exp_id, condition = disk_fallback
                disk_rewards = fetch_rewards_disk(exp_id, env, condition)
                if len(disk_rewards) > len(rewards):
                    rewards = disk_rewards
            if not rewards:
                data[(env, llm_name)] = (0.0, 0.0, 0.0, 0)
            else:
                arr = np.array(rewards)
                mean = float(arr.mean())
                n = len(arr)
                sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                wr = float(np.sum(arr > WR_THR)) / n
                data[(env, llm_name)] = (mean, sem, wr, n)

    # Plot : 2 env groups, 3 bars per group
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    n_envs = len(envs)
    n_llms = len(llms)
    bar_w = 0.25
    x_base = np.arange(n_envs)

    for i, llm_tup in enumerate(llms):
        llm_name = llm_tup[0]
        means = [data[(env, llm_name)][0] for env in envs]
        sems = [data[(env, llm_name)][1] for env in envs]
        wrs = [data[(env, llm_name)][2] for env in envs]
        x_pos = x_base + (i - 1) * bar_w
        ax.bar(
            x_pos, means, bar_w, yerr=sems, color=colors[i],
            label=f"{llm_name}", edgecolor="black", linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )
        # WR annotation above each bar
        for j, (m, s, wr) in enumerate(zip(means, sems, wrs)):
            ax.text(
                x_pos[j], m + s + 0.02, f"{wr:.0%}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels([e.replace("_", " ").title() for e in envs])
    ax.set_ylabel("Mean episode reward")
    ax.set_title("E4 — LLM ablation (mean ± SEM ; WR above bar, n=30)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()

    # Save
    pdf = OUT / "E4_3llms_2envs_grouped.pdf"
    png = OUT / "E4_3llms_2envs_grouped.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    print()
    print("Data table :")
    print(f"  {'env':<8} {'LLM':<16} {'mean':>6} {'sem':>6} {'WR':>6} {'n':>4}")
    for env in envs:
        for llm_tup in llms:
            llm_name = llm_tup[0]
            mean, sem, wr, n = data[(env, llm_name)]
            print(f"  {env:<8} {llm_name:<16} {mean:>6.3f} {sem:>6.3f} {wr:>5.1%} {n:>4d}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
