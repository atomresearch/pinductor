"""Plot learning curve E2_online_K10 : reward vs episode_idx.

Pour chaque (env, condition), une ligne avec shaded CI95.
Filtre : skip une condition si moins de 7 done à n'importe quel ep_idx
(7 = 70 % de 10 seeds attendus par ep_idx).

Usage : python scripts/paper/plot_progression.py
Output : outputs/paper_runs/plots/E2_online_K10_progression_<env>.{png,pdf}
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
DB = REPO / "outputs" / "paper_runs" / "registry.db"
OUT = REPO / "outputs" / "paper_runs" / "plots"

EXP = "E2_online_K10"
N_EPISODES = 10
MIN_DONE_PER_EP = 7  # 70 % de 10 seeds
COLOR = {"curtis_their": "#1f77b4", "ours": "#ff7f0e"}
LABEL = {"curtis_their": "curtis", "ours": "ours (REx)"}


def _ci95(values):
    if len(values) < 2:
        return 0.0
    s = stdev(values)
    return 1.96 * s / (len(values) ** 0.5)


def fetch_per_ep(conn, exp, env, cond):
    """Retourne dict ep_idx -> list[reward] pour les rows status=done."""
    cur = conn.cursor()
    cur.execute(
        "SELECT episode_idx, reward FROM runs "
        "WHERE exp_id=? AND env=? AND condition=? AND status=\"done\" "
        "AND reward IS NOT NULL "
        "ORDER BY episode_idx",
        (exp, env, cond),
    )
    out = {}
    for r in cur.fetchall():
        out.setdefault(int(r[0]), []).append(float(r[1]))
    return out


def is_admissible(per_ep):
    """True si chaque ep_idx 0..N_EPISODES-1 a >= MIN_DONE_PER_EP done."""
    if not per_ep:
        return False
    for ep in range(N_EPISODES):
        if len(per_ep.get(ep, [])) < MIN_DONE_PER_EP:
            return False
    return True


def plot_env(conn, env):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = []
    for cond in ("curtis_their", "ours"):
        per_ep = fetch_per_ep(conn, EXP, env, cond)
        if not is_admissible(per_ep):
            ns = [len(per_ep.get(ep, [])) for ep in range(N_EPISODES)]
            print(
                f"[skip] {EXP}/{env}/{cond}: n_done per ep = {ns} "
                f"(some < {MIN_DONE_PER_EP})"
            )
            continue
        eps = list(range(N_EPISODES))
        means = [mean(per_ep[e]) for e in eps]
        cis = [_ci95(per_ep[e]) for e in eps]
        ns = [len(per_ep[e]) for e in eps]
        means = np.array(means)
        cis = np.array(cis)
        ax.plot(
            eps, means, "-o", label=LABEL[cond], color=COLOR[cond],
            linewidth=2, markersize=6,
        )
        ax.fill_between(
            eps, means - cis, means + cis, color=COLOR[cond], alpha=0.18,
        )
        plotted.append((cond, ns))
        print(
            f"[ok] {EXP}/{env}/{cond}: ns={ns} "
            f"means={[f'{m:.3f}' for m in means]}"
        )

    if not plotted:
        print(f"[skip plot] {env}: no admissible curve")
        plt.close(fig)
        return

    ax.set_xlabel("episode index (online learning step)")
    ax.set_ylabel("mean reward")
    ax.set_title(f"E2 online K=10 — {env} — learning curve")
    ax.set_xticks(range(N_EPISODES))
    ax.set_xticklabels([str(i) for i in range(N_EPISODES)])
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    # Tag n in caption
    n_caption = " | ".join(
        f"{LABEL[c]} n_per_ep ∈ [{min(ns)},{max(ns)}]"
        for c, ns in plotted
    )
    ax.text(
        0.5, -0.2, n_caption, transform=ax.transAxes, ha="center", va="top",
        fontsize=8, color="#444",
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{EXP}_progression_{env}.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {EXP}_progression_{env}.png/pdf")


def main():
    conn = sqlite3.connect(DB)
    print(f"--- learning curves {EXP} (threshold {MIN_DONE_PER_EP} done / ep) ---")
    for env in ("lava", "unlock"):
        plot_env(conn, env)
    conn.close()


if __name__ == "__main__":
    main()
