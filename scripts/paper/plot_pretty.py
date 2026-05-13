"""plot_pretty.py — Paper-ready plots with colorblind-safe palette + raw dots.

Improvements over plot_clean.py :
  - NDS extended to {2,4,6,8,10,12}
  - Conditions = (curtis_their, ours, tabular) where applicable
  - Wong colorblind-safe palette
  - CI95 error bars + raw episode dots (jittered)
  - Consistent serif font
  - Larger axes, no cryptic legends
  - Output in PNG (web) + PDF (paper)

Usage : python scripts/paper/plot_pretty.py
Output : outputs/paper_runs/plots/pretty/
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paper-style plot bundle from tueplots (figure size, fontsizes,
# constrained_layout). usetex disabled so LaTeX is not required at runtime.
try:
    from tueplots import bundles
    plt.rcParams.update(bundles.neurips2024(usetex=False, family="serif"))
except Exception:
    # Fallback: tueplots not installed — keep matplotlib defaults below.
    pass

# Wong (Nature Methods 2011) colorblind-safe palette.
WONG = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
}

REPO = Path(__file__).resolve().parent.parent.parent
DB = REPO / "outputs" / "paper_runs" / "registry.db"
OUT = REPO / "outputs" / "paper_runs" / "plots" / "pretty"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD_PCT = 0.0  # legacy, unused — see _is_admissible
EXPECTED_N_PER_CELL = 30  # 10 seeds × 3 ep — strict requirement 
COLOR = {
    "curtis_their": WONG["skyblue"],
    "ours":         WONG["vermilion"],
    "tabular":      WONG["green"],
}
LABEL = {
    "curtis_their": "Baseline",
    "ours":         "Ours (REx)",
    "tabular":      "Tabular",
}

# Consistent paper-style font.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _ci95(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    s = stdev(values)
    return 1.96 * s / (len(values) ** 0.5)


def fetch_group(conn: sqlite3.Connection, exp_id: str, env: str, cond: str
                ) -> Tuple[List[float], int]:
    """Returns (rewards_done, total) — excludes dedup-marked rows from done count."""
    cur = conn.cursor()
    cur.execute(
        "SELECT status, reward FROM runs "
        "WHERE exp_id=? AND env=? AND condition=? "
        "AND status NOT IN ('superseded', 'skipped_dedup')",
        (exp_id, env, cond),
    )
    rows = cur.fetchall()
    rewards = [float(r[1]) for r in rows if r[0] == "done" and r[1] is not None]
    return rewards, len(rows)


def _is_admissible(rewards_done: List[float], total: int) -> bool:
    """A cell is admissible iff it has the full expected episode count."""
    return len(rewards_done) >= EXPECTED_N_PER_CELL


def _add_raw_dots(ax, x_center: float, rewards: List[float], color: str) -> None:
    """Overlay individual episode rewards as small jittered dots."""
    rng = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.06, 0.06, size=len(rewards))
    ax.scatter(
        np.full(len(rewards), x_center) + jitter,
        rewards,
        s=14, color=color, edgecolor="black",
        linewidth=0.4, alpha=0.55, zorder=3,
    )


def plot_e2_offline(conn: sqlite3.Connection) -> None:
    """Bar chart per ND, conditions side-by-side, one figure per env."""
    NDS = [2, 4, 6, 8, 10, 12]
    CONDS = ("curtis_their", "ours", "tabular")
    # Cells without the full expected count are skipped silently so this
    # remains safe to run mid-bench.
    for env in ("lava", "unlock", "four_rooms", "corners"):
        bars: List[Tuple[int, str, float, float, int, List[float]]] = []
        for nd in NDS:
            exp = f"E2_offline_ND{nd}"
            for cond in CONDS:
                rewards, total = fetch_group(conn, exp, env, cond)
                if not _is_admissible(rewards, total):
                    print(
                        f"  [skip] {exp}/{env}/{cond}: "
                        f"{len(rewards)}/{total} done < {THRESHOLD_PCT:.0f}%"
                    )
                    continue
                bars.append(
                    (nd, cond, mean(rewards), _ci95(rewards),
                     len(rewards), rewards)
                )

        if not bars:
            print(f"[skip plot] E2_offline {env}: no admissible cell")
            continue

        nds_present = sorted({b[0] for b in bars})
        # Conditions effectively present (across NDs)
        conds_present = [c for c in CONDS if any(b[1] == c for b in bars)]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(nds_present), dtype=float)
        width = 0.85 / max(1, len(conds_present))
        offset_base = -(len(conds_present) - 1) * width / 2.0

        for i, cond in enumerate(conds_present):
            xs = []
            ys = []
            errs = []
            ns = []
            raws = []
            for j, nd in enumerate(nds_present):
                match = next((b for b in bars if b[0] == nd and b[1] == cond), None)
                if match is None:
                    continue
                xs.append(x[j] + offset_base + i * width)
                ys.append(match[2])
                errs.append(match[3])
                ns.append(match[4])
                raws.append(match[5])
            ax.bar(
                xs, ys, width, yerr=errs, capsize=3,
                label=LABEL[cond], color=COLOR[cond], edgecolor="black",
                linewidth=0.6, alpha=0.92,
            )
            for xc, yval, n, raw in zip(xs, ys, ns, raws):
                _add_raw_dots(ax, xc, raw, color=COLOR[cond])
                ax.text(
                    xc, yval + 0.02, f"n={n}", ha="center", va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={nd}" for nd in nds_present])
        ax.set_xlabel("Number of demonstrations $N_D$")
        ax.set_ylabel("Mean episode reward")
        ax.set_title(f"E2 offline — {env} — sample efficiency")
        ymax = max(b[2] + b[3] for b in bars)
        ax.set_ylim(0, max(1.0, ymax * 1.18))
        ax.legend(loc="upper left", frameon=False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"E2_offline_{env}.{ext}")
        plt.close(fig)
        print(
            f"[saved] E2_offline {env}: {len(nds_present)} NDs × "
            f"{len(conds_present)} conds = {len(bars)} bars"
        )


def plot_e2_offline_grid(conn: sqlite3.Connection) -> None:
    """Single 2x2 figure (4 envs) for paper main figure.

    Same data as `plot_e2_offline`, but in a single shared-axes grid so
    the paper has one canonical sample-efficiency figure instead of 4
    standalone PNGs. Saves both PNG (web) and PDF (paper).
    """
    NDS = [2, 4, 6, 8, 10, 12]
    CONDS = ("curtis_their", "ours", "tabular")
    ENVS = ("lava", "unlock", "four_rooms", "corners")
    ENV_TITLE = {
        "lava": "Lava-Wall",
        "unlock": "Unlock",
        "four_rooms": "Four-Rooms",
        "corners": "Corners",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes_flat = axes.flat

    any_data = False
    for ax, env in zip(axes_flat, ENVS):
        bars: List[Tuple[int, str, float, float, int, List[float]]] = []
        for nd in NDS:
            exp = f"E2_offline_ND{nd}"
            for cond in CONDS:
                rewards, total = fetch_group(conn, exp, env, cond)
                if not _is_admissible(rewards, total):
                    continue
                bars.append((nd, cond, mean(rewards), _ci95(rewards),
                             len(rewards), rewards))

        if not bars:
            ax.set_title(f"{ENV_TITLE[env]} (no data)")
            ax.set_xticks([])
            continue
        any_data = True

        nds_present = sorted({b[0] for b in bars})
        conds_present = [c for c in CONDS if any(b[1] == c for b in bars)]

        x = np.arange(len(nds_present), dtype=float)
        width = 0.85 / max(1, len(conds_present))
        offset_base = -(len(conds_present) - 1) * width / 2.0

        for i, cond in enumerate(conds_present):
            xs, ys, errs, ns, raws = [], [], [], [], []
            for j, nd in enumerate(nds_present):
                match = next((b for b in bars if b[0] == nd and b[1] == cond), None)
                if match is None:
                    continue
                xs.append(x[j] + offset_base + i * width)
                ys.append(match[2])
                errs.append(match[3])
                ns.append(match[4])
                raws.append(match[5])
            ax.bar(
                xs, ys, width, yerr=errs, capsize=2,
                label=LABEL[cond], color=COLOR[cond], edgecolor="black",
                linewidth=0.5, alpha=0.92,
            )
            for xc, raw in zip(xs, raws):
                _add_raw_dots(ax, xc, raw, color=COLOR[cond])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{nd}" for nd in nds_present])
        ax.set_title(ENV_TITLE[env])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Shared X label / Y label on outer
    for ax in axes[1]:
        ax.set_xlabel("$N_D$ (number of demonstrations)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean episode reward")

    # Single shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        # Pick first axis with any handles
        for ax in axes_flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("E2 offline — sample efficiency across 4 environments (n=30 / cell)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if any_data:
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"E2_offline_grid_4envs.{ext}")
        print(f"[saved] E2_offline_grid_4envs.{{png,pdf}}")
    plt.close(fig)


def plot_single_exp(conn: sqlite3.Connection, exp_id: str, label: str) -> None:
    """Bar chart, 1 figure per env, conditions side-by-side."""
    CONDS = ("curtis_their", "ours", "tabular")
    for env in ("lava", "unlock"):
        bars: List[Tuple[str, float, float, int, List[float]]] = []
        for cond in CONDS:
            rewards, total = fetch_group(conn, exp_id, env, cond)
            if not _is_admissible(rewards, total):
                continue
            bars.append((cond, mean(rewards), _ci95(rewards), len(rewards), rewards))

        if not bars:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(bars))
        for i, (cond, m, ci, n, raw) in enumerate(bars):
            ax.bar(
                x[i], m, 0.65, yerr=ci, capsize=4,
                label=LABEL[cond], color=COLOR[cond], edgecolor="black",
                linewidth=0.6, alpha=0.92,
            )
            _add_raw_dots(ax, x[i], raw, color=COLOR[cond])
            ax.text(x[i], m + 0.02, f"n={n}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[b[0]] for b in bars])
        ax.set_ylabel("Mean episode reward")
        ax.set_title(f"{label} — {env}")
        ymax = max(b[1] + b[2] for b in bars)
        ax.set_ylim(0, max(1.0, ymax * 1.18))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{exp_id}_{env}.{ext}")
        plt.close(fig)
        print(f"[saved] {exp_id} {env}: {len(bars)} bars")


def main() -> int:
    if not DB.exists():
        print(f"ERROR : registry not found at {DB}")
        return 1
    conn = sqlite3.connect(DB)
    print(f"--- E2_offline per-env (4 envs) ---")
    plot_e2_offline(conn)
    print(f"\n--- E2_offline grid 2x2 (paper main figure) ---")
    plot_e2_offline_grid(conn)
    print(f"\n--- E4_qwen__qwen3_14b ---")
    plot_single_exp(conn, "E4_qwen__qwen3_14b", "E4 — Qwen3-14B small model")
    print(f"\n--- E4_anthropic__claude_opus_4_7 ---")
    plot_single_exp(conn, "E4_anthropic__claude_opus_4_7", "E4 — Claude Opus 4.7 frontier")
    conn.close()
    print(f"\nAll plots in {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
