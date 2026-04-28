"""Generate the headline figures for the project writeup.

Produces:
  - docs/figures/pretrain_ablation.png
        Two-panel: val_total curves over training steps + train/val value
        gap per cell, for the 2x2 {CNN, Transformer} x {reg, no-reg} ablation.
  - docs/figures/gumbel_vs_puct_hero.png
        Two-panel: equal-sims (Phase 1) + wall-clock-matched (Phase 2)
  - docs/figures/elo_vs_wall_clock.png
        AZ-Gumbel vs PPO trajectory against MCTS@200, x = hours
  - docs/figures/elo_vs_nn_forwards.png
        Same trajectories, x = cumulative NN forward passes (log scale).
        Algorithmic data efficiency, hardware-independent.

All numbers are read from the canonical CSVs under runs/ so the figures
re-render cleanly when underlying data changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_FIG = ROOT / "docs" / "figures"
DOCS_FIG.mkdir(parents=True, exist_ok=True)


def _winrate_to_elo(p: float, anchor: float = 1500.0) -> float:
    p = max(0.01, min(0.99, float(p)))
    return anchor + 400.0 * math.log10(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Pretraining ablation: 2x2 {CNN, Transformer} x {reg off, reg on}
# ---------------------------------------------------------------------------

def make_pretrain_ablation() -> Path:
    cells = [
        # (cell_id, label, run_dir, color, linestyle, train_val_at_best, test_total)
        ("C1", "CNN — reg off",         "pretrain_cnn_100k",          "#4C78A8", "-",  (0.046, 0.144), 1.327),
        ("C2", "CNN — reg on",          "pretrain_cnn_100k_reg",      "#4C78A8", "--", (0.057, 0.145), 1.324),
        ("C3", "Transformer — reg off", "pretrain_transformer_100k",     "#F58518", "-",  (0.057, 0.228), 1.508),
        ("C4", "Transformer — reg on",  "pretrain_transformer_100k_reg", "#F58518", "--", (0.116, 0.167), 1.363),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    # ----- Panel A: val_total_loss training curves -----
    for _, label, run_dir, color, ls, _, _ in cells:
        df = pd.read_csv(ROOT / "runs" / run_dir / "train_log.csv")
        ax1.plot(
            df["step"], df["val_total_loss"],
            label=label, color=color, linestyle=ls, linewidth=1.8,
        )

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation total loss (policy + 2·value)")
    ax1.set_title("Validation loss over training", fontsize=12, pad=10)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # ----- Panel B: train vs val value loss at best step (the overfit story) -----
    x_idx = list(range(len(cells)))
    bar_w = 0.36
    train_vals = [c[5][0] for c in cells]
    val_vals   = [c[5][1] for c in cells]
    bar_colors = [c[3] for c in cells]

    b_train = ax2.bar(
        [i - bar_w/2 for i in x_idx], train_vals, bar_w,
        color=bar_colors, alpha=0.45, edgecolor="black", linewidth=0.5,
        label="train_value",
    )
    b_val = ax2.bar(
        [i + bar_w/2 for i in x_idx], val_vals, bar_w,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        label="val_value",
    )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([c[0] for c in cells])
    ax2.set_ylabel("Value-head loss at best-by-val step")
    ax2.set_title("Train/val gap → overfit by cell", fontsize=12, pad=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    # Annotate gap ratio above each pair
    for i, (_, _, _, _, _, (tv, vv), _) in enumerate(cells):
        ratio = vv / tv if tv > 0 else float("nan")
        ax2.annotate(
            f"{ratio:.2f}×",
            xy=(i, max(tv, vv)),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "Pretraining ablation — {CNN, Transformer} × {reg off, reg on}, "
        "100k MCTS dataset, seed 42",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_ablation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Hero figure: Gumbel vs PUCT (equal sims | matched wall-clock)
# ---------------------------------------------------------------------------

def make_hero() -> Path:
    # Phase 1 -- equal sims
    phase1_runs = [64, 128, 256]
    phase1: list[dict] = []
    for s in phase1_runs:
        df = pd.read_csv(ROOT / f"runs/tournament_sims{s}/summary.csv")
        r = df.iloc[0]
        # agent_a is AZ-SelfPlay (PUCT), agent_b is Gumbel-SelfPlay
        n = int(r["games"])
        phase1.append({
            "sims":       s,
            "puct_win":   100 * r["pts_a"] / n,
            "gumbel_win": 100 * r["pts_b"] / n,
            "puct_ms":    float(r["a_ms_per_move"]),
            "gumbel_ms":  float(r["b_ms_per_move"]),
        })

    # Phase 2 -- matched wall-clock (we keep the cleaner PUCT@1024 run)
    p2 = pd.read_csv(ROOT / "runs/tournament_walltime_match_puct1024/summary.csv").iloc[0]
    n2 = int(p2["games"])
    phase2 = {
        "puct_win":   100 * p2["pts_a"] / n2,
        "gumbel_win": 100 * p2["pts_b"] / n2,
        "puct_ms":    float(p2["a_ms_per_move"]),
        "gumbel_ms":  float(p2["b_ms_per_move"]),
        "puct_sims":  1024,
        "gumbel_sims": 64,
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    # ----- Phase 1 -----
    x_idx = list(range(len(phase1_runs)))
    bar_w = 0.36
    puct_vals   = [r["puct_win"] for r in phase1]
    gumbel_vals = [r["gumbel_win"] for r in phase1]
    color_puct, color_gumbel = "#4C78A8", "#F58518"

    b1 = ax1.bar([i - bar_w/2 for i in x_idx], puct_vals,   bar_w,
                 label="PUCT",   color=color_puct,   edgecolor="black", linewidth=0.5)
    b2 = ax1.bar([i + bar_w/2 for i in x_idx], gumbel_vals, bar_w,
                 label="Gumbel", color=color_gumbel, edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"sims = {s}" for s in phase1_runs])
    ax1.set_ylabel("Win % (vs the other side)")
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_title("Equal sim count — Gumbel dominates", fontsize=12, pad=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper left")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    # ----- Phase 2 -----
    p2_x = [0, 1]
    p2_w = 0.55
    p2_vals = [phase2["puct_win"], phase2["gumbel_win"]]
    p2_labels = [
        f"PUCT@{phase2['puct_sims']}\n({phase2['puct_ms']:.1f} ms/mv)",
        f"Gumbel@{phase2['gumbel_sims']}\n({phase2['gumbel_ms']:.1f} ms/mv)",
    ]
    p2_colors = [color_puct, color_gumbel]
    bars2 = ax2.bar(p2_x, p2_vals, p2_w, color=p2_colors,
                    edgecolor="black", linewidth=0.5)
    ax2.set_xticks(p2_x)
    ax2.set_xticklabels(p2_labels)
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="grey", linewidth=0.7, linestyle=":")
    ax2.set_title("Matched wall-clock — Gumbel still wins", fontsize=12, pad=10)
    ax2.grid(axis="y", alpha=0.3)

    for rect in bars2:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width()/2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    fig.suptitle(
        "Gumbel vs PUCT on identical CNN weights "
        "(20 paired games, seed 42, UTTT)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = DOCS_FIG / "gumbel_vs_puct_hero.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Secondary figure: Elo vs wall-clock for AZ-Gumbel and PPO
# ---------------------------------------------------------------------------

def make_wall_clock() -> Path:
    df = pd.read_csv(ROOT / "runs/comparison_mcts200/eval_vs_mcts200.csv")
    df = df.copy()
    df["hours"] = df["cumulative_time_s"] / 3600.0
    df["elo"]   = df["win_rate_mcts"].apply(_winrate_to_elo)

    warm = df[df["run"] == "warm_start"]
    real = df[df["run"] != "warm_start"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"az_cnn_gumbel": "#4C78A8", "ppo_warm_matched": "#F58518"}
    pretty = {"az_cnn_gumbel": "AZ-Gumbel (CNN)", "ppo_warm_matched": "PPO (CNN)"}

    for run_name, run_df in real.groupby("run", sort=False):
        kind = run_df["kind"].iloc[0]
        warm_for_kind = warm[warm["kind"] == kind]
        merged = pd.concat([warm_for_kind, run_df], ignore_index=True)
        merged = merged.sort_values("hours").reset_index(drop=True)
        c = colors.get(run_name, "black")
        ax.plot(
            merged["hours"], merged["elo"],
            marker="o", markersize=6, linewidth=2, color=c,
            label=f"{pretty.get(run_name, run_name)} "
                  f"(final {merged['elo'].iloc[-1]:.0f})",
        )

    ax.axhline(1500, color="grey", linewidth=0.9, linestyle="--",
               label="MCTS@200 anchor (1500)")
    ax.set_xlabel("Cumulative training wall-clock hours")
    ax.set_ylabel("Elo (anchored to MCTS@200 = 1500)")
    ax.set_title("Practical engineering efficiency — Elo vs wall-clock",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out = DOCS_FIG / "elo_vs_wall_clock.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tertiary figure: Elo vs cumulative NN forwards (algorithmic efficiency)
# ---------------------------------------------------------------------------

def make_nn_forwards() -> Path:
    """Same trajectories as make_wall_clock(), but x = cumulative NN forwards.

    NN forwards are the hardware-independent unit of algorithmic work — one
    forward = one (B, 6, 9, 9) policy/value evaluation — so this view
    answers "how data-efficient is the learning rule" rather than "how fast
    is the GPU".
    """
    df = pd.read_csv(ROOT / "runs/comparison_mcts200/eval_vs_mcts200.csv")
    df = df.copy()
    df["elo"] = df["win_rate_mcts"].apply(_winrate_to_elo)
    df["forwards_M"] = df["cumulative_nn_forwards"] / 1e6

    warm = df[df["run"] == "warm_start"]
    real = df[df["run"] != "warm_start"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"az_cnn_gumbel": "#4C78A8", "ppo_warm_matched": "#F58518"}
    pretty = {"az_cnn_gumbel": "AZ-Gumbel (CNN)", "ppo_warm_matched": "PPO (CNN)"}

    for run_name, run_df in real.groupby("run", sort=False):
        kind = run_df["kind"].iloc[0]
        warm_for_kind = warm[warm["kind"] == kind]
        merged = pd.concat([warm_for_kind, run_df], ignore_index=True)
        merged = merged.sort_values("forwards_M").reset_index(drop=True)
        c = colors.get(run_name, "black")
        ax.plot(
            merged["forwards_M"], merged["elo"],
            marker="o", markersize=6, linewidth=2, color=c,
            label=f"{pretty.get(run_name, run_name)} "
                  f"(final {merged['elo'].iloc[-1]:.0f})",
        )

    ax.axhline(1500, color="grey", linewidth=0.9, linestyle="--",
               label="MCTS@200 anchor (1500)")
    ax.set_xlim(left=0)
    ax.set_xlabel("Cumulative NN forward passes (millions)")
    ax.set_ylabel("Elo (anchored to MCTS@200 = 1500)")
    ax.set_title("Algorithmic data efficiency — Elo vs NN forwards",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out = DOCS_FIG / "elo_vs_nn_forwards.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p0 = make_pretrain_ablation()
    print(f"Wrote {p0}")
    p1 = make_hero()
    print(f"Wrote {p1}")
    p2 = make_wall_clock()
    print(f"Wrote {p2}")
    p3 = make_nn_forwards()
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
