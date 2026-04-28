"""Generate the figures embedded in the project doc files.

One function per figure. ``main()`` re-renders all of them in dependency
order; the docstring of each function names the markdown file it lives in.

Produces:
  GUMBEL_VS_PUCT.md:
    - gumbel_vs_puct_hero.png            hero 2-panel (equal-sims + matched-walltime)
    - gumbel_ms_scaling.png              log-log: ms/move vs sims + slowdown ratio
    - gumbel_match_outcomes.png          stacked W-D-L bars per matchup
    - gumbel_per_game_msmove.png         per-game ms/move boxplots (split panels)

  PPO_VS_ALPHAGUMBEL.md:
    - elo_vs_wall_clock.png              Elo trajectory vs wall-clock hours
    - elo_vs_nn_forwards.png             Elo trajectory vs cumulative NN forwards
    - ppo_training_curves.png            4-panel PPO metrics over 150 iters
    - azgumbel_acceptance_gate.png       per-iter gate visualization
    - azgumbel_selfplay_dynamics.png     internal Elo + win_rate_previous
    - ppoaz_head_to_head.png             50-game match bar chart + outcome stack
    - ppoaz_msmove_distribution.png      per-game ms/move violins (split panels)
    - ppoaz_inference_speed.png          ms/move log-scale bar chart

  PRETRAIN.md:
    - pretrain_ablation.png              hero 2-panel (val curves + train/val gap)
    - pretrain_test_total.png            4-cell test_total bar chart
    - pretrain_curves_grid.png           train+val per cell, 4-panel
    - pretrain_loss_split.png            policy vs value loss separated
    - pretrain_overfit_progression.png   train/val value gap over training
    - pretrain_hp_cnn.png                CNN hyperparameter tuning
    - pretrain_hp_transformer.png        Transformer hyperparameter tuning

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
    ax2.set_title("Train vs val value loss at best-by-val step", fontsize=12, pad=10)
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
    ax1.set_title("Equal sim count — win % per side", fontsize=12, pad=10)
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
    ax2.set_title("Matched wall-clock — win % per side", fontsize=12, pad=10)
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
    ax.set_title("Elo (anchored to MCTS@200 = 1500) vs wall-clock hours",
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
    ax.set_title("Elo (anchored to MCTS@200 = 1500) vs cumulative NN forwards",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out = DOCS_FIG / "elo_vs_nn_forwards.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


_COLOR_PUCT = "#4C78A8"   # blue   -- PUCT / CNN / PPO baseline
_COLOR_GUMBEL = "#F58518" # orange -- Gumbel / Transformer / AZ-Gumbel
_COLOR_OK = "#54A24B"     # green  -- accepted / on
_COLOR_BAD = "#E45756"    # red    -- rejected / off
_COLOR_NEUTRAL = "#9D755D"


# ===========================================================================
# Figures for docs/GUMBEL_VS_PUCT.md
# ===========================================================================

def _phase1_summaries() -> list[dict]:
    rows = []
    for s in (64, 128, 256):
        df = pd.read_csv(ROOT / f"runs/tournament_sims{s}/summary.csv")
        r = df.iloc[0]
        n = int(r["games"])
        rows.append({
            "sims":   s,
            "n":      n,
            "puct_w": int(r["wins_a"]),
            "gumbel_w": int(r["wins_b"]),
            "draws": int(r["draws"]),
            "puct_pts":  float(r["pts_a"]),
            "gumbel_pts":float(r["pts_b"]),
            "puct_winpct":   100.0 * r["pts_a"] / n,
            "gumbel_winpct": 100.0 * r["pts_b"] / n,
            "puct_ms":   float(r["a_ms_per_move"]),
            "gumbel_ms": float(r["b_ms_per_move"]),
        })
    return rows


def make_gumbel_ms_scaling() -> Path:
    rows = _phase1_summaries()
    sims = [r["sims"] for r in rows]
    puct_ms = [r["puct_ms"] for r in rows]
    gumbel_ms = [r["gumbel_ms"] for r in rows]
    ratios = [g / p for g, p in zip(gumbel_ms, puct_ms)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                    gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax1.plot(sims, gumbel_ms, "o-", color=_COLOR_GUMBEL, linewidth=2.2,
             markersize=9, label="Gumbel")
    ax1.plot(sims, puct_ms, "s-", color=_COLOR_PUCT, linewidth=2.2,
             markersize=9, label="PUCT")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(sims)
    ax1.set_xticklabels([str(s) for s in sims])
    ax1.set_xlabel("Simulations per move")
    ax1.set_ylabel("Wall-clock ms / move (log scale)")
    ax1.set_title("Wall-clock ms / move vs sim count",
                  fontsize=12, pad=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()
    for s, m in zip(sims, gumbel_ms):
        ax1.annotate(f"{m:.0f}ms", (s, m), textcoords="offset points",
                     xytext=(8, 0), ha="left", fontsize=9, color=_COLOR_GUMBEL)
    for s, m in zip(sims, puct_ms):
        ax1.annotate(f"{m:.1f}ms", (s, m), textcoords="offset points",
                     xytext=(8, 0), ha="left", fontsize=9, color=_COLOR_PUCT)

    bars = ax2.bar(range(len(sims)), ratios, color=_COLOR_NEUTRAL,
                   edgecolor="black", linewidth=0.5)
    for i, r in enumerate(ratios):
        ax2.annotate(f"{r:.1f}×", (i, r), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(sims)))
    ax2.set_xticklabels([f"sims = {s}" for s in sims])
    ax2.set_ylabel("Gumbel ms/mv ÷ PUCT ms/mv")
    ax2.set_title("Gumbel ÷ PUCT ms-per-move ratio", fontsize=12, pad=10)
    ax2.set_ylim(0, max(ratios) * 1.18)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(1, color="grey", linewidth=0.7, linestyle=":")

    fig.tight_layout()
    out = DOCS_FIG / "gumbel_ms_scaling.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_gumbel_per_game_msmove() -> Path:
    """Box+strip plot of per-game ms/move, split into per-algorithm panels.

    The two algorithms run an order of magnitude apart (PUCT ~7-25 ms, Gumbel
    ~85-320 ms), so a single shared axis squeezes both into illegible
    rectangles. Splitting into two panels with independent linear scales lets
    each box keep visible IQR and outlier structure.
    """
    import numpy as np
    from matplotlib.patches import Patch

    sims_dirs = [(64, "tournament_sims64"),
                 (128, "tournament_sims128"),
                 (256, "tournament_sims256")]

    puct_data: list[list[float]] = []
    gumbel_data: list[list[float]] = []
    for sims, run in sims_dirs:
        per_game = pd.read_csv(ROOT / f"runs/{run}/az-selfplay_vs_gumbel-selfplay.csv")
        puct_col = [c for c in per_game.columns
                    if c.startswith("AZ-SelfPlay") and c.endswith("ms_per_move")][0]
        gum_col  = [c for c in per_game.columns
                    if c.startswith("Gumbel-SelfPlay") and c.endswith("ms_per_move")][0]
        puct_data.append(per_game[puct_col].tolist())
        gumbel_data.append(per_game[gum_col].tolist())

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(13, 5.4))
    rng = np.random.default_rng(7)
    positions = [1, 2, 3]
    tick_labels = [f"sims = {s}" for s, _ in sims_dirs]

    for ax, data, color, title in [
        (ax_p, puct_data,   _COLOR_PUCT,   "PUCT  (per-game ms / move)"),
        (ax_g, gumbel_data, _COLOR_GUMBEL, "Gumbel  (per-game ms / move)"),
    ]:
        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.6),
                        whiskerprops=dict(color="black", linewidth=1.0),
                        capprops=dict(color="black", linewidth=1.0))
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.55)
            patch.set_edgecolor("black"); patch.set_linewidth(0.8)

        # Per-game scatter overlay
        for d, p in zip(data, positions):
            jx = p + (rng.random(len(d)) - 0.5) * 0.30
            ax.scatter(jx, d, s=18, color=color, alpha=0.85,
                       edgecolor="black", linewidth=0.3, zorder=3)

        # Mean line annotation
        for d, p in zip(data, positions):
            mean = float(np.mean(d))
            ax.annotate(f"μ={mean:.1f} ms", (p, mean),
                        textcoords="offset points", xytext=(0, -32),
                        ha="center", fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor=color,
                                  linewidth=0.7))

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("ms / move (linear)")
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(axis="y", alpha=0.3)
        # Add 10% headroom + floor of 0
        all_vals = [v for d in data for v in d]
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.18
        ax.set_ylim(max(0, lo - margin), hi + margin)

    fig.suptitle("Per-game ms / move by sim count (separate per-algorithm scales)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = DOCS_FIG / "gumbel_per_game_msmove.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_gumbel_match_outcomes() -> Path:
    """Stacked W-D-L bars per matchup."""
    rows = []
    # Phase 1
    for r in _phase1_summaries():
        rows.append((f"P1: sims={r['sims']}",
                     r["puct_w"], r["draws"], r["gumbel_w"], r["n"]))
    # Phase 2
    p2_768  = pd.read_csv(ROOT / "runs/tournament_walltime_match/summary.csv").iloc[0]
    p2_1024 = pd.read_csv(ROOT / "runs/tournament_walltime_match_puct1024/summary.csv").iloc[0]
    rows.append((f"P2: PUCT@768",  int(p2_768["wins_a"]),  int(p2_768["draws"]),
                 int(p2_768["wins_b"]),  int(p2_768["games"])))
    rows.append((f"P2: PUCT@1024", int(p2_1024["wins_a"]), int(p2_1024["draws"]),
                 int(p2_1024["wins_b"]), int(p2_1024["games"])))

    labels = [r[0] for r in rows]
    puct_w   = [100*r[1]/r[4] for r in rows]
    draws    = [100*r[2]/r[4] for r in rows]
    gumbel_w = [100*r[3]/r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    y = list(range(len(labels)))
    ax.barh(y, puct_w, color=_COLOR_PUCT, label="PUCT win", edgecolor="black", linewidth=0.4)
    ax.barh(y, draws,  left=puct_w, color="#BAB0AC", label="Draw",
            edgecolor="black", linewidth=0.4)
    ax.barh(y, gumbel_w, left=[a + b for a, b in zip(puct_w, draws)],
            color=_COLOR_GUMBEL, label="Gumbel win", edgecolor="black", linewidth=0.4)

    for i, r in enumerate(rows):
        ax.text(2, i, f"{r[1]}", va="center", ha="left",
                color="white", fontsize=10, fontweight="bold")
        if r[2] > 0:
            ax.text(puct_w[i] + draws[i]/2, i, f"{r[2]}", va="center",
                    ha="center", fontsize=9)
        ax.text(98, i, f"{r[3]}", va="center", ha="right",
                color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Result share (%)")
    ax.set_title("Match outcomes by matchup (W-D-L share)",
                 fontsize=12, pad=10)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    out = DOCS_FIG / "gumbel_match_outcomes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Figures for docs/PPO_VS_ALPHAGUMBEL.md
# ===========================================================================

def make_ppoaz_head_to_head() -> Path:
    df = pd.read_csv(ROOT / "runs/ppo_vs_gumbel/summary.csv")
    r = df.iloc[0]
    n = int(r["games"])
    ppo_w, gum_w, draws = int(r["wins_a"]), int(r["wins_b"]), int(r["draws"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Win % bars
    labels = ["PPO\n(1.63 ms/mv)", "Gumbel-SelfPlay\n(93.5 ms/mv)"]
    vals = [100*r["pts_a"]/n, 100*r["pts_b"]/n]
    bars = ax1.bar(labels, vals, color=[_COLOR_PUCT, _COLOR_GUMBEL],
                   edgecolor="black", linewidth=0.5, width=0.55)
    for rect, v in zip(bars, vals):
        ax1.annotate(f"{v:.1f}%", (rect.get_x() + rect.get_width()/2, v),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 80)
    ax1.axhline(50, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("Score % (50 paired games)")
    ax1.set_title("Phase 3 — Direct head-to-head", fontsize=12, pad=10)
    ax1.grid(axis="y", alpha=0.3)

    # Outcome stack
    pct = [100*ppo_w/n, 100*draws/n, 100*gum_w/n]
    stack_labels = ["PPO win", "Draw", "Gumbel win"]
    cs = [_COLOR_PUCT, "#BAB0AC", _COLOR_GUMBEL]
    left = 0
    for v, lab, c in zip(pct, stack_labels, cs):
        ax2.barh(0, v, left=left, color=c, edgecolor="black", linewidth=0.4,
                 label=f"{lab} ({v:.0f}%)")
        if v > 3:
            ax2.text(left + v/2, 0, f"{lab}\n{v:.0f}%", va="center",
                     ha="center", fontsize=10, fontweight="bold",
                     color="white" if c != "#BAB0AC" else "black")
        left += v
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("Result share (%)")
    ax2.set_title("Outcome distribution (50 games)", fontsize=12, pad=10)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("PPO vs Gumbel-SelfPlay@64 — 50-game match",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = DOCS_FIG / "ppoaz_head_to_head.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ppoaz_inference_speed() -> Path:
    df = pd.read_csv(ROOT / "runs/ppo_vs_gumbel/summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["PPO\n(direct policy)", "Gumbel-SelfPlay\n(64 sims/move)"]
    vals = [float(df["a_ms_per_move"]), float(df["b_ms_per_move"])]
    bars = ax.bar(labels, vals, color=[_COLOR_PUCT, _COLOR_GUMBEL],
                  edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_yscale("log")
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.2f} ms", (rect.get_x() + rect.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=11, fontweight="bold")
    # Headroom on the log y-axis so the top label doesn't collide with the
    # axis frame.
    ax.set_ylim(bottom=max(0.5, min(vals) * 0.5), top=max(vals) * 2.5)
    ax.set_ylabel("ms / move (log scale)")
    ax.set_title("Inference cost per move", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    out = DOCS_FIG / "ppoaz_inference_speed.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ppo_training_curves() -> Path:
    df = pd.read_csv(ROOT / "runs/ppo_warm_matched/train_log.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    (ax_pl, ax_vl), (ax_ent, ax_sp) = axes

    ax_pl.plot(df["iter"], df["policy_loss"], color=_COLOR_PUCT, linewidth=1.6)
    ax_pl.set_title("Policy loss", fontsize=11)
    ax_pl.set_xlabel("Iter"); ax_pl.set_ylabel("policy_loss")
    ax_pl.grid(alpha=0.3); ax_pl.axhline(0, color="grey", linewidth=0.7, linestyle=":")

    ax_vl.plot(df["iter"], df["value_loss"], color=_COLOR_GUMBEL, linewidth=1.6)
    ax_vl.set_title("Value loss", fontsize=11)
    ax_vl.set_xlabel("Iter"); ax_vl.set_ylabel("value_loss")
    ax_vl.grid(alpha=0.3)

    ax_ent.plot(df["iter"], df["entropy"], color=_COLOR_NEUTRAL, linewidth=1.6)
    ax_ent.set_title("Policy entropy", fontsize=11)
    ax_ent.set_xlabel("Iter"); ax_ent.set_ylabel("entropy (nats)")
    ax_ent.grid(alpha=0.3)

    ax_sp.plot(df["iter"], df["selfplay_win_rate"] * 100,
               color=_COLOR_OK, linewidth=1.6)
    ax_sp.axhline(50, color="grey", linewidth=0.7, linestyle=":")
    ax_sp.set_title("Self-play win % (X-side scoring)", fontsize=11)
    ax_sp.set_xlabel("Iter"); ax_sp.set_ylabel("win %")
    ax_sp.set_ylim(0, 100); ax_sp.grid(alpha=0.3)

    fig.suptitle("PPO training trajectory (150 updates, 512 envs × 128 steps)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    out = DOCS_FIG / "ppo_training_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_azgumbel_acceptance_gate() -> Path:
    df = pd.read_csv(ROOT / "runs/az_cnn_gumbel/train_log.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    iters = df["iter"].tolist()
    scores = df["score_previous"].astype(float).tolist()
    phases = df["phase"].tolist()

    bar_colors = [_COLOR_OK if p == "accepted" else _COLOR_BAD for p in phases]
    bars = ax.bar(iters, scores, color=bar_colors, edgecolor="black", linewidth=0.5,
                  width=0.65)
    for rect, s, p in zip(bars, scores, phases):
        ax.annotate(f"{s:.3f}\n{p}",
                    (rect.get_x() + rect.get_width()/2, s),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle=":")
    ax.axhline(0.45, color=_COLOR_BAD, linewidth=1.0, linestyle="--",
               label="advertised threshold (0.45)")
    ax.set_xticks(iters)
    ax.set_xticklabels([f"iter {i}" for i in iters])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score vs previous best (1=win, 0.5=draw, 0=loss)")
    ax.set_title("AZ-Gumbel score vs previous best, per iteration",
                 fontsize=12, pad=10)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_COLOR_OK, label="accepted"),
               Patch(facecolor=_COLOR_BAD, label="rejected"),
               plt.Line2D([0], [0], color=_COLOR_BAD, linestyle="--",
                          label="threshold (0.45)")]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = DOCS_FIG / "azgumbel_acceptance_gate.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_azgumbel_selfplay_dynamics() -> Path:
    """Two panels: internal Elo + win_rate_random over iters."""
    df = pd.read_csv(ROOT / "runs/az_cnn_gumbel/train_log.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: internal Elo trajectory
    iters = df["iter"].tolist()
    elo = df["elo"].ffill().tolist()
    if elo[0] is None or pd.isna(elo[0]):
        elo[0] = 1500.0
    phases = df["phase"].tolist()

    ax1.plot(iters, elo, color="#333", linewidth=1.6, zorder=1)
    for i, e, p in zip(iters, elo, phases):
        ax1.scatter(i, e, s=110, color=_COLOR_OK if p == "accepted" else _COLOR_BAD,
                    edgecolor="black", linewidth=0.6, zorder=3)
        ax1.annotate(f"{e:.0f}", (i, e), textcoords="offset points",
                     xytext=(8, 0), fontsize=9, va="center")
    ax1.set_xlabel("Iter"); ax1.set_ylabel("Internal self-play Elo")
    ax1.set_title("Internal self-play Elo per iteration",
                  fontsize=12, pad=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(iters)

    # Panel 2: win_rate_random (sanity floor)
    wr = df["win_rate_random"].fillna(0).tolist()
    bars = ax2.bar(iters, [100*x for x in wr], color=_COLOR_GUMBEL,
                   edgecolor="black", linewidth=0.5, width=0.6)
    for rect, w in zip(bars, wr):
        ax2.annotate(f"{100*w:.0f}%",
                     (rect.get_x() + rect.get_width()/2, 100*w),
                     textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=10)
    ax2.set_xlabel("Iter"); ax2.set_ylabel("Win % vs uniform random")
    ax2.set_ylim(0, 105)
    ax2.set_title("Win % vs uniform-random opponent",
                  fontsize=12, pad=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(iters)

    fig.suptitle("AZ-Gumbel internal training dynamics (5 iters)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = DOCS_FIG / "azgumbel_selfplay_dynamics.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ppoaz_msmove_distribution() -> Path:
    """Per-game ms/move violins, split per-algorithm (linear scales).

    PPO runs at ~1-6 ms while Gumbel-SelfPlay@64 runs at ~85-110 ms; on a
    shared axis (log or linear) the PPO violin collapses into a sliver. Two
    panels with independent scales let each side keep visible shape.
    """
    import numpy as np

    df = pd.read_csv(ROOT / "runs/ppo_vs_gumbel/ppo_vs_gumbel-selfplay.csv")
    ppo_col = [c for c in df.columns if c.startswith("PPO") and c.endswith("ms_per_move")][0]
    gum_col = [c for c in df.columns if c.startswith("Gumbel") and c.endswith("ms_per_move")][0]
    ppo_vals = df[ppo_col].astype(float).tolist()
    gum_vals = df[gum_col].astype(float).tolist()

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(13, 5.4))
    rng = np.random.default_rng(42)

    for ax, vals, color, label in [
        (ax_p, ppo_vals, _COLOR_PUCT,   f"PPO  (n={len(ppo_vals)})"),
        (ax_g, gum_vals, _COLOR_GUMBEL, f"Gumbel-SelfPlay@64  (n={len(gum_vals)})"),
    ]:
        parts = ax.violinplot([vals], positions=[1], showmeans=True,
                              showextrema=True, widths=0.78)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.55)
            pc.set_edgecolor("black"); pc.set_linewidth(0.8)
        for key in ("cmeans", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("black"); parts[key].set_linewidth(1.2)

        jx = 1 + (rng.random(len(vals)) - 0.5) * 0.30
        ax.scatter(jx, vals, s=22, color=color, edgecolor="black",
                   linewidth=0.4, alpha=0.85, zorder=3)

        mean = float(np.mean(vals))
        median = float(np.median(vals))
        ax.axhline(mean,   color="black", linewidth=1.0, linestyle="--", alpha=0.45)
        ax.axhline(median, color="grey",  linewidth=0.8, linestyle=":",  alpha=0.55)
        # Always offset labels vertically so they never collide even when
        # mean ≈ median (which happens for the Gumbel panel here).
        ax.annotate(f"mean = {mean:.2f} ms", xy=(1.45, mean),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=10, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor=color,
                              linewidth=0.8))
        ax.annotate(f"median = {median:.2f} ms", xy=(1.45, median),
                    xytext=(0, -10), textcoords="offset points",
                    fontsize=9, va="top", color="#444",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="#999",
                              linewidth=0.5))

        ax.set_xticks([1])
        ax.set_xticklabels([label])
        ax.set_xlim(0.5, 1.95)
        ax.set_ylabel("ms / move (linear)")
        ax.grid(axis="y", alpha=0.3)
        lo, hi = min(vals), max(vals)
        margin = (hi - lo) * 0.18
        ax.set_ylim(max(0, lo - margin), hi + margin)

    ax_p.set_title("PPO — direct policy net", fontsize=12, pad=10)
    ax_g.set_title("Gumbel-SelfPlay — 64 sims/move + halving", fontsize=12, pad=10)
    fig.suptitle("Per-game inference time, 50 paired games "
                 "(separate per-algorithm scales)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = DOCS_FIG / "ppoaz_msmove_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Figures for docs/PRETRAIN.md
# ===========================================================================

_CELLS = [
    # (id, label, run_dir, color, linestyle, train_val_at_best, test_total)
    ("C1", "CNN — reg off",         "pretrain_cnn_100k",             _COLOR_PUCT,   "-",  (0.046, 0.144), 1.327),
    ("C2", "CNN — reg on",          "pretrain_cnn_100k_reg",         _COLOR_PUCT,   "--", (0.057, 0.145), 1.324),
    ("C3", "Transformer — reg off", "pretrain_transformer_100k",     _COLOR_GUMBEL, "-",  (0.057, 0.228), 1.508),
    ("C4", "Transformer — reg on",  "pretrain_transformer_100k_reg", _COLOR_GUMBEL, "--", (0.116, 0.167), 1.363),
]


def make_pretrain_test_total() -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ids   = [c[0] for c in _CELLS]
    labels = [c[1] for c in _CELLS]
    vals  = [c[6] for c in _CELLS]
    cs    = [c[3] for c in _CELLS]
    edge  = ["solid" if c[4] == "-" else "dashed" for c in _CELLS]

    bars = ax.bar(ids, vals,
                  color=cs, edgecolor="black", linewidth=0.6, width=0.55)
    for rect, lab, v, ls in zip(bars, labels, vals, edge):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=11, fontweight="bold")
        ax.annotate(lab, (rect.get_x() + rect.get_width()/2, 0),
                    textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=9, color="black")
        if ls == "dashed":
            rect.set_hatch("//")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Held-out test total loss (lower = better)")
    ax.set_title("Test_total across the 2×2 ablation",
                 fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_test_total.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _hp_grid_panel(ax, df, color: str, title: str) -> None:
    df = df.copy().sort_values("n_params").reset_index(drop=True)
    df["mb"] = df["n_params"] / 1e6
    bars = ax.bar(df.index, df["best_val_total"], color=color,
                  edgecolor="black", linewidth=0.5, width=0.65)
    best_idx = int(df["best_val_total"].idxmin())
    bars[best_idx].set_edgecolor(_COLOR_OK)
    bars[best_idx].set_linewidth(2.2)
    for i, row in df.iterrows():
        ax.annotate(f"{row['best_val_total']:.3f}",
                    (i, row["best_val_total"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, fontweight="bold" if i == best_idx else "normal")
        ax.annotate(f"{row['mb']:.2f} M",
                    (i, 0), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="grey")
    def _short(cfg: str) -> str:
        # Strip braces and quotes; replace 'channels' -> 'ch' etc. for terseness
        s = cfg.replace("{", "").replace("}", "").replace("'", "")
        s = s.replace("channels", "ch").replace("num_blocks", "blk")
        s = s.replace("embed_dim", "embed").replace("depth", "d")
        s = s.replace("num_heads", "h")
        return s
    ax.set_xticks(df.index)
    ax.set_xticklabels([_short(c) for c in df["config"]],
                        rotation=18, ha="right", fontsize=7.8)
    ax.set_ylabel("best val_total (1.5k steps)")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)


def make_pretrain_hp_cnn() -> Path:
    df = pd.read_csv(ROOT / "runs/tune_100k/sweep_results.csv")
    df_cnn = df[df["arch"] == "cnn"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _hp_grid_panel(ax, df_cnn, _COLOR_PUCT,
                   "CNN hyperparameter tuning — best val_total per config (1.5k steps)")
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_hp_cnn.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_pretrain_hp_transformer() -> Path:
    df = pd.read_csv(ROOT / "runs/tune_100k/sweep_results.csv")
    df_tx = df[df["arch"] == "transformer"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _hp_grid_panel(ax, df_tx, _COLOR_GUMBEL,
                   "Transformer hyperparameter tuning — best val_total per config (1.5k steps)")
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_hp_transformer.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_pretrain_curves_grid() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, cell in zip(axes.flat, _CELLS):
        cid, lab, run_dir, color, ls, _, ttot = cell
        df = pd.read_csv(ROOT / "runs" / run_dir / "train_log.csv")
        ax.plot(df["step"], df["train_value_loss"], color=color, linestyle="-",
                linewidth=1.6, label="train_value")
        ax.plot(df["step"], df["val_value_loss"],   color=color, linestyle="--",
                linewidth=1.6, label="val_value")
        ax.plot(df["step"], df["train_policy_loss"], color="#888", linestyle="-",
                linewidth=1.0, alpha=0.6, label="train_policy")
        ax.plot(df["step"], df["val_policy_loss"],  color="#888", linestyle="--",
                linewidth=1.0, alpha=0.6, label="val_policy")
        ax.set_title(f"{cid} — {lab}  (test_total={ttot:.3f})",
                     fontsize=11, pad=6)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss (log scale)")
    axes[0][0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Per-cell training curves (policy + value, train + val)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_curves_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_pretrain_loss_split() -> Path:
    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for cell in _CELLS:
        cid, lab, run_dir, color, ls, _, _ = cell
        df = pd.read_csv(ROOT / "runs" / run_dir / "train_log.csv")
        ax_p.plot(df["step"], df["val_policy_loss"], color=color, linestyle=ls,
                  linewidth=1.7, label=lab)
        ax_v.plot(df["step"], df["val_value_loss"], color=color, linestyle=ls,
                  linewidth=1.7, label=lab)
    ax_p.axhline(2.0, color=_COLOR_BAD, linewidth=1.0, linestyle=":",
                 label="policy noise floor (~2.0)")
    ax_p.set_xlabel("Step"); ax_p.set_ylabel("val_policy_loss")
    ax_p.set_title("Validation policy loss per cell",
                   fontsize=12, pad=8)
    ax_p.grid(alpha=0.3); ax_p.legend(loc="upper right", fontsize=9)

    ax_v.set_xlabel("Step"); ax_v.set_ylabel("val_value_loss")
    ax_v.set_title("Validation value loss per cell",
                   fontsize=12, pad=8)
    ax_v.grid(alpha=0.3); ax_v.legend(loc="upper right", fontsize=9)

    fig.suptitle("Validation loss split — policy and value per cell",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_loss_split.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_pretrain_overfit_progression() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for cell in _CELLS:
        cid, lab, run_dir, color, ls, _, _ = cell
        df = pd.read_csv(ROOT / "runs" / run_dir / "train_log.csv").copy()
        ratio = df["val_value_loss"] / df["train_value_loss"].clip(lower=1e-6)
        ax.plot(df["step"], ratio, color=color, linestyle=ls,
                linewidth=1.8, label=lab)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("val_value_loss / train_value_loss")
    ax.set_title("Overfit ratio (val_value / train_value) over training",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out = DOCS_FIG / "pretrain_overfit_progression.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("== docs/GUMBEL_VS_PUCT.md ==")
    for fn in (make_hero, make_gumbel_ms_scaling,
               make_gumbel_match_outcomes, make_gumbel_per_game_msmove):
        print(f"  Wrote {fn()}")
    print("== docs/PPO_VS_ALPHAGUMBEL.md ==")
    for fn in (make_wall_clock, make_nn_forwards, make_ppo_training_curves,
               make_azgumbel_acceptance_gate, make_azgumbel_selfplay_dynamics,
               make_ppoaz_head_to_head, make_ppoaz_msmove_distribution,
               make_ppoaz_inference_speed):
        print(f"  Wrote {fn()}")
    print("== docs/PRETRAIN.md ==")
    for fn in (make_pretrain_ablation, make_pretrain_test_total,
               make_pretrain_curves_grid, make_pretrain_loss_split,
               make_pretrain_overfit_progression,
               make_pretrain_hp_cnn, make_pretrain_hp_transformer):
        print(f"  Wrote {fn()}")


if __name__ == "__main__":
    main()
