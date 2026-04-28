"""Wall-clock-matched head-to-head: PUCT@N vs Gumbel@M with shared weights.

`head_to_head.py` exposes a single global `--az-sims`, so equal-sims runs are
trivial but asymmetric-sims runs (the kind needed for "match wall time, not
sim count") require a custom driver. This is that driver.

The match logic, per-move timing, paired openings, and summary printing
all reuse `head_to_head.py` -- this script just wires the two agents with
different sim counts and calls the shared `run_match`.

Usage
-----
    uv run python src/scripts/walltime_match.py \
        --ckpt models/cnn/cnn_c128b3_100k_reg_best.pt \
        --puct-sims 768 --gumbel-sims 64 \
        --games 20 --seed 42 \
        --out-dir runs/tournament_walltime_match
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from agents.alphazero.mcts import AlphaZeroMCTS
from agents.common.network import AlphaZeroNet

# Reuse the match infrastructure from the tournament tool.
from tournament.head_to_head import (
    _detect_device,
    _load_cnn,
    _print_summary,
    _write_csv,
    run_match,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wall-clock-matched PUCT vs Gumbel head-to-head with shared weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Shared CNN checkpoint (both agents load the same weights).")
    p.add_argument("--puct-sims",   type=int, default=768)
    p.add_argument("--gumbel-sims", type=int, default=64)
    p.add_argument("--max-root-actions", type=int, default=16,
                   help="Gumbel-only: max root actions in sequential halving.")
    p.add_argument("--leaf-batch", type=int, default=64,
                   help="MCTS leaf batch (PUCT only).")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs/tournament_walltime_match")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else _detect_device()
    print(f"Device: {device}")

    # Load identical weights for both sides.
    net_puct   = _load_cnn(args.ckpt, device, cls=AlphaZeroNet)
    net_gumbel = _load_cnn(args.ckpt, device, cls=AlphaZeroNet)

    agent_puct = AlphaZeroMCTS(
        net_puct,
        num_simulations=args.puct_sims,
        batch_size=args.leaf_batch,
        device=device,
    )
    agent_gumbel = AlphaGumbelMCTS(
        net_gumbel,
        num_simulations=args.gumbel_sims,
        max_root_actions=args.max_root_actions,
        device=device,
    )

    label_puct   = f"PUCT@{args.puct_sims}({Path(args.ckpt).stem})"
    label_gumbel = f"Gumbel@{args.gumbel_sims}({Path(args.ckpt).stem})"

    sep = "=" * 70
    print()
    print(sep)
    print(f"  Wall-clock match: {label_puct}  vs  {label_gumbel}")
    print(f"  {args.games} games, paired openings, seed={args.seed}")
    print(sep)

    t0 = time.time()
    result = run_match(agent_puct, agent_gumbel, label_puct, label_gumbel, args.games)
    elapsed = time.time() - t0
    _print_summary(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result, out_dir / f"puct{args.puct_sims}_vs_gumbel{args.gumbel_sims}.csv")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "agent_a,agent_b,games,wins_a,wins_b,draws,pts_a,pts_b,"
            "a_ms_per_move,b_ms_per_move,elapsed_s\n"
        )
        f.write(
            f"{label_puct},{label_gumbel},{result['games']},"
            f"{result['wins_a']},{result['wins_b']},{result['draws']},"
            f"{result['wins_a']+0.5*result['draws']},{result['wins_b']+0.5*result['draws']},"
            f"{result['timer_a'].avg_ms:.2f},{result['timer_b'].avg_ms:.2f},{elapsed:.2f}\n"
        )
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
