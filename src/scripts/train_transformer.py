"""AlphaZero-Transformer training entry point.

Usage:
    python src/scripts/train_transformer.py --run-dir runs/transformer_run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.alphazero.transformer_trainer import TransformerTrainer
from utils.runtime import default_run_dir, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AlphaZero with the Transformer policy/value backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loop sizing
    p.add_argument("--iterations",      type=int,   default=75)
    p.add_argument("--games",           type=int,   default=1_000, help="Self-play games / iter")
    p.add_argument("--sims",            type=int,   default=200,   help="MCTS sims per self-play move")
    p.add_argument("--train-steps",     type=int,   default=500)

    # Transformer architecture
    p.add_argument("--embed-dim",       type=int,   default=128)
    p.add_argument("--depth",           type=int,   default=4)
    p.add_argument("--num-heads",       type=int,   default=4)
    p.add_argument("--mlp-ratio",       type=float, default=4.0)
    p.add_argument("--dropout",         type=float, default=0.0)

    # Optimisation
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--leaf-batch",      type=int,   default=8)
    p.add_argument("--eval-leaf-batch", type=int,   default=8)
    p.add_argument("--buffer-size",     type=int,   default=500_000)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lr-min",          type=float, default=1e-4)
    p.add_argument("--temp-threshold",  type=int,   default=10)

    # Evaluation panel
    p.add_argument("--eval-sims",       type=int,   default=100)
    p.add_argument("--eval-games",      type=int,   default=40)
    p.add_argument("--win-threshold",   type=float, default=0.50)
    p.add_argument("--eval-random-games", type=int, default=20)
    p.add_argument("--eval-mcts-games",   type=int, default=20)
    p.add_argument("--eval-mcts-sims",    type=int, default=1_000)

    # I/O + reproducibility
    p.add_argument("--run-dir",       type=str, default=None,
                   help="Output directory. Defaults to runs/transformer_<timestamp>")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="Resume from this Transformer checkpoint")
    p.add_argument("--mcts-dataset",  type=str, default=None,
                   help="MCTS bootstrap dataset .npz")
    p.add_argument("--pretrain-steps",         type=int, default=0)
    p.add_argument("--mcts-refresh-positions", type=int, default=5_000)

    # Search algorithm — Gumbel is an add-on usable on the Transformer backbone
    p.add_argument("--search", type=str, default="puct", choices=["puct", "gumbel"],
                   help="Tree-search algorithm for self-play and gating eval")
    p.add_argument("--max-root-actions", type=int, default=16,
                   help="Gumbel-only: max root actions considered in sequential halving")

    # Comparison-study knob (see train_alphazero.py for full notes).
    p.add_argument("--eval-at-forwards", type=int, nargs="+", default=None,
                   help="Strength-panel eval triggers when cumulative NN forwards "
                        "crosses each value (e.g. 1e7 3e7 1e8). Gating eval still "
                        "runs every iter. Default: panel every iter.")

    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / mps / cpu (auto-detect if omitted)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    set_seed(args.seed)
    device  = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("transformer")
    print(f"Using device: {device}")

    trainer = TransformerTrainer(
        run_dir                = run_dir,
        games_per_iter         = args.games,
        train_steps            = args.train_steps,
        sims                   = args.sims,
        eval_sims              = args.eval_sims,
        eval_games             = args.eval_games,
        win_threshold          = args.win_threshold,
        batch_size             = args.batch_size,
        leaf_batch_size        = args.leaf_batch,
        eval_leaf_batch_size   = args.eval_leaf_batch,
        buffer_size            = args.buffer_size,
        temp_threshold         = args.temp_threshold,
        lr_init                = args.lr,
        lr_min                 = args.lr_min,
        total_iters            = args.iterations,
        embed_dim              = args.embed_dim,
        depth                  = args.depth,
        num_heads              = args.num_heads,
        mlp_ratio              = args.mlp_ratio,
        dropout                = args.dropout,
        device                 = device,
        mcts_dataset_path      = args.mcts_dataset,
        mcts_refresh_positions = args.mcts_refresh_positions,
        eval_random_games      = args.eval_random_games,
        eval_mcts_games        = args.eval_mcts_games,
        eval_mcts_sims         = args.eval_mcts_sims,
        search                 = args.search,
        max_root_actions       = args.max_root_actions,
        eval_at_forwards       = args.eval_at_forwards,
    )

    start_iter = 0
    if args.checkpoint:
        start_iter = trainer.load_checkpoint(args.checkpoint)

    if args.mcts_dataset:
        trainer.load_mcts_dataset(args.mcts_dataset, pretrain_steps=args.pretrain_steps)

    trainer.run(start_iter=start_iter)


if __name__ == "__main__":
    main()
