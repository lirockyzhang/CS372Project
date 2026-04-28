"""AlphaGumbel training entry point.

Usage:
    python src/scripts/train_alphagumbel.py --run-dir runs/alphagumbel/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.alphagumbel.trainer import AlphaGumbelTrainer
from utils.runtime import default_run_dir, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AlphaGumbel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--total-iters",     type=int,   default=75)
    p.add_argument("--games-per-iter",  type=int,   default=1_000)
    p.add_argument("--train-steps",     type=int,   default=500)
    p.add_argument("--sims",            type=int,   default=64)

    p.add_argument("--channels",        type=int,   default=128)
    p.add_argument("--num-blocks",      type=int,   default=6)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--max-root-actions",type=int,   default=16)
    p.add_argument("--buffer-size",     type=int,   default=500_000)
    p.add_argument("--min-buffer",      type=int,   default=10_000)
    p.add_argument("--temp-threshold",  type=int,   default=10)
    p.add_argument("--lr-init",         type=float, default=1e-3)
    p.add_argument("--lr-min",          type=float, default=1e-4)

    p.add_argument("--eval-sims",       type=int,   default=64)
    p.add_argument("--eval-games",      type=int,   default=40)
    p.add_argument("--win-threshold",   type=float, default=0.50)
    p.add_argument("--eval-random-games", type=int, default=20)
    p.add_argument("--eval-mcts-games",   type=int, default=20)
    p.add_argument("--eval-mcts-sims",    type=int, default=1_000)

    p.add_argument("--run-dir",       type=str, default=None,
                   help="Output directory. Defaults to runs/alphagumbel_<timestamp>")
    p.add_argument("--mcts-dataset",  type=str, default=None,
                   help="MCTS bootstrap dataset .npz")
    p.add_argument("--pretrain-steps",         type=int, default=0)
    p.add_argument("--mcts-refresh-positions", type=int, default=5_000)
    p.add_argument("--resume",        type=str, default=None,
                   help="Resume from this AlphaGumbel checkpoint")

    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / mps / cpu (auto-detect if omitted)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("alphagumbel")
    print(f"Using device: {device}")

    trainer = AlphaGumbelTrainer(
        run_dir                = run_dir,
        games_per_iter         = args.games_per_iter,
        train_steps            = args.train_steps,
        sims                   = args.sims,
        eval_sims              = args.eval_sims,
        eval_games             = args.eval_games,
        win_threshold          = args.win_threshold,
        batch_size             = args.batch_size,
        max_root_actions       = args.max_root_actions,
        buffer_size            = args.buffer_size,
        min_buffer             = args.min_buffer,
        temp_threshold         = args.temp_threshold,
        lr_init                = args.lr_init,
        lr_min                 = args.lr_min,
        total_iters            = args.total_iters,
        channels               = args.channels,
        num_blocks             = args.num_blocks,
        device                 = device,
        mcts_dataset_path      = args.mcts_dataset,
        mcts_refresh_positions = args.mcts_refresh_positions,
        eval_random_games      = args.eval_random_games,
        eval_mcts_games        = args.eval_mcts_games,
        eval_mcts_sims         = args.eval_mcts_sims,
    )

    if args.mcts_dataset is not None:
        trainer.load_mcts_dataset(args.mcts_dataset, pretrain_steps=args.pretrain_steps)

    start_iter = 0
    if args.resume is not None:
        start_iter = trainer.load_checkpoint(args.resume)

    trainer.run(start_iter=start_iter)


if __name__ == "__main__":
    main()
