"""AlphaZero training entry point.

Usage
-----
    # Fresh run, all defaults
    python src/scripts/train_alphazero.py

    # Custom options
    python src/scripts/train_alphazero.py --iterations 75 --games 1000 --sims 200

    # Resume from a checkpoint
    python src/scripts/train_alphazero.py --checkpoint runs/alphazero_<date>/iter_020_accepted.pt

    # Bootstrap from a pre-generated MCTS dataset (recommended for iteration 1)
    python src/scripts/train_alphazero.py --mcts-dataset data/mcts_selfplay/dataset.npz

    # Bootstrap + warm-start the network with 1000 gradient steps before self-play
    python src/scripts/train_alphazero.py --mcts-dataset data/mcts_selfplay/dataset.npz --pretrain-steps 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.alphazero.trainer import AlphaZeroTrainer
from utils.runtime import default_run_dir, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an AlphaZero agent for Ultimate Tic-Tac-Toe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loop sizing
    p.add_argument("--iterations",    type=int,   default=75,   help="Self-play / train iterations")
    p.add_argument("--games",         type=int,   default=1_000, help="Self-play games per iteration")
    p.add_argument("--sims",          type=int,   default=200,   help="MCTS sims per self-play move")
    p.add_argument("--train-steps",   type=int,   default=500,   help="Gradient steps per iteration")

    # Network + optimisation
    p.add_argument("--channels",       type=int,   default=64,    help="CNN trunk channels")
    p.add_argument("--num-blocks",     type=int,   default=3,     help="Residual blocks in trunk")
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--leaf-batch",     type=int,   default=8,     help="MCTS leaf batch (self-play)")
    p.add_argument("--eval-leaf-batch",type=int,   default=8,     help="MCTS leaf batch (eval)")
    p.add_argument("--buffer-size",    type=int,   default=500_000)
    p.add_argument("--lr",             type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--lr-min",         type=float, default=1e-4,  help="Cosine-anneal floor")
    p.add_argument("--temp-threshold", type=int,   default=10,    help="Self-play moves with temperature")

    # Gating evaluation (vs previous accepted network)
    p.add_argument("--eval-sims",      type=int,   default=100,   help="MCTS sims per eval move")
    p.add_argument("--eval-games",     type=int,   default=50,    help="Gating eval games / iter")
    p.add_argument("--win-threshold",  type=float, default=0.50,  help="Accept new network above this win rate")

    # Strength evaluation panel
    p.add_argument("--eval-random-games", type=int, default=20,
                   help="Games against uniform random opponent each iteration")
    p.add_argument("--eval-mcts-games",   type=int, default=20,
                   help="Games against pure MCTS (anchored benchmark)")
    p.add_argument("--eval-mcts-sims",    type=int, default=1_000,
                   help="MCTS sims for the anchored opponent")

    # I/O + reproducibility
    p.add_argument("--run-dir",       type=str, default=None,
                   help="Output directory. Defaults to runs/alphazero_<timestamp>")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="Resume from this AlphaZero checkpoint")
    p.add_argument("--ppo-init",      type=str, default=None,
                   help="Warm-start network weights from a PPO checkpoint")
    p.add_argument("--mcts-dataset",  type=str, default=None,
                   help="MCTS bootstrap dataset .npz")
    p.add_argument("--pretrain-steps",      type=int, default=0,
                   help="Pre-training gradient steps on MCTS data before iter 0")
    p.add_argument("--mcts-refresh-positions", type=int, default=5_000,
                   help="MCTS positions re-injected each iteration")

    # Search algorithm — Gumbel is an add-on usable on the same backbone
    p.add_argument("--search", type=str, default="puct", choices=["puct", "gumbel"],
                   help="Tree-search algorithm for self-play and gating eval")
    p.add_argument("--max-root-actions", type=int, default=16,
                   help="Gumbel-only: max root actions considered in sequential halving")

    # Comparison-study knob — fire the strength panel (vs random + vs MCTS@N)
    # only when cumulative NN forwards crosses each milestone instead of every
    # iteration. Aligns x-axis between PPO and AZ runs.
    p.add_argument("--eval-at-forwards", type=int, nargs="+", default=None,
                   help="Strength-panel eval triggers when cumulative NN forwards "
                        "crosses each value (e.g. 1e7 3e7 1e8). Gating eval still "
                        "runs every iter. Default: panel every iter.")

    p.add_argument("--seed",   type=int, default=None, help="Global RNG seed")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / mps / cpu (auto-detect if omitted)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("alphazero")
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(
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
        channels               = args.channels,
        num_blocks             = args.num_blocks,
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
    elif args.ppo_init:
        trainer.load_ppo_weights(args.ppo_init)

    if args.mcts_dataset:
        trainer.load_mcts_dataset(args.mcts_dataset, pretrain_steps=args.pretrain_steps)

    trainer.run(start_iter=start_iter)


if __name__ == "__main__":
    main()
