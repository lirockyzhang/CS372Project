"""PPO training entry point.

Usage
-----
    # Fresh run, all defaults
    python src/scripts/train_ppo.py

    # Custom options
    python src/scripts/train_ppo.py --num-updates 1000 --num-envs 1024 --lr 1e-4

    # Resume from a PPO checkpoint
    python src/scripts/train_ppo.py --resume runs/ppo_<date>/ppo_update_500.pt

    # Warm-start from a CNN MCTS-pretrained checkpoint (no manual adapter needed --
    # PPOTrainer.run accepts both PPO-format and pretrain-format checkpoints).
    python src/scripts/train_ppo.py \
        --channels 128 --num-blocks 3 \
        --resume models/cnn/cnn_c128b3_100k_reg_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.common.network import UTTTNet
from agents.ppo.trainer import PPOTrainer
from utils.runtime import default_run_dir, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PPO agent for Ultimate Tic-Tac-Toe via self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loop sizing
    p.add_argument("--num-updates",       type=int, default=500,
                   help="PPO updates (rollout + gradient pass cycles)")
    p.add_argument("--num-envs",          type=int, default=512,
                   help="Parallel self-play envs per rollout")
    p.add_argument("--steps-per-rollout", type=int, default=128,
                   help="Env steps collected per env per rollout")

    # Network
    p.add_argument("--channels",   type=int, default=64, help="CNN trunk channels")
    p.add_argument("--num-blocks", type=int, default=3,  help="Residual blocks in trunk")

    # Optimisation
    p.add_argument("--num-epochs",      type=int,   default=4,
                   help="PPO epochs per rollout")
    p.add_argument("--num-minibatches", type=int,   default=4,
                   help="Minibatches per epoch")
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--gae-lambda",      type=float, default=0.95,
                   help="GAE lambda")
    p.add_argument("--clip-eps",        type=float, default=0.2,
                   help="PPO clipping epsilon")
    p.add_argument("--vf-coef",         type=float, default=0.5,
                   help="Value-loss coefficient")
    p.add_argument("--ent-coef",        type=float, default=0.01,
                   help="Entropy bonus coefficient")
    p.add_argument("--max-grad-norm",   type=float, default=0.5)

    # Strength evaluation panel
    p.add_argument("--eval-every",        type=int, default=50,
                   help="Run strength eval every N updates")
    p.add_argument("--eval-random-games", type=int, default=20,
                   help="Games against uniform random opponent each eval")
    p.add_argument("--eval-mcts-games",   type=int, default=20,
                   help="Games against pure MCTS (anchored benchmark)")
    p.add_argument("--eval-mcts-sims",    type=int, default=1_000,
                   help="MCTS sims for the anchored opponent")

    # I/O + reproducibility
    p.add_argument("--run-dir",    type=str, default=None,
                   help="Output directory. Defaults to runs/ppo_<timestamp>")
    p.add_argument("--resume",     type=str, default=None,
                   help="Resume / warm-start: accepts PPO-format checkpoints "
                        "({'model': ...}) or AlphaZero / Transformer pretrain "
                        "checkpoints ({'network_state': ...}). PPO-specific state "
                        "(elo, update count) only restored from the PPO format.")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every",  type=int, default=10)
    p.add_argument("--eval-at-forwards", type=int, nargs="+", default=None,
                   help="Strength-panel eval triggers when cumulative NN forwards "
                        "crosses each value (e.g. 1e7 3e7 1e8). Default: every "
                        "--eval-every updates.")
    p.add_argument("--seed",       type=int, default=None, help="Global RNG seed")
    p.add_argument("--device",     type=str, default=None,
                   help="cuda / mps / cpu (auto-detect if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device  = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("ppo")
    print(f"Using device: {device}")

    net = UTTTNet(channels=args.channels, num_blocks=args.num_blocks)

    trainer = PPOTrainer(
        net,
        num_envs          = args.num_envs,
        steps_per_rollout = args.steps_per_rollout,
        num_epochs        = args.num_epochs,
        num_minibatches   = args.num_minibatches,
        lr                = args.lr,
        gamma             = args.gamma,
        gae_lambda        = args.gae_lambda,
        clip_eps          = args.clip_eps,
        vf_coef           = args.vf_coef,
        ent_coef          = args.ent_coef,
        max_grad_norm     = args.max_grad_norm,
        device            = str(device),
    )

    trainer.run(
        run_dir           = run_dir,
        num_updates       = args.num_updates,
        save_every        = args.save_every,
        log_every         = args.log_every,
        eval_every        = args.eval_every,
        eval_random_games = args.eval_random_games,
        eval_mcts_games   = args.eval_mcts_games,
        eval_mcts_sims    = args.eval_mcts_sims,
        resume_from       = args.resume,
        eval_at_forwards  = args.eval_at_forwards,
    )


if __name__ == "__main__":
    main()
