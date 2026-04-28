"""Pre-train an AlphaZero policy/value network on an MCTS bootstrap dataset.

Supports both backbones the project ships with:

    --arch cnn          AlphaZeroNet            (channels, num_blocks)
    --arch transformer  AlphaZeroTransformerNet (embed_dim, depth, num_heads, ...)

Runs supervised policy+value training on the .npz dataset and writes a
checkpoint that the matching ``train_*.py --checkpoint <path>`` resumes from.
Logs train/val loss every ``--eval-every`` steps via ``TrainingLogger`` so the
run produces ``train_log.csv`` and a TensorBoard trace alongside the .pt.

Data split
----------
The .npz is shuffled with the run seed and partitioned into three disjoint
sample-level splits (matches how the AlphaZero replay buffer treats positions
independently):

    train : 80 %   -- gradient steps sample from this buffer
    val   : 10 %   -- evaluated every --eval-every steps to monitor overfit
    test  : 10 %   -- evaluated ONCE at the end as an unbiased final score

Defaults are configurable via ``--val-frac`` and ``--test-frac``.

Module layout
-------------
The script is structured around small single-purpose functions so each piece
can be read or replaced in isolation:

    build_network(arch, args)        -> (network, model_type, model_config)
    make_splits(data, ...)           -> (train_buf, val_buf, test_arrays)
    eval_random_batches / eval_full_split  -> shared loss helpers
    run_pretrain(network, ..., logger) -> trains and logs in-place
    main()                           -> argparse + orchestration

Usage:
    uv run python src/scripts/pretrain_transformer.py \
        --arch transformer \
        --dataset data/mcts_selfplay/dataset_100k.npz \
        --steps 10000 --batch-size 256 \
        --run-dir runs/pretrain_transformer_100k \
        --out models/transformer/pretrained_100k.pt

    uv run python src/scripts/pretrain_transformer.py \
        --arch cnn \
        --dataset data/mcts_selfplay/dataset_100k.npz \
        --steps 10000 --batch-size 256 \
        --run-dir runs/pretrain_cnn_100k \
        --out models/cnn/pretrained_100k.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.alphazero.transformer import AlphaZeroTransformerNet
from agents.alphazero.transformer_trainer import train_step
from agents.common.network import AlphaZeroNet, PolicyValueNet
from agents.common.replay_buffer import ReplayBuffer
from utils.runtime import default_run_dir, detect_device, set_seed
from utils.training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Architecture registry — one place to plug in new backbones
# ---------------------------------------------------------------------------

ArchBuilder = Callable[[argparse.Namespace], tuple[PolicyValueNet, str, dict[str, Any]]]


def _build_cnn(args: argparse.Namespace) -> tuple[PolicyValueNet, str, dict[str, Any]]:
    cfg = {
        "channels":      args.channels,
        "num_blocks":    args.num_blocks,
        "value_dropout": getattr(args, "value_dropout", 0.0),
    }
    return AlphaZeroNet(**cfg), "cnn", cfg


def _build_transformer(args: argparse.Namespace) -> tuple[PolicyValueNet, str, dict[str, Any]]:
    cfg = {
        "embed_dim":     args.embed_dim,
        "depth":         args.depth,
        "num_heads":     args.num_heads,
        "mlp_ratio":     args.mlp_ratio,
        "dropout":       args.dropout,
        "value_dropout": getattr(args, "value_dropout", 0.0),
    }
    return AlphaZeroTransformerNet(**cfg), "transformer", cfg


ARCHS: dict[str, ArchBuilder] = {
    "cnn":         _build_cnn,
    "transformer": _build_transformer,
}


def build_network(args: argparse.Namespace) -> tuple[PolicyValueNet, str, dict[str, Any]]:
    if args.arch not in ARCHS:
        raise ValueError(f"--arch must be one of {sorted(ARCHS)}, got {args.arch!r}")
    return ARCHS[args.arch](args)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def make_splits(
    obs:        np.ndarray,
    policy:     np.ndarray,
    value:      np.ndarray,
    val_frac:   float,
    test_frac:  float,
    batch_size: int,
    seed:       int | None,
) -> tuple[ReplayBuffer, ReplayBuffer, dict[str, np.ndarray]]:
    """Shuffle and partition into train/val/test (no overlap, sample-level)."""
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0"
        )
    n_total = len(value)
    rng     = np.random.default_rng(seed if seed is not None else 0)
    perm    = rng.permutation(n_total)

    n_val   = max(batch_size, int(n_total * val_frac))
    n_test  = max(batch_size, int(n_total * test_frac))
    val_idx   = perm[:n_val]
    test_idx  = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]

    train_buf = ReplayBuffer(max_size=len(train_idx))
    train_buf.add_game(obs[train_idx], policy[train_idx], value[train_idx])
    val_buf = ReplayBuffer(max_size=len(val_idx))
    val_buf.add_game(obs[val_idx], policy[val_idx], value[val_idx])

    test = {
        "obs":    obs[test_idx],
        "policy": policy[test_idx],
        "value":  value[test_idx],
    }
    return train_buf, val_buf, test


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _loss_on_batch(
    network:   PolicyValueNet,
    obs_np:    np.ndarray,
    policy_np: np.ndarray,
    value_np:  np.ndarray,
    device:    torch.device,
) -> tuple[float, float]:
    obs_t    = torch.from_numpy(obs_np).to(device)
    policy_t = torch.from_numpy(policy_np).to(device)
    value_t  = torch.from_numpy(value_np).to(device)

    logits, value_pred = network(obs_t)
    log_probs   = torch.log_softmax(logits.reshape(-1, 81), dim=-1)
    policy_flat = policy_t.reshape(-1, 81)
    policy_loss = -(policy_flat * log_probs).sum(dim=-1).mean()
    value_loss  = nn.functional.mse_loss(value_pred, value_t)
    return policy_loss.item(), value_loss.item()


@torch.no_grad()
def eval_random_batches(
    network:    PolicyValueNet,
    buffer:     ReplayBuffer,
    batch_size: int,
    n_batches:  int,
    device:     torch.device,
) -> tuple[float, float]:
    network.eval()
    pls: list[float] = []
    vls: list[float] = []
    for _ in range(n_batches):
        obs_np, policy_np, value_np = buffer.sample(batch_size)
        pl, vl = _loss_on_batch(network, obs_np, policy_np, value_np, device)
        pls.append(pl)
        vls.append(vl)
    network.train()
    return float(np.mean(pls)), float(np.mean(vls))


@torch.no_grad()
def eval_full_split(
    network:    PolicyValueNet,
    obs:        np.ndarray,
    policy:     np.ndarray,
    value:      np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> tuple[float, float]:
    network.eval()
    n = len(value)
    pl_sum = 0.0
    vl_sum = 0.0
    seen   = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs  = end - start
        pl, vl = _loss_on_batch(
            network, obs[start:end], policy[start:end], value[start:end], device,
        )
        pl_sum += pl * bs
        vl_sum += vl * bs
        seen   += bs
    network.train()
    return pl_sum / seen, vl_sum / seen


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_pretrain(
    network:     PolicyValueNet,
    optimizer:   optim.Optimizer,
    train_buf:   ReplayBuffer,
    val_buf:     ReplayBuffer,
    *,
    steps:       int,
    batch_size:  int,
    eval_every:  int,
    val_batches: int,
    device:      torch.device,
    logger:      TrainingLogger,
    scheduler:   optim.lr_scheduler._LRScheduler | None = None,
    on_best_val: Any = None,   # called(val_total, step) when val improves
) -> float:
    """Run the supervised pretraining loop. Returns best val total loss seen."""
    network.train()
    train_pl: list[float] = []
    train_vl: list[float] = []
    t0 = time.time()
    log_iter = 0
    best_val_total = float("inf")

    for step in range(1, steps + 1):
        pl, vl = train_step(network, optimizer, train_buf, batch_size, device)
        train_pl.append(pl)
        train_vl.append(vl)
        if scheduler is not None:
            scheduler.step()

        if step % eval_every == 0 or step == steps:
            val_pl, val_vl = eval_random_batches(
                network, val_buf, batch_size, val_batches, device,
            )
            dt  = time.time() - t0
            val_total = 0.5 * val_pl + 2.0 * val_vl
            metrics = {
                "step":              step,
                "lr":                optimizer.param_groups[0]["lr"],
                "train_policy_loss": float(np.mean(train_pl)),
                "train_value_loss":  float(np.mean(train_vl)),
                "val_policy_loss":   val_pl,
                "val_value_loss":    val_vl,
                "val_total_loss":    val_total,
                "steps_per_s":       step / dt,
                "elapsed_s":         dt,
            }
            logger.log_iteration(log_iter, metrics)
            if val_total < best_val_total:
                best_val_total = val_total
                if on_best_val is not None:
                    on_best_val(val_total, step)
            log_iter += 1
            train_pl.clear()
            train_vl.clear()

    return best_val_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised pre-training of AlphaZero CNN or Transformer on MCTS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--arch",    type=str, default="transformer", choices=sorted(ARCHS),
                   help="Backbone architecture to pretrain")
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to .npz with obs/policy/value")
    p.add_argument("--out",     type=str, default=None,
                   help="Checkpoint output. Defaults to models/<arch>/pretrained.pt")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Logger output dir. Defaults to runs/pretrain_<arch>_<timestamp>")

    p.add_argument("--steps",      type=int,   default=10_000)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lr-min",     type=float, default=None,
                   help="If set, cosine-decay LR from --lr to --lr-min over --steps")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (decoupled L2). 0 reduces to plain Adam")
    p.add_argument("--value-dropout", type=float, default=0.0,
                   help="Dropout rate inside the value-head FC bottleneck (both archs)")
    p.add_argument("--no-save-best", action="store_true",
                   help="Disable saving the best-by-val checkpoint alongside the final one")

    p.add_argument("--val-frac",     type=float, default=0.10,
                   help="Held-out fraction used for in-training val loss")
    p.add_argument("--test-frac",    type=float, default=0.10,
                   help="Held-out fraction evaluated ONCE at the end")
    p.add_argument("--val-batches",  type=int,   default=20,
                   help="Minibatches per in-training val pass")
    p.add_argument("--test-batches", type=int,   default=0,
                   help="Minibatches for final test pass. 0 = sweep entire test split")
    p.add_argument("--eval-every",   type=int,   default=200,
                   help="Run val + log a CSV/TensorBoard row every N steps")

    # CNN args
    p.add_argument("--channels",   type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=3)

    # Transformer args
    p.add_argument("--embed-dim", type=int,   default=128)
    p.add_argument("--depth",     type=int,   default=4)
    p.add_argument("--num-heads", type=int,   default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout",   type=float, default=0.0)

    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device  = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(f"pretrain_{args.arch}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else Path(f"models/{args.arch}/pretrained.pt")
    print(f"Arch:    {args.arch}")
    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    print(f"Loading dataset: {args.dataset}")
    data   = np.load(args.dataset)
    obs    = data["obs"].astype(np.float32)
    policy = data["policy"].astype(np.float32)
    value  = data["value"].astype(np.float32)
    n_total = len(value)

    train_buf, val_buf, test = make_splits(
        obs, policy, value,
        val_frac=args.val_frac, test_frac=args.test_frac,
        batch_size=args.batch_size, seed=args.seed,
    )
    print(
        f"  total={n_total:,}  obs={obs.shape}\n"
        f"  split (seed={args.seed}):  "
        f"train={train_buf.size:,} ({train_buf.size/n_total:.1%})  "
        f"val={val_buf.size:,} ({val_buf.size/n_total:.1%})  "
        f"test={len(test['value']):,} ({len(test['value'])/n_total:.1%})"
    )

    network, model_type, model_config = build_network(args)
    network = network.to(device)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"Network: {type(network).__name__}  params={n_params:,}  config={model_config}")

    # AdamW (decoupled weight decay) when --weight-decay > 0; reduces to Adam otherwise.
    optimizer = optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_min is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr_min,
        )
    logger = TrainingLogger(
        run_dir=run_dir,
        headline_keys=[
            "iter", "step", "lr", "train_policy_loss", "train_value_loss",
            "val_policy_loss", "val_value_loss", "steps_per_s",
        ],
    )

    # ``foo.pt``  ->  ``foo_best.pt`` so the suffix isn't a confusing ``.best.pt``.
    best_path = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    save_best = not args.no_save_best
    best_meta: dict[str, Any] = {"saved": False, "val_total": float("inf"), "step": -1}

    def _save_best(val_total: float, step: int) -> None:
        if not save_best:
            return
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration":     0,
                "network_state": network.state_dict(),
                "model_type":    model_type,
                "model_config":  model_config,
                "pretrain": {
                    "arch":         args.arch,
                    "dataset":      str(args.dataset),
                    "step":         step,
                    "val_total":    val_total,
                    "selection":    "best_by_val",
                },
            },
            best_path,
        )
        best_meta["saved"]     = True
        best_meta["val_total"] = val_total
        best_meta["step"]      = step

    sched_str = "cosine" if scheduler is not None else "constant"
    print(
        f"Pre-training for {args.steps:,} steps  "
        f"batch_size={args.batch_size}  lr={args.lr} ({sched_str}"
        f"{f'->{args.lr_min}' if scheduler is not None else ''})  "
        f"weight_decay={args.weight_decay}  value_dropout={args.value_dropout}  "
        f"eval_every={args.eval_every}  save_best={save_best}"
    )
    try:
        best_val_total = run_pretrain(
            network, optimizer, train_buf, val_buf,
            steps=args.steps, batch_size=args.batch_size,
            eval_every=args.eval_every, val_batches=args.val_batches,
            device=device, logger=logger,
            scheduler=scheduler,
            on_best_val=_save_best,
        )
    finally:
        logger.close()
    if save_best and best_meta["saved"]:
        print(
            f"Best-by-val checkpoint -> {best_path}  "
            f"(step={best_meta['step']}, val_total={best_meta['val_total']:.4f})"
        )

    print("Evaluating on held-out test split ...")
    if args.test_batches > 0:
        test_buf = ReplayBuffer(max_size=len(test["value"]))
        test_buf.add_game(test["obs"], test["policy"], test["value"])
        test_pl, test_vl = eval_random_batches(
            network, test_buf, args.batch_size, args.test_batches, device,
        )
        test_n_used = args.test_batches * args.batch_size
    else:
        test_pl, test_vl = eval_full_split(
            network, test["obs"], test["policy"], test["value"],
            args.batch_size, device,
        )
        test_n_used = len(test["value"])
    test_total = 0.5 * test_pl + 2.0 * test_vl
    print(
        f"  test_policy_loss={test_pl:.4f}  "
        f"test_value_loss={test_vl:.4f}  "
        f"test_total_loss={test_total:.4f}  "
        f"(n={test_n_used:,})"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration":       0,
            "network_state":   network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_type":      model_type,
            "model_config":    model_config,
            "pretrain": {
                "arch":       args.arch,
                "dataset":    str(args.dataset),
                "steps":      args.steps,
                "batch_size": args.batch_size,
                "lr":         args.lr,
                "split": {
                    "seed":            args.seed,
                    "val_frac":        args.val_frac,
                    "test_frac":       args.test_frac,
                    "train_positions": int(train_buf.size),
                    "val_positions":   int(val_buf.size),
                    "test_positions":  int(len(test["value"])),
                },
                "best_val_total":   best_val_total,
                "test_policy_loss": test_pl,
                "test_value_loss":  test_vl,
                "test_total_loss":  test_total,
                "run_dir":          str(run_dir),
            },
        },
        out_path,
    )
    print(f"Saved pre-trained checkpoint -> {out_path}")
    print(f"Logs:                         -> {run_dir}/train_log.csv (+ tensorboard/)")


if __name__ == "__main__":
    main()
