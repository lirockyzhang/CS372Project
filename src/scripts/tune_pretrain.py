"""Hyperparameter tuning for AlphaZero-style pretraining (CNN + Transformer).

For each architecture this runs a small grid of configurations on the SAME
train/val/test split, logs per-trial best val loss + final test loss to a
CSV, and prints a ranked summary so the best config can be selected for a
longer full pretraining run.

Methodology
-----------
* All trials share a single shuffled split (train 80% / val 10% / test 10%)
  derived from ``--seed`` so val/test scores are directly comparable.
* Each trial trains for ``--steps-per-trial`` steps (short) and reports
  ``best_val_total`` (the lowest combined val loss seen during training).
* The held-out test split is evaluated ONCE per trial after training and
  reported alongside.  Test scores are for reporting only -- selection is
  done on val to keep the test set unbiased.
* Per-trial training-curve CSVs land in ``<run-dir>/<trial>/train_log.csv``;
  the cross-trial summary lands in ``<run-dir>/sweep_results.csv``.

Grid
----
The grid lives at the top of this file (CNN_GRID, TRANSFORMER_GRID) so it is
trivial to edit.  Each grid contains at least three configurations.

Usage:
    uv run python src/scripts/tune_pretrain.py \
        --dataset data/mcts_selfplay/dataset_100k.npz \
        --steps-per-trial 1500 \
        --run-dir runs/tune_100k

    # Restrict to one arch
    uv run python src/scripts/tune_pretrain.py --arch cnn ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.optim as optim

from utils.runtime import default_run_dir, detect_device, set_seed
from utils.training_logger import TrainingLogger

# Reuse the building blocks from the single-config pretrainer so the tuning
# run trains the model the exact same way -- only HPs differ.
from scripts.pretrain_transformer import (
    build_network,
    eval_full_split,
    make_splits,
    run_pretrain,
)


# ---------------------------------------------------------------------------
# Hyperparameter grids -- edit these to broaden / narrow the tuning run
# ---------------------------------------------------------------------------

# Each entry is the *delta* over the arch's default; missing keys keep
# their argparse defaults from pretrain_transformer.py.
CNN_GRID: list[dict[str, Any]] = [
    {"channels":  32, "num_blocks": 3, "lr": 1e-3},   # tiny
    {"channels":  64, "num_blocks": 3, "lr": 1e-3},   # default
    {"channels":  64, "num_blocks": 6, "lr": 1e-3},   # deeper
    {"channels": 128, "num_blocks": 3, "lr": 5e-4},   # wider, lower lr
]

TRANSFORMER_GRID: list[dict[str, Any]] = [
    {"embed_dim":  64, "depth": 4, "num_heads": 4, "lr": 1e-3},   # tiny
    {"embed_dim": 128, "depth": 4, "num_heads": 4, "lr": 1e-3},   # default
    {"embed_dim": 128, "depth": 6, "num_heads": 4, "lr": 1e-3},   # deeper
    {"embed_dim": 192, "depth": 4, "num_heads": 6, "lr": 5e-4},   # wider, lower lr
]


# ---------------------------------------------------------------------------
# Defaults that build_network() reads from the Namespace
# ---------------------------------------------------------------------------

ARCH_DEFAULTS: dict[str, dict[str, Any]] = {
    "cnn":         {"channels": 64, "num_blocks": 3},
    "transformer": {"embed_dim": 128, "depth": 4, "num_heads": 4,
                    "mlp_ratio": 4.0, "dropout": 0.0},
}


def make_trial_namespace(arch: str, config: dict[str, Any]) -> Namespace:
    """Build a Namespace that ``build_network`` accepts, with config overrides."""
    ns = Namespace(arch=arch)
    for k, v in ARCH_DEFAULTS[arch].items():
        setattr(ns, k, v)
    for k, v in config.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for pretrain_transformer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--run-dir", type=str, default=None,
                   help="Tuning-run root. Defaults to runs/tune_<timestamp>")
    p.add_argument("--arch",    type=str, default="both",
                   choices=["both", "cnn", "transformer"])

    p.add_argument("--steps-per-trial", type=int, default=1_500,
                   help="Training steps per trial (kept short for tuning speed)")
    p.add_argument("--batch-size",      type=int, default=256)
    p.add_argument("--eval-every",      type=int, default=150)
    p.add_argument("--val-batches",     type=int, default=10)
    p.add_argument("--val-frac",        type=float, default=0.10)
    p.add_argument("--test-frac",       type=float, default=0.10)

    p.add_argument("--seed",   type=int, default=42,
                   help="Shared seed for split + per-trial init")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_trial(
    arch:        str,
    config:      dict[str, Any],
    trial_dir:   Path,
    *,
    train_buf, val_buf, test,
    args:        argparse.Namespace,
    device:      torch.device,
) -> dict[str, Any]:
    """Train one config and return summary metrics."""
    set_seed(args.seed)  # same init seed across trials so HP is the only diff

    ns = make_trial_namespace(arch, config)
    network, model_type, model_config = build_network(ns)
    network = network.to(device)
    n_params  = sum(p.numel() for p in network.parameters())
    optimizer = optim.Adam(network.parameters(), lr=config["lr"])

    trial_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(
        run_dir=trial_dir,
        headline_keys=["iter", "step", "val_policy_loss",
                       "val_value_loss", "val_total_loss"],
    )

    t0 = time.time()
    try:
        best_val_total = run_pretrain(
            network, optimizer, train_buf, val_buf,
            steps=args.steps_per_trial, batch_size=args.batch_size,
            eval_every=args.eval_every, val_batches=args.val_batches,
            device=device, logger=logger,
        )
    finally:
        logger.close()

    test_pl, test_vl = eval_full_split(
        network, test["obs"], test["policy"], test["value"],
        args.batch_size, device,
    )
    test_total = 0.5 * test_pl + 2.0 * test_vl
    elapsed = time.time() - t0

    return {
        "arch":             arch,
        "config":           config,
        "model_type":       model_type,
        "model_config":     model_config,
        "n_params":         n_params,
        "best_val_total":   best_val_total,
        "test_policy_loss": test_pl,
        "test_value_loss":  test_vl,
        "test_total_loss":  test_total,
        "elapsed_s":        elapsed,
        "trial_dir":        str(trial_dir),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device  = detect_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir("tune")
    run_dir.mkdir(parents=True, exist_ok=True)
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
        f"  total={n_total:,}  split (seed={args.seed}):  "
        f"train={train_buf.size:,}  val={val_buf.size:,}  test={len(test['value']):,}"
    )

    arches = ["cnn", "transformer"] if args.arch == "both" else [args.arch]
    grids  = {"cnn": CNN_GRID, "transformer": TRANSFORMER_GRID}

    summary_path = run_dir / "sweep_results.csv"
    summary_csv  = open(summary_path, "w", newline="", encoding="utf-8")
    summary_w    = csv.writer(summary_csv)
    summary_w.writerow([
        "arch", "trial", "config", "n_params",
        "best_val_total", "test_policy_loss", "test_value_loss",
        "test_total_loss", "elapsed_s", "trial_dir",
    ])

    all_results: list[dict[str, Any]] = []
    try:
        for arch in arches:
            grid = grids[arch]
            print(f"\n===== {arch.upper()} hyperparameter tuning ({len(grid)} configs) =====")
            for i, config in enumerate(grid):
                trial_name = f"{arch}_t{i:02d}"
                trial_dir  = run_dir / trial_name
                print(f"\n[{trial_name}] config={config}")
                result = run_trial(
                    arch, config, trial_dir,
                    train_buf=train_buf, val_buf=val_buf, test=test,
                    args=args, device=device,
                )
                print(
                    f"  -> params={result['n_params']:,}  "
                    f"best_val_total={result['best_val_total']:.4f}  "
                    f"test_total={result['test_total_loss']:.4f}  "
                    f"({result['elapsed_s']:.1f}s)"
                )
                summary_w.writerow([
                    arch, trial_name, str(config), result["n_params"],
                    f"{result['best_val_total']:.6g}",
                    f"{result['test_policy_loss']:.6g}",
                    f"{result['test_value_loss']:.6g}",
                    f"{result['test_total_loss']:.6g}",
                    f"{result['elapsed_s']:.2f}",
                    result["trial_dir"],
                ])
                summary_csv.flush()
                all_results.append(result)
    finally:
        summary_csv.close()

    # Per-arch ranking by best_val_total (lower = better).
    print("\n===== Hyperparameter tuning summary (ranked by best_val_total) =====")
    for arch in arches:
        rows = sorted(
            [r for r in all_results if r["arch"] == arch],
            key=lambda r: r["best_val_total"],
        )
        if not rows:
            continue
        print(f"\n{arch.upper()}:")
        print(f"  {'rank':>4}  {'val':>8}  {'test':>8}  {'params':>9}  config")
        for rank, r in enumerate(rows, 1):
            print(
                f"  {rank:>4}  "
                f"{r['best_val_total']:>8.4f}  "
                f"{r['test_total_loss']:>8.4f}  "
                f"{r['n_params']:>9,}  "
                f"{r['config']}"
            )
        best = rows[0]
        print(f"  best: {best['config']}  ->  trial_dir={best['trial_dir']}")

    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
