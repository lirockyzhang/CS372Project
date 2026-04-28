"""MCTS self-play dataset generator for AlphaZero training.

Two backends:

  cpu   (default fallback)  Multi-process pool of independent single-game
                            workers using the pure-Python MCTSAgent. Suitable
                            on machines without a CUDA GPU.

  cuda  (default)           GPU-resident MCTS via the vendored Numba CUDA
                            engine in src/agents/mcts_cuda. Single process
                            (the GPU does the parallelism). Required for the
                            high-budget settings (10k games × 100k sims/move)
                            this script defaults to.

Output is identical for both backends:

  <out>/dataset.npz
    obs     (N, 6, 9, 9)  float32   board observation (current player POV)
    policy  (N, 9, 9)     float32   MCTS visit-count distribution
    value   (N,)          float32   game outcome from mover's POV (+1/-1/0)

Recommended invocation (matches the project's intended bootstrap-dataset run):

    python data/generate_mcts_data.py \\
        --backend cuda --games 10000 --sims 100000 \\
        --cuda-n-trees 8 --cuda-n-playouts 128 \\
        --cuda-variant acp_prodigal --cuda-device-memory 4.0 \\
        --out data/mcts_selfplay

CPU smoke for a quick check:

    python data/generate_mcts_data.py \\
        --backend cpu --games 64 --sims 200 --workers 8 \\
        --out runs/_smoke_cpu

Run src/scripts/calibrate_mcts_throughput.py first to size --sims and --workers
on the CPU backend.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.mcts.self_play import play_game
from utils.parallel_selfplay import (
    _merge_chunks,
    _save_chunk,
    run_parallel_self_play,
)


# ---------------------------------------------------------------------------
# CPU backend worker (must be a top-level function for mp.Pool pickling)
# ---------------------------------------------------------------------------

def _worker(args: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one self-play game on CPU.

    Args:
        args: (num_simulations, temp_threshold, seed)

    Returns:
        (obs, policy, value) arrays for every position in the game.
    """
    import random

    num_sims, temp_threshold, seed = args
    random.seed(seed)
    np.random.seed(seed)

    samples = play_game(num_sims, temp_threshold)
    obs    = np.stack([s[0] for s in samples])
    policy = np.stack([s[1] for s in samples])
    value  = np.array([s[2] for s in samples], dtype=np.float32)
    return obs, policy, value


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_workers = max(1, (os.cpu_count() or 4) - 2)

    p = argparse.ArgumentParser(
        description="Generate MCTS self-play dataset for AlphaZero training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Common knobs
    p.add_argument("--backend",        type=str, default="cuda",
                   choices=["cpu", "cuda"],
                   help="MCTS backend. 'cuda' uses the vendored Numba CUDA engine "
                        "(requires NVIDIA GPU); 'cpu' uses the pure-Python MCTSAgent "
                        "with a multiprocessing pool.")
    p.add_argument("--games",          type=int, default=10_000,
                   help="Total self-play games to generate")
    p.add_argument("--sims",           type=int, default=100_000,
                   help="MCTS simulations per move (search_steps_limit on the cuda backend)")
    p.add_argument("--out",            type=str, default="data/mcts_selfplay",
                   help="Output directory")
    p.add_argument("--chunk-size",     type=int, default=500,
                   help="Flush to disk every N games")
    p.add_argument("--temp-threshold", type=int, default=30,
                   help="Moves before switching to greedy action selection")
    p.add_argument("--seed",           type=int, default=42,
                   help="Base random seed (each game gets seed+game_idx)")
    p.add_argument("--keep-chunks",    action="store_true",
                   help="Keep intermediate chunk files after merge")

    # CPU-only knobs
    p.add_argument("--workers",        type=int, default=default_workers,
                   help="[cpu backend] Parallel worker processes")

    # CUDA-only knobs
    p.add_argument("--cuda-n-trees",       type=int,   default=8,
                   help="[cuda backend] Independent MCTS trees grown concurrently (power of 2 recommended)")
    p.add_argument("--cuda-n-playouts",    type=int,   default=128,
                   help="[cuda backend] Random rollouts per leaf expansion (power of 2)")
    p.add_argument("--cuda-variant",       type=str,   default="acp_prodigal",
                   choices=["ocp_thrifty", "ocp_prodigal", "acp_thrifty", "acp_prodigal"],
                   help="[cuda backend] MCTSNC algorithmic variant")
    p.add_argument("--cuda-device-memory", type=float, default=4.0,
                   help="[cuda backend] GiB of GPU memory budget for tree storage")
    p.add_argument("--cuda-ucb-c",         type=float, default=2.0,
                   help="[cuda backend] UCB exploration constant")
    p.add_argument("--cuda-verbose",       action="store_true",
                   help="[cuda backend] Print MCTSNC verbose-info each move (slow)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _run_cpu(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    print(f"  workers:        {args.workers}\n")

    tasks = [
        (args.sims, args.temp_threshold, args.seed + i)
        for i in range(args.games)
    ]

    run_parallel_self_play(
        worker_fn   = _worker,
        task_args   = tasks,
        num_games   = args.games,
        workers     = args.workers,
        out_dir     = out_dir,
        chunk_size  = args.chunk_size,
        desc        = "MCTS self-play (cpu)",
        keep_chunks = args.keep_chunks,
    )


def _run_cuda(args: argparse.Namespace) -> None:
    """Single-process CUDA path. The GPU IS the parallelism; mp.Pool would just
    contend for one device, so we drive games sequentially and feed each
    finished game into the same _save_chunk / _merge_chunks pipeline the
    CPU backend uses."""
    from agents.mcts_cuda import play_games_cuda  # imports numba; do it lazily

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  cuda variant:        {args.cuda_variant}")
    print(f"  cuda n_trees:        {args.cuda_n_trees}")
    print(f"  cuda n_playouts:     {args.cuda_n_playouts}")
    print(f"  cuda device memory:  {args.cuda_device_memory:.1f} GiB")
    print(f"  cuda ucb_c:          {args.cuda_ucb_c}\n")

    chunk_paths: list[Path] = []
    chunk_idx     = 0
    obs_buf:    list[np.ndarray] = []
    policy_buf: list[np.ndarray] = []
    value_buf:  list[np.ndarray] = []
    games_done  = 0
    total_pos   = 0

    import time
    from tqdm import tqdm

    start = time.perf_counter()
    pbar  = tqdm(total=args.games, unit="game", desc="MCTS self-play (cuda)")
    try:
        for samples in play_games_cuda(
            num_games      = args.games,
            sims_per_move  = args.sims,
            n_trees        = args.cuda_n_trees,
            n_playouts     = args.cuda_n_playouts,
            variant        = args.cuda_variant,
            device_memory  = args.cuda_device_memory,
            ucb_c          = args.cuda_ucb_c,
            temp_threshold = args.temp_threshold,
            seed           = args.seed,
            verbose        = args.cuda_verbose,
        ):
            obs    = np.stack([s[0] for s in samples])
            policy = np.stack([s[1] for s in samples])
            value  = np.array([s[2] for s in samples], dtype=np.float32)

            obs_buf.append(obs)
            policy_buf.append(policy)
            value_buf.append(value)
            games_done += 1
            total_pos  += len(value)

            elapsed = time.perf_counter() - start
            pbar.set_postfix({
                "games/s":  f"{games_done / max(elapsed, 1e-9):.2f}",
                "positions": f"{total_pos:,}",
            })
            pbar.update(1)

            if len(obs_buf) >= args.chunk_size:
                path = _save_chunk(obs_buf, policy_buf, value_buf, out_dir, chunk_idx)
                chunk_paths.append(path)
                chunk_idx += 1
                obs_buf, policy_buf, value_buf = [], [], []
    finally:
        pbar.close()

    if obs_buf:
        path = _save_chunk(obs_buf, policy_buf, value_buf, out_dir, chunk_idx)
        chunk_paths.append(path)

    print("\nMerging chunks ...")
    out_path = out_dir / "dataset.npz"
    _merge_chunks(chunk_paths, out_path)

    if not args.keep_chunks:
        for p in chunk_paths:
            p.unlink()

    elapsed = time.perf_counter() - start
    print(f"\nFinished in {elapsed:.0f}s  ({elapsed / 3600:.2f}h)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    print(f"Generating {args.games:,} games  [backend={args.backend}]")
    print(f"  sims/move:      {args.sims:,}")
    print(f"  temp threshold: {args.temp_threshold} moves")
    print(f"  chunk size:     {args.chunk_size} games")
    print(f"  output:         {out_dir.resolve()}")

    if args.backend == "cuda":
        _run_cuda(args)
    else:
        _run_cpu(args)


if __name__ == "__main__":
    main()
