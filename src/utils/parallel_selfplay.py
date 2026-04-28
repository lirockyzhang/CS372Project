"""Multiprocessing self-play scaffold shared by ``generate_*_data.py`` scripts.

Both the pure-MCTS and the AlphaGumbel data generators boil down to:

    1. Build a list of per-game task arguments.
    2. Fan them out across a multiprocessing pool.
    3. Append (obs, policy, value) arrays to in-memory buffers.
    4. Periodically flush a chunk to ``chunk_NNNN.npz`` for crash safety.
    5. Merge all chunk files into one ``dataset.npz`` and (optionally) clean up.

Only the worker function differs between scripts, so it is passed in.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm


# A worker takes one opaque task tuple and returns the (obs, policy, value)
# arrays for the game it played.
WorkerFn = Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray]]


def run_parallel_self_play(
    *,
    worker_fn:   WorkerFn,
    task_args:   Iterable[tuple],
    num_games:   int,
    workers:     int,
    out_dir:     str | Path,
    chunk_size:  int = 500,
    desc:        str = "Self-play",
    keep_chunks: bool = False,
) -> Path:
    """Run worker_fn across `workers` processes and merge the results.

    Returns the path to the final ``dataset.npz``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    chunk_idx       = 0
    obs_buf:    list[np.ndarray] = []
    policy_buf: list[np.ndarray] = []
    value_buf:  list[np.ndarray] = []
    games_done = 0
    total_pos  = 0
    start      = time.perf_counter()

    with mp.Pool(workers) as pool:
        with tqdm(total=num_games, unit="game", desc=desc) as pbar:
            for obs, policy, value in pool.imap_unordered(worker_fn, task_args, chunksize=1):
                obs_buf.append(obs)
                policy_buf.append(policy)
                value_buf.append(value)
                games_done += 1
                total_pos  += len(value)

                elapsed = time.perf_counter() - start
                pbar.set_postfix({
                    "games/s":  f"{games_done / max(elapsed, 1e-9):.1f}",
                    "positions": f"{total_pos:,}",
                })
                pbar.update(1)

                if len(obs_buf) >= chunk_size:
                    path = _save_chunk(obs_buf, policy_buf, value_buf, out_dir, chunk_idx)
                    chunk_paths.append(path)
                    chunk_idx += 1
                    obs_buf, policy_buf, value_buf = [], [], []

    # Flush any remaining games.
    if obs_buf:
        path = _save_chunk(obs_buf, policy_buf, value_buf, out_dir, chunk_idx)
        chunk_paths.append(path)

    print("\nMerging chunks ...")
    out_path = out_dir / "dataset.npz"
    _merge_chunks(chunk_paths, out_path)

    if not keep_chunks:
        for p in chunk_paths:
            p.unlink()

    elapsed = time.perf_counter() - start
    print(f"\nFinished in {elapsed:.0f}s  ({elapsed / 3600:.2f}h)")
    return out_path


# ----------------------------------------------------------------------
# Chunk I/O helpers
# ----------------------------------------------------------------------


def _save_chunk(
    obs_buf:    list[np.ndarray],
    policy_buf: list[np.ndarray],
    value_buf:  list[np.ndarray],
    out_dir:    Path,
    chunk_idx:  int,
) -> Path:
    path = out_dir / f"chunk_{chunk_idx:04d}.npz"
    np.savez_compressed(
        path,
        obs    = np.concatenate(obs_buf,    axis=0),
        policy = np.concatenate(policy_buf, axis=0),
        value  = np.concatenate(value_buf,  axis=0),
    )
    return path


def _merge_chunks(chunk_paths: list[Path], out_path: Path) -> None:
    all_obs:    list[np.ndarray] = []
    all_policy: list[np.ndarray] = []
    all_value:  list[np.ndarray] = []
    for p in chunk_paths:
        d = np.load(p)
        all_obs.append(d["obs"])
        all_policy.append(d["policy"])
        all_value.append(d["value"])

    obs    = np.concatenate(all_obs,    axis=0)
    policy = np.concatenate(all_policy, axis=0)
    value  = np.concatenate(all_value,  axis=0)
    np.savez_compressed(out_path, obs=obs, policy=policy, value=value)

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"\nSaved {len(value):,} positions -> {out_path}  ({size_mb:.1f} MB)")
    print(f"  obs:    {obs.shape}  {obs.dtype}")
    print(f"  policy: {policy.shape}  {policy.dtype}")
    print(f"  value:  {value.shape}  {value.dtype}")

    n = len(value)
    wins   = int((value ==  1.0).sum())
    losses = int((value == -1.0).sum())
    draws  = int((value ==  0.0).sum())
    print("\n  Value distribution (from mover's POV):")
    print(f"    wins={wins/n:.1%}  losses={losses/n:.1%}  draws={draws/n:.1%}")
