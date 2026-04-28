"""Quick benchmark to calibrate MCTS self-play speed on this machine.

Run this before data/generate_mcts_data.py to pick --sims and --workers — the
output table estimates how many games / positions you can produce overnight
at various worker counts.

Usage:
    python src/scripts/calibrate_mcts_throughput.py --sims 200 --games 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.mcts.self_play import play_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MCTS self-play speed.")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    parser.add_argument("--games", type=int, default=20, help="Games to benchmark")
    args = parser.parse_args()

    print(f"Benchmarking {args.games} games at {args.sims} sims/move ...")
    print("(single process — multiply by workers for parallel estimate)\n")

    total_positions = 0
    start = time.perf_counter()

    for i in range(args.games):
        samples = play_game(args.sims)
        total_positions += len(samples)
        elapsed = time.perf_counter() - start
        print(
            f"  game {i + 1:>{len(str(args.games))}}/{args.games}"
            f"  positions={len(samples)}"
            f"  elapsed={elapsed:.1f}s",
            end="\r",
        )

    elapsed = time.perf_counter() - start
    games_per_sec = args.games / elapsed
    avg_positions = total_positions / args.games
    positions_per_sec = total_positions / elapsed

    print(f"\n\n--- Results ({args.games} games, {args.sims} sims/move) ---")
    print(f"  Total time:         {elapsed:.1f}s")
    print(f"  Games/sec:          {games_per_sec:.3f}")
    print(f"  Avg positions/game: {avg_positions:.1f}")
    print(f"  Positions/sec:      {positions_per_sec:.1f}")

    print("\n--- Overnight estimates (assumes linear scaling with workers) ---")
    print(f"  {'Hours':>5}  {'Workers':>7}  {'Games':>10}  {'Positions':>12}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*12}")
    for hours in [1, 2, 4, 8]:
        for workers in [1, 4, 8, 12]:
            g = int(games_per_sec * workers * hours * 3600)
            p = int(g * avg_positions)
            print(f"  {hours:>5}h  {workers:>7}w  {g:>10,}  {p:>12,}")
        print()


if __name__ == "__main__":
    main()
