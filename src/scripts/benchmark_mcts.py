"""Benchmark MCTS agent: time per move and time per self-play game.

Usage
-----
    # Defaults: 100 games, 200 rollouts
    python src/scripts/benchmark_mcts.py

    # Custom
    python src/scripts/benchmark_mcts.py --games 50 --sims 800

    # Multiple rollout counts
    python src/scripts/benchmark_mcts.py --sims 100 400 800 --games 30
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.mcts.mcts_agent import MCTSAgent
from env.logic import legal_actions, step
from env.state import initial_state


def _play_game(agent: MCTSAgent) -> tuple[int, int]:
    """Play one full game. Returns (num_moves, winner)."""
    state = initial_state()
    moves = 0
    while not state.terminated:
        action = agent.act(state)
        step(state, action)
        moves += 1
    return moves, state.winner


def benchmark(num_sims: int, num_games: int) -> None:
    agent = MCTSAgent(num_simulations=num_sims)

    move_times:  list[float] = []
    game_times:  list[float] = []
    move_counts: list[int]   = []

    print(f"\n  sims={num_sims}  games={num_games}")
    print(f"  {'game':>5}  {'moves':>6}  {'game_time':>10}  {'ms/move':>9}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}")

    for g in range(num_games):
        state = initial_state()
        moves = 0
        game_start = time.perf_counter()

        while not state.terminated:
            move_start = time.perf_counter()
            action = agent.act(state)
            move_times.append(time.perf_counter() - move_start)
            step(state, action)
            moves += 1

        game_elapsed = time.perf_counter() - game_start
        game_times.append(game_elapsed)
        move_counts.append(moves)

        print(
            f"  {g+1:>5}  {moves:>6}  {game_elapsed:>9.3f}s"
            f"  {1000 * game_elapsed / moves:>8.1f}ms"
        )

    total_moves = sum(move_counts)
    total_time  = sum(game_times)

    avg_game  = total_time / num_games
    avg_move  = total_time / total_moves
    avg_moves = total_moves / num_games

    print()
    print(f"  --- Summary (sims={num_sims}) ---")
    print(f"  Games played      : {num_games}")
    print(f"  Total moves       : {total_moves}")
    print(f"  Avg moves/game    : {avg_moves:.1f}")
    print(f"  Total time        : {total_time:.2f}s")
    print(f"  Avg time/game     : {avg_game:.3f}s  ({1000*avg_game:.1f}ms)")
    print(f"  Avg time/move     : {avg_move*1000:.2f}ms")
    print(f"  Throughput        : {num_games/total_time:.2f} games/s"
          f"  |  {total_moves/total_time:.1f} moves/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS agent speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sims", type=int, nargs="+", default=[200],
        help="MCTS rollouts per move. Pass multiple values to sweep.",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to play per rollout count.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 55)
    print("  MCTS Benchmark")
    print(f"  rollout counts: {args.sims}")
    print(f"  games per count: {args.games}")
    print("=" * 55)

    for sims in args.sims:
        benchmark(num_sims=sims, num_games=args.games)
        print()

    print("=" * 55)
    print("  Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
