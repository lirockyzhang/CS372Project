"""Head-to-head match between two (or more) AlphaZero checkpoints.

Usage
-----
    # One match: agent A vs agent B
    python src/tournament/az_vs_az.py --a runs/.../iter_020_accepted.pt \
                                  --b runs/.../iter_040_accepted.pt

    # Custom sims and game count
    python src/tournament/az_vs_az.py --a A.pt --b B.pt --sims 400 --games 100

    # Round-robin tournament across multiple checkpoints
    python src/tournament/az_vs_az.py --tournament A.pt B.pt C.pt D.pt --games 50 --sims 200

Options
-------
    --a           Path to checkpoint A (head-to-head mode)
    --b           Path to checkpoint B (head-to-head mode)
    --tournament  Two or more checkpoint paths —plays every pair (round-robin)
    --sims        MCTS simulations per move for both agents  (default: 200)
    --sims-a      Override sims for agent A only
    --sims-b      Override sims for agent B only
    --games       Games per match                            (default: 50)
    --leaf-batch  MCTS leaf batch size                      (default: 8)
    --seed        Random seed
    --report      Print progress every N games              (default: 10)
    --device      Torch device (auto-detected if omitted)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from agents.alphazero.mcts import AlphaZeroMCTS
from agents.common.network import AlphaZeroNet
from env.logic import legal_actions, step
from env.state import initial_state


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_agent(
    path: str,
    sims: int,
    leaf_batch: int,
    device: torch.device,
) -> tuple[AlphaZeroMCTS, str]:
    """Load a checkpoint and return (agent, label).

    Label is '<filename> (iter N)' when the checkpoint stores an iteration.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg = ckpt.get("model_config", {})
    channels   = cfg.get("channels",   64)
    num_blocks = cfg.get("num_blocks",  3)

    network = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(device)
    network.load_state_dict(ckpt["network_state"])
    network.eval()

    iteration = ckpt.get("iteration", "?")
    name      = Path(path).stem
    label     = f"{name} (iter {iteration}, {sims} sims)"

    agent = AlphaZeroMCTS(
        network,
        num_simulations=sims,
        batch_size=leaf_batch,
        device=device,
    )
    return agent, label


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _play_game(
    agent_x:      AlphaZeroMCTS,
    agent_o:      AlphaZeroMCTS,
    opening_move: tuple[int, int] | None = None,
) -> int:
    """Play one game. Returns winner: -1 draw, 0 X wins, 1 O wins."""
    state = initial_state()
    if opening_move is not None:
        step(state, opening_move)
    while not state.terminated:
        agent = agent_x if state.current_player == 0 else agent_o
        step(state, agent.act(state))
    return state.winner


def _score(winner: int, a_is_x: bool) -> tuple[float, float, float]:
    """Return (a_score, b_score, draw_flag) for one game result.

    Wins count as 1.0, draws as 0.5 each.
    """
    a_player = 0 if a_is_x else 1
    if winner == -1:
        return 0.5, 0.5, 1.0
    elif winner == a_player:
        return 1.0, 0.0, 0.0
    else:
        return 0.0, 1.0, 0.0


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match(
    agent_a:      AlphaZeroMCTS,
    agent_b:      AlphaZeroMCTS,
    label_a:      str,
    label_b:      str,
    n_games:      int,
    report_every: int,
) -> tuple[float, float, int]:
    """Play *n_games* in side-alternating pairs sharing a random opening.

    Returns (score_a, score_b, draws).
    """
    score_a = score_b = draws = 0.0
    games_done = 0
    t_start = time.time()

    col_w = max(len(label_a), len(label_b), 14)

    def _report() -> None:
        if games_done == 0:
            return
        elapsed = time.time() - t_start
        rate    = games_done / elapsed
        eta_s   = (n_games - games_done) / rate if rate > 0 else 0
        eta_str = f"{eta_s/60:.1f}min" if eta_s >= 60 else f"{eta_s:.0f}s"
        print(
            f"  [{games_done:>{len(str(n_games))}}/{n_games}]"
            f"  {label_a:<{col_w}} {score_a:>5.1f} pts"
            f"  {label_b:<{col_w}} {score_b:>5.1f} pts"
            f"  draws={int(draws)}"
            f"  {rate:.2f} g/s  ETA {eta_str}"
        )

    n_pairs = n_games // 2
    for _ in range(n_pairs):
        opening = random.choice(legal_actions(initial_state()))

        # Game A: agent_a plays X
        w = _play_game(agent_a, agent_b, opening_move=opening)
        sa, sb, d = _score(w, a_is_x=True)
        score_a += sa; score_b += sb; draws += d
        games_done += 1

        # Game B: agent_a plays O (same opening, sides swapped)
        w = _play_game(agent_b, agent_a, opening_move=opening)
        sa, sb, d = _score(w, a_is_x=False)
        score_a += sa; score_b += sb; draws += d
        games_done += 1

        if games_done % report_every == 0:
            _report()

    # Odd game: fresh opening, agent_a as X
    if n_games % 2 == 1:
        opening = random.choice(legal_actions(initial_state()))
        w = _play_game(agent_a, agent_b, opening_move=opening)
        sa, sb, d = _score(w, a_is_x=True)
        score_a += sa; score_b += sb; draws += d
        games_done += 1
        if games_done % report_every == 0:
            _report()

    return score_a, score_b, int(draws)


def _print_match_result(
    label_a: str,
    label_b: str,
    score_a: float,
    score_b: float,
    draws:   int,
    n_games: int,
    elapsed: float,
) -> None:
    col_w = max(len(label_a), len(label_b), 14)
    sep   = "=" * (col_w * 2 + 40)
    print()
    print(sep)
    print("  MATCH RESULT")
    print(sep)
    print(f"  {'Agent':<{col_w}}  {'Score':>6}  {'Win%':>6}  {'Pts/Game':>9}")
    print(f"  {'-'*col_w}  {'-'*6}  {'-'*6}  {'-'*9}")
    for label, score in [(label_a, score_a), (label_b, score_b)]:
        wins = int(score - draws * 0.5)   # approximate integer wins (draws give 0.5)
        print(
            f"  {label:<{col_w}}  {score:>6.1f}  {100*score/n_games:>5.1f}%"
            f"  {score/n_games:>9.3f}"
        )
    print(f"  {'Draws':<{col_w}}  {draws:>6}")
    print(f"  Games: {n_games}  |  Time: {elapsed/60:.1f}min  ({n_games/elapsed:.2f} g/s)")
    print(sep)


# ---------------------------------------------------------------------------
# Round-robin tournament
# ---------------------------------------------------------------------------

def run_tournament(
    agents:       list[tuple[AlphaZeroMCTS, str]],
    n_games:      int,
    report_every: int,
) -> None:
    n = len(agents)
    # Elo-style score table
    scores  = {label: 0.0 for _, label in agents}
    results: list[tuple[str, str, float, float, int, int]] = []

    pairs = list(combinations(range(n), 2))
    print(f"\n  Round-robin: {n} agents, {len(pairs)} matches × {n_games} games each")

    for match_idx, (i, j) in enumerate(pairs, 1):
        agent_a, label_a = agents[i]
        agent_b, label_b = agents[j]

        print(f"\n{'—'*60}")
        print(f"  Match {match_idx}/{len(pairs)}: {label_a}  vs  {label_b}")
        print(f"{'—'*60}")

        t0 = time.time()
        sa, sb, draws = run_match(
            agent_a, agent_b, label_a, label_b, n_games, report_every
        )
        elapsed = time.time() - t0

        _print_match_result(label_a, label_b, sa, sb, draws, n_games, elapsed)
        scores[label_a] += sa
        scores[label_b] += sb
        results.append((label_a, label_b, sa, sb, draws, n_games))

    # Final standings
    max_possible = n_games * (n - 1)  # points available per agent
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print()
    print("=" * 60)
    print("  TOURNAMENT STANDINGS")
    print("=" * 60)
    col_w = max(len(lbl) for lbl in scores) + 2
    print(f"  {'Agent':<{col_w}}  {'Score':>7}  {'Max':>5}  {'%':>6}")
    print(f"  {'-'*col_w}  {'-'*7}  {'-'*5}  {'-'*6}")
    for rank, (label, score) in enumerate(ranked, 1):
        print(
            f"  {rank}. {label:<{col_w-3}}  {score:>7.1f}"
            f"  {max_possible:>5}  {100*score/max_possible:>5.1f}%"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head or round-robin between AlphaZero checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--a", type=str, metavar="CKPT",
                      help="Checkpoint A (head-to-head mode)")
    mode.add_argument("--tournament", type=str, nargs="+", metavar="CKPT",
                      help="Two or more checkpoints for a round-robin tournament")

    parser.add_argument("--b",          type=str, default=None, metavar="CKPT",
                        help="Checkpoint B (required in head-to-head mode)")
    parser.add_argument("--sims",       type=int, default=200,
                        help="MCTS sims per move (both agents)")
    parser.add_argument("--sims-a",     type=int, default=None,
                        help="Override sims for agent A")
    parser.add_argument("--sims-b",     type=int, default=None,
                        help="Override sims for agent B")
    parser.add_argument("--games",      type=int, default=50,
                        help="Games per match")
    parser.add_argument("--leaf-batch", type=int, default=8,
                        help="MCTS leaf batch size")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Random seed")
    parser.add_argument("--report",     type=int, default=10,
                        help="Print progress every N games")
    parser.add_argument("--device",     type=str, default=None,
                        help="Torch device (auto-detected if omitted)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else _detect_device()
    print(f"Device: {device}")

    # ---- Head-to-head mode ----
    if args.a is not None:
        if args.b is None:
            parser.error("--b is required in head-to-head mode")

        sims_a = args.sims_a or args.sims
        sims_b = args.sims_b or args.sims

        agent_a, label_a = _load_agent(args.a, sims_a, args.leaf_batch, device)
        agent_b, label_b = _load_agent(args.b, sims_b, args.leaf_batch, device)

        print()
        print("=" * 60)
        print(f"  {label_a}")
        print(f"  vs")
        print(f"  {label_b}")
        print(f"  {args.games} games")
        print("=" * 60)

        t0 = time.time()
        sa, sb, draws = run_match(
            agent_a, agent_b, label_a, label_b, args.games, args.report
        )
        elapsed = time.time() - t0
        _print_match_result(label_a, label_b, sa, sb, draws, args.games, elapsed)

    # ---- Tournament mode ----
    else:
        if len(args.tournament) < 2:
            parser.error("--tournament requires at least 2 checkpoints")

        agents = [
            _load_agent(path, args.sims, args.leaf_batch, device)
            for path in args.tournament
        ]
        run_tournament(agents, n_games=args.games, report_every=args.report)


if __name__ == "__main__":
    main()
